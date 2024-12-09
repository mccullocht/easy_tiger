//! Tools to bulk load an index.
//!
//! For a brand new index we can take a set of vectors and build the graph in memory,
//! then bulk load it into WiredTiger. This is much faster than incrementally building
//! the graph through a series of transactions as there are fewer, simpler abstractions
//! around vector access and graph edge state.
//!
//! Caveats:
//! * Only `numpy` little-endian formatted `f32` vectors are accepted.
//! * Row keys are assigned densely beginning at zero.
use core::f64;
use std::{
    borrow::Cow,
    num::NonZero,
    ops::Range,
    sync::{
        atomic::{self, AtomicI64},
        Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard,
    },
};

use crossbeam_skiplist::SkipSet;
use memmap2::{Mmap, MmapMut};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use wt_mdb::{Connection, Record, Result, Session};

use crate::{
    graph::{prune_edges, Graph, GraphConfig, GraphVectorIndexReader, GraphVertex, NavVectorStore},
    input::{DerefVectorStore, VectorStore},
    quantization::{binary_quantize, binary_quantized_bytes},
    scoring::F32VectorScorer,
    search::GraphSearcher,
    wt::{
        encode_graph_node, CursorNavVectorStore, TableGraphVectorIndex, CONFIG_KEY, ENTRY_POINT_KEY,
    },
    Neighbor,
};

/// Stats for the built graph.
#[derive(Debug)]
pub struct GraphStats {
    /// Number of vertices in the graph.
    pub vertices: usize,
    /// Total number of out edges across all nodes.
    /// The graph is undirected so the actual edge count would be half of this.
    pub edges: usize,
    /// Number of vertices that have no out edges.
    /// The graph is undirected so these nodes also should not have in edges.
    pub unconnected: usize,
}

// TODO: rather than relying on users calling these methods in the right order, instead
// consume/perform/generate, e.g.:
//
// pub fn QuantizedVectorLoader::new(...) -> Self
// pub fn QuantizedVectorLoader::load(self, progress_cb) -> Result<BulkGraphBuilder>
// pub fn BulkGraphBuilder::build(self, progress_cb) -> Result<BulkGraphPruner>
// pub fn BulkGraphPruner::load(self, progress_cb) -> Result<()>
//
// if you do this the graph build can be locked, but then transform into a representation that
// is better for single-threaded access for cleanup.

// TODO: tests for bulk load builder, mostly the built graph.

/// Builds a Vamana graph for a bulk load.
pub struct BulkLoadBuilder<D> {
    connection: Arc<Connection>,
    index: TableGraphVectorIndex,
    limit: usize,

    vectors: DerefVectorStore<f32, D>,
    centroid: Vec<f32>,

    memory_quantized_vectors: bool,
    quantized_vectors: Option<DerefVectorStore<u8, Mmap>>,

    graph: Box<[RwLock<Vec<Neighbor>>]>,
    entry_vertex: AtomicI64,
    scorer: Box<dyn F32VectorScorer>,
}

impl<D> BulkLoadBuilder<D>
where
    D: Send + Sync,
{
    /// Create a new bulk graph builder with the passed vector set and configuration.
    /// `limit` limits the number of vectors processed to less than the full set.
    pub fn new(
        connection: Arc<Connection>,
        index: TableGraphVectorIndex,
        vectors: DerefVectorStore<f32, D>,
        memory_quantized_vectors: bool,
        limit: usize,
    ) -> Self {
        let mut graph_vec = Vec::with_capacity(vectors.len());
        graph_vec.resize_with(vectors.len(), || {
            RwLock::new(Vec::with_capacity(index.config().max_edges.get() * 2))
        });
        let scorer = index.config().new_scorer();
        Self {
            connection,
            index,
            limit,
            vectors,
            centroid: Vec::new(),
            memory_quantized_vectors,
            quantized_vectors: None,
            graph: graph_vec.into_boxed_slice(),
            entry_vertex: AtomicI64::new(-1),
            scorer,
        }
    }

    /// Load binary quantized vector data into the nav vectors table.
    pub fn load_nav_vectors<P: Fn()>(&mut self, progress: P) -> Result<()> {
        let session = self.connection.open_session()?;
        let dim = self.index.config().dimensions.get();
        let mut sum = vec![0.0; dim];
        let mut quantized_vectors = if self.memory_quantized_vectors {
            Some(MmapMut::map_anon(binary_quantized_bytes(dim) * self.vectors.len()).unwrap())
        } else {
            None
        };
        session.bulk_load(
            self.index.nav_table_name(),
            None,
            self.vectors
                .iter()
                .enumerate()
                .take(self.limit)
                .map(|(i, v)| {
                    progress();
                    for (i, o) in v.iter().zip(sum.iter_mut()) {
                        *o += *i as f64;
                    }
                    let quantized = binary_quantize(v);
                    if let Some(q) = quantized_vectors.as_mut() {
                        let start = i * quantized.len();
                        q[start..(start + quantized.len())].copy_from_slice(&quantized);
                    }
                    Record::new(i as i64, quantized)
                }),
        )?;

        self.quantized_vectors = quantized_vectors.map(|m| {
            DerefVectorStore::new(
                m.make_read_only().unwrap(),
                NonZero::new(binary_quantized_bytes(dim)).unwrap(),
            )
            .unwrap()
        });
        self.centroid = sum
            .into_iter()
            .map(|s| (s / self.limit as f64) as f32)
            .collect();
        self.scorer.normalize(&mut self.centroid);
        Ok(())
    }

    /// Insert all vectors from the passed vector store into the graph.
    ///
    /// This operation uses rayon to parallelize large parts of the graph build.
    pub fn insert_all<P>(&self, progress: P) -> Result<()>
    where
        P: Fn() + Send + Sync,
    {
        // apply_mu is used to ensure that only one thread is mutating the graph at a time so we can maintain
        // the "undirected graph" invariant. apply_mu contains the entry point vertex id and score against the
        // centroid, we may update this if we find a closer point then reflect it back into entry_vertex.
        let apply_mu = Mutex::new((0i64, self.scorer.score(&self.vectors[0], &self.centroid)));
        self.entry_vertex.store(0, atomic::Ordering::SeqCst);

        // Keep track of all in-flight concurrent insertions. These nodes will be processed at the
        // end of each insertion search to ensure that we are linking these nodes when appropriate as those
        // links would not be generated by a search of the graph.
        let in_flight = SkipSet::new();
        (0..self.limit)
            .into_par_iter()
            .skip(1) // vertex 0 has been implicitly inserted.
            .chunks(1_000)
            .try_for_each(|nodes| {
                // NB: we create a new session and cursor for each chunk. Relevant rayon APIs require these
                // objects to be Send + Sync, but Session is only Send and wrapping it in a Mutex does not
                // work because any RecordCursor objects returned have to be destroyed before the Mutex is
                // released.
                let mut session = self.connection.open_session()?;
                let mut searcher = GraphSearcher::new(self.index.config().index_search_params);
                for i in nodes {
                    // Use a transaction for each search. Without this each lookup will be a separate transaction
                    // which obtains a reader lock inside the session. Overhead for that is ~10x.
                    session.begin_transaction(None)?;
                    let mut reader = BulkLoadGraphVectorIndexReader(self, session);
                    in_flight.insert(i);
                    let mut edges = self.search_for_insert(i, &mut searcher, &mut reader)?;
                    // Insert any other in-flight edges into the candidate queue. These are vertices we may have missed because
                    // they are being inserted concurrently in another thread.
                    self.insert_in_flight_edges(i, in_flight.iter().map(|e| *e), &mut edges);
                    let centroid_score = self.scorer.score(&self.vectors[i], &self.centroid);
                    {
                        let mut entry_point = apply_mu.lock().unwrap();
                        self.apply_insert(i, edges)?;
                        if centroid_score > entry_point.1 {
                            entry_point.0 = i as i64;
                            entry_point.1 = centroid_score;
                            self.entry_vertex.store(i as i64, atomic::Ordering::SeqCst);
                        }
                    }
                    // Close out the transaction. There should be no conflicts as we did not write to the database.
                    session = reader.into_session();
                    session.rollback_transaction(None)?;
                    in_flight.remove(&i);
                    progress();
                }

                Ok::<(), wt_mdb::Error>(())
            })
    }

    /// Cleanup the graph.
    ///
    /// This may prune edges and/or ensure graph connectivity.
    pub fn cleanup<P>(&self, progress: P) -> Result<()>
    where
        P: Fn(),
    {
        // NB: this must not be run concurrently so that we can ensure edges are reciprocal.
        for (i, n) in self.graph.iter().enumerate().take(self.limit) {
            self.maybe_prune_node(i, n.write().unwrap(), self.index.config().max_edges)?;
            progress();
        }
        Ok(())
    }

    /// Bulk load the graph table with raw vectors and graph edges.
    ///
    /// Returns statistics about the generated graph.
    pub fn load_graph<P>(&self, progress: P) -> Result<GraphStats>
    where
        P: Fn(),
    {
        let mut stats = GraphStats {
            vertices: 0,
            edges: 0,
            unconnected: 0,
        };
        let config_rows = vec![
            Record::new(
                CONFIG_KEY,
                serde_json::to_vec(&self.index.config()).unwrap(),
            ),
            Record::new(
                ENTRY_POINT_KEY,
                self.entry_vertex
                    .load(atomic::Ordering::Relaxed)
                    .to_le_bytes()
                    .to_vec(),
            ),
        ];
        let session = self.connection.open_session()?;
        session.bulk_load(
            self.index.graph_table_name(),
            None,
            config_rows.into_iter().chain(
                self.vectors
                    .iter()
                    .zip(self.graph.iter())
                    .enumerate()
                    .take(self.limit)
                    .map(|(i, (v, n))| {
                        progress();
                        let vertex = n.read().unwrap();
                        stats.vertices += 1;
                        stats.edges += vertex.len();
                        if vertex.is_empty() {
                            stats.unconnected += 1;
                        }
                        Record::new(
                            i as i64,
                            encode_graph_node(v, vertex.iter().map(|n| n.vertex()).collect()),
                        )
                    }),
            ),
        )?;
        Ok(stats)
    }

    fn search_for_insert(
        &self,
        vertex_id: usize,
        searcher: &mut GraphSearcher,
        reader: &mut BulkLoadGraphVectorIndexReader<'_, D>,
    ) -> Result<Vec<Neighbor>> {
        let mut graph = BulkLoadBuilderGraph(self);
        let mut candidates = searcher.search_for_insert(vertex_id as i64, reader)?;
        let split = prune_edges(
            &mut candidates,
            self.index.config().max_edges,
            &mut graph,
            self.scorer.as_ref(),
        )?;
        candidates.truncate(split);
        Ok(candidates)
    }

    fn insert_in_flight_edges(
        &self,
        vertex_id: usize,
        in_flight: impl Iterator<Item = usize>,
        edges: &mut Vec<Neighbor>,
    ) {
        let limit = self.index.config().index_search_params.beam_width.get();
        for in_flight_vertex in in_flight.filter(|v| *v != vertex_id) {
            let n = Neighbor::new(
                in_flight_vertex as i64,
                self.scorer
                    .score(&self.vectors[vertex_id], &self.vectors[in_flight_vertex]),
            );
            // If the queue is full and n is worse than all other edges, skip.
            if edges.len() >= limit && n >= *edges.last().unwrap() {
                continue;
            }

            if let Err(index) = edges.binary_search(&n) {
                if edges.len() >= limit {
                    edges.pop();
                }
                edges.insert(index, n);
            }
        }
    }

    /// This function is the only mutator of self.graph and must not be run concurrently.
    fn apply_insert(&self, index: usize, edges: Vec<Neighbor>) -> Result<()> {
        assert!(
            !edges.iter().any(|n| n.vertex() == index as i64),
            "Candidate edges for vertex {} contains self-edge.",
            index
        );
        self.graph[index].write().unwrap().extend_from_slice(&edges);
        for e in edges.iter() {
            let mut guard = self.graph[e.vertex() as usize].write().unwrap();
            guard.push(Neighbor::new(index as i64, e.score()));
            let max_edges = NonZero::new(guard.capacity() - 1).unwrap();
            self.maybe_prune_node(index, guard, max_edges)?;
        }
        Ok(())
    }

    fn maybe_prune_node(
        &self,
        index: usize,
        mut guard: RwLockWriteGuard<'_, Vec<Neighbor>>,
        max_edges: NonZero<usize>,
    ) -> Result<()> {
        if guard.len() <= max_edges.get() {
            return Ok(());
        }

        guard.sort();
        let split = prune_edges(
            &mut guard,
            self.index.config().max_edges,
            &mut BulkLoadBuilderGraph(self),
            self.scorer.as_ref(),
        )?;
        let dropped = guard.split_off(split);
        drop(guard);

        // Remove in-links from nodes that we dropped out-links to.
        // If we maintain the invariant that all links are reciprocated then it will be easier
        // to mutate the index without requiring a cleaning process.
        for n in dropped {
            self.graph[n.vertex() as usize]
                .write()
                .unwrap()
                .retain(|e| e.vertex() != index as i64);
        }
        Ok(())
    }
}

struct BulkLoadGraphVectorIndexReader<'a, D: Send>(&'a BulkLoadBuilder<D>, Session);

impl<D> BulkLoadGraphVectorIndexReader<'_, D>
where
    D: Send,
{
    fn into_session(self) -> Session {
        self.1
    }
}

impl<D> GraphVectorIndexReader for BulkLoadGraphVectorIndexReader<'_, D>
where
    D: Send,
{
    type Graph<'b>
        = BulkLoadBuilderGraph<'b, D>
    where
        Self: 'b;
    type NavVectorStore<'b>
        = BulkLoadNavVectorStore<'b>
    where
        Self: 'b;

    fn config(&self) -> &GraphConfig {
        self.0.index.config()
    }

    fn graph(&self) -> Result<Self::Graph<'_>> {
        Ok(BulkLoadBuilderGraph(self.0))
    }

    fn nav_vectors(&self) -> Result<Self::NavVectorStore<'_>> {
        if let Some(s) = self.0.quantized_vectors.as_ref() {
            Ok(BulkLoadNavVectorStore::Memory(s))
        } else {
            Ok(BulkLoadNavVectorStore::Cursor(CursorNavVectorStore::new(
                self.1.get_record_cursor(self.0.index.nav_table_name())?,
            )))
        }
    }
}

struct BulkLoadBuilderGraph<'a, D: Send>(&'a BulkLoadBuilder<D>);

impl<D> Graph for BulkLoadBuilderGraph<'_, D>
where
    D: Send,
{
    type Vertex<'c>
        = BulkLoadGraphVertex<'c, D>
    where
        Self: 'c;

    fn entry_point(&mut self) -> Option<Result<i64>> {
        let vertex = self.0.entry_vertex.load(atomic::Ordering::Relaxed);
        if vertex >= 0 {
            Some(Ok(vertex))
        } else {
            None
        }
    }

    fn get(&mut self, vertex_id: i64) -> Option<Result<Self::Vertex<'_>>> {
        Some(Ok(BulkLoadGraphVertex {
            builder: self.0,
            vertex_id,
        }))
    }
}

struct BulkLoadGraphVertex<'a, D> {
    builder: &'a BulkLoadBuilder<D>,
    vertex_id: i64,
}

impl<D> GraphVertex for BulkLoadGraphVertex<'_, D> {
    type EdgeIterator<'c>
        = BulkNodeEdgesIterator<'c>
    where
        Self: 'c;

    fn vector(&self) -> Cow<'_, [f32]> {
        self.builder.vectors[self.vertex_id as usize].into()
    }

    fn edges(&self) -> Self::EdgeIterator<'_> {
        BulkNodeEdgesIterator::new(self.builder.graph[self.vertex_id as usize].read().unwrap())
    }
}

struct BulkNodeEdgesIterator<'a> {
    guard: RwLockReadGuard<'a, Vec<Neighbor>>,
    range: Range<usize>,
}

impl<'a> BulkNodeEdgesIterator<'a> {
    fn new(guard: RwLockReadGuard<'a, Vec<Neighbor>>) -> Self {
        let len = guard.len();
        Self {
            guard,
            range: 0..len,
        }
    }
}

impl Iterator for BulkNodeEdgesIterator<'_> {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(|i| self.guard[i].vertex())
    }
}

enum BulkLoadNavVectorStore<'a> {
    Cursor(CursorNavVectorStore<'a>),
    Memory(&'a DerefVectorStore<u8, memmap2::Mmap>),
}

impl NavVectorStore for BulkLoadNavVectorStore<'_> {
    fn get(&mut self, vertex_id: i64) -> Option<Result<Cow<'_, [u8]>>> {
        match self {
            Self::Cursor(c) => c.get(vertex_id),
            Self::Memory(m) => {
                if vertex_id >= 0 && (vertex_id as usize) < m.len() {
                    Some(Ok(m[vertex_id as usize].into()))
                } else {
                    None
                }
            }
        }
    }
}
