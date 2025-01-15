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
use rayon::prelude::*;
use wt_mdb::{Connection, Record, Result, Session};

use crate::{
    graph::{
        prune_edges, select_pruned_edges, Graph, GraphConfig, GraphVectorIndexReader, GraphVertex,
        NavVectorStore,
    },
    input::{DerefVectorStore, VectorStore},
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

/// Options for bulk loading a data set.
#[derive(Debug, Copy, Clone)]
pub struct Options {
    /// If true, write the quantized vectors into an anonymous memory mapped segment.
    /// This ensures that they will stay in memory (modulo swap) and also provides
    /// faster access than WT.
    pub memory_quantized_vectors: bool,
    /// If true, load vectors into a WT table and use that backing store during build.
    /// This may be faster for dot similarity as the vectors will only be normalized
    /// once on input. This also allows measuring cache efficiency during build when
    /// the data set is larger than available memory.
    pub wt_vector_store: bool,
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

    options: Options,
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
        options: Options,
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
            options,
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
        let quantizer = self.index.config().new_quantizer();
        let mut sum = vec![0.0; dim];
        let mut quantized_vectors = if self.options.memory_quantized_vectors {
            Some(MmapMut::map_anon(quantizer.doc_bytes(dim) * self.vectors.len()).unwrap())
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
                    let quantized = quantizer.for_doc(v);
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
                NonZero::new(quantizer.doc_bytes(dim)).unwrap(),
            )
            .unwrap()
        });
        self.centroid = self
            .scorer
            .normalize_vector(
                sum.into_iter()
                    .map(|s| (s / self.limit as f64) as f32)
                    .collect::<Vec<_>>()
                    .into(),
            )
            .into_owned();
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
        let apply_mu = Mutex::new((0i64, self.scorer.score(&self.get_vector(0), &self.centroid)));
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
                    assert!(
                        !edges.iter().any(|n| n.vertex() == i as i64),
                        "Candidate edges for vertex {} contains self-edge.",
                        i
                    );
                    let centroid_score = self.scorer.score(&self.get_vector(i), &self.centroid);
                    // Add each edge to this vertex and a reciprocal edge to make the graph
                    // undirected. If an edge does not fit on either vertex, save it for later.
                    // We will prune any vertices in this state, but put together the pruned edge
                    // list outside of `apply_mu` to maximize concurrency.
                    loop {
                        let mut entry_point = apply_mu.lock().unwrap();

                        edges.retain(|e| {
                            let (mut iv, mut ev) = self.lock_edge(i, e.vertex() as usize);
                            if iv.len() == iv.capacity() || ev.len() == ev.capacity() {
                                true
                            } else {
                                iv.push(*e);
                                let backedge = Neighbor::new(i as i64, e.score());
                                if !ev.contains(&backedge) {
                                    ev.push(backedge);
                                }
                                false
                            }
                        });

                        if edges.is_empty() {
                            if centroid_score > entry_point.1 {
                                entry_point.0 = i as i64;
                                entry_point.1 = centroid_score;
                                self.entry_vertex.store(i as i64, atomic::Ordering::SeqCst);
                            }
                            break;
                        }

                        drop(entry_point);
                        // Any edge still in the list is overful, so prune it.
                        self.prune_and_apply(i, &apply_mu)?;
                        for e in edges.iter() {
                            self.prune_and_apply(e.vertex() as usize, &apply_mu)?;
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
        P: Fn() + Send + Sync,
    {
        // synchronize application of changes to the graph.
        // this is necessary to ensure the graph remains undirected.
        let apply_mu = Mutex::new(());

        (0..self.limit).into_par_iter().try_for_each(|v| {
            self.prune_and_apply(v, &apply_mu)?;
            progress();
            Ok::<_, wt_mdb::Error>(())
        })
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
                            encode_graph_node(
                                &self.scorer.normalize_vector(v.into()),
                                vertex.iter().map(|n| n.vertex()).collect(),
                            ),
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
        // TODO: return any vectors used for re-ranking here so that we can use them for pruning.
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
        let vertex_vector = self.get_vector(vertex_id);
        let limit = self.index.config().index_search_params.beam_width.get();
        for in_flight_vertex in in_flight.filter(|v| *v != vertex_id) {
            let n = Neighbor::new(
                in_flight_vertex as i64,
                self.scorer
                    .score(&vertex_vector, &self.get_vector(in_flight_vertex)),
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

    fn prune_and_apply<T>(&self, vertex: usize, apply_mu: &Mutex<T>) -> Result<()> {
        // Get the set of edges to prune while only holding a read lock on the vertex.
        let max_edges = self.index.config().max_edges;
        let pruned_edges = {
            let v = self.graph[vertex].read().unwrap();
            if v.len() > max_edges.get() {
                // We copy the contents of the graph because they need to be sorted to prune.
                let mut edges = v.clone();
                drop(v);

                edges.sort_unstable();
                let selected = select_pruned_edges(
                    &edges,
                    max_edges,
                    &mut BulkLoadBuilderGraph(self),
                    self.scorer.as_ref(),
                )?;
                edges
                    .iter()
                    .enumerate()
                    .filter_map(|(i, e)| {
                        if !selected.contains(&i) {
                            Some(e.vertex())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            } else {
                vec![]
            }
        };

        if pruned_edges.is_empty() {
            return Ok(());
        }

        // Apply pruned_edges, starting with the lowest scoring edge.
        let _a = apply_mu.lock().unwrap();
        for e in pruned_edges.into_iter().rev() {
            let (mut iv, mut ev) = self.lock_edge(vertex, e as usize);
            if iv.len() > max_edges.get() || ev.len() > max_edges.get() {
                iv.retain(|n| n.vertex() != e);
                ev.retain(|n| n.vertex() != vertex as i64);
            }
        }
        Ok(())
    }

    // Lock the vertices related to the edge in a consistent order (lowest ord first).
    fn lock_edge(
        &self,
        vertex0: usize,
        vertex1: usize,
    ) -> (
        RwLockWriteGuard<'_, Vec<Neighbor>>,
        RwLockWriteGuard<'_, Vec<Neighbor>>,
    ) {
        assert_ne!(vertex0, vertex1);
        if vertex0 < vertex1 {
            (
                self.graph[vertex0].write().unwrap(),
                self.graph[vertex1].write().unwrap(),
            )
        } else {
            let g1 = self.graph[vertex1].write().unwrap();
            (self.graph[vertex0].write().unwrap(), g1)
        }
    }

    fn get_vector(&self, index: usize) -> Cow<'_, [f32]> {
        self.scorer.normalize_vector(self.vectors[index].into())
    }
}

struct BulkLoadGraphVectorIndexReader<'a, D: Send>(&'a BulkLoadBuilder<D>, Session);

impl<D: Send + Sync> BulkLoadGraphVectorIndexReader<'_, D> {
    fn into_session(self) -> Session {
        self.1
    }
}

impl<D: Send + Sync> GraphVectorIndexReader for BulkLoadGraphVectorIndexReader<'_, D> {
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

impl<D: Send + Sync> Graph for BulkLoadBuilderGraph<'_, D> {
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

impl<D: Send + Sync> GraphVertex for BulkLoadGraphVertex<'_, D> {
    type EdgeIterator<'c>
        = BulkNodeEdgesIterator<'c>
    where
        Self: 'c;

    fn vector(&self) -> Cow<'_, [f32]> {
        self.builder.get_vector(self.vertex_id as usize)
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
