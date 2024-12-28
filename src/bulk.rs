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
        Arc, RwLock, RwLockReadGuard, RwLockWriteGuard,
    },
};

use memmap2::{Mmap, MmapMut};
use rand::prelude::*;
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

    graph: Vec<RwLock<Vec<Neighbor>>>,
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
        let mut graph = Vec::with_capacity(vectors.len());
        graph.resize_with(vectors.len(), || {
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
            graph,
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

    pub fn init_graph<P>(&self, progress: P) -> Result<()>
    where
        P: Fn() + Send + Sync,
    {
        let ep = (0i64..self.limit as i64)
            .into_par_iter()
            .map_init(
                || thread_rng(),
                |r, i| {
                    let mut edges = self.graph[i as usize].write().unwrap();
                    for v in (0..self.index.config().max_edges.get())
                        .map(|_| r.gen_range(0..self.graph.len() as i64))
                        .filter(|x| *x != i)
                    {
                        edges.push(Neighbor::new(v, 0.0));
                    }
                    let ep_neighbor = Neighbor::new(
                        i,
                        self.scorer.score(&self.vectors[i as usize], &self.centroid),
                    );
                    progress();
                    ep_neighbor
                },
            )
            .max()
            .unwrap();
        self.entry_vertex
            .store(ep.vertex(), atomic::Ordering::SeqCst);
        Ok(())
    }

    /// Insert all vectors from the passed vector store into the graph.
    ///
    /// This operation uses rayon to parallelize large parts of the graph build.
    pub fn insert_all<P>(&self, progress: P) -> Result<()>
    where
        P: Fn() + Send + Sync,
    {
        // XXX this results in a directed graph.
        (0..self.limit)
            .into_par_iter()
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
                    let new_edges = self.search_for_insert(i, &mut searcher, &mut reader)?;
                    // XXX this is not what I intended. i guess a session per query?
                    let mut old_edges = self.graph[i].write().unwrap();
                    old_edges.clear();
                    old_edges.extend_from_slice(&new_edges);
                    drop(old_edges);
                    // Close out the transaction. There should be no conflicts as we did not write to the database.
                    session = reader.into_session();
                    session.rollback_transaction(None)?;
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
