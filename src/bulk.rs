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
    cell::RefCell,
    num::NonZero,
    ops::{Deref, Range},
    sync::{
        atomic::{self, AtomicI64},
        Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard,
    },
};

use crossbeam_skiplist::SkipSet;
use memmap2::{Mmap, MmapMut};
use rayon::prelude::*;
use thread_local::ThreadLocal;
use wt_mdb::{options::DropOptionsBuilder, Connection, Record, Result, Session};

use crate::{
    graph::{
        prune_edges, select_pruned_edges, Graph, GraphConfig, GraphLayout, GraphVectorIndexReader,
        GraphVertex, NavVectorStore, RawVectorStore,
    },
    input::{DerefVectorStore, VectorStore},
    scoring::F32VectorScorer,
    search::GraphSearcher,
    wt::{
        encode_graph_vertex, encode_raw_vector, CursorGraph, CursorNavVectorStore,
        TableGraphVectorIndex, CONFIG_KEY, ENTRY_POINT_KEY,
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

#[derive(Debug, Copy, Clone)]
pub enum BulkLoadPhase {
    LoadNavVectors,
    LoadRawVectors,
    BuildGraph,
    CleanupGraph,
    LoadGraph,
}

impl BulkLoadPhase {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::LoadNavVectors => "load nav vectors",
            Self::LoadRawVectors => "load raw vectors",
            Self::BuildGraph => "build graph",
            Self::CleanupGraph => "cleanup graph",
            Self::LoadGraph => "load graph",
        }
    }
}

// TODO: this should use some sort of state machine pattern.

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

    graph_stats: Option<GraphStats>,
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
            graph_stats: None,
        }
    }

    /// Phases to be executed by the builder.
    /// This can vary depending on the options.
    pub fn phases(&self) -> Vec<BulkLoadPhase> {
        if self.options.wt_vector_store || self.index.config().layout == GraphLayout::Split {
            vec![
                BulkLoadPhase::LoadNavVectors,
                BulkLoadPhase::LoadRawVectors,
                BulkLoadPhase::BuildGraph,
                BulkLoadPhase::CleanupGraph,
                BulkLoadPhase::LoadGraph,
            ]
        } else {
            vec![
                BulkLoadPhase::LoadNavVectors,
                BulkLoadPhase::BuildGraph,
                BulkLoadPhase::CleanupGraph,
                BulkLoadPhase::LoadGraph,
            ]
        }
    }

    /// Total number of vectors to process. Useful for status reporting.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.limit
    }

    /// Execute a single named phase, calling progress() as each vector is processed.
    pub fn execute_phase<P>(&mut self, phase: BulkLoadPhase, progress: P) -> Result<()>
    where
        P: Fn() + Send + Sync,
    {
        match phase {
            BulkLoadPhase::LoadNavVectors => self.load_nav_vectors(progress),
            BulkLoadPhase::LoadRawVectors => self.load_raw_vectors(progress),
            BulkLoadPhase::BuildGraph => self.insert_all(progress),
            BulkLoadPhase::CleanupGraph => self.cleanup(progress),
            BulkLoadPhase::LoadGraph => {
                self.graph_stats = Some(self.load_graph(progress)?);
                Ok(())
            }
        }
    }

    /// Return graph stats after execution has completed.
    pub fn graph_stats(&self) -> Option<&GraphStats> {
        self.graph_stats.as_ref()
    }

    /// Load binary quantized vector data into the nav vectors table.
    fn load_nav_vectors<P: Fn()>(&mut self, progress: P) -> Result<()> {
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

    fn load_raw_vectors<P: Fn() + Send + Sync>(&mut self, progress: P) -> Result<()> {
        let session = self.connection.open_session()?;
        session.bulk_load(
            self.index.raw_table_name(),
            None,
            self.vectors
                .iter()
                .enumerate()
                .take(self.limit)
                .map(|(i, v)| {
                    let normalized = self.scorer.normalize_vector(v.into());
                    let value = encode_raw_vector(&normalized);
                    progress();
                    Record::new(i as i64, value)
                }),
        )
    }

    /// Insert all vectors from the passed vector store into the graph.
    ///
    /// This operation uses rayon to parallelize large parts of the graph build.
    fn insert_all<P>(&self, progress: P) -> Result<()>
    where
        P: Fn() + Send + Sync,
    {
        // apply_mu is used to ensure that only one thread is mutating the graph at a time so we can maintain
        // the "undirected graph" invariant. apply_mu contains the entry point vertex id and score against the
        // centroid, we may update this if we find a closer point then reflect it back into entry_vertex.
        let apply_mu = Mutex::new((
            0i64,
            self.scorer.distance(&self.get_vector(0), &self.centroid),
        ));
        self.entry_vertex.store(0, atomic::Ordering::SeqCst);

        // Use thread locals to avoid recreating Session and GraphSearcher per vector. Rayon
        // doesn't provide a good way to initialize something falliable once per thread, and the
        // alternative is chunking which limits work-stealing.
        let tl_session = ThreadLocalSession::new(self.connection.clone());
        let tl_searcher = ThreadLocal::new();

        // Keep track of all in-flight concurrent insertions. These nodes will be processed at the
        // end of each insertion search to ensure that we are linking these nodes when appropriate as those
        // links would not be generated by a search of the graph.
        let in_flight = SkipSet::new();
        (0..self.limit)
            .into_par_iter()
            .skip(1) // vertex 0 has been implicitly inserted.
            .try_for_each(|v| {
                let session = tl_session.get()?;
                let mut searcher = tl_searcher
                    .get_or(|| {
                        RefCell::new(GraphSearcher::new(self.index.config().index_search_params))
                    })
                    .borrow_mut();
                // Use a transaction for each search. Without this each lookup will be a separate transaction
                // which obtains a reader lock inside the session. Overhead for that is ~10x.
                // TODO: add session.do_in_transaction() or similar to avoid getting session into a bad state.
                session.begin_transaction(None)?;
                let mut reader = BulkLoadGraphVectorIndexReader(self, &session);
                in_flight.insert(v);
                let mut edges = self.search_for_insert(v, &mut searcher, &mut reader)?;
                // Insert any other in-flight edges into the candidate queue. These are vertices we may have missed because
                // they are being inserted concurrently in another thread.
                self.insert_in_flight_edges(v, in_flight.iter().map(|e| *e), &mut edges);
                assert!(
                    !edges.iter().any(|n| n.vertex() == v as i64),
                    "Candidate edges for vertex {} contains self-edge.",
                    v
                );

                let mut raw_vectors = reader.raw_vectors()?;
                let centroid_score = self.scorer.distance(
                    &raw_vectors.get_raw_vector(v as i64).unwrap().unwrap(),
                    &self.centroid,
                );
                drop(raw_vectors);

                // Add each edge to this vertex and a reciprocal edge to make the graph
                // undirected. If an edge does not fit on either vertex, save it for later.
                // We will prune any vertices in this state, but put together the pruned edge
                // list outside of `apply_mu` to maximize concurrency.
                loop {
                    let mut entry_point = apply_mu.lock().unwrap();

                    edges.retain(|e| {
                        let (mut iv, mut ev) = self.lock_edge(v, e.vertex() as usize);
                        if iv.len() == iv.capacity() || ev.len() == ev.capacity() {
                            true
                        } else {
                            iv.push(*e);
                            let backedge = Neighbor::new(v as i64, e.distance());
                            if !ev.contains(&backedge) {
                                ev.push(backedge);
                            }
                            false
                        }
                    });

                    if edges.is_empty() {
                        if centroid_score > entry_point.1 {
                            entry_point.0 = v as i64;
                            entry_point.1 = centroid_score;
                            self.entry_vertex.store(v as i64, atomic::Ordering::SeqCst);
                        }
                        break;
                    }

                    drop(entry_point);
                    // Any edge still in the list is overful, so prune it.
                    self.prune_and_apply(v, &mut reader, &apply_mu)?;
                    for e in edges.iter() {
                        self.prune_and_apply(e.vertex() as usize, &mut reader, &apply_mu)?;
                    }
                }

                // Close out the transaction. There should be no conflicts as we did not write to the database.
                session.rollback_transaction(None)?;
                in_flight.remove(&v);
                progress();

                Ok::<(), wt_mdb::Error>(())
            })
    }

    /// Cleanup the graph.
    ///
    /// This may prune edges and/or ensure graph connectivity.
    fn cleanup<P>(&self, progress: P) -> Result<()>
    where
        P: Fn() + Send + Sync,
    {
        // synchronize application of changes to the graph.
        // this is necessary to ensure the graph remains undirected.
        let apply_mu = Mutex::new(());

        let tl_session = ThreadLocalSession::new(self.connection.clone());

        (0..self.limit).into_par_iter().try_for_each(|v| {
            let session = tl_session.get()?;
            session.begin_transaction(None)?;
            let mut reader = BulkLoadGraphVectorIndexReader(self, &session);
            self.prune_and_apply(v, &mut reader, &apply_mu)?;
            session.rollback_transaction(None)?;
            progress();
            Ok::<_, wt_mdb::Error>(())
        })
    }

    /// Bulk load the graph table with raw vectors and graph edges.
    ///
    /// Returns statistics about the generated graph.
    fn load_graph<P>(&self, progress: P) -> Result<GraphStats>
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
        if self.options.wt_vector_store {
            session.drop_table(
                self.index.graph_table_name(),
                Some(DropOptionsBuilder::default().set_force().into()),
            )?;
        }
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
                        let vertex_vector =
                            if self.index.config().layout == GraphLayout::RawVectorInGraph {
                                Some(self.scorer.normalize_vector(v.into()))
                            } else {
                                None
                            };
                        Record::new(
                            i as i64,
                            encode_graph_vertex(
                                vertex.iter().map(|n| n.vertex()).collect(),
                                vertex_vector.as_ref().map(|vv| vv.as_ref()),
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
        reader: &mut BulkLoadGraphVectorIndexReader<'_, '_, D>,
    ) -> Result<Vec<Neighbor>> {
        // TODO: return any vectors used for re-ranking here so that we can use them for pruning.
        let mut candidates = searcher.search_for_insert(vertex_id as i64, reader)?;
        let mut graph = reader.graph()?;
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
                    .distance(&vertex_vector, &self.get_vector(in_flight_vertex)),
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

    fn prune_and_apply<T>(
        &self,
        vertex: usize,
        reader: &mut BulkLoadGraphVectorIndexReader<'_, '_, D>,
        apply_mu: &Mutex<T>,
    ) -> Result<()> {
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
                    &mut reader.graph()?,
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

// TODO: move this, it could be useful elsewhere.
struct ThreadLocalSession {
    connection: Arc<Connection>,
    tl_session: ThreadLocal<Session>,
}

impl ThreadLocalSession {
    fn new(connection: Arc<Connection>) -> Self {
        ThreadLocalSession {
            connection,
            tl_session: ThreadLocal::new(),
        }
    }

    fn get(&self) -> Result<SessionGuard<'_>> {
        self.tl_session
            .get_or_try(|| self.connection.open_session())
            .map(SessionGuard)
    }
}

struct SessionGuard<'a>(&'a Session);

impl Deref for SessionGuard<'_> {
    type Target = Session;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl Drop for SessionGuard<'_> {
    fn drop(&mut self) {
        let _ = self.0.reset();
    }
}

struct BulkLoadGraphVectorIndexReader<'a, 'b, D: Send>(&'a BulkLoadBuilder<D>, &'b Session);

impl<D: Send + Sync> GraphVectorIndexReader for BulkLoadGraphVectorIndexReader<'_, '_, D> {
    type Graph<'b>
        = BulkLoadBuilderGraph<'b, D>
    where
        Self: 'b;
    type RawVectorStore<'b>
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
        Ok(BulkLoadBuilderGraph(self.0, None))
    }

    fn raw_vectors(&self) -> Result<Self::RawVectorStore<'_>> {
        let cursor_graph = if self.0.options.wt_vector_store {
            Some(CursorGraph::new(
                *self.0.index.config(),
                self.1.get_record_cursor(self.0.index.raw_table_name())?,
            ))
        } else {
            None
        };
        Ok(BulkLoadBuilderGraph(self.0, cursor_graph))
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

struct BulkLoadBuilderGraph<'a, D: Send>(&'a BulkLoadBuilder<D>, Option<CursorGraph<'a>>);

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

    fn get_vertex(&mut self, vertex_id: i64) -> Option<Result<Self::Vertex<'_>>> {
        Some(Ok(BulkLoadGraphVertex {
            builder: self.0,
            vertex_id,
        }))
    }
}

impl<D: Send + Sync> RawVectorStore for BulkLoadBuilderGraph<'_, D> {
    fn get_raw_vector(&mut self, vertex_id: i64) -> Option<Result<crate::graph::RawVector<'_>>> {
        if let Some(cursor) = self.1.as_mut() {
            cursor.get_raw_vector(vertex_id)
        } else {
            Some(Ok(self.0.get_vector(vertex_id as usize).into()))
        }
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

    fn vector(&self) -> Option<Cow<'_, [f32]>> {
        None
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
    fn get_nav_vector(&mut self, vertex_id: i64) -> Option<Result<Cow<'_, [u8]>>> {
        match self {
            Self::Cursor(c) => c.get_nav_vector(vertex_id),
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
