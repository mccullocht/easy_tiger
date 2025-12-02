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
use rustix::io::Errno;
use thread_local::ThreadLocal;
use vectors::{F32VectorCoding, VectorSimilarity};
use wt_mdb::{options::CreateOptionsBuilder, Connection, Error, Result, Session};

use crate::{
    graph_clustering,
    input::{DerefVectorStore, SubsetViewVectorStore, VectorStore},
    vamana::search::GraphSearcher,
    vamana::wt::{encode_graph_vertex, CursorVectorStore, TableGraphVectorIndex, ENTRY_POINT_KEY},
    vamana::{
        prune_edges, select_pruned_edges, EdgeSetDistanceComputer, Graph, GraphConfig,
        GraphVectorIndex, GraphVectorStore,
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
#[derive(Debug, Default, Copy, Clone)]
pub struct Options {
    /// If true, write the quantized vectors into an anonymous memory mapped segment.
    /// This ensures that they will stay in memory (modulo swap) and also provides
    /// faster access than WT.
    pub memory_quantized_vectors: bool,
    /// If true, cluster the input data set to choose insertion order. This improves locality
    /// during the insertion step, yielding higher cache hit rates and graph build times, at the
    /// expense of a compute intensive k-means clustering step.
    pub cluster_ordered_insert: bool,
}

#[derive(Debug, Copy, Clone)]
pub enum BulkLoadPhase {
    // TODO: combine vector loading phases now that bulk_load APIs have been refactored to allow
    // this in a useful/meaningful way.
    LoadVectors,
    ClusterVectors,
    BuildGraph,
    CleanupGraph,
    LoadGraph,
}

impl BulkLoadPhase {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::LoadVectors => "load vectors",
            Self::ClusterVectors => "cluster vectors",
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

    vectors: D,
    centroid: Vec<u8>,

    options: Options,
    quantized_vectors: Option<DerefVectorStore<u8, Mmap>>,
    clustered_order: Option<Vec<usize>>,

    graph: Box<[RwLock<Vec<Neighbor>>]>,
    entry_vertex: AtomicI64,

    graph_stats: Option<GraphStats>,
}

impl<D> BulkLoadBuilder<D>
where
    D: VectorStore<Elem = f32> + Send + Sync,
{
    /// Create a new bulk graph builder with the passed vector set and configuration.
    /// `limit` limits the number of vectors processed to less than the full set.
    pub fn new(
        connection: Arc<Connection>,
        index: TableGraphVectorIndex,
        vectors: D,
        options: Options,
        limit: usize,
    ) -> Self {
        let mut graph_vec = Vec::with_capacity(vectors.len());
        graph_vec.resize_with(vectors.len(), || {
            RwLock::new(Vec::with_capacity(
                index.config().pruning.max_edges.get() * 2,
            ))
        });
        Self {
            connection,
            index,
            limit,
            vectors,
            centroid: Vec::new(),
            options,
            quantized_vectors: None,
            clustered_order: None,
            graph: graph_vec.into_boxed_slice(),
            entry_vertex: AtomicI64::new(-1),
            graph_stats: None,
        }
    }

    /// Phases to be executed by the builder.
    /// This can vary depending on the options.
    pub fn phases(&self) -> Vec<BulkLoadPhase> {
        let mut phases = vec![BulkLoadPhase::LoadVectors];
        if self.options.cluster_ordered_insert {
            phases.push(BulkLoadPhase::ClusterVectors);
        }
        phases.extend_from_slice(&[
            BulkLoadPhase::BuildGraph,
            BulkLoadPhase::CleanupGraph,
            BulkLoadPhase::LoadGraph,
        ]);
        phases
    }

    /// Total number of vectors to process. Useful for status reporting.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.limit
    }

    /// Execute a single named phase, calling progress() as each vector is processed.
    pub fn execute_phase<P>(&mut self, phase: BulkLoadPhase, progress: P) -> Result<()>
    where
        P: Fn(u64) + Send + Sync,
    {
        match phase {
            BulkLoadPhase::LoadVectors => self.load_vectors(progress),
            BulkLoadPhase::ClusterVectors => self.cluster_vectors(progress),
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

    /// Load nav and rerank vectors into tables.
    fn load_vectors<P: Fn(u64)>(&mut self, progress: P) -> Result<()> {
        let session = self.connection.open_session()?;
        let dim = self.index.config().dimensions.get();
        let nav_coder = self.index.nav_table().new_coder();
        let mut nav_vector = vec![0u8; nav_coder.byte_len(dim)];
        let mut sum = vec![0.0; dim];
        let mut quantized_vectors = if self.options.memory_quantized_vectors {
            Some(MmapMut::map_anon(nav_coder.byte_len(dim) * self.vectors.len()).unwrap())
        } else {
            None
        };
        let mut nav_cursor =
            session.new_bulk_load_cursor::<i64, Vec<u8>>(self.index.nav_table().name(), None)?;

        let mut rerank = if let Some(rerank_table) = self.index.rerank_table() {
            let rerank_coder = rerank_table.new_coder();
            let rerank_vector = vec![0u8; rerank_coder.byte_len(dim)];
            let rerank_cursor =
                session.new_bulk_load_cursor::<i64, Vec<u8>>(rerank_table.name(), None)?;
            Some((rerank_coder, rerank_vector, rerank_cursor))
        } else {
            None
        };

        for (i, v) in self.vectors.iter().enumerate().take(self.limit) {
            for (i, o) in v.iter().zip(sum.iter_mut()) {
                *o += *i as f64;
            }
            nav_coder.encode_to(v, &mut nav_vector);
            if let Some(q) = quantized_vectors.as_mut() {
                let start = i * nav_vector.len();
                q[start..(start + nav_vector.len())].copy_from_slice(&nav_vector);
            }
            nav_cursor.insert(i as i64, &nav_vector)?;

            if let Some((coder, vector, cursor)) = rerank.as_mut() {
                coder.encode_to(v, vector);
                cursor.insert(i as i64, vector)?;
            }
            progress(1);
        }

        self.quantized_vectors = quantized_vectors.map(|m| {
            DerefVectorStore::new(
                m.make_read_only().unwrap(),
                NonZero::new(nav_coder.byte_len(dim)).unwrap(),
            )
            .unwrap()
        });
        let centroid = sum
            .into_iter()
            .map(|s| (s / self.limit as f64) as f32)
            .collect::<Vec<_>>();
        self.centroid = self
            .index
            .rerank_table()
            .unwrap_or(self.index.nav_table())
            .new_coder()
            .encode(&centroid);
        Ok(())
    }

    fn cluster_vectors<P: Fn(u64) + Send + Sync>(&mut self, progress: P) -> Result<()> {
        // TODO: this should accept a random number generator
        let subset = SubsetViewVectorStore::new(&self.vectors, (0..self.limit).collect());
        self.clustered_order = Some(graph_clustering::cluster_for_reordering(
            &subset,
            self.index.config().pruning.max_edges.get(),
            &crate::kmeans::Params {
                iters: 100,
                init_iters: 10,
                epsilon: 0.01,
                ..Default::default()
            },
            &mut rand::rng(),
            progress,
        ));
        Ok(())
    }

    fn insert_all<P: Fn(u64) + Send + Sync>(&self, progress: P) -> Result<()> {
        // apply_mu is used to ensure that only one thread is mutating the graph at a time so we can maintain
        // the "undirected graph" invariant. apply_mu contains the entry point vertex id and score against the
        // centroid, we may update this if we find a closer point then reflect it back into entry_vertex.
        let distance_fn = self.index.high_fidelity_table().new_distance_function();
        let apply_mu = {
            let session = self.connection.open_session()?;
            let reader = BulkLoadGraphVectorIndexReader(self, &session);
            let mut vectors = reader.high_fidelity_vectors()?;
            Mutex::new((
                0i64,
                distance_fn.distance(&self.centroid, vectors.get(0).expect("vector 0 exists")?),
            ))
        };
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
        let order = self
            .clustered_order
            .as_ref()
            .map(Cow::from)
            .unwrap_or_else(|| Cow::from((0..self.limit).collect::<Vec<_>>()));
        order
            .par_iter()
            .by_uniform_blocks(self.index.config().pruning.max_edges.get() * 2)
            .copied()
            .filter(|i| *i != 0)
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
                let centroid_distance = {
                    // Insert any other in-flight edges into the candidate queue. These are vertices we may have missed because
                    // they are being inserted concurrently in another thread.
                    self.insert_in_flight_edges(
                        v,
                        in_flight.iter().map(|e| *e),
                        &mut reader,
                        &mut edges,
                    )?;
                    assert!(
                        !edges.iter().any(|n| n.vertex() == v as i64),
                        "Candidate edges for vertex {v} contains self-edge."
                    );

                    // TODO: consider using quantized scores here to avoid reading f32 vectors when
                    // reranking is turned off.
                    let mut vectors = reader.high_fidelity_vectors()?;
                    distance_fn.distance(&self.centroid, vectors.get(v as i64).unwrap()?)
                };

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
                        if centroid_distance < entry_point.1 {
                            entry_point.0 = v as i64;
                            entry_point.1 = centroid_distance;
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
                progress(1);

                Ok::<(), wt_mdb::Error>(())
            })
    }

    /// Cleanup the graph.
    ///
    /// This may prune edges and/or ensure graph connectivity.
    fn cleanup<P: Fn(u64) + Send + Sync>(&self, progress: P) -> Result<()> {
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
            progress(1);
            Ok::<_, wt_mdb::Error>(())
        })
    }

    /// Bulk load the graph table with raw vectors and graph edges.
    ///
    /// Returns statistics about the generated graph.
    fn load_graph<P: Fn(u64)>(&self, progress: P) -> Result<GraphStats> {
        let mut stats = GraphStats {
            vertices: 0,
            edges: 0,
            unconnected: 0,
        };
        let session = self.connection.open_session()?;
        let mut cursor = session.new_bulk_load_cursor::<i64, Vec<u8>>(
            self.index.graph_table_name(),
            Some(
                CreateOptionsBuilder::default()
                    .app_metadata(&serde_json::to_string(&self.index.config()).unwrap()),
            ),
        )?;
        cursor.insert(
            ENTRY_POINT_KEY,
            &self
                .entry_vertex
                .load(atomic::Ordering::Relaxed)
                .to_le_bytes(),
        )?;
        for (i, n) in self.graph.iter().enumerate().take(self.limit) {
            let vertex = n.read().unwrap();
            stats.vertices += 1;
            stats.edges += vertex.len();
            if vertex.is_empty() {
                stats.unconnected += 1;
            }
            let edges = encode_graph_vertex(vertex.iter().map(|n| n.vertex()).collect());
            cursor.insert(i as i64, &edges)?;
            progress(1);
        }
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
        let edge_set_distance_computer = EdgeSetDistanceComputer::new(reader, &candidates)?;
        let split = prune_edges(
            &mut candidates,
            // XXX pass whole config to pruning function.
            self.index.config().pruning.max_edges,
            edge_set_distance_computer,
        );
        candidates.truncate(split);
        Ok(candidates)
    }

    fn insert_in_flight_edges(
        &self,
        vertex_id: usize,
        in_flight: impl Iterator<Item = usize>,
        reader: &mut BulkLoadGraphVectorIndexReader<'_, '_, D>,
        edges: &mut Vec<Neighbor>,
    ) -> Result<()> {
        // TODO: consider moving this to search_for_insert() where it makes more sense to consider
        // the additional vectors and there's more control over how it is done.
        if self.index.config().index_search_params.num_rerank > 0 {
            self.insert_in_flight_edges_from(
                vertex_id,
                in_flight,
                &mut reader
                    .rerank_vectors()
                    .expect("must have rerank table if rerank is configured")?,
                self.index
                    .config()
                    .rerank_format
                    .expect("must have rerank table if rerank is configured"),
                edges,
            )
        } else {
            self.insert_in_flight_edges_from(
                vertex_id,
                in_flight,
                &mut reader.nav_vectors()?,
                self.index.config().nav_format,
                edges,
            )
        }
    }

    fn insert_in_flight_edges_from(
        &self,
        vertex_id: usize,
        in_flight: impl Iterator<Item = usize>,
        vector_store: &mut impl GraphVectorStore,
        vector_format: F32VectorCoding,
        edges: &mut Vec<Neighbor>,
    ) -> Result<()> {
        let vertex_vector = vector_store.get(vertex_id as i64).unwrap()?.to_vec();
        let vertex_dist_fn = vector_format
            .query_vector_distance_indexing(&vertex_vector, self.index.config().similarity);
        let limit = self.index.config().index_search_params.beam_width.get();
        for in_flight_vertex in in_flight.filter(|v| *v != vertex_id) {
            let in_flight_vertex_vector = vector_store.get(in_flight_vertex as i64).unwrap()?;
            let n = Neighbor::new(
                in_flight_vertex as i64,
                vertex_dist_fn.distance(in_flight_vertex_vector),
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

        Ok(())
    }

    fn prune_and_apply<T>(
        &self,
        vertex: usize,
        reader: &mut BulkLoadGraphVectorIndexReader<'_, '_, D>,
        apply_mu: &Mutex<T>,
    ) -> Result<()> {
        // Get the set of edges to prune while only holding a read lock on the vertex.
        let max_edges = self.index.config().pruning.max_edges;
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
                    EdgeSetDistanceComputer::new(reader, &edges)?,
                );
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

impl<D: VectorStore<Elem = f32> + Send + Sync> GraphVectorIndex
    for BulkLoadGraphVectorIndexReader<'_, '_, D>
{
    type Graph<'b>
        = BulkLoadBuilderGraph<'b, D>
    where
        Self: 'b;
    type VectorStore<'b>
        = BulkLoadGraphVectorStore<'b>
    where
        Self: 'b;

    fn config(&self) -> &GraphConfig {
        self.0.index.config()
    }

    fn graph(&self) -> Result<Self::Graph<'_>> {
        Ok(BulkLoadBuilderGraph(self.0))
    }

    fn nav_vectors(&self) -> Result<Self::VectorStore<'_>> {
        if let Some(s) = self.0.quantized_vectors.as_ref() {
            Ok(BulkLoadGraphVectorStore::Memory(
                s,
                self.config().similarity,
                self.config().nav_format,
            ))
        } else {
            Ok(BulkLoadGraphVectorStore::Cursor(CursorVectorStore::new(
                self.1.get_record_cursor(self.0.index.nav_table().name())?,
                self.config().similarity,
                self.config().nav_format,
            )))
        }
    }

    fn rerank_vectors(&self) -> Option<Result<Self::VectorStore<'_>>> {
        self.0.index.rerank_table().map(|t| {
            self.1.get_record_cursor(t.name()).map(|c| {
                BulkLoadGraphVectorStore::Cursor(CursorVectorStore::new(
                    c,
                    self.0.index.config().similarity,
                    t.format(),
                ))
            })
        })
    }
}

struct BulkLoadBuilderGraph<'a, D: Send>(&'a BulkLoadBuilder<D>);

impl<D: Send + Sync> Graph for BulkLoadBuilderGraph<'_, D> {
    type EdgeIterator<'c>
        = BulkNodeEdgesIterator<'c>
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

    fn edges(&mut self, vertex_id: i64) -> Option<Result<Self::EdgeIterator<'_>>> {
        if vertex_id >= 0 && (vertex_id as usize) < self.0.graph.len() {
            Some(Ok(BulkNodeEdgesIterator::new(
                self.0.graph[vertex_id as usize].read().unwrap(),
            )))
        } else {
            None
        }
    }

    fn estimated_vertex_count(&mut self) -> Result<usize> {
        Ok(self.0.graph.len())
    }

    fn set_entry_point(&mut self, _: i64) -> Result<()> {
        Err(Error::Errno(Errno::NOTSUP))
    }

    fn remove_entry_point(&mut self) -> Result<()> {
        Err(Error::Errno(Errno::NOTSUP))
    }

    fn set_edges(&mut self, _: i64, _: impl Into<Vec<i64>>) -> Result<()> {
        Err(Error::Errno(Errno::NOTSUP))
    }

    fn remove_vertex(&mut self, _: i64) -> Result<Vec<i64>> {
        Err(Error::Errno(Errno::NOTSUP))
    }

    fn next_available_vertex_id(&mut self) -> Result<i64> {
        Err(Error::Errno(Errno::NOTSUP))
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

enum BulkLoadGraphVectorStore<'a> {
    Cursor(CursorVectorStore<'a>),
    Memory(
        &'a DerefVectorStore<u8, memmap2::Mmap>,
        VectorSimilarity,
        F32VectorCoding,
    ),
}

impl GraphVectorStore for BulkLoadGraphVectorStore<'_> {
    fn format(&self) -> F32VectorCoding {
        match self {
            Self::Cursor(c) => c.format(),
            Self::Memory(_, _, f) => *f,
        }
    }

    fn similarity(&self) -> VectorSimilarity {
        match self {
            Self::Cursor(c) => c.similarity(),
            Self::Memory(_, s, _) => *s,
        }
    }

    fn get(&mut self, vertex_id: i64) -> Option<Result<&[u8]>> {
        match self {
            Self::Cursor(c) => c.get(vertex_id),
            Self::Memory(m, _, _) => {
                if vertex_id >= 0 && (vertex_id as usize) < m.len() {
                    Some(Ok(&m[vertex_id as usize]))
                } else {
                    None
                }
            }
        }
    }

    fn set(&mut self, _: i64, _: impl AsRef<[u8]>) -> Result<()> {
        Err(Error::Errno(Errno::NOTSUP))
    }

    fn remove(&mut self, _: i64) -> Result<Vec<u8>> {
        Err(Error::Errno(Errno::NOTSUP))
    }
}
