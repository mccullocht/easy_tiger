use core::f64;
use std::{
    borrow::Cow,
    collections::BTreeSet,
    num::NonZero,
    ops::Range,
    sync::{
        atomic::{self, AtomicI64},
        Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard,
    },
};

use crossbeam_skiplist::SkipSet;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use wt_mdb::{Connection, Record, Result, Session};

use crate::{
    graph::{Graph, GraphMetadata, GraphVectorIndexReader, GraphVertex},
    input::NumpyF32VectorStore,
    quantization::binary_quantize,
    scoring::F32VectorScorer,
    search::GraphSearcher,
    wt::{
        encode_graph_node, WiredTigerGraphVectorIndex, WiredTigerNavVectorStore, ENTRY_POINT_KEY,
        METADATA_KEY,
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
    index: WiredTigerGraphVectorIndex,
    limit: usize,

    vectors: NumpyF32VectorStore<D>,
    centroid: Vec<f32>,

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
        index: WiredTigerGraphVectorIndex,
        vectors: NumpyF32VectorStore<D>,
        limit: usize,
    ) -> Self {
        let mut graph_vec = Vec::with_capacity(vectors.len());
        graph_vec.resize_with(vectors.len(), || {
            RwLock::new(Vec::with_capacity(index.metadata().max_edges.get() * 2))
        });
        let scorer = index.metadata().new_scorer();
        Self {
            connection,
            index,
            limit,
            vectors,
            centroid: Vec::new(),
            graph: graph_vec.into_boxed_slice(),
            entry_vertex: AtomicI64::new(-1),
            scorer: scorer,
        }
    }

    /// Load binary quantized vector data into the nav vectors table.
    pub fn load_nav_vectors<P>(&mut self, progress: P) -> Result<()>
    where
        P: Fn(),
    {
        let session = self.connection.open_session()?;
        let mut sum = vec![0.0; self.index.metadata().dimensions.get()];
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
                    Record::new(i as i64, binary_quantize(v))
                }),
        )?;

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
                let mut searcher = GraphSearcher::new(self.index.metadata().index_search_params);
                for i in nodes {
                    // Use a transaction for each search. Without this each lookup will be a separate transaction
                    // which obtains a reader lock inside the session. Overhead for that is ~10x.
                    session.begin_transaction(None)?;
                    let mut reader = BulkLoadGraphVectorIndexReader(self, session);
                    in_flight.insert(i);
                    let mut edges = self.search_for_insert(i, &mut searcher, &mut reader)?;
                    let worst_score = edges.last().map(|n| n.score()).unwrap_or(f64::MIN);
                    edges.extend(in_flight.iter().filter_map(|v| {
                        if *v == i {
                            return None;
                        }
                        let p = Neighbor::new(
                            *v as i64,
                            self.scorer.score(&self.vectors[i], &self.vectors[*v]),
                        );
                        if p.score < worst_score {
                            None
                        } else {
                            Some(p)
                        }
                    }));
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
            self.maybe_prune_node(i, n.write().unwrap(), self.index.metadata().max_edges)?;
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
        let metadata_rows = vec![
            Record::new(
                METADATA_KEY,
                serde_json::to_vec(&self.index.metadata()).unwrap(),
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
            metadata_rows.into_iter().chain(
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
        let pruned_len = self
            .prune(&mut candidates, &mut graph, self.scorer.as_ref())?
            .0
            .len();
        candidates.truncate(pruned_len);
        Ok(candidates)
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
            let max_edges = NonZero::new(guard.capacity()).unwrap();
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

        let (selected, dropped) = self.prune(
            &mut guard,
            &mut BulkLoadBuilderGraph(self),
            self.scorer.as_ref(),
        )?;
        let pruned_len = selected.len();
        let dropped = dropped.to_vec();
        guard.truncate(pruned_len);
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

    /// Prune `edges`, enforcing RNG properties with alpha parameter.
    ///
    /// Returns two slices: one containing the selected nodes and one containing the unselected nodes.
    fn prune<'a>(
        &self,
        edges: &'a mut [Neighbor],
        graph: &mut BulkLoadBuilderGraph<'_, D>,
        scorer: &dyn F32VectorScorer,
    ) -> Result<(&'a [Neighbor], &'a [Neighbor])> {
        if edges.is_empty() {
            return Ok((&[], &[]));
        }
        edges.sort();
        // TODO: replace with a fixed length bitset
        let mut selected = BTreeSet::new();
        selected.insert(0); // we always keep the first node.
        for alpha in [1.0, 1.2] {
            for (i, e) in edges.iter().enumerate().skip(1) {
                if selected.contains(&i) {
                    continue;
                }

                // TODO: fix error handling so we can reuse this elsewhere.
                let e_vec = graph
                    .get(e.vertex())
                    .expect("bulk load")
                    .expect("numpy vector store")
                    .vector()
                    .into_owned();
                let mut select = false;
                for p in selected.iter().take_while(|j| **j < i).map(|j| edges[*j]) {
                    let p_node = graph
                        .get(p.vertex())
                        .expect("bulk load")
                        .expect("numpy vector store");
                    if scorer.score(&e_vec, &p_node.vector()) > e.score * alpha {
                        select = true;
                        break;
                    }
                }

                if select {
                    selected.insert(i);
                    if selected.len() >= self.index.metadata().max_edges.get() {
                        break;
                    }
                }
            }

            if selected.len() >= self.index.metadata().max_edges.get() {
                break;
            }
        }

        // Partition edges into selected and unselected.
        for (i, j) in selected.iter().enumerate() {
            edges.swap(i, *j);
        }

        Ok(edges.split_at(selected.len()))
    }
}

struct BulkLoadGraphVectorIndexReader<'a, D: Send>(&'a BulkLoadBuilder<D>, Session);

impl<'a, D> BulkLoadGraphVectorIndexReader<'a, D>
where
    D: Send,
{
    fn into_session(self) -> Session {
        self.1
    }
}

impl<'a, D> GraphVectorIndexReader for BulkLoadGraphVectorIndexReader<'a, D>
where
    D: Send,
{
    type Graph<'b> = BulkLoadBuilderGraph<'b, D> where Self: 'b;
    type NavVectorStore<'b> = WiredTigerNavVectorStore<'b> where Self: 'b;

    fn metadata(&self) -> &GraphMetadata {
        self.0.index.metadata()
    }

    fn graph(&self) -> Result<Self::Graph<'_>> {
        Ok(BulkLoadBuilderGraph(self.0))
    }

    fn nav_vectors(&self) -> Result<Self::NavVectorStore<'_>> {
        Ok(WiredTigerNavVectorStore::new(
            self.1.get_record_cursor(self.0.index.nav_table_name())?,
        ))
    }
}

struct BulkLoadBuilderGraph<'a, D: Send>(&'a BulkLoadBuilder<D>);

impl<'a, D> Graph for BulkLoadBuilderGraph<'a, D>
where
    D: Send,
{
    type Vertex<'c> = BulkLoadGraphVertex<'c, D> where Self: 'c;

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

impl<'a, D> GraphVertex for BulkLoadGraphVertex<'a, D> {
    type EdgeIterator<'c> = BulkNodeEdgesIterator<'c> where Self: 'c;

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

impl<'a> Iterator for BulkNodeEdgesIterator<'a> {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(|i| self.guard[i].vertex())
    }
}
