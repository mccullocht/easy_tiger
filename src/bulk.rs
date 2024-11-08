use std::{
    borrow::Cow,
    collections::BTreeSet,
    num::NonZero,
    ops::Range,
    sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use wt_mdb::{Record, Result};

use crate::{
    graph::{Graph, GraphNode, NavVectorStore},
    input::NumpyF32VectorStore,
    quantization::binary_quantize,
    scoring::{DotProductScorer, HammingScorer, VectorScorer},
    search::{GraphSearchParams, GraphSearcher},
    wt::{encode_graph_node, GraphMetadata, WiredTigerIndexParams, WiredTigerNavVectorStore},
    Neighbor,
};

// XXX cleverly -- i want the loader to be public to allow injecting progress bars, but maybe
// I can just pass one for each step. alternatively write an abstraction for progress update.
// it could be structured as multiple structs that that consume/perform/generate:
//
// pub fn QuantizedVectorLoader::new(...) -> Self
// pub fn QuantizedVectorLoader::load(self, progress_cb) -> Result<BulkGraphBuilder>
// pub fn BulkGraphBuilder::build(self, progress_cb) -> Result<BulkGraphPruner>
// pub fn BulkGraphPruner::load(self, progress_cb) -> Result<()>

// TODO: tests for bulk load builder, mostly the built graph.

/// Builds a Vamana graph for a bulk load.
pub struct BulkLoadBuilder<D> {
    metadata: GraphMetadata,
    wt_params: WiredTigerIndexParams,
    vectors: NumpyF32VectorStore<D>,
    graph: Box<[RwLock<Vec<Neighbor>>]>,
}

impl<D> BulkLoadBuilder<D>
where
    D: Send + Sync,
{
    /// Create a new bulk graph builder with the passed vector set and max edge count limit.
    pub fn new(
        metadata: GraphMetadata,
        wt_params: WiredTigerIndexParams,
        vectors: NumpyF32VectorStore<D>,
    ) -> Self {
        let mut graph_vec = Vec::with_capacity(vectors.len());
        graph_vec.resize_with(vectors.len(), || {
            RwLock::new(Vec::with_capacity(metadata.max_edges.get() * 2))
        });
        Self {
            metadata,
            wt_params,
            vectors,
            graph: graph_vec.into_boxed_slice(),
        }
    }

    /// Load binary quantized vector data into the nav vectors table.
    pub fn load_nav_vectors<P>(&self, progress: P) -> Result<()>
    where
        P: Fn(),
    {
        let session = self.wt_params.connection.open_session()?;
        session.bulk_load(
            &self.wt_params.nav_table_name,
            None,
            self.vectors.iter().enumerate().map(|(i, v)| {
                progress();
                Record::new(i as i64, binary_quantize(v))
            }),
        )
    }

    /// Insert all vectors from the passed vector store into the graph.
    ///
    /// This operation uses rayon to parallelize large parts of the graph build.
    pub fn insert_all<P>(&self, progress: P) -> Result<()>
    where
        P: Fn() + Send + Sync,
    {
        // TODO: track in-flight inserts and and be sure to include those in the results.
        let apply_mu = Mutex::new(());
        (0..self.vectors.len())
            .into_par_iter()
            .chunks(1_000)
            .try_for_each(|nodes| {
                // NB: we create a new session and cursor for each chunk. Relevant rayon APIs require these
                // objects to be Send + Sync, but Session is only Send and wrapping it in a Mutex does not
                // work because any RecordCursor objects returned have to be destroyed before the Mutex is
                // released.
                let session = self.wt_params.connection.open_session()?;
                let mut nav = WiredTigerNavVectorStore::new(
                    session
                        .open_record_cursor(&self.wt_params.nav_table_name)
                        .unwrap(),
                );
                // XXX make this configurable! should be in GraphMetadata, right?
                let mut searcher = GraphSearcher::new(GraphSearchParams {
                    beam_width: NonZero::new(128).unwrap(),
                    num_rerank: 128,
                });
                for i in nodes {
                    let edges = self.search_for_insert(i, &mut searcher, &mut nav)?;
                    {
                        // XXX i could move this inside apply_insert() maybe?
                        let _unused = apply_mu.lock().unwrap();
                        self.apply_insert(i, edges)?;
                    }
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
        // NB: this must not be run concurrently so that we can ensure
        for (i, n) in self.graph.iter().enumerate() {
            self.maybe_prune_node(i, n.write().unwrap(), self.metadata.max_edges)?;
            progress();
        }
        Ok(())
    }

    /// Bulk load the graph table with raw vectors and graph edges.
    pub fn load_graph<P>(&self, progress: P) -> Result<()>
    where
        P: Fn(),
    {
        // TODO: write graph metadata, select a real entry point.
        let session = self.wt_params.connection.open_session()?;
        session.bulk_load(
            &self.wt_params.graph_table_name,
            None,
            self.vectors
                .iter()
                .zip(self.graph.iter())
                .enumerate()
                .map(|(i, (v, n))| {
                    progress();
                    Record::new(
                        i as i64,
                        encode_graph_node(v, n.read().unwrap().iter().map(|n| n.node()).collect()),
                    )
                }),
        )
    }

    fn search_for_insert<N>(
        &self,
        index: usize,
        searcher: &mut GraphSearcher,
        nav_vectors: &mut N,
    ) -> Result<Vec<Neighbor>>
    where
        N: NavVectorStore,
    {
        let query = &self.vectors[index];
        let mut graph = BulkLoadBuilderGraph(self);
        let mut candidates = searcher.search(
            query,
            &mut graph,
            &DotProductScorer,
            nav_vectors,
            &HammingScorer,
        )?;

        let pruned_len = self
            .prune(&mut candidates, &mut graph, &DotProductScorer)?
            .0
            .len();
        candidates.truncate(pruned_len);
        Ok(candidates)
    }

    /// This function is the only mutator of self.graph and must not be run concurrently.
    fn apply_insert(&self, index: usize, edges: Vec<Neighbor>) -> Result<()> {
        self.graph[index].write().unwrap().extend_from_slice(&edges);
        for e in edges.iter() {
            let mut guard = self.graph[e.node() as usize].write().unwrap();
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

        /*
        let (selected, dropped) = self.prune(
            &mut guard,
            &mut BulkLoadBuilderGraph(self),
            &DotProductScorer,
        )?;
         */
        let (selected, dropped) = self.prune_trivial(&mut guard);
        let pruned_len = selected.len();
        let dropped = dropped.to_vec();
        guard.truncate(pruned_len);
        drop(guard);

        // Remove in-links from nodes that we dropped out-links to.
        // If we maintain the invariant that all links are reciprocated then it will be easier
        // to mutate the index without requiring a cleaning process.
        for n in dropped {
            self.graph[n.node() as usize]
                .write()
                .unwrap()
                .retain(|e| e.node() != index as i64);
        }
        Ok(())
    }

    /// Prune `edges`, enforcing RNG properties with alpha parameter.
    ///
    /// Returns two slices: one containing the selected nodes and one containing the unselected nodes.
    #[allow(dead_code)]
    fn prune<'a, S>(
        &self,
        edges: &'a mut [Neighbor],
        graph: &mut BulkLoadBuilderGraph<'_, D>,
        scorer: &S,
    ) -> Result<(&'a mut [Neighbor], &'a mut [Neighbor])>
    where
        S: VectorScorer<Elem = f32>,
    {
        // XXX it would be nice to stop passing graph and inject some sort of cache.
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
                // TODO: consider caching id -> vector as right now we might repeatedly
                // lookup, particularly at the beginning of the list.
                let e_vec = graph
                    .get(e.node())
                    .expect("bulk load")
                    .expect("numpy vector store")
                    .vector()
                    .into_owned();
                let mut select = false;
                for p in selected.iter().take_while(|j| **j < i).map(|j| edges[*j]) {
                    let p_node = graph
                        .get(p.node())
                        .expect("bulk load")
                        .expect("numpy vector store");
                    if scorer.score(&e_vec, &p_node.vector()) > e.score * alpha {
                        select = true;
                        break;
                    }
                }

                if select {
                    selected.insert(i);
                    if selected.len() >= self.metadata.max_edges.get() {
                        break;
                    }
                }
            }

            if selected.len() >= self.metadata.max_edges.get() {
                break;
            }
        }

        // Partition edges into selected and unselected.
        for (i, j) in selected.iter().enumerate() {
            edges.swap(i, *j);
        }

        Ok(edges.split_at_mut(selected.len()))
    }

    fn prune_trivial<'a>(
        &self,
        edges: &'a mut [Neighbor],
    ) -> (&'a mut [Neighbor], &'a mut [Neighbor]) {
        edges.sort();
        edges.split_at_mut(self.metadata.max_edges.get())
    }
}

struct BulkLoadBuilderGraph<'a, D>(&'a BulkLoadBuilder<D>);

impl<'a, D> Graph for BulkLoadBuilderGraph<'a, D> {
    type Node<'c> = BulkLoadGraphNode<'c, D> where Self: 'c;

    fn entry_point(&self) -> Option<i64> {
        // TODO: store an entry point in the builder and optimize for distance to a centroid.
        if self.0.vectors.is_empty() {
            None
        } else {
            Some(0)
        }
    }

    fn get(&mut self, node: i64) -> Option<Result<Self::Node<'_>>> {
        Some(Ok(BulkLoadGraphNode {
            builder: self.0,
            node,
        }))
    }
}

struct BulkLoadGraphNode<'a, D> {
    builder: &'a BulkLoadBuilder<D>,
    node: i64,
}

impl<'a, D> GraphNode for BulkLoadGraphNode<'a, D> {
    type EdgeIterator<'c> = BulkNodeEdgesIterator<'c> where Self: 'c;

    fn vector(&self) -> Cow<'_, [f32]> {
        self.builder.vectors[self.node as usize].into()
    }

    fn edges(&self) -> Self::EdgeIterator<'_> {
        BulkNodeEdgesIterator::new(self.builder.graph[self.node as usize].read().unwrap())
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
        self.range.next().map(|i| self.guard[i].node())
    }
}
