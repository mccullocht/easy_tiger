use std::{
    borrow::Cow,
    collections::BTreeSet,
    num::NonZero,
    ops::Range,
    sync::{RwLock, RwLockReadGuard},
};

use wt_mdb::Result;

use crate::{
    graph::{Graph, GraphNode, NavVectorStore},
    input::NumpyF32VectorStore,
    scoring::{DotProductScorer, HammingScorer, VectorScorer},
    search::GraphSearcher,
    Neighbor,
};

/// Builds a Vamana graph for a bulk load.
pub struct BulkLoadBuilder<D> {
    vectors: NumpyF32VectorStore<D>,
    graph: Box<[RwLock<Vec<Neighbor>>]>,
    max_edges: NonZero<usize>,
}

impl<D> BulkLoadBuilder<D> {
    /// Create a new bulk graph builder with the passed vector set and max edge count limit.
    // XXX need a session and a table name for the quantized vectors.
    pub fn new(vectors: NumpyF32VectorStore<D>, max_edges: NonZero<usize>) -> Self {
        let mut graph_vec = Vec::with_capacity(vectors.len());
        graph_vec.resize_with(vectors.len(), || {
            RwLock::new(Vec::with_capacity(max_edges.get() * 2))
        });
        Self {
            vectors,
            graph: graph_vec.into_boxed_slice(),
            max_edges,
        }
    }

    // TODO: consider forcing links to reciprocal. This means that if you prune a link you
    // remove the backlink to yourself. i think this forces writing to be single threaded
    // although you could position this as queues: threads concurrently search and then a
    // single thread is responsible for actually updating. i think this ca
    fn insert<N>(
        &self,
        index: usize,
        searcher: &mut GraphSearcher,
        nav_vectors: &mut N,
    ) -> Result<()>
    where
        N: NavVectorStore,
    {
        // separate this into two halves: the first half does the search to generate a candidate list
        // and the second manipulates the graph. The first part is run concurrently; the second part
        // runs sequentially. the first part sends results to the second via mpsc. to avoid holes where
        // concurrently processed nodes don't have links to each other we will maintain a skiplist of
        // in-flight inserts. a node is in-flight when it is received by the thread that performs the
        // search and finished when the graph update processor applies the update.

        let query = &self.vectors[index];
        let mut graph = BulkLoadBuilderGraph(self);
        let mut candidates = searcher.search(
            query,
            &mut graph,
            &DotProductScorer,
            nav_vectors,
            &HammingScorer,
        )?;

        // XXX if we allow concurrent search this is where we try to insert the concurrent nodes into the result set.

        let (edges, _) = self.prune(&mut candidates, &mut graph, &DotProductScorer)?;
        // if we are not allowing concurrent insert then this gets a lot easier: just replace the set of neighbors
        // and do backlinks. If we _are_ allowing concurrent insertion then we might have to merge this with any
        // backlinks that have already been built. if we hit vector capacity we may decide we need to prune.
        let mut node = self.graph[index].write().unwrap();
        node.extend_from_slice(edges);
        drop(node);

        // XXX at this point we are working off of a potentially outdated list -- another inserter may have pruned
        // some of these outlinks. this means we cannot enforce reciprocal backlinks.

        todo!()
    }

    fn search_for_insert<N>(
        &self,
        index: usize,
        searcher: &mut GraphSearcher,
        nav_vectors: &mut N,
    ) -> Result<(usize, Vec<Neighbor>)>
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
        Ok((index, candidates))
    }

    /// This function is the only mutator of self.graph and must not be run concurrently.
    fn apply_insert(&self, index: usize, edges: Vec<Neighbor>) -> Result<()> {
        self.graph[index].write().unwrap().extend_from_slice(&edges);
        for e in edges.iter() {
            let mut guard = self.graph[e.node() as usize].write().unwrap();
            guard.push(Neighbor::new(index as i64, e.score()));
            if guard.len() >= guard.capacity() {
                let (selected, dropped) = self.prune(
                    &mut guard,
                    &mut BulkLoadBuilderGraph(self),
                    &DotProductScorer,
                )?;
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
            }
        }
        Ok(())
    }

    /// Prune `edges`, enforcing RNG properties with alpha parameter.
    ///
    /// Returns two slices: one containing the selected nodes and one containing the unselected nodes.
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
                    if selected.len() >= self.max_edges.get() {
                        break;
                    }
                }
            }

            if selected.len() >= self.max_edges.get() {
                break;
            }
        }

        // Partition edges into selected and unselected.
        for (i, j) in selected.iter().enumerate() {
            edges.swap(i, *j);
        }

        Ok(edges.split_at_mut(selected.len()))
    }

    // insertion flow:
    // * search the graph with some parameters, re-score, prune(?)
    // * insert the results into the graph for the given node (write locked)
    // * iterate over the results and insert backlinks (write locked on each node)
    // i think maybe we should insert without pruning and do the pruning as it comes up
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

struct BulkNodeEdges {
    edges: Vec<Neighbor>,
    first_unpruned: usize,
}

impl BulkNodeEdges {
    fn new(max_edges: NonZero<usize>) -> Self {
        Self {
            edges: Vec::with_capacity(max_edges.get() * 2),
            first_unpruned: 0,
        }
    }

    fn insert(&mut self, n: Neighbor) -> Option<usize> {
        match self.edges.binary_search(&n) {
            Err(i) if i < self.capacity() => {
                self.edges.insert(i, n);
                self.first_unpruned = std::cmp::min(self.first_unpruned, i);
                Some(i)
            }
            // Don't insert on exact match or if capacity has been reached.
            _ => None,
        }
    }

    fn prune<I>(&mut self, selected: I)
    where
        I: Iterator<Item = usize>,
    {
        let mut len = 0;
        for (i, o) in selected.zip(0..self.len()) {
            self.edges.swap(i, o);
            len += 1;
        }
        self.edges.truncate(len);
        self.first_unpruned = self.edges.len();
    }

    fn iter(&self) -> std::slice::Iter<'_, Neighbor> {
        self.edges.iter()
    }

    fn len(&self) -> usize {
        self.edges.len()
    }

    fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    fn capacity(&self) -> usize {
        self.edges.capacity()
    }

    fn first_unpruned(&self) -> usize {
        self.first_unpruned
    }
}
