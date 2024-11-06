use std::{
    num::NonZero,
    ops::Range,
    path::Iter,
    sync::{RwLock, RwLockReadGuard},
};

use crate::{
    graph::{Graph, GraphNode},
    input::NumpyF32VectorStore,
    Neighbor,
};

/// Builds a Vamana graph for a bulk load.
pub struct BulkLoadBuilder<D> {
    vectors: NumpyF32VectorStore<D>,
    graph: Box<[RwLock<BulkNodeEdges>]>,
}

impl<D> BulkLoadBuilder<D> {
    /// Create a new bulk graph builder with the passed vector set and max edge count limit.
    pub fn new(vectors: NumpyF32VectorStore<D>, max_edges: NonZero<usize>) -> Self {
        let mut graph_vec = Vec::with_capacity(vectors.len());
        graph_vec.resize_with(vectors.len(), || RwLock::new(BulkNodeEdges::new(max_edges)));
        Self {
            vectors,
            graph: graph_vec.into_boxed_slice(),
        }
    }

    // insertion flow:
    // * search the graph with some parameters, re-score, prune(?)
    // * insert the results into the graph for the given node (write locked)
    // * iterate over the results and insert backlinks (write locked on each node)
    // i think maybe we should insert without pruning and do the pruning as it comes up
}

impl<D> Graph for BulkLoadBuilder<D> {
    type Node<'c> = BulkLoadGraphNode<'c, D> where Self: 'c;

    fn get(&mut self, node: i64) -> Option<wt_mdb::Result<Self::Node<'_>>> {
        Some(Ok(BulkLoadGraphNode {
            builder: self,
            node,
        }))
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

impl AsRef<[Neighbor]> for BulkNodeEdges {
    fn as_ref(&self) -> &[Neighbor] {
        self.edges.as_ref()
    }
}

pub struct BulkLoadGraphNode<'a, D> {
    builder: &'a BulkLoadBuilder<D>,
    node: i64,
}

impl<'a, D> GraphNode for BulkLoadGraphNode<'a, D> {
    type EdgeIterator<'c> = BulkNodeEdgesIterator<'c> where Self: 'c;

    fn vector(&self) -> std::borrow::Cow<'_, [f32]> {
        self.builder.vectors[self.node as usize].into()
    }

    fn edges(&self) -> Self::EdgeIterator<'_> {
        BulkNodeEdgesIterator::new(self.builder.graph[self.node as usize].read().unwrap())
    }
}

pub struct BulkNodeEdgesIterator<'a> {
    guard: RwLockReadGuard<'a, BulkNodeEdges>,
    range: Range<usize>,
}

impl<'a> BulkNodeEdgesIterator<'a> {
    fn new(guard: RwLockReadGuard<'a, BulkNodeEdges>) -> Self {
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
        self.range.next().map(|i| self.guard.as_ref()[i].node())
    }
}
