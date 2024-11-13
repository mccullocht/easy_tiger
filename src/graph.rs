use std::{borrow::Cow, num::NonZero};

use serde::{Deserialize, Serialize};
use wt_mdb::Result;

/// Parameters for a search over a Vamana graph.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct GraphSearchParams {
    /// Width of the graph search beam -- the number of candidates considered.
    /// We will return this many results.
    pub beam_width: NonZero<usize>,
    /// Number of results to re-rank using the vectors in the graph.
    pub num_rerank: usize,
}

/// Metadata about graph shape and construction.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct GraphMetadata {
    pub dimensions: NonZero<usize>,
    pub max_edges: NonZero<usize>,
    pub index_search_params: GraphSearchParams,
}

/// A node in the Vamana graph.
pub trait GraphNode {
    type EdgeIterator<'a>: Iterator<Item = i64>
    where
        Self: 'a;

    /// Access the raw float vector.
    fn vector(&self) -> Cow<'_, [f32]>;

    /// Access the edges of the graph. These may be returned in an arbitrary order.
    fn edges(&self) -> Self::EdgeIterator<'_>;
}

/// A Vamana graph.
pub trait Graph {
    type Node<'c>: GraphNode
    where
        Self: 'c;

    /// Return the graph entry point, or None if the graph is empty.
    fn entry_point(&mut self) -> Option<i64>;

    /// Get the contents of a single node.
    // NB: self is mutable to allow reading from a WT cursor.
    fn get(&mut self, node: i64) -> Option<Result<Self::Node<'_>>>;
}

/// Vector store for vectors used to navigate the graph.
pub trait NavVectorStore {
    /// Get the navigation vector for a single node.
    // NB: self is mutable to allow reading from a WT cursor.
    fn get(&mut self, node: i64) -> Option<Result<Cow<'_, [u8]>>>;
}
