use std::{borrow::Cow, num::NonZero};

use serde::{Deserialize, Serialize};
use wt_mdb::Result;

use crate::scoring::{F32VectorScorer, QuantizedVectorScorer};

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

/// `GraphVectorIndexReader` is used to generate objects for graph navigation.
pub trait GraphVectorIndexReader {
    type Graph: Graph;
    type NavVectorStore: NavVectorStore;

    /// Return the scorer used for vectors returned by the graph.
    ///
    /// This is only used at the reranking step at the end.
    fn scorer(&self) -> Box<dyn F32VectorScorer>;

    /// Return an object that can be used to navigate the graph.
    fn graph(&mut self) -> Result<Self::Graph>;

    /// Return the scorer used for navigation vectors.
    fn nav_scorer(&self) -> Box<dyn QuantizedVectorScorer>;

    /// Return an object that can be used to read navigational vectors.
    fn nav_vectors(&mut self) -> Result<Self::NavVectorStore>;
}
