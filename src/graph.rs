use std::{borrow::Cow, num::NonZero};

use serde::{Deserialize, Serialize};
use wt_mdb::Result;

use crate::scoring::{
    DotProductScorer, F32VectorScorer, HammingScorer, QuantizedVectorScorer, VectorSimilarity,
};

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
// TODO: rename to GraphConfig.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct GraphMetadata {
    pub dimensions: NonZero<usize>,
    #[serde(default)]
    pub similarity: VectorSimilarity,
    pub max_edges: NonZero<usize>,
    pub index_search_params: GraphSearchParams,
}

impl GraphMetadata {
    /// Return a scorer for high fidelity vectors in the index.
    pub fn new_scorer(&self) -> Box<dyn F32VectorScorer> {
        Box::new(DotProductScorer)
    }

    /// Return a scorer for quantized navigational vectors in the index.
    pub fn new_nav_scorer(&self) -> Box<dyn QuantizedVectorScorer> {
        Box::new(HammingScorer)
    }
}

/// `GraphVectorIndexReader` is used to generate objects for graph navigation.
pub trait GraphVectorIndexReader {
    type Graph<'a>: Graph + 'a
    where
        Self: 'a;
    type NavVectorStore<'a>: NavVectorStore + 'a
    where
        Self: 'a;

    /// Return metadata for this graph.
    fn metadata(&self) -> &GraphMetadata;

    /// Return an object that can be used to navigate the graph.
    fn graph(&self) -> Result<Self::Graph<'_>>;

    /// Return an object that can be used to read navigational vectors.
    fn nav_vectors(&self) -> Result<Self::NavVectorStore<'_>>;

    /// If true, `lookup()` calls may be performed in parallel.
    fn parallel_lookup(&self) -> bool {
        false
    }

    /// Lookup `vertex_id` and when the read is complete invoke `done` with the vertex contents.
    ///
    /// `done` may be invoked from the calling thread or another thread; callers are responsible for
    /// marshalling the data to wherever it needs to be.
    // TODO: callers should be able to pass a timestamp token so that lookups are performed at the
    // same timestamp as any other reads performed for the query.
    fn lookup<D>(&self, vertex_id: i64, done: D)
    where
        D: FnOnce(
                Option<Result<<<Self as GraphVectorIndexReader>::Graph<'_> as Graph>::Vertex<'_>>>,
            ) + Send
            + Sync
            + 'static,
    {
        match self.graph() {
            Ok(mut graph) => done(graph.get(vertex_id)),
            Err(e) => done(Some(Err(e))),
        }
    }
}

/// A Vamana graph.
pub trait Graph {
    type Vertex<'c>: GraphVertex
    where
        Self: 'c;

    /// Return the graph entry point, or None if the graph is empty.
    fn entry_point(&mut self) -> Option<Result<i64>>;

    /// Get the contents of a single vertex.
    // NB: self is mutable to allow reading from a WT cursor.
    fn get(&mut self, vertex_id: i64) -> Option<Result<Self::Vertex<'_>>>;
}

/// A node in the Vamana graph.
pub trait GraphVertex {
    type EdgeIterator<'a>: Iterator<Item = i64>
    where
        Self: 'a;

    /// Access the raw float vector.
    fn vector(&self) -> Cow<'_, [f32]>;

    /// Access the edges of the graph. These may be returned in an arbitrary order.
    fn edges(&self) -> Self::EdgeIterator<'_>;
}

/// Vector store for vectors used to navigate the graph.
pub trait NavVectorStore {
    /// Get the navigation vector for a single vertex.
    fn get(&mut self, vertex_id: i64) -> Option<Result<Cow<'_, [u8]>>>;
}
