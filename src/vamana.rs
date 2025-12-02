//! A DiskANN/Vamana graph ANN implementation
//!

pub mod bulk;
pub mod mutate;
pub mod search;
pub mod wt;

use std::{collections::BTreeSet, num::NonZero};

use rustix::io::Errno;
use serde::{Deserialize, Serialize};
use vectors::{F32VectorCoder, F32VectorCoding, VectorDistance, VectorSimilarity};
use wt_mdb::{Error, Result};

use crate::Neighbor;

/// Parameters for a search over a Vamana graph.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct GraphSearchParams {
    /// Width of the graph search beam -- the number of candidates considered.
    /// We will return this many results.
    pub beam_width: NonZero<usize>,
    /// Number of results to re-rank using the vectors in the graph.
    pub num_rerank: usize,
}

/// Configuration describing edge pruning policy for a graph index.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct EdgePruningConfig {
    /// Number of edges to keep for each vertex.
    pub max_edges: NonZero<usize>,
    /// Maximum alpha value to use for edge pruning.
    ///
    /// Larger values result in retain longer edges while still saying below `max_edges`.
    ///
    /// Must be >= 1.0. Default value is 1.2.
    pub max_alpha: f64,
    /// Alpha value scaling factor.
    ///
    /// This is the rate at which we scale from an alpha value of 1.0 up to `max_alpha`. Slowly
    /// scaling up the alpha value gives preference to shorter edges, but may result in more
    /// iterations in pruning.
    ///
    /// Must be >= 1.0. Default value is 1.2.
    pub alpha_scale: f64,
}

impl EdgePruningConfig {
    pub const DEFAULT_MAX_ALPHA: f64 = 1.2;
    pub const DEFAULT_ALPHA_SCALE: f64 = 1.2;

    pub fn new(max_edges: NonZero<usize>) -> Self {
        Self {
            max_edges,
            max_alpha: Self::DEFAULT_MAX_ALPHA,
            alpha_scale: Self::DEFAULT_ALPHA_SCALE,
        }
    }
}

/// Configuration describing graph shape and construction. Used to read and mutate the graph.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct GraphConfig {
    /// Number of vector dimensions.
    pub dimensions: NonZero<usize>,
    /// Function to use for vector distance computations.
    ///
    /// If using [`VectorSimilarity::Dot`] along with [`F32VectorCoding::Raw`], consider using
    /// [`F32VectorCoding::RawL2Normalized`] instead unless you are certain that all input vectors
    /// in the read and write path are _already_ l2 normalized.
    pub similarity: VectorSimilarity,
    /// Vector coding to use in the nav table.
    ///
    /// Nav vectors are used to compute edge distance during graph traversal, so we may examine
    /// ~O(log(graph_size) * num_candidates) vectors for each query.
    pub nav_format: F32VectorCoding,
    /// Vector coding to use in the rerank table.
    ///
    /// If re-ranking is turned on during graph construction or search this will be used to compute
    /// ~O(num_candidates) query distances. If this format is quantized you should typically choose
    /// a high fidelity quantization function.
    pub rerank_format: Option<F32VectorCoding>,
    /// Edge pruning configuration.
    ///
    /// This controls how many edges are placed on each vertex and how they are selected.
    pub pruning: EdgePruningConfig,
    /// Search parameters to use during graph construction.
    ///
    /// This dictates the set of candidate edges provided to pruning.
    pub index_search_params: GraphSearchParams,
}

/// `GraphVectorIndex` is used to generate objects for graph navigation and mutation.
pub trait GraphVectorIndex {
    type Graph<'a>: Graph + 'a
    where
        Self: 'a;
    type VectorStore<'a>: GraphVectorStore + 'a
    where
        Self: 'a;

    /// Return config for this graph.
    fn config(&self) -> &GraphConfig;

    /// Return an object that can be used to navigate the graph.
    fn graph(&self) -> Result<Self::Graph<'_>>;

    /// Return an object that can be used to read navigational vectors.
    fn nav_vectors(&self) -> Result<Self::VectorStore<'_>>;

    /// Return an object that can be used to read vectors for re-ranking.
    ///
    /// Not all graphs will have a set of rerank vectors; use high_fidelity_vectors()
    /// to get most accurate set of vectors available.
    fn rerank_vectors(&self) -> Option<Result<Self::VectorStore<'_>>>;

    /// Return an object that can be used to read the most accurate set of vectors stored
    /// for this index.
    fn high_fidelity_vectors(&self) -> Result<Self::VectorStore<'_>> {
        self.rerank_vectors().unwrap_or_else(|| self.nav_vectors())
    }
}

/// A Vamana graph.
pub trait Graph {
    type EdgeIterator<'a>: Iterator<Item = i64>
    where
        Self: 'a;

    /// Return the graph entry point, or None if the graph is empty.
    fn entry_point(&mut self) -> Option<Result<i64>>;

    /// Access the edges of the vertex. These may be returned in an arbitrary order.
    fn edges(&mut self, vertex_id: i64) -> Option<Result<Self::EdgeIterator<'_>>>;

    /// Return an estimate of the number of vertices in the graph.
    fn estimated_vertex_count(&mut self) -> Result<usize>;

    /// Set the entry point of the graph.
    ///
    /// The caller is responsible for ensuring that the named `vertex_id` exists in the graph.
    ///
    /// Returns a NOTSUP error if the graph does not support mutation.
    fn set_entry_point(&mut self, vertex_id: i64) -> Result<()>;

    /// Remove the entry point from the graph.
    ///
    /// The caller is responsible for ensuring that the graph is empty when the entry point is
    /// removed or this will create search/reachability issues.
    ///
    /// Returns a NOTSUP error if the graph does not support mutation.
    /// May return a NOT_FOUND error if the graph is empty.
    fn remove_entry_point(&mut self) -> Result<()>;

    /// Set the outbound edges from `vertex_id`.
    ///
    /// The caller is responsible for ensuring that the listed edges point to vertexes that exist
    /// as well as maintaining the undirected property of the graph.
    ///
    /// Returns a NOTSUP error if the graph does not support mutation.
    fn set_edges(&mut self, vertex_id: i64, edges: impl Into<Vec<i64>>) -> Result<()>;

    /// Rmoves the vertex and returns the outbound edges associated with it.
    ///
    /// Returns a NOTSUP error if the graph does not support mutation.
    /// May return a NOT_FOUND error if the vertex does not exist.
    fn remove_vertex(&mut self, vertex_id: i64) -> Result<Vec<i64>>;

    /// Returns the next available vertex id.
    ///
    /// This vertex id is guaranteed to be unused but may not be sequentially assigned.
    /// Returns a NOTSUP error if the graph does not support mutation.
    fn next_available_vertex_id(&mut self) -> Result<i64>;
}

/// Vector store for known vector formats accessible by a record id.
pub trait GraphVectorStore {
    /// Similarity function used for vectors in this store.
    fn similarity(&self) -> VectorSimilarity;

    /// Return the format that vectors in the store are encoded in.
    fn format(&self) -> F32VectorCoding;

    /// Create a new distance function that operates over vectors on this table.
    fn new_distance_function(&self) -> Box<dyn VectorDistance> {
        self.format().new_vector_distance(self.similarity())
    }

    /// Create a new coder for vectors of this type.
    fn new_coder(&self) -> Box<dyn F32VectorCoder> {
        self.format().new_coder(self.similarity())
    }

    /// Return the contents of the vector at vertex, or `None` if the vertex is unknown.
    // TODO: consider replacing this method as it is unsafe in the event of a rollback.
    // The replacement would accept a closure to transform the vector reference in place.
    fn get(&mut self, vertex_id: i64) -> Option<Result<&[u8]>>;

    /// Set the contents of the vector at vertex.
    ///
    /// Returns a NOTSUP error if the graph does not support mutation.
    fn set(&mut self, vertex_id: i64, vector: impl AsRef<[u8]>) -> Result<()>;

    /// Remove the vector at vertex and returns its contents.
    ///
    /// Returns a NOTSUP error if the graph does not support mutation.
    /// May return a NOT_FOUND error if the vertex does not exist.
    fn remove(&mut self, vertex_id: i64) -> Result<Vec<u8>>;

    // TODO: extract many vectors into VecVectorStore.
    // TODO: method to turn self into a QueryVectorDistance wrapper.
}

/// Computes the distance between two edges in a set to assist in pruning.
///
/// This abstraction allows us to switch between raw (float) and nav (quantized) vectors.
pub struct EdgeSetDistanceComputer {
    distance_fn: Box<dyn VectorDistance>,
    // TODO: flat representation of the vectors instead of nesting.
    vectors: Vec<Vec<u8>>,
}

impl EdgeSetDistanceComputer {
    pub fn new<R: GraphVectorIndex>(reader: &R, edges: &[Neighbor]) -> Result<Self> {
        if reader.config().index_search_params.num_rerank > 0 {
            Self::from_store_and_edges(
                &mut reader
                    .rerank_vectors()
                    .unwrap_or(Err(Error::Errno(Errno::NOTSUP)))?,
                edges,
            )
        } else {
            Self::from_store_and_edges(&mut reader.nav_vectors()?, edges)
        }
    }

    fn from_store_and_edges(store: &mut impl GraphVectorStore, edges: &[Neighbor]) -> Result<Self> {
        let vectors = edges
            .iter()
            .map(|n| {
                store
                    .get(n.vertex())
                    .unwrap_or(Err(Error::not_found_error()))
                    .map(|v| v.to_vec())
            })
            .collect::<Result<Vec<_>>>()?;
        let distance_fn = store.new_distance_function();
        Ok(Self {
            distance_fn,
            vectors,
        })
    }

    pub fn distance(&self, i: usize, j: usize) -> f64 {
        self.distance_fn
            .distance(&self.vectors[i], &self.vectors[j])
    }
}

/// Select the indices of `edges` that should remain when pruning down to at most `max_edges`.
/// `graph` is used to access vectors and `distance_fn` is used to compare vectors when making pruning
/// decisions.
/// REQUIRES: `edges.is_sorted()`.
fn select_pruned_edges(
    edges: &[Neighbor],
    config: &EdgePruningConfig,
    edge_distance_computer: EdgeSetDistanceComputer,
) -> BTreeSet<usize> {
    if edges.is_empty() {
        return BTreeSet::new();
    }

    debug_assert!(edges.is_sorted());

    // XXX use alpha value.

    // TODO: replace with a fixed length bitset
    let mut selected = BTreeSet::new();
    selected.insert(0); // we always keep the first node.
    for alpha in [1.0, 1.2] {
        for (i, e) in edges.iter().enumerate().skip(1) {
            if selected.contains(&i) {
                continue;
            }

            if !selected
                .iter()
                .take_while(|s| **s < i)
                .any(|s| edge_distance_computer.distance(i, *s) < e.distance * alpha)
            {
                selected.insert(i);
                if selected.len() >= config.max_edges.get() {
                    break;
                }
            }
        }

        if selected.len() >= config.max_edges.get() {
            break;
        }
    }

    selected
}

/// Prune `edges` down to at most `max_edges`. Use `graph` and `distance_fn` to inform this decision.
/// Returns a split point: all edges before that point are selected, all after are to be dropped.
/// REQUIRES: `edges.is_sorted()`.
fn prune_edges(
    edges: &mut [Neighbor],
    config: &EdgePruningConfig,
    edge_distance_computer: EdgeSetDistanceComputer,
) -> usize {
    let selected = select_pruned_edges(edges, config, edge_distance_computer);
    // Partition edges into selected and unselected.
    for (i, j) in selected.iter().enumerate() {
        edges.swap(i, *j);
    }
    selected.len()
}
