//! Interfaces for graph configuration and access abstractions.
//!
//! Graph access traits provided here are used during graph search, and allow us to
//! build indices with both WiredTiger backing and in-memory backing for bulk loads.

use std::{collections::BTreeSet, io, num::NonZero, str::FromStr};

use serde::{Deserialize, Serialize};
use wt_mdb::{Error, Result};

use crate::{
    vectors::{
        F32VectorCoder, F32VectorCoding, F32VectorDistance, VectorDistance, VectorSimilarity,
    },
    Neighbor,
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

/// Describes how fields within the vector index are laid out -- split completely or with some
/// colocated fields.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum GraphLayout {
    /// Each field appears in its own table.
    Split,
}

impl Default for GraphLayout {
    fn default() -> Self {
        Self::Split
    }
}

impl FromStr for GraphLayout {
    type Err = io::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "split" => Ok(Self::Split),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Unknown graph layout {s}"),
            )),
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
    pub rerank_format: F32VectorCoding,
    pub layout: GraphLayout,
    /// Maximum number of edges at each vertex.
    pub max_edges: NonZero<usize>,
    /// Search parameters to use during graph construction.
    pub index_search_params: GraphSearchParams,
}

impl GraphConfig {
    /// Return a distance function for high fidelity vectors in the index.
    pub fn new_distance_function(&self) -> Box<dyn F32VectorDistance> {
        self.similarity.new_distance_function()
    }

    /// Return a new vector coder for the nav vector format.
    pub fn new_nav_coder(&self) -> Box<dyn F32VectorCoder> {
        self.nav_format.new_coder(self.similarity)
    }

    /// Return a distance function for quantized navigational vectors in the index.
    pub fn new_nav_distance_function(&self) -> Box<dyn VectorDistance> {
        self.nav_format.new_vector_distance(self.similarity)
    }
}

/// `GraphVectorIndexReader` is used to generate objects for graph navigation.
pub trait GraphVectorIndexReader {
    type Graph<'a>: Graph + 'a
    where
        Self: 'a;
    type NavVectorStore<'a>: GraphVectorStore + 'a
    where
        Self: 'a;
    type RerankVectorStore<'a>: GraphVectorStore + 'a
    where
        Self: 'a;

    /// Return config for this graph.
    fn config(&self) -> &GraphConfig;

    /// Return an object that can be used to navigate the graph.
    fn graph(&self) -> Result<Self::Graph<'_>>;

    /// Return an object that can be used to read navigational vectors.
    fn nav_vectors(&self) -> Result<Self::NavVectorStore<'_>>;

    /// Return an object that can be used to read vectors for re-ranking.
    fn rerank_vectors(&self) -> Result<Self::RerankVectorStore<'_>>;
}

/// A Vamana graph.
pub trait Graph {
    type Vertex<'c>: GraphVertex
    where
        Self: 'c;

    /// Return the graph entry point, or None if the graph is empty.
    fn entry_point(&mut self) -> Option<Result<i64>>;

    /// Get the contents of a single vertex.
    fn get_vertex(&mut self, vertex_id: i64) -> Option<Result<Self::Vertex<'_>>>;
}

/// A node in the Vamana graph.
pub trait GraphVertex {
    type EdgeIterator<'a>: Iterator<Item = i64>
    where
        Self: 'a;

    /// Access the edges of the graph. These may be returned in an arbitrary order.
    fn edges(&self) -> Self::EdgeIterator<'_>;
}

/// Vector store for known vector formats accessible by a record id.
pub trait GraphVectorStore {
    /// Return the format that vectors in the store are encoded in.
    fn format(&self) -> F32VectorCoding;

    /// Return the contents of the vector at vertex, or `None` if the vertex is unknown.
    // TODO: consider removing this method as it is _unsafe_ in the event of a rollback.
    fn get(&mut self, vertex_id: i64) -> Option<Result<&[u8]>>;

    // TODO: extract many vectors into VecVectorStore.
    // TODO: method to turn self into a QueryVectorDistance.
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
    pub fn new<R: GraphVectorIndexReader>(reader: &R, edges: &[Neighbor]) -> Result<Self> {
        if reader.config().index_search_params.num_rerank > 0 {
            let vectors = Self::extract_vectors(&mut reader.rerank_vectors()?, edges)?;
            Ok(Self {
                distance_fn: reader.config().new_distance_function(),
                vectors,
            })
        } else {
            let vectors = Self::extract_vectors(&mut reader.nav_vectors()?, edges)?;
            Ok(Self {
                distance_fn: reader.config().new_nav_distance_function(),
                vectors,
            })
        }
    }

    fn extract_vectors(
        store: &mut impl GraphVectorStore,
        edges: &[Neighbor],
    ) -> Result<Vec<Vec<u8>>> {
        edges
            .iter()
            .map(|n| {
                store
                    .get(n.vertex())
                    .unwrap_or(Err(Error::not_found_error()))
                    .map(|v| v.to_vec())
            })
            .collect::<Result<Vec<_>>>()
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
// TODO: alpha value(s) should be tuneable.
pub(crate) fn select_pruned_edges(
    edges: &[Neighbor],
    max_edges: NonZero<usize>,
    edge_distance_computer: EdgeSetDistanceComputer,
) -> BTreeSet<usize> {
    if edges.is_empty() {
        return BTreeSet::new();
    }

    debug_assert!(edges.is_sorted());

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
                if selected.len() >= max_edges.get() {
                    break;
                }
            }
        }

        if selected.len() >= max_edges.get() {
            break;
        }
    }

    selected
}

/// Prune `edges` down to at most `max_edges`. Use `graph` and `distance_fn` to inform this decision.
/// Returns a split point: all edges before that point are selected, all after are to be dropped.
/// REQUIRES: `edges.is_sorted()`.
// TODO: alpha value(s) should be tuneable.
pub(crate) fn prune_edges(
    edges: &mut [Neighbor],
    max_edges: NonZero<usize>,
    edge_distance_computer: EdgeSetDistanceComputer,
) -> usize {
    let selected = select_pruned_edges(edges, max_edges, edge_distance_computer);
    // Partition edges into selected and unselected.
    for (i, j) in selected.iter().enumerate() {
        edges.swap(i, *j);
    }
    selected.len()
}
