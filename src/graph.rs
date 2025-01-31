//! Interfaces for graph configuration and access abstractions.
//!
//! Graph access traits provided here are used during graph search, and allow us to
//! build indices with both WiredTiger backing and in-memory backing for bulk loads.

use std::{borrow::Cow, collections::BTreeSet, io, num::NonZero, ops::Deref, str::FromStr};

use serde::{Deserialize, Serialize};
use wt_mdb::{Error, Result};

use crate::{
    quantization::{Quantizer, VectorQuantizer},
    scoring::{F32VectorScorer, QuantizedVectorScorer, VectorSimilarity},
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
    /// Raw vector is stored in the graph alongside the edges.
    RawVectorInGraph,
}

impl Default for GraphLayout {
    fn default() -> Self {
        Self::RawVectorInGraph
    }
}

impl FromStr for GraphLayout {
    type Err = io::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "split" => Ok(Self::Split),
            "raw_vector_in_graph" => Ok(Self::RawVectorInGraph),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Unknown graph layout {}", s),
            )),
        }
    }
}

/// Configuration describing graph shape and construction. Used to read and mutate the graph.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct GraphConfig {
    pub dimensions: NonZero<usize>,
    pub similarity: VectorSimilarity,
    pub quantizer: VectorQuantizer,
    #[serde(default)]
    pub layout: GraphLayout,
    pub max_edges: NonZero<usize>,
    pub index_search_params: GraphSearchParams,
}

impl GraphConfig {
    /// Return a scorer for high fidelity vectors in the index.
    pub fn new_scorer(&self) -> Box<dyn F32VectorScorer> {
        self.similarity.new_scorer()
    }

    pub fn new_quantizer(&self) -> Box<dyn Quantizer> {
        self.quantizer.new_quantizer()
    }

    /// Return a scorer for quantized navigational vectors in the index.
    pub fn new_nav_scorer(&self) -> Box<dyn QuantizedVectorScorer> {
        self.quantizer.new_scorer(&self.similarity)
    }
}

/// `GraphVectorIndexReader` is used to generate objects for graph navigation.
pub trait GraphVectorIndexReader {
    type Graph<'a>: Graph + 'a
    where
        Self: 'a;
    type RawVectorStore<'a>: RawVectorStore + 'a
    where
        Self: 'a;
    type NavVectorStore<'a>: NavVectorStore + 'a
    where
        Self: 'a;

    /// Return config for this graph.
    fn config(&self) -> &GraphConfig;

    /// Return an object that can be used to navigate the graph.
    fn graph(&self) -> Result<Self::Graph<'_>>;

    /// Return an object that can be used to read raw vectors.
    fn raw_vectors(&self) -> Result<Self::RawVectorStore<'_>>;

    /// Return an object that can be used to read navigational vectors.
    fn nav_vectors(&self) -> Result<Self::NavVectorStore<'_>>;
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

    /// Access the raw float vector if stored with the edges.
    fn vector(&self) -> Option<Cow<'_, [f32]>>;
}

/// A raw float vector.
///
/// This implementation may have Cow reference to backing bytes (if possible) or have a copy of
/// the vector data.
pub enum RawVector<'a> {
    Bytes { bytes: Cow<'a, [u8]>, dim: usize },
    Cow(Cow<'a, [f32]>),
}

impl<'a> RawVector<'a> {
    pub fn from_cow_partial(bytes: Cow<'a, [u8]>, dim: usize) -> Self {
        #[cfg(target_endian = "little")]
        {
            // WiredTiger does not guarantee that the returned memory will be aligned, a
            // Try to align it and if that fails, copy the data.
            let (prefix, _, _) = unsafe { bytes.align_to::<f32>() };
            if prefix.is_empty() {
                return Self::Bytes { bytes, dim };
            }
        }

        Self::Cow(Self::bytes_to_vec(bytes, dim).into())
    }

    pub fn to_vec(self) -> Vec<f32> {
        match self {
            Self::Bytes { bytes, dim } => Self::bytes_to_vec(bytes, dim),
            Self::Cow(v) => v.to_vec(),
        }
    }

    pub fn bytes_len(&self) -> usize {
        let dim = match self {
            Self::Bytes { bytes: _, dim } => *dim,
            Self::Cow(v) => v.len(),
        };
        dim * std::mem::size_of::<f32>()
    }

    fn bytes_to_vec(bytes: Cow<'_, [u8]>, dim: usize) -> Vec<f32> {
        bytes
            .chunks(std::mem::size_of::<f32>())
            .take(dim)
            .map(|b| f32::from_le_bytes(b.try_into().expect("array of 4 conversion")))
            .collect::<Vec<_>>()
    }
}

impl<'a> From<Cow<'a, [f32]>> for RawVector<'a> {
    fn from(value: Cow<'a, [f32]>) -> Self {
        Self::Cow(value)
    }
}

impl From<Vec<f32>> for RawVector<'_> {
    fn from(value: Vec<f32>) -> Self {
        Self::Cow(value.into())
    }
}

impl<'a> From<&'a [f32]> for RawVector<'a> {
    fn from(value: &'a [f32]) -> Self {
        Self::Cow(value.into())
    }
}

impl Deref for RawVector<'_> {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        match self {
            // Safety: From conversion ensures that bytes is aligned before producing Borrowed.
            Self::Bytes { bytes, dim } => unsafe { &bytes.align_to::<f32>().1[..*dim] },
            Self::Cow(v) => v,
        }
    }
}

/// Vector store for raw vectors used to produce the highest fidelity scores.
pub trait RawVectorStore {
    /// Get the raw vector for the given vertex.
    fn get_raw_vector(&mut self, vertex_id: i64) -> Option<Result<RawVector<'_>>>;
}

/// Vector store for vectors used to navigate the graph.
pub trait NavVectorStore {
    /// Get the navigation vector for the given vertex.
    fn get_nav_vector(&mut self, vertex_id: i64) -> Option<Result<Cow<'_, [u8]>>>;
}

/// Select the indices of `edges` that should remain when pruning down to at most `max_edges`.
/// `graph` is used to access vectors and `scorer` is used to compare vectors when making pruning
/// decisions.
/// REQUIRES: `edges.is_sorted()`.
// TODO: alpha value(s) should be tuneable.
pub(crate) fn select_pruned_edges(
    edges: &[Neighbor],
    max_edges: NonZero<usize>,
    raw_vectors: &mut impl RawVectorStore,
    scorer: &dyn F32VectorScorer,
) -> Result<BTreeSet<usize>> {
    if edges.is_empty() {
        return Ok(BTreeSet::new());
    }

    debug_assert!(edges.is_sorted());

    // Obtain all the vectors to make relative neighbor graph scoring easier.
    let vectors = edges
        .iter()
        .map(|n| {
            raw_vectors
                .get_raw_vector(n.vertex())
                .unwrap_or(Err(Error::not_found_error()))
                .map(|v| v.to_vec())
        })
        .collect::<Result<Vec<_>>>()?;

    // TODO: replace with a fixed length bitset
    let mut selected = BTreeSet::new();
    selected.insert(0); // we always keep the first node.
    for alpha in [1.0, 1.2] {
        for (i, e) in edges.iter().enumerate().skip(1) {
            if selected.contains(&i) {
                continue;
            }

            let e_vec = &vectors[i];
            if !selected
                .iter()
                .take_while(|s| **s < i)
                .any(|s| scorer.distance(e_vec, &vectors[*s]) < e.distance * alpha)
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

    Ok(selected)
}

/// Prune `edges` down to at most `max_edges`. Use `graph` and `scorer` to inform this decision.
/// Returns a split point: all edges before that point are selected, all after are to be dropped.
/// REQUIRES: `edges.is_sorted()`.
// TODO: alpha value(s) should be tuneable.
pub(crate) fn prune_edges(
    edges: &mut [Neighbor],
    max_edges: NonZero<usize>,
    raw_vectors: &mut impl RawVectorStore,
    scorer: &dyn F32VectorScorer,
) -> Result<usize> {
    let selected = select_pruned_edges(edges, max_edges, raw_vectors, scorer)?;

    // Partition edges into selected and unselected.
    for (i, j) in selected.iter().enumerate() {
        edges.swap(i, *j);
    }

    Ok(selected.len())
}
