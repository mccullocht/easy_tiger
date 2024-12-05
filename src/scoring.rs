//! Dense vector scoring traits and implementations.

use std::{io, str::FromStr};

use serde::{Deserialize, Serialize};
use simsimd::{BinarySimilarity, SpatialSimilarity};

/// Scorer for `f32` vectors.
///
/// This trait is object-safe; it may be instantiated at runtime based on
/// data that appears in a file or other backing store.
pub trait F32VectorScorer: Send + Sync {
    /// Score vectors `a` and `b` against one another. Returns a score
    /// where larger values are better matches.
    ///
    /// Input vectors must be the same length or this function may panic.
    fn score(&self, a: &[f32], b: &[f32]) -> f64;

    /// Normalize a vector for use with this scoring function.
    /// By default, does nothing.
    fn normalize(&self, _vector: &mut [f32]) {}
}

/// Scorer for quantized vectors.
///
/// This trait is object-safe; it may be instantiated at runtime based on
/// data that appears in a file or other backing store.
pub trait QuantizedVectorScorer {
    /// Score the `query` vector against the `doc` vector. Returns a score
    /// where larger values are better matches.
    ///
    /// This function is not required to be commutative and may panic if
    /// one of the inputs is misshapen.
    fn score(&self, query: &[u8], doc: &[u8]) -> f64;
}

/// Functions used for computing a similarity score for high fidelity vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorSimilarity {
    /// Euclidean (l2) distance.
    Euclidean,
    /// Dot product scoring, an approximation of cosine scoring.
    /// Vectors used for this scorer must be normalized.
    Dot,
}

impl VectorSimilarity {
    pub fn new_scorer(self) -> Box<dyn F32VectorScorer> {
        match self {
            Self::Euclidean => Box::new(EuclideanScorer),
            Self::Dot => Box::new(DotProductScorer),
        }
    }
}

impl Default for VectorSimilarity {
    fn default() -> Self {
        Self::Dot
    }
}

impl FromStr for VectorSimilarity {
    type Err = io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "euclidean" => Ok(VectorSimilarity::Euclidean),
            "dot" => Ok(VectorSimilarity::Dot),
            x => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown similarity fuction {}", x),
            )),
        }
    }
}

/// Computes a score based on l2 distance.
#[derive(Debug, Copy, Clone)]
pub struct EuclideanScorer;

impl F32VectorScorer for EuclideanScorer {
    fn score(&self, a: &[f32], b: &[f32]) -> f64 {
        1f64 / (1f64 + SpatialSimilarity::l2sq(a, b).unwrap())
    }
}

/// Computes a score based on the dot product.
#[derive(Debug, Copy, Clone)]
pub struct DotProductScorer;

impl F32VectorScorer for DotProductScorer {
    fn score(&self, a: &[f32], b: &[f32]) -> f64 {
        // Assuming values are normalized, this will produce a score in [0,1]
        (1f64 + SpatialSimilarity::dot(a, b).unwrap()) / 2f64
    }

    fn normalize(&self, vector: &mut [f32]) {
        let norm = SpatialSimilarity::dot(vector, vector).unwrap().sqrt() as f32;
        for d in vector.iter_mut() {
            *d /= norm;
        }
    }
}

/// Computes a score from two bitmaps using hamming distance.
#[derive(Debug, Copy, Clone)]
pub struct HammingScorer;

impl QuantizedVectorScorer for HammingScorer {
    fn score(&self, a: &[u8], b: &[u8]) -> f64 {
        let dim = (a.len() * 8) as f64;
        (dim - BinarySimilarity::hamming(a, b).unwrap()) / dim
    }
}
