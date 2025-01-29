//! Dense vector scoring traits and implementations.

use std::{borrow::Cow, io, str::FromStr};

use serde::{Deserialize, Serialize};
use simsimd::{BinarySimilarity, SpatialSimilarity};

/// Distance function for `f32` vectors.
///
/// This trait is object-safe; it may be instantiated at runtime based on
/// data that appears in a file or other backing store.
pub trait F32VectorDistance: Send + Sync {
    /// Score vectors `a` and `b` against one another. Returns a score
    /// where larger values are better matches.
    ///
    /// Input vectors must be the same length or this function may panic.
    fn distance(&self, a: &[f32], b: &[f32]) -> f64;

    /// Normalize a vector for use with this scoring function.
    /// By default, does nothing.
    fn normalize_vector<'a>(&self, vector: Cow<'a, [f32]>) -> Cow<'a, [f32]> {
        vector
    }
}

/// Distance function for quantized vectors.
///
/// This trait is object-safe; it may be instantiated at runtime based on
/// data that appears in a file or other backing store.
pub trait QuantizedVectorDistance {
    /// Score the `query` vector against the `doc` vector. Returns a score
    /// where larger values are better matches.
    ///
    /// This function is not required to be commutative and may panic if
    /// one of the inputs is misshapen.
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64;
}

/// Functions used for computing a similarity score for high fidelity vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorSimilarity {
    /// Euclidean (l2) distance.
    Euclidean,
    /// Dot product scoring, an approximation of cosine scoring.
    /// Vectors used for this distance function must be normalized.
    Dot,
}

impl VectorSimilarity {
    pub fn new_distance_function(self) -> Box<dyn F32VectorDistance> {
        match self {
            Self::Euclidean => Box::new(EuclideanDistance),
            Self::Dot => Box::new(DotProductDistance),
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
pub struct EuclideanDistance;

impl F32VectorDistance for EuclideanDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f64 {
        SpatialSimilarity::l2sq(a, b).unwrap()
    }
}

/// Computes a score based on the dot product.
#[derive(Debug, Copy, Clone)]
pub struct DotProductDistance;

impl F32VectorDistance for DotProductDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f64 {
        // Assuming values are normalized, this will produce a distance in [0,1]
        (-SpatialSimilarity::dot(a, b).unwrap() + 1.0) / 2.0
    }

    fn normalize_vector<'a>(&self, mut vector: Cow<'a, [f32]>) -> Cow<'a, [f32]> {
        let norm = SpatialSimilarity::dot(&vector, &vector).unwrap().sqrt() as f32;
        for d in vector.to_mut().iter_mut() {
            *d /= norm;
        }
        vector
    }
}

/// Computes a score from two bitmaps using hamming distance.
#[derive(Debug, Copy, Clone)]
pub struct HammingDistance;

impl QuantizedVectorDistance for HammingDistance {
    fn distance(&self, a: &[u8], b: &[u8]) -> f64 {
        BinarySimilarity::hamming(a, b).unwrap()
    }
}

/// Computes a score between a query and doc vectors produced by [crate::quantization::AsymmetricBinaryQuantizer]
#[derive(Debug, Copy, Clone)]
pub struct AsymmetricHammingDistance;

impl QuantizedVectorDistance for AsymmetricHammingDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        assert_eq!(query.len() % doc.len(), 0);
        query
            .chunks(doc.len())
            .enumerate()
            .map(|(i, v)| {
                BinarySimilarity::hamming(doc, v).expect("same vector length") as usize
                    * (1usize << i)
            })
            .sum::<usize>() as f64
    }
}

struct I8NaiveVector<'a> {
    vector: &'a [i8],
    #[allow(dead_code)]
    squared_l2_norm: f32,
}

impl<'a> From<&'a [u8]> for I8NaiveVector<'a> {
    fn from(value: &'a [u8]) -> Self {
        assert!(value.len() >= std::mem::size_of::<f32>());
        let (vector_bytes, norm_bytes) = value.split_at(value.len() - std::mem::size_of::<f32>());
        Self {
            vector: unsafe {
                std::slice::from_raw_parts(vector_bytes.as_ptr() as *const i8, vector_bytes.len())
            },
            squared_l2_norm: f32::from_le_bytes(norm_bytes.try_into().unwrap()),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I8DotScorer;

impl QuantizedVectorScorer for I8DotScorer {
    fn score(&self, query: &[u8], doc: &[u8]) -> f64 {
        let qv = I8NaiveVector::from(query);
        let dv = I8NaiveVector::from(doc);
        SpatialSimilarity::dot(qv.vector, dv.vector).unwrap()
    }
}
