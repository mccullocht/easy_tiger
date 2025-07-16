//! Abstractions for computing the distance between a fixed query vector and other vectors.
//!
//! This abstraction is particularly useful in query paths where we are computing the distance to
//! a query and allows us to hide details about the packing of quantized vectors or implement
//! asymmetric scoring schemes.

use crate::{
    distance::{
        F32DotProductDistance, F32EuclideanDistance, HammingDistance, I8NaiveDistance,
        VectorDistance, VectorSimilarity,
    },
    quantization::VectorQuantizer,
};

// XXX I actually have two separate problems: one is that I have no mechanism for replacing the
// format of the "raw" vectors, which overlaps with quantization (except the default is lossless),
// the second is that implementing scoring in this kind of scheme is annoying. Should I focus on
// the former? Seems moree productive.

// XXX format/quantization and distance are closely interrelated and it probably doesn't make
// sense for the formatting code to live in a different place from the distance code. i might need
// both to get anything done.

/// Compute the distance between a fixed vector provided at creation time and other vectors.
/// This is often useful in query flows where everything references a specific point.
pub trait QueryVectorDistance: Send + Sync {
    fn distance(&self, vector: &[u8]) -> f64;
}

#[derive(Debug, Copy, Clone)]
struct GenericQueryVectorDistance<'a, D> {
    distance_fn: D,
    query: &'a [u8],
}

impl<'a, D: VectorDistance> QueryVectorDistance for GenericQueryVectorDistance<'a, D> {
    fn distance(&self, vector: &[u8]) -> f64 {
        self.distance_fn.distance(self.query, vector)
    }
}

/// Create a new [QueryVectorDistance] given a query, similarity function, and quantizer.
pub fn new_query_vector_distance<'a>(
    query: &'a [u8],
    similarity: VectorSimilarity,
    quantizer: Option<VectorQuantizer>,
) -> Box<dyn QueryVectorDistance + 'a> {
    match (similarity, quantizer) {
        (VectorSimilarity::Dot, None) => Box::new(GenericQueryVectorDistance {
            distance_fn: F32DotProductDistance,
            query,
        }),
        (VectorSimilarity::Euclidean, None) => Box::new(GenericQueryVectorDistance {
            distance_fn: F32EuclideanDistance,
            query,
        }),
        (_, Some(VectorQuantizer::Binary)) => Box::new(GenericQueryVectorDistance {
            distance_fn: HammingDistance,
            query,
        }),
        // TODO: this should be implemented by some other means and GenericQueryVectorDistance
        // should be called "SymmetricQueryVectorDistance"
        (_, Some(VectorQuantizer::AsymmetricBinary { n: _ })) => unimplemented!(),
        (_, Some(VectorQuantizer::I8Naive)) => Box::new(GenericQueryVectorDistance {
            distance_fn: I8NaiveDistance(similarity),
            query,
        }),
    }
}
