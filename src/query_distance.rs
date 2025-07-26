//! Abstractions for computing the distance between a fixed query vector and other vectors.
//!
//! This abstraction is particularly useful in query paths where we are computing the distance to
//! a query and allows us to hide details about the packing of quantized vectors or implement
//! asymmetric scoring schemes.

use std::borrow::Cow;

use crate::{
    distance::{
        AsymmetricHammingDistance, F32DotProductDistance, F32EuclideanDistance, F32VectorDistance,
        HammingDistance, I8NaiveDistance, I8ScaledUniformDotProduct,
        I8ScaledUniformDotProductQueryDistance, I8ScaledUniformEuclidean,
        I8ScaledUniformEuclideanQueryDistance, VectorDistance, VectorSimilarity,
    },
    vectors::{
        AsymmetricBinaryQuantizedVectorCoder, BinaryQuantizedVectorCoder, F32VectorCoder,
        F32VectorCoding, I8NaiveVectorCoder,
    },
};

/// Compute the distance between a fixed vector provided at creation time and other vectors.
/// This is often useful in query flows where everything references a specific point.
pub trait QueryVectorDistance: Send + Sync {
    fn distance(&self, vector: &[u8]) -> f64;
}

#[derive(Debug, Clone)]
struct F32QueryVectorDistance<'a, D> {
    distance_fn: D,
    query: Cow<'a, [f32]>,
}

impl<'a, D: F32VectorDistance> F32QueryVectorDistance<'a, D> {
    fn new(distance_fn: D, query: &'a [f32]) -> Self {
        let query = distance_fn.normalize(query.into());
        Self { distance_fn, query }
    }
}

impl<'a, D: F32VectorDistance> QueryVectorDistance for F32QueryVectorDistance<'a, D> {
    fn distance(&self, vector: &[u8]) -> f64 {
        self.distance_fn
            .distance(bytemuck::cast_slice(self.query.as_ref()), vector)
    }
}

#[derive(Debug, Clone)]
struct QuantizedQueryVectorDistance<'a, D> {
    distance_fn: D,
    query: Cow<'a, [u8]>,
}

impl<'a, D: VectorDistance> QuantizedQueryVectorDistance<'a, D> {
    fn from_f32(distance_fn: D, query: &'a [f32], coder: impl F32VectorCoder) -> Self {
        let query = coder.encode(query).into();
        Self { distance_fn, query }
    }

    fn from_quantized(distance_fn: D, query: &'a [u8]) -> Self {
        Self {
            distance_fn,
            query: query.into(),
        }
    }
}

impl<'a, D: VectorDistance> QueryVectorDistance for QuantizedQueryVectorDistance<'a, D> {
    fn distance(&self, vector: &[u8]) -> f64 {
        self.distance_fn.distance(self.query.as_ref(), vector)
    }
}

/// Create a new [QueryVectorDistance] given a query, similarity function, and quantizer.
pub fn new_query_vector_distance_f32<'a>(
    query: &'a [f32],
    similarity: VectorSimilarity,
    coding: F32VectorCoding,
) -> Box<dyn QueryVectorDistance + 'a> {
    match (similarity, coding) {
        // XXX treating raw and raw l2norm as equivalent -- a feature or a bug???
        // in this case F32VectorDistance is papering over the problem by normalizing for dot.
        // XXX this could theoretically be problematic if we use l2 norm with euclidean distance
        // as i might have normalized the on-disk vectors without normalizing the query.
        (VectorSimilarity::Dot, F32VectorCoding::Raw)
        | (VectorSimilarity::Dot, F32VectorCoding::RawL2Normalized) => {
            Box::new(F32QueryVectorDistance::new(F32DotProductDistance, query))
        }
        (VectorSimilarity::Euclidean, F32VectorCoding::Raw)
        | (VectorSimilarity::Euclidean, F32VectorCoding::RawL2Normalized) => {
            Box::new(F32QueryVectorDistance::new(F32EuclideanDistance, query))
        }
        (_, F32VectorCoding::BinaryQuantized) => Box::new(QuantizedQueryVectorDistance::from_f32(
            HammingDistance,
            query,
            BinaryQuantizedVectorCoder,
        )),
        (_, F32VectorCoding::NBitBinaryQuantized(n)) => {
            Box::new(QuantizedQueryVectorDistance::from_f32(
                AsymmetricHammingDistance,
                query,
                AsymmetricBinaryQuantizedVectorCoder::new(n),
            ))
        }
        (_, F32VectorCoding::I8NaiveQuantized) => Box::new(QuantizedQueryVectorDistance::from_f32(
            I8NaiveDistance(similarity),
            query,
            I8NaiveVectorCoder,
        )),
        (VectorSimilarity::Dot, F32VectorCoding::I8ScaledUniformQuantized) => {
            Box::new(I8ScaledUniformDotProductQueryDistance::new(query))
        }
        (VectorSimilarity::Euclidean, F32VectorCoding::I8ScaledUniformQuantized) => {
            Box::new(I8ScaledUniformEuclideanQueryDistance::new(query))
        }
    }
}

/// Create a new [QueryVectorDistance] for indexing that _requires_ symmetrical distance computation.
// TODO: ideally we would indicate the on-disk target format (today specified as Option<VectorQuantizer>)
pub fn new_query_vector_distance_indexing<'a>(
    query: &'a [u8],
    similarity: VectorSimilarity,
    coding: F32VectorCoding,
) -> Box<dyn QueryVectorDistance + 'a> {
    match (similarity, coding) {
        (VectorSimilarity::Dot, F32VectorCoding::Raw)
        | (VectorSimilarity::Dot, F32VectorCoding::RawL2Normalized) => Box::new(
            QuantizedQueryVectorDistance::from_quantized(F32DotProductDistance, query),
        ),
        (VectorSimilarity::Euclidean, F32VectorCoding::Raw)
        | (VectorSimilarity::Euclidean, F32VectorCoding::RawL2Normalized) => Box::new(
            QuantizedQueryVectorDistance::from_quantized(F32EuclideanDistance, query),
        ),
        (_, F32VectorCoding::BinaryQuantized) => Box::new(
            QuantizedQueryVectorDistance::from_quantized(HammingDistance, query),
        ),
        (_, F32VectorCoding::NBitBinaryQuantized(_)) => Box::new(
            QuantizedQueryVectorDistance::from_quantized(HammingDistance, query),
        ),
        (_, F32VectorCoding::I8NaiveQuantized) => Box::new(
            QuantizedQueryVectorDistance::from_quantized(I8NaiveDistance(similarity), query),
        ),
        (VectorSimilarity::Dot, F32VectorCoding::I8ScaledUniformQuantized) => Box::new(
            QuantizedQueryVectorDistance::from_quantized(I8ScaledUniformDotProduct, query),
        ),
        (VectorSimilarity::Euclidean, F32VectorCoding::I8ScaledUniformQuantized) => Box::new(
            QuantizedQueryVectorDistance::from_quantized(I8ScaledUniformEuclidean, query),
        ),
    }
}
