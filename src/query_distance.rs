//! Abstractions for computing the distance between a fixed query vector and other vectors.
//!
//! This abstraction is particularly useful in query paths where we are computing the distance to
//! a query and allows us to hide details about the packing of quantized vectors or implement
//! asymmetric scoring schemes.

use std::borrow::Cow;

use crate::{
    distance::{
        AsymmetricHammingDistance, F32DotProductDistance, F32EuclideanDistance, HammingDistance,
        I8NaiveDistance, I8ScaledUniformDotProduct, I8ScaledUniformDotProductQueryDistance,
        I8ScaledUniformEuclidean, I8ScaledUniformEuclideanQueryDistance, VectorDistance,
        VectorSimilarity,
    },
    quantization::{
        AsymmetricBinaryQuantizer, BinaryQuantizer, I8NaiveQuantizer, Quantizer, VectorQuantizer,
    },
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

#[derive(Debug, Clone)]
struct GenericQueryVectorDistance<'a, D> {
    distance_fn: D,
    query: Cow<'a, [u8]>,
}

impl<'a, D: VectorDistance> GenericQueryVectorDistance<'a, D> {
    fn new(distance_fn: D, query: &'a [f32]) -> Self {
        Self {
            distance_fn,
            query: bytemuck::cast_slice(query).into(),
        }
    }

    fn new_quantized(distance_fn: D, query: &'a [f32], quantizer: impl Quantizer) -> Self {
        let query = quantizer.for_query(query).into();
        Self { distance_fn, query }
    }

    fn from_quantized(distance_fn: D, query: &'a [u8]) -> Self {
        Self {
            distance_fn,
            query: query.into(),
        }
    }
}

impl<'a, D: VectorDistance> QueryVectorDistance for GenericQueryVectorDistance<'a, D> {
    fn distance(&self, vector: &[u8]) -> f64 {
        self.distance_fn.distance(self.query.as_ref(), vector)
    }
}

/// Create a new [QueryVectorDistance] given a query, similarity function, and quantizer.
// TODO: ideally we would indicate the on-disk target format (today specified as Option<VectorQuantizer>)
pub fn new_query_vector_distance_f32<'a>(
    query: &'a [f32],
    similarity: VectorSimilarity,
    quantizer: Option<VectorQuantizer>,
) -> Box<dyn QueryVectorDistance + 'a> {
    match (similarity, quantizer) {
        (VectorSimilarity::Dot, None) => Box::new(GenericQueryVectorDistance::new(
            F32DotProductDistance,
            query,
        )),
        (VectorSimilarity::Euclidean, None) => {
            Box::new(GenericQueryVectorDistance::new(F32EuclideanDistance, query))
        }
        (_, Some(VectorQuantizer::Binary)) => Box::new(GenericQueryVectorDistance::new_quantized(
            HammingDistance,
            query,
            BinaryQuantizer,
        )),
        (_, Some(VectorQuantizer::AsymmetricBinary { n })) => {
            Box::new(GenericQueryVectorDistance::new_quantized(
                AsymmetricHammingDistance,
                query,
                AsymmetricBinaryQuantizer::new(n),
            ))
        }
        (_, Some(VectorQuantizer::I8Naive)) => Box::new(GenericQueryVectorDistance::new_quantized(
            I8NaiveDistance(similarity),
            query,
            I8NaiveQuantizer,
        )),
        (VectorSimilarity::Dot, Some(VectorQuantizer::I8ScaledUniform)) => {
            Box::new(I8ScaledUniformDotProductQueryDistance::new(query))
        }
        (VectorSimilarity::Euclidean, Some(VectorQuantizer::I8ScaledUniform)) => {
            Box::new(I8ScaledUniformEuclideanQueryDistance::new(query))
        }
    }
}

/// Create a new [QueryVectorDistance] that accepts a pre-quantized query.
// TODO: ideally we would indicate the on-disk target format (today specified as Option<VectorQuantizer>)
pub fn new_query_vector_distance_quantized<'a>(
    query: &'a [u8],
    similarity: VectorSimilarity,
    quantizer: Option<VectorQuantizer>,
) -> Box<dyn QueryVectorDistance + 'a> {
    match (similarity, quantizer) {
        (VectorSimilarity::Dot, None) => Box::new(GenericQueryVectorDistance::from_quantized(
            F32DotProductDistance,
            query,
        )),
        (VectorSimilarity::Euclidean, None) => Box::new(
            GenericQueryVectorDistance::from_quantized(F32EuclideanDistance, query),
        ),
        (_, Some(VectorQuantizer::Binary)) => Box::new(GenericQueryVectorDistance::from_quantized(
            HammingDistance,
            query,
        )),
        (_, Some(VectorQuantizer::AsymmetricBinary { n: _ })) => Box::new(
            // XXX this is probably all wrong if you use AsymmetricBinary during indexing.
            // i may just delete this because i don't think it's all that useful.
            // maybe it should just be named "for indexing" and use HammingDistance?
            GenericQueryVectorDistance::from_quantized(AsymmetricHammingDistance, query),
        ),
        (_, Some(VectorQuantizer::I8Naive)) => Box::new(
            GenericQueryVectorDistance::from_quantized(I8NaiveDistance(similarity), query),
        ),
        (VectorSimilarity::Dot, Some(VectorQuantizer::I8ScaledUniform)) => Box::new(
            GenericQueryVectorDistance::from_quantized(I8ScaledUniformDotProduct, query),
        ),
        (VectorSimilarity::Euclidean, Some(VectorQuantizer::I8ScaledUniform)) => Box::new(
            GenericQueryVectorDistance::from_quantized(I8ScaledUniformEuclidean, query),
        ),
    }
}
