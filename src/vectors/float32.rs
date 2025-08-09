//! Raw float 32 vector coding and distance computation.
//!
//! Vectors are stored as a sequence of raw little-endian coded f32 values.
//!
//! For Cosine similarity the vector will be normalized during encoding. When scoring float vectors
//! we will assume the vectors are unnormalized.

use std::borrow::Cow;

use simsimd::SpatialSimilarity;

use crate::{
    distance::{dot_f32, dot_f32_bytes, l2_normalize, l2sq_f32, l2sq_f32_bytes},
    vectors::{
        F32VectorCoder, F32VectorDistance, QueryVectorDistance as QueryVectorDistanceT,
        VectorDistance, VectorSimilarity,
    },
};

#[derive(Debug, Copy, Clone)]
pub struct VectorCoder(VectorSimilarity);

impl VectorCoder {
    pub fn new(similarity: VectorSimilarity) -> Self {
        Self(similarity)
    }

    fn encode_it(vector: impl ExactSizeIterator<Item = f32>, out: &mut [u8]) {
        for (d, o) in vector.zip(out.as_chunks_mut::<{ std::mem::size_of::<f32>() }>().0) {
            *o = d.to_le_bytes();
        }
    }
}

impl F32VectorCoder for VectorCoder {
    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions * std::mem::size_of::<f32>()
    }

    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        assert!(out.len() >= std::mem::size_of_val(vector));
        let vector_it = vector.iter().copied();
        if self.0.l2_normalize() {
            let scale = (1.0
                / SpatialSimilarity::dot(vector, vector)
                    .expect("identical vectors")
                    .sqrt()) as f32;
            Self::encode_it(vector_it.map(|d| d * scale), out);
        } else {
            Self::encode_it(vector_it, out);
        }
    }

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        // NB: if the input value was l2 normalized we can't recreate that value -- we've already
        // discarded the norm.
        let f32_len = std::mem::size_of::<f32>();
        assert!(encoded.len() % f32_len == 0);
        Some(
            encoded
                .chunks(f32_len)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        )
    }
}

/// Computes a score based on l2 distance.
#[derive(Debug, Copy, Clone)]
pub struct EuclideanDistance;

impl VectorDistance for EuclideanDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        l2sq_f32_bytes(query, doc)
    }
}

impl F32VectorDistance for EuclideanDistance {
    fn distance_f32(&self, a: &[f32], b: &[f32]) -> f64 {
        l2sq_f32(a, b)
    }
}

/// Computes a score based on the dot product.
#[derive(Debug, Copy, Clone)]
pub struct DotProductDistance;

impl VectorDistance for DotProductDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        // Assuming values are normalized, this will produce a distance in [0,1]
        (-dot_f32_bytes(query, doc) + 1.0) / 2.0
    }
}

impl F32VectorDistance for DotProductDistance {
    fn distance_f32(&self, a: &[f32], b: &[f32]) -> f64 {
        // Assuming values are normalized, this will produce a distance in [0,1]
        (-dot_f32(a, b) + 1.0) / 2.0
    }
}

#[derive(Debug, Copy, Clone)]
pub struct CosineDistance;

impl VectorDistance for CosineDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        // Vectors are normalized during encoding so we can make this fast.
        DotProductDistance.distance(query, doc)
    }
}

impl F32VectorDistance for CosineDistance {
    fn distance_f32(&self, a: &[f32], b: &[f32]) -> f64 {
        // We can't assume the vectors have been processed/normalized here so we have to perform
        // full cosine similarity.
        let (ab, a2, b2) = a
            .iter()
            .zip(b.iter())
            .map(|(a, b)| (*a * b, *a * *a, *b * *b))
            .fold((0.0, 0.0, 0.0), |s, x| (s.0 + x.0, s.1 + x.1, s.2 + x.2));
        let cos = ab / (a2.sqrt() * b2.sqrt());
        (-cos as f64 + 1.0) / 2.0
    }
}

#[derive(Debug, Clone)]
pub struct QueryVectorDistance<'a, D> {
    distance_fn: D,
    query: Cow<'a, [f32]>,
}

impl<'a, D: F32VectorDistance> QueryVectorDistance<'a, D> {
    pub fn new(distance_fn: D, query: Cow<'a, [f32]>) -> Self {
        Self { distance_fn, query }
    }
}

impl<'a, D: F32VectorDistance> QueryVectorDistanceT for QueryVectorDistance<'a, D> {
    fn distance(&self, vector: &[u8]) -> f64 {
        self.distance_fn
            .distance(bytemuck::cast_slice(self.query.as_ref()), vector)
    }
}

pub fn new_query_vector_distance<'a>(
    similarity: VectorSimilarity,
    query: Cow<'a, [f32]>,
) -> Box<dyn QueryVectorDistanceT + 'a> {
    match similarity {
        VectorSimilarity::Cosine => Box::new(QueryVectorDistance::new(
            CosineDistance,
            l2_normalize(query),
        )),
        VectorSimilarity::Dot => Box::new(QueryVectorDistance::new(DotProductDistance, query)),
        VectorSimilarity::Euclidean => Box::new(QueryVectorDistance::new(EuclideanDistance, query)),
    }
}
