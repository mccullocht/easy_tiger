//! Raw float 32 vector coding and distance computation.
//!
//! Vectors are stored as a sequence of raw little-endian f32 values without any additional metadata.
//! The L2 Normalized coding performs l2 normalization before storing the vector, making it a better
//! fit for angular distance computation.

use std::borrow::Cow;

use simsimd::SpatialSimilarity;

use crate::{
    distance::{dot_f32, dot_f32_bytes, l2sq_f32, l2sq_f32_bytes},
    vectors::{F32VectorCoder, F32VectorDistance, VectorDistance, VectorSimilarity},
};

#[derive(Debug, Copy, Clone)]
pub struct VectorCoder(bool);

impl VectorCoder {
    pub fn new(similarity: VectorSimilarity) -> Self {
        Self(similarity.l2_normalize())
    }
}

impl F32VectorCoder for VectorCoder {
    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions * std::mem::size_of::<f32>()
    }

    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        assert!(out.len() >= std::mem::size_of_val(vector));
        let encode_it = vector
            .iter()
            .zip(out.chunks_mut(std::mem::size_of::<f32>()));
        if self.0 {
            let scale = (1.0
                / SpatialSimilarity::dot(vector, vector)
                    .expect("identical vectors")
                    .sqrt()) as f32;
            for (d, o) in encode_it {
                o.copy_from_slice(&(*d * scale).to_le_bytes());
            }
        } else {
            for (d, o) in encode_it {
                o.copy_from_slice(&d.to_le_bytes());
            }
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

#[derive(Debug, Clone)]
pub struct QueryVectorDistance<'a, D> {
    distance_fn: D,
    query: Cow<'a, [f32]>,
}

impl<'a, D: F32VectorDistance> QueryVectorDistance<'a, D> {
    pub fn new(distance_fn: D, query: Cow<'a, [f32]>, l2_normalize: bool) -> Self {
        let query = if l2_normalize {
            crate::distance::l2_normalize(query)
        } else {
            query.into()
        };
        Self { distance_fn, query }
    }
}

impl<'a, D: F32VectorDistance> super::QueryVectorDistance for QueryVectorDistance<'a, D> {
    fn distance(&self, vector: &[u8]) -> f64 {
        self.distance_fn
            .distance(bytemuck::cast_slice(self.query.as_ref()), vector)
    }
}
