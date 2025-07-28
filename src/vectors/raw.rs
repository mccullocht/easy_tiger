//! Raw float 32 vector coding and distance computation.
//!
//! Vectors are stored as a sequence of raw little-endian f32 values without any additional metadata.
//! The L2 Normalized coding performs l2 normalization before storing the vector, making it a better
//! fit for angular distance computation.

use std::borrow::Cow;

use simsimd::SpatialSimilarity;

use crate::{
    distance::{dot_f32, dot_f32_bytes, l2sq_f32, l2sq_f32_bytes},
    vectors::{F32VectorCoder, F32VectorDistance, QueryVectorDistance, VectorDistance},
};

#[derive(Debug, Copy, Clone)]
pub struct RawF32VectorCoder;

impl F32VectorCoder for RawF32VectorCoder {
    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions * std::mem::size_of::<f32>()
    }

    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        assert!(out.len() >= std::mem::size_of_val(vector));
        for (d, o) in vector
            .iter()
            .zip(out.chunks_mut(std::mem::size_of::<f32>()))
        {
            o.copy_from_slice(&d.to_le_bytes());
        }
    }

    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        vector.iter().flat_map(|d| d.to_le_bytes()).collect()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct RawL2NormalizedF32VectorCoder;

impl RawL2NormalizedF32VectorCoder {
    fn scale(v: &[f32]) -> f32 {
        let l2_norm = SpatialSimilarity::dot(v, v).unwrap().sqrt() as f32;
        l2_norm.recip()
    }
}

impl F32VectorCoder for RawL2NormalizedF32VectorCoder {
    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions * std::mem::size_of::<f32>()
    }

    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        assert!(out.len() >= std::mem::size_of_val(vector));
        let scale = Self::scale(vector);
        for (d, o) in vector
            .iter()
            .zip(out.chunks_mut(std::mem::size_of::<f32>()))
        {
            o.copy_from_slice(&(*d * scale).to_le_bytes());
        }
    }

    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let scale = Self::scale(vector);
        vector
            .iter()
            .flat_map(|d| (*d * scale).to_le_bytes())
            .collect()
    }
}

/// Computes a score based on l2 distance.
#[derive(Debug, Copy, Clone)]
pub struct F32EuclideanDistance;

impl VectorDistance for F32EuclideanDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        l2sq_f32_bytes(query, doc)
    }
}

impl F32VectorDistance for F32EuclideanDistance {
    fn distance_f32(&self, a: &[f32], b: &[f32]) -> f64 {
        l2sq_f32(a, b)
    }
}

/// Computes a score based on the dot product.
#[derive(Debug, Copy, Clone)]
pub struct F32DotProductDistance;

impl VectorDistance for F32DotProductDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        // Assuming values are normalized, this will produce a distance in [0,1]
        (-dot_f32_bytes(query, doc) + 1.0) / 2.0
    }
}

impl F32VectorDistance for F32DotProductDistance {
    fn distance_f32(&self, a: &[f32], b: &[f32]) -> f64 {
        // Assuming values are normalized, this will produce a distance in [0,1]
        (-dot_f32(a, b) + 1.0) / 2.0
    }

    fn normalize<'a>(&self, mut vector: Cow<'a, [f32]>) -> Cow<'a, [f32]> {
        let norm = dot_f32(&vector, &vector).sqrt() as f32;
        for d in vector.to_mut().iter_mut() {
            *d /= norm;
        }
        vector
    }
}

#[derive(Debug, Clone)]
pub struct F32QueryVectorDistance<'a, D> {
    distance_fn: D,
    query: Cow<'a, [f32]>,
}

impl<'a, D: F32VectorDistance> F32QueryVectorDistance<'a, D> {
    pub fn new(distance_fn: D, query: &'a [f32], l2_normalize: bool) -> Self {
        let query = if l2_normalize {
            crate::distance::l2_normalize(query)
        } else {
            query.into()
        };
        Self { distance_fn, query }
    }
}

impl<'a, D: F32VectorDistance> QueryVectorDistance for F32QueryVectorDistance<'a, D> {
    fn distance(&self, vector: &[u8]) -> f64 {
        self.distance_fn
            .distance(bytemuck::cast_slice(self.query.as_ref()), vector)
    }
}
