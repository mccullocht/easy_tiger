//! Scaled uniform coding uses the maximum absolute value across all dimensions for a given vector
//! to produce an i8 value in [-127,127]. It stores both a value to invert the scaling as well as
//! the squared l2 norm for computing euclidean distance using dot product.
//!
//! Unlike some other quantization schemes this does not rely on anything computed across a sample
//! of the data set -- no means or centroids, no quantiles. For transformer models which produce
//! relatively well centered vectors this seems to be effective enough that we may discard the
//! original f32 vectors.
use std::borrow::Cow;

use crate::{
    distance::{dot_f32, l2_normalize},
    vectors::{F32VectorCoder, QueryVectorDistance, VectorDistance},
};

#[derive(Debug, Copy, Clone)]
pub struct I8ScaledUniformVectorCoder;

impl F32VectorCoder for I8ScaledUniformVectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let l2_norm = crate::distance::dot_f32(vector, vector).sqrt() as f32;
        let (scale, inv_scale) =
            if let Some(max) = vector.iter().map(|d| d.abs()).max_by(|a, b| a.total_cmp(b)) {
                (
                    (f64::from(i8::MAX) / max as f64) as f32,
                    (max as f64 / f64::from(i8::MAX)) as f32,
                )
            } else {
                (0.0, 0.0)
            };

        out[0..4].copy_from_slice(&inv_scale.to_le_bytes());
        out[4..8].copy_from_slice(&l2_norm.to_le_bytes());
        for (d, o) in vector.iter().zip(out[8..].iter_mut()) {
            *o = ((*d * scale).round() as i8).to_le_bytes()[0];
        }
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions + std::mem::size_of::<f32>() * 2
    }
}

// TODO: quantizer that is non-uniform for MRL vectors.
// Bonus points if it can still be scored on the quantized rep instead of de-quantizing.

#[derive(Debug, Copy, Clone)]
struct I8ScaledUniformVector<'a>(&'a [u8]);

impl I8ScaledUniformVector<'_> {
    fn dot_unnormalized(&self, other: &Self) -> f64 {
        self.vector()
            .iter()
            .zip(other.vector().iter())
            .map(|(s, o)| *s as i32 * *o as i32)
            .sum::<i32>() as f64
            * self.scale()
            * other.scale()
    }

    fn dequantized_unnormalized_iter(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        self.vector()
            .iter()
            .map(|d| *d as f32 * self.scale() as f32)
    }

    fn dequantized_normalized_iter(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        let scale = (self.scale() / self.l2_norm()) as f32;
        self.vector().iter().map(move |d| *d as f32 * scale)
    }

    fn scale(&self) -> f64 {
        f32::from_le_bytes(self.0[0..4].try_into().unwrap()).into()
    }

    fn l2_norm_sq(&self) -> f64 {
        self.l2_norm() * self.l2_norm()
    }

    fn l2_norm(&self) -> f64 {
        f32::from_le_bytes(self.0[4..8].try_into().unwrap()).into()
    }

    fn vector(&self) -> &[i8] {
        bytemuck::cast_slice(&self.0[8..])
    }
}

impl<'a> From<&'a [u8]> for I8ScaledUniformVector<'a> {
    fn from(value: &'a [u8]) -> Self {
        assert!(value.len() >= 8);
        Self(value)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I8ScaledUniformDotProduct;

impl VectorDistance for I8ScaledUniformDotProduct {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = I8ScaledUniformVector::from(query);
        let doc = I8ScaledUniformVector::from(doc);
        let dot = query.dot_unnormalized(&doc) * query.l2_norm().recip() * doc.l2_norm().recip();
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Clone)]
pub struct I8ScaledUniformDotProductQueryDistance<'a>(Cow<'a, [f32]>);

impl<'a> I8ScaledUniformDotProductQueryDistance<'a> {
    pub fn new(query: &'a [f32]) -> Self {
        Self(l2_normalize(query))
    }
}

impl QueryVectorDistance for I8ScaledUniformDotProductQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        // TODO: benchmark performing dot product of query and doc without scaling, then scaling
        // afterward. This would avoid a multiplication per dimension.
        let vector = I8ScaledUniformVector::from(vector);
        let dot = self
            .0
            .iter()
            .zip(vector.dequantized_normalized_iter())
            .map(|(q, d)| *q * d)
            .sum::<f32>() as f64;
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I8ScaledUniformEuclidean;

impl VectorDistance for I8ScaledUniformEuclidean {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = I8ScaledUniformVector::from(query);
        let doc = I8ScaledUniformVector::from(doc);
        let dot = query.dot_unnormalized(&doc);
        query.l2_norm_sq() + doc.l2_norm_sq() - (2.0 * dot)
    }
}

#[derive(Debug, Clone)]
pub struct I8ScaledUniformEuclideanQueryDistance<'a>(&'a [f32], f64);

impl<'a> I8ScaledUniformEuclideanQueryDistance<'a> {
    pub fn new(query: &'a [f32]) -> Self {
        let l2_norm_sq = dot_f32(query, query);
        Self(query, l2_norm_sq)
    }
}

impl QueryVectorDistance for I8ScaledUniformEuclideanQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        // TODO: benchmark performing dot product of query and doc without scaling, then scaling
        // afterward. This would avoid a multiplication per dimension.
        let vector = I8ScaledUniformVector::from(vector);
        let dot = self
            .0
            .iter()
            .zip(vector.dequantized_unnormalized_iter())
            .map(|(q, d)| *q * d)
            .sum::<f32>() as f64;
        self.1 + vector.l2_norm_sq() - (2.0 * dot)
    }
}
