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

fn compute_scale<const M: i8>(vector: &[f32]) -> (f32, f32) {
    if let Some(max) = vector.iter().map(|d| d.abs()).max_by(|a, b| a.total_cmp(b)) {
        (
            (f64::from(M) / max as f64) as f32,
            (max as f64 / f64::from(M)) as f32,
        )
    } else {
        (0.0, 0.0)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I8ScaledUniformVectorCoder;

impl F32VectorCoder for I8ScaledUniformVectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let l2_norm = crate::distance::dot_f32(vector, vector).sqrt() as f32;
        let (scale, inv_scale) = compute_scale::<{ i8::MAX }>(vector);
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

#[derive(Debug, Copy, Clone)]
pub struct I4PackedVectorCoder;

impl F32VectorCoder for I4PackedVectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let l2_norm = crate::distance::dot_f32(vector, vector).sqrt() as f32;
        let (scale, inv_scale) = compute_scale::<7>(vector);
        out[0..4].copy_from_slice(&inv_scale.to_le_bytes());
        out[4..8].copy_from_slice(&l2_norm.to_le_bytes());
        // Encode two at a time and pack them together in a single byte. Use offset binary coding
        // to avoid problems with sign extension happening or not happening when intended.
        for (c, o) in vector.chunks(2).zip(out[8..].iter_mut()) {
            let lo = (c[0] * scale).round() as i8;
            let hi = (c.get(1).unwrap_or(&0.0) * scale).round() as i8;
            *o = ((lo + 7) as u8) | (((hi + 7) as u8) << 4)
        }
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions.div_ceil(2) + std::mem::size_of::<f32>() * 2
    }
}

struct I4PackedVector<'a>(&'a [u8]);

impl<'a> I4PackedVector<'a> {
    fn new(vector: &'a [u8]) -> Option<Self> {
        if vector.len() <= std::mem::size_of::<f32>() * 2 {
            None
        } else {
            Some(Self(vector))
        }
    }

    fn scale(&self) -> f64 {
        f32::from_le_bytes(self.0[0..4].try_into().expect("4 bytes")).into()
    }

    fn l2_norm(&self) -> f64 {
        f32::from_le_bytes(self.0[4..8].try_into().expect("4 bytes")).into()
    }

    fn l2_norm_sq(&self) -> f64 {
        self.l2_norm() * self.l2_norm()
    }

    fn dimensions(&self) -> &[u8] {
        &self.0[8..]
    }

    fn unpack(x: u8) -> [i8; 2] {
        [((x & 0xf) as i8) - 7, ((x >> 4) as i8) - 7]
    }

    fn dot_unnormalized(&self, other: &Self) -> f64 {
        self.dimensions()
            .iter()
            .zip(other.dimensions().iter())
            .map(|(q, d)| {
                let q = Self::unpack(*q);
                let d = Self::unpack(*d);
                (q[0] * d[0] + q[1] * d[1]) as i32
            })
            .sum::<i32>() as f64
            * self.scale()
            * other.scale()
    }

    fn dot_unnormalized_f32(&self, other: &[f32]) -> f64 {
        let mut dim_it = self.dimensions().iter();
        let mut other_it = other.chunks_exact(2);
        let mut dot = dim_it
            .by_ref()
            .zip(other_it.by_ref())
            .map(|(d, c)| {
                let d = Self::unpack(*d);
                d[0] as f32 * c[0] + d[1] as f32 * c[1]
            })
            .sum::<f32>();
        dot += dim_it
            .zip(other_it.remainder())
            .map(|(d, o)| (*d - 15) as f32 * *o)
            .sum::<f32>();
        // NB: other.scale() is implicitly 1.
        dot as f64 * self.scale()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I4PackedDotProductDistance;

impl VectorDistance for I4PackedDotProductDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = I4PackedVector::new(query).expect("valid format");
        let doc = I4PackedVector::new(doc).expect("valid format");
        let dot = query.dot_unnormalized(&doc) * query.l2_norm().recip() * doc.l2_norm().recip();
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Clone)]
pub struct I4PackedDotProductQueryDistance<'a>(Cow<'a, [f32]>);

impl<'a> I4PackedDotProductQueryDistance<'a> {
    pub fn new(query: &'a [f32]) -> Self {
        Self(l2_normalize(query))
    }
}

impl QueryVectorDistance for I4PackedDotProductQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = I4PackedVector::new(vector).expect("valid format");
        let dot = vector.dot_unnormalized_f32(self.0.as_ref()) * vector.l2_norm().recip();
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I4PackedEuclideanDistance;

impl VectorDistance for I4PackedEuclideanDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = I4PackedVector::new(query).expect("valid format");
        let doc = I4PackedVector::new(doc).expect("valid format");
        let dot = query.dot_unnormalized(&doc);
        query.l2_norm_sq() + doc.l2_norm_sq() - (2.0 * dot)
    }
}

#[derive(Debug, Clone)]
pub struct I4PackedEuclideanQueryDistance<'a>(&'a [f32]);

impl<'a> I4PackedEuclideanQueryDistance<'a> {
    pub fn new(query: &'a [f32]) -> Self {
        Self(query)
    }
}

impl QueryVectorDistance for I4PackedEuclideanQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = I4PackedVector::new(vector).expect("valid format");
        let dot = vector.dot_unnormalized_f32(self.0.as_ref());
        vector.l2_norm_sq() - (2.0 * dot)
    }
}
