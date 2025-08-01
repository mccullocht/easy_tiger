//! Scaled non-uniform coding divides the segment into several splits based on manually specified
//! split points. Within each segment we compute the maximum absolute value across all dimensions
//! and use this to produce an i8 value in [-127,127]. Each segment has its own stored scaling
//! factor; we also store the squared l2 norm for adjusting the final distances to produces scores
//! in the same ranges as float scoring. This may be useful for MRL representation vectors that are
//! designed to be truncated at fix points, where later segments are "less important".
//!
//! Unlike some other quantization schemes this does not rely on anything computed across a sample
//! of the data set -- no means or centroids, no quantiles. For transformer models which produce
//! relatively well centered vectors this seems to be effective enough that we may discard the
//! original f32 vectors.

use std::{borrow::Cow, ops::Range};

use super::dot_unnormalized_i8_f32;
use crate::{
    distance::{dot_f32, l2_normalize},
    vectors::{F32VectorCoder, NonUniformQuantizedDimensions, QueryVectorDistance, VectorDistance},
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

fn split_dim_iterator<'a>(
    splits: &'a [u16],
    vector_len: usize,
) -> impl Iterator<Item = Range<usize>> + 'a {
    std::iter::once(0)
        .chain(splits.iter().map(|s| *s as usize))
        .zip(
            splits
                .iter()
                .map(|s| *s as usize)
                .chain(std::iter::once(vector_len)),
        )
        .map(|(s, e)| s..e)
}

#[derive(Debug, Copy, Clone)]
pub struct I8VectorCoder(NonUniformQuantizedDimensions);

impl I8VectorCoder {
    pub fn new(splits: NonUniformQuantizedDimensions) -> Self {
        Self(splits)
    }

    fn split_iterator<'a, 'b>(
        &'b self,
        vector: &'a [f32],
    ) -> impl Iterator<Item = &'a [f32]> + use<'a, 'b> {
        split_dim_iterator(&self.0, vector.len()).map(|r| &vector[r])
    }
}

impl F32VectorCoder for I8VectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let l2_norm = crate::distance::dot_f32(vector, vector).sqrt() as f32;
        let scales = self
            .split_iterator(vector)
            .map(compute_scale::<{ i8::MAX }>)
            .collect::<Vec<_>>();
        out[0..4].copy_from_slice(&l2_norm.to_le_bytes());
        let mut output_index = 4usize;
        for inv_scale in scales.iter().map(|(_, s)| *s) {
            out[output_index..(output_index + 4)].copy_from_slice(&inv_scale.to_le_bytes());
            output_index += 4;
        }
        for (scale, v) in scales
            .iter()
            .map(|(s, _)| *s)
            .zip(self.split_iterator(vector))
        {
            for (d, o) in v.iter().zip(out[output_index..].iter_mut()) {
                *o = ((*d * scale).round() as i8).to_le_bytes()[0];
            }
            output_index += v.len();
        }
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        // Store l2 norm and a scaling factor for each segment.
        dimensions + std::mem::size_of::<f32>() * (self.0.len() + 2)
    }

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        let v = I8Vector::new(&self.0, encoded);
        Some(
            v.segments()
                .flat_map(|(scale, bytes)| bytes.iter().map(move |d| *d as f32 * scale))
                .collect(),
        )
    }
}

#[derive(Debug, Copy, Clone)]
struct I8Vector<'a, 'b> {
    splits: &'a [u16],
    raw_vector: &'b [u8],
    vector_base: usize,
}

impl<'a, 'b> I8Vector<'a, 'b> {
    fn new(splits: &'a [u16], raw_vector: &'b [u8]) -> Self {
        let vector_base = std::mem::size_of::<f32>() * (splits.len() + 2);
        Self {
            splits,
            raw_vector,
            vector_base,
        }
    }

    fn l2_norm_sq(&self) -> f64 {
        self.l2_norm() * self.l2_norm()
    }

    fn l2_norm(&self) -> f64 {
        f32::from_le_bytes(self.raw_vector[0..4].try_into().unwrap()).into()
    }

    fn segments(&self) -> impl Iterator<Item = (f32, &[i8])> {
        let scale_it = self.raw_vector[std::mem::size_of::<f32>()..].chunks(4);
        let vector = &self.raw_vector[self.vector_base..];
        scale_it
            .zip(split_dim_iterator(self.splits, vector.len()))
            .map(|(s, r)| {
                (
                    f32::from_le_bytes(s.try_into().expect("4 bytes")),
                    bytemuck::cast_slice(&vector[r]),
                )
            })
    }

    fn segment_dot_unnormalized(a: (f32, &[i8]), b: (f32, &[i8])) -> f64 {
        a.1.iter()
            .zip(b.1.iter())
            .map(|(a, b)| *a as i32 * *b as i32)
            .sum::<i32>() as f64
            * a.0 as f64
            * b.0 as f64
    }

    fn dot_unnormalized(&self, other: &Self) -> f64 {
        self.segments()
            .zip(other.segments())
            .map(|(a, b)| Self::segment_dot_unnormalized(a, b))
            .sum::<f64>()
    }

    fn dot_unnormalized_f32(&self, other: &[f32]) -> f64 {
        self.segments()
            .zip(split_dim_iterator(self.splits, other.len()).map(|r| &other[r]))
            .map(|(q, f)| dot_unnormalized_i8_f32(q.1, q.0 as f64, f))
            .sum::<f64>()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I8DotProductDistance(NonUniformQuantizedDimensions);

impl I8DotProductDistance {
    pub fn new(splits: NonUniformQuantizedDimensions) -> Self {
        Self(splits)
    }
}

impl VectorDistance for I8DotProductDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = I8Vector::new(&self.0, query);
        let doc = I8Vector::new(&self.0, doc);
        let dot = query.dot_unnormalized(&doc) * query.l2_norm().recip() * doc.l2_norm().recip();
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Clone)]
pub struct I8DotProductQueryDistance<'a> {
    splits: NonUniformQuantizedDimensions,
    query: Cow<'a, [f32]>,
}

impl<'a> I8DotProductQueryDistance<'a> {
    pub fn new(splits: NonUniformQuantizedDimensions, query: &'a [f32]) -> Self {
        Self {
            splits,
            query: l2_normalize(query),
        }
    }
}

impl QueryVectorDistance for I8DotProductQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = I8Vector::new(&self.splits, vector);
        let dot = vector.dot_unnormalized_f32(&self.query) / vector.l2_norm();
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I8EuclideanDistance(NonUniformQuantizedDimensions);

impl I8EuclideanDistance {
    pub fn new(splits: NonUniformQuantizedDimensions) -> Self {
        Self(splits)
    }
}

impl VectorDistance for I8EuclideanDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = I8Vector::new(&self.0, query);
        let doc = I8Vector::new(&self.0, doc);
        let dot = query.dot_unnormalized(&doc);
        query.l2_norm_sq() + doc.l2_norm_sq() - (2.0 * dot)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I8EuclideanQueryDistance<'a> {
    splits: NonUniformQuantizedDimensions,
    query: &'a [f32],
    l2_norm_sq: f64,
}

impl<'a> I8EuclideanQueryDistance<'a> {
    pub fn new(splits: NonUniformQuantizedDimensions, query: &'a [f32]) -> Self {
        Self {
            splits,
            query,
            l2_norm_sq: dot_f32(query, query),
        }
    }
}

impl QueryVectorDistance for I8EuclideanQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = I8Vector::new(&self.splits, vector);
        let dot = vector.dot_unnormalized_f32(self.query);
        self.l2_norm_sq + vector.l2_norm_sq() - (2.0 * dot)
    }
}
