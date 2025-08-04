use std::borrow::Cow;

use simsimd::SpatialSimilarity;

use crate::{
    distance::l2_normalize,
    vectors::{F32VectorCoder, QueryVectorDistance, VectorDistance},
};

/// Encodes a mixed representatio of the vector split at a particular dimensions.
/// Format:
/// * l2 norm (f32 le)
/// * scaling factor (f32 le)
/// * first split dimensions as f32 le
/// * remaining dimensions as i8.
#[derive(Debug, Copy, Clone)]
pub struct MixedRepVectorCoder(usize);

impl MixedRepVectorCoder {
    pub fn new(split: usize) -> Self {
        Self(split)
    }
}

impl F32VectorCoder for MixedRepVectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        assert!(self.0 < vector.len());

        let l2_norm = SpatialSimilarity::dot(vector, vector).unwrap().sqrt() as f32;
        let (f32_vec, i8_vec) = vector.split_at(self.0);
        let i8_abs_max = i8_vec
            .iter()
            .copied()
            .map(f32::abs)
            .max_by(f32::total_cmp)
            .unwrap() as f64;
        let scale = (f64::from(i8::MAX) / i8_abs_max) as f32;
        let inv_scale = (i8_abs_max / f64::from(i8::MAX)) as f32;

        out[0..4].copy_from_slice(&l2_norm.to_le_bytes());
        out[4..8].copy_from_slice(&inv_scale.to_le_bytes());

        let (f32_out, i8_out) = out[8..].split_at_mut(self.0 * std::mem::size_of::<f32>());
        for (i, o) in f32_vec
            .iter()
            .zip(f32_out.chunks_mut(std::mem::size_of::<f32>()))
        {
            o.copy_from_slice(&i.to_le_bytes());
        }
        for (i, o) in i8_vec.iter().zip(i8_out.iter_mut()) {
            *o = ((*i * scale).round() as i8).to_le_bytes()[0];
        }
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        let split = self.0.min(dimensions);
        std::mem::size_of::<f32>() * (split + 2) + (dimensions - split)
    }

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        let scale = f32::from_le_bytes(encoded[4..8].try_into().unwrap());
        // XXX we're really leaning on the dimension count being fixed here and it's awkward as hell.
        let (f32_vec, i8_vec) = encoded[8..].split_at(self.0 * std::mem::size_of::<f32>());
        Some(
            f32_vec
                .chunks(4)
                .map(|d| f32::from_le_bytes(d.try_into().unwrap()))
                .chain(
                    bytemuck::cast_slice::<_, i8>(i8_vec)
                        .iter()
                        .map(|d| *d as f32 * scale),
                )
                .collect(),
        )
    }
}

struct MixedRepVector<'a> {
    split: usize,
    rep: &'a [u8],
}

impl<'a> MixedRepVector<'a> {
    fn new(split: usize, rep: &'a [u8]) -> Self {
        Self { split, rep }
    }

    fn l2_norm(&self) -> f64 {
        f32::from_le_bytes(self.rep[0..4].try_into().unwrap()).into()
    }

    fn scale(&self) -> f64 {
        f32::from_le_bytes(self.rep[4..8].try_into().unwrap()).into()
    }

    fn f32_vector(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        self.rep[8..(8 + std::mem::size_of::<f32>() * self.split)]
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
    }

    fn i8_vector(&self) -> impl ExactSizeIterator<Item = i8> + '_ {
        bytemuck::cast_slice(&self.rep[(8 + std::mem::size_of::<f32>() * self.split)..])
            .iter()
            .copied()
    }

    fn dot_unnormalized(&self, other: &Self) -> f64 {
        let f32_segment = self
            .f32_vector()
            .zip(other.f32_vector())
            .map(|(q, d)| q * d)
            .sum::<f32>() as f64;
        let i8_segment = self
            .i8_vector()
            .zip(other.i8_vector())
            .map(|(q, d)| q as i32 * d as i32)
            .sum::<i32>() as f64
            * self.scale()
            * other.scale();
        f32_segment + i8_segment
    }

    fn dot_unnormalized_f32(&self, other: &[f32]) -> f64 {
        let f32_segment = self
            .f32_vector()
            .zip(other[..self.split].iter())
            .map(|(s, o)| s * *o)
            .sum::<f32>() as f64;
        let i8_segment = self
            .i8_vector()
            .zip(other[self.split..].iter())
            .map(|(s, o)| s as f32 * *o)
            .sum::<f32>() as f64
            * self.scale();
        f32_segment + i8_segment
    }
}

#[derive(Debug, Copy, Clone)]
pub struct MixedRepDotProductDistance(usize);

impl MixedRepDotProductDistance {
    pub fn new(split: usize) -> Self {
        Self(split)
    }
}

impl VectorDistance for MixedRepDotProductDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = MixedRepVector::new(self.0, query);
        let doc = MixedRepVector::new(self.0, doc);
        let dot = query.dot_unnormalized(&doc) / (query.l2_norm() * doc.l2_norm());
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Clone)]
pub struct MixedRepDotProductQueryDistance<'a> {
    split: usize,
    query: Cow<'a, [f32]>,
}

impl<'a> MixedRepDotProductQueryDistance<'a> {
    pub fn new(split: usize, query: Cow<'a, [f32]>) -> Self {
        Self {
            split,
            query: l2_normalize(query),
        }
    }
}

impl QueryVectorDistance for MixedRepDotProductQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = MixedRepVector::new(self.split, vector);
        let dot = vector.dot_unnormalized_f32(&self.query) / vector.l2_norm();
        (-dot + 1.0) / 2.0
    }
}
