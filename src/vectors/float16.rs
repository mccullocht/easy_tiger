use std::borrow::Cow;

use half::f16;

use crate::{
    distance::l2_normalize,
    vectors::{F32VectorCoder, QueryVectorDistance, VectorDistance},
};

// XXX creators should be forced to provide a similarity so I can normalize for dot.
#[derive(Debug, Copy, Clone)]
pub struct F16VectorCoder;

impl F32VectorCoder for F16VectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        for (d, o) in vector.iter().zip(out.chunks_mut(2)) {
            o.copy_from_slice(&f16::from_f32(*d).to_le_bytes());
        }
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions * 2
    }

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        Some(
            encoded
                .chunks(2)
                .map(|h| f16::from_le_bytes(h.try_into().unwrap()).to_f32())
                .collect(),
        )
    }
}

fn f16_iter(raw: &[u8]) -> impl ExactSizeIterator<Item = f16> + '_ {
    raw.chunks_exact(2)
        .map(|c| f16::from_le_bytes(c.try_into().unwrap()))
}

#[derive(Debug, Copy, Clone)]
pub struct F16DotProductDistance;

impl VectorDistance for F16DotProductDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        // XXX lol this makes f32 dot product look _cheap_. 35x more accurate than i8-su qxq
        // in all native f16 math it is less accurate than i8 _and_ 3x more expensive.
        let dot = f16_iter(query)
            .zip(f16_iter(doc))
            .map(|(q, d)| q * d)
            .sum::<f16>()
            .to_f64();
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Clone)]
pub struct F16DotProductQueryDistance<'a>(Cow<'a, [f32]>);

impl<'a> F16DotProductQueryDistance<'a> {
    pub fn new(query: &'a [f32]) -> Self {
        Self(l2_normalize(query))
    }
}

impl QueryVectorDistance for F16DotProductQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        // XXX just as stupidly expensive as f16xf16 dot product, but now 50x more accurate than
        // i8-su qxq and 35x more accurate than f32xq
        let dot = self
            .0
            .iter()
            .zip(vector.chunks_exact(2))
            .map(|(s, o)| *s * f16::from_le_bytes(o.try_into().unwrap()).to_f32())
            .sum::<f32>() as f64;
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Copy, Clone)]
pub struct F16EuclideanDistance;

impl VectorDistance for F16EuclideanDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        f16_iter(query)
            .zip(f16_iter(doc))
            .map(|(q, d)| {
                let diff = q.to_f32() - d.to_f32();
                diff * diff
            })
            .sum::<f32>() as f64
    }
}

#[derive(Debug, Clone)]
pub struct F16EuclideanQueryDistance<'a>(&'a [f32]);

impl<'a> F16EuclideanQueryDistance<'a> {
    pub fn new(query: &'a [f32]) -> Self {
        Self(query)
    }
}

impl QueryVectorDistance for F16EuclideanQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        self.0
            .iter()
            .zip(vector.chunks_exact(2))
            .map(|(s, o)| {
                let diff = *s * f16::from_le_bytes(o.try_into().unwrap()).to_f32();
                diff * diff
            })
            .sum::<f32>() as f64
    }
}

// XXX direct f16 x f32 distance computation as QueryVectorDistance.
