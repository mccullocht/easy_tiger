use std::borrow::Cow;

use half::f16;
use simsimd::SpatialSimilarity;

use crate::{
    distance::l2_normalize,
    vectors::{F32VectorCoder, QueryVectorDistance, VectorDistance, VectorSimilarity},
};

#[derive(Debug, Copy, Clone)]
pub struct VectorCoder(bool);

impl VectorCoder {
    pub fn new(similarity: VectorSimilarity) -> Self {
        Self(similarity.l2_normalize())
    }
}

impl F32VectorCoder for VectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let encode_it = vector.iter().zip(out.chunks_mut(2));
        if self.0 {
            let scale = (1.0
                / SpatialSimilarity::dot(vector, vector)
                    .expect("identical vectors")
                    .sqrt()) as f32;
            for (d, o) in encode_it {
                o.copy_from_slice(&f16::from_f32(*d * scale).to_le_bytes());
            }
        } else {
            for (d, o) in encode_it {
                o.copy_from_slice(&f16::from_f32(*d).to_le_bytes());
            }
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

#[allow(dead_code)]
unsafe extern "C" {
    unsafe fn et_dot_f16_f16(a: *const u16, b: *const u16, len: usize) -> f32;
    unsafe fn et_l2_f16_f16(a: *const u16, b: *const u16, len: usize) -> f32;
}

#[derive(Debug, Copy, Clone)]
pub struct DotProductDistance;

impl DotProductDistance {
    #[allow(dead_code)]
    fn dot_scalar(&self, a: &[u8], b: &[u8]) -> f32 {
        f16_iter(a)
            .zip(f16_iter(b))
            .map(|(a, b)| a.to_f32() * b.to_f32())
            .sum::<f32>()
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn dot(&self, a: &[u8], b: &[u8]) -> f32 {
        self.dot_scalar(a, b)
    }

    #[cfg(target_arch = "aarch64")]
    fn dot(&self, a: &[u8], b: &[u8]) -> f32 {
        unsafe {
            et_dot_f16_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() / 2,
            )
        }
    }
}

impl VectorDistance for DotProductDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let dot = self.dot(query, doc) as f64;
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Clone)]
pub struct DotProductQueryDistance<'a>(Cow<'a, [f32]>);

impl<'a> DotProductQueryDistance<'a> {
    pub fn new(query: Cow<'a, [f32]>) -> Self {
        Self(l2_normalize(query))
    }
}

impl QueryVectorDistance for DotProductQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        // TODO: vector accelerate this when necessary bits stabilize (or use C).
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
pub struct EuclideanDistance;

impl EuclideanDistance {
    #[allow(dead_code)]
    fn l2_scalar(&self, a: &[u8], b: &[u8]) -> f32 {
        f16_iter(a)
            .zip(f16_iter(b))
            .map(|(a, b)| {
                let diff = a.to_f32() - b.to_f32();
                diff * diff
            })
            .sum::<f32>()
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn l2(&self, a: &[u8], b: &[u8]) -> f32 {
        self.l2_scalar(a, b)
    }

    #[cfg(target_arch = "aarch64")]
    fn l2(&self, a: &[u8], b: &[u8]) -> f32 {
        unsafe {
            et_l2_f16_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() / 2,
            )
        }
    }
}

impl VectorDistance for EuclideanDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        self.l2(query, doc) as f64
    }
}

#[derive(Debug, Clone)]
pub struct EuclideanQueryDistance<'a>(Cow<'a, [f32]>);

impl<'a> EuclideanQueryDistance<'a> {
    pub fn new(query: Cow<'a, [f32]>) -> Self {
        Self(query)
    }
}

impl QueryVectorDistance for EuclideanQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        // TODO: vector accelerate this when necessary bits stabilize (or use C).
        self.0
            .iter()
            .zip(vector.chunks_exact(2))
            .map(|(s, o)| {
                let diff = *s - f16::from_le_bytes(o.try_into().unwrap()).to_f32();
                diff * diff
            })
            .sum::<f32>() as f64
    }
}
