use half::f16;

use crate::vectors::{F32VectorCoder, VectorDistance};

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
        f16_iter(query)
            .zip(f16_iter(doc))
            .map(|(q, d)| q * d)
            .sum::<f16>()
            .to_f64()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct F16EuclideanDistance;

impl VectorDistance for F16EuclideanDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        f16_iter(query)
            .zip(f16_iter(doc))
            .map(|(q, d)| {
                let diff = q - d;
                diff * diff
            })
            .sum::<f16>()
            .to_f64()
    }
}

// XXX direct f16 x f32 distance computation as QueryVectorDistance.
