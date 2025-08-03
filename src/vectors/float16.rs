use half::f16;

use crate::vectors::F32VectorCoder;

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
