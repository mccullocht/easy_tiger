use simsimd::SpatialSimilarity;

use crate::vectors::F32VectorCoder;

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
