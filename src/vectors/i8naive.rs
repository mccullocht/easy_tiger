use crate::vectors::{F32VectorCoder, VectorDistance, VectorSimilarity};

#[derive(Debug, Copy, Clone)]
pub struct I8NaiveVectorCoder;

impl F32VectorCoder for I8NaiveVectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let l2_norm = crate::distance::dot_f32(vector, vector).sqrt() as f32;

        // Use the l2 norm to produce a value in [-1,1] then multiply by i8::MAX, round, and cast to
        // an i8 value to quantize. This is quite conservative in larger dimensions at it is
        // extremely unlikely that one dimension dominates the distance calculation.
        //
        // Save the squared l2 norm for use computing l2 distance.
        let scale = l2_norm.recip() * i8::MAX as f32;
        for (d, o) in vector.iter().zip(out.iter_mut()) {
            *o = ((*d * scale).round() as i8).to_le_bytes()[0];
        }
        out[vector.len()..(vector.len() + 4)].copy_from_slice(&(l2_norm * l2_norm).to_le_bytes());
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions + std::mem::size_of::<f32>()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I8NaiveDistance(pub(crate) VectorSimilarity);

impl I8NaiveDistance {
    fn unpack(raw: &[u8]) -> (&[i8], f32) {
        assert!(raw.len() >= std::mem::size_of::<f32>());
        let (vector_bytes, norm_bytes) = raw.split_at(raw.len() - std::mem::size_of::<f32>());
        (
            bytemuck::cast_slice(vector_bytes),
            f32::from_le_bytes(norm_bytes.try_into().unwrap()),
        )
    }
}

impl VectorDistance for I8NaiveDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let (qv, qnorm) = Self::unpack(query);
        let (dv, dnorm) = Self::unpack(doc);
        let divisor = i8::MAX as f32 * i8::MAX as f32;
        // XXX this might not be correct, IIRC I need to scale dot based on the norms like in the
        // scaled uniform implementation.
        // NB: we may be able to accelerate this further with manual SIMD implementations.
        let dot = qv
            .iter()
            .zip(dv.iter())
            .map(|(q, d)| *q as i32 * *d as i32)
            .sum::<i32>() as f64
            / divisor as f64;
        match self.0 {
            VectorSimilarity::Dot => (-dot + 1.0) / 2.0,
            VectorSimilarity::Euclidean => qnorm as f64 + dnorm as f64 - (2.0 * dot),
        }
    }
}
