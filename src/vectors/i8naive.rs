//! i8 naive vector coding and distance computation.
//!
//! This scheme l2 normalizes the input vector and then maps values into fixed size buckets in
//! [-127,127]. The vector is emitted as an i8 sequence along with the l2 norm. This is naive in
//! that we don't attempt to use a sample of the data set to improve quantization accuracy; we don't
//! even really use any features in the input vector either. This is a lower bound for how accurate
//! an i8 quantization scheme.
//!
//! Distance can be computed directly on the i8 vector for both dot product and l2 distance so it
//! is generally much faster than raw vector encodings.

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

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        let (vector, l2_norm_sq) = I8NaiveDistance::unpack(encoded);
        let scale = l2_norm_sq.sqrt().recip();
        Some(vector.iter().map(|d| *d as f32 * scale).collect())
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I8NaiveDistance(pub(crate) VectorSimilarity);

impl I8NaiveDistance {
    const SCALE: f64 = (i8::MAX as f64 * i8::MAX as f64).recip();

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
        let dot = qv
            .iter()
            .zip(dv.iter())
            .map(|(q, d)| *q as i32 * *d as i32)
            .sum::<i32>() as f64
            * Self::SCALE;
        // TODO: store l2 norm instead of squared l2 norm to speed up euclidean distance.
        match self.0 {
            VectorSimilarity::Dot => (-dot + 1.0) / 2.0,
            VectorSimilarity::Euclidean => {
                qnorm as f64 + dnorm as f64
                    - (2.0 * dot * qnorm.sqrt() as f64 * dnorm.sqrt() as f64)
            }
        }
    }
}
