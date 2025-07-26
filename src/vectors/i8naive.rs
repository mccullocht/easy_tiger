use crate::vectors::F32VectorCoder;

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
