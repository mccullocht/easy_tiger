//! Scaled uniform coding uses the maximum absolute value across all dimensions for a given vector
//! to produce an i8 value in [-127,127]. It stores both a value to invert the scaling as well as
//! the squared l2 norm for computing euclidean distance using dot product.
//!
//! Unlike some other quantization schemes this does not rely on anything computed across a sample
//! of the data set -- no means or centroids, no quantiles. For transformer models which produce
//! relatively well centered vectors this seems to be effective enough that we may discard the
//! original f32 vectors.
use crate::vectors::F32VectorCoder;

#[derive(Debug, Copy, Clone)]
pub struct I8ScaledUniformVectorCoder;

impl F32VectorCoder for I8ScaledUniformVectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let l2_norm = crate::distance::dot_f32(vector, vector).sqrt() as f32;
        let (scale, inv_scale) =
            if let Some(max) = vector.iter().map(|d| d.abs()).max_by(|a, b| a.total_cmp(b)) {
                (
                    (f64::from(i8::MAX) / max as f64) as f32,
                    (max as f64 / f64::from(i8::MAX)) as f32,
                )
            } else {
                (0.0, 0.0)
            };

        out[0..4].copy_from_slice(&inv_scale.to_le_bytes());
        out[4..8].copy_from_slice(&l2_norm.to_le_bytes());
        for (d, o) in vector.iter().zip(out[8..].iter_mut()) {
            *o = ((*d * scale).round() as i8).to_le_bytes()[0];
        }
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions + std::mem::size_of::<f32>() * 2
    }
}

// TODO: quantizer that is non-uniform for MRL vectors.
// Bonus points if it can still be scored on the quantized rep instead of de-quantizing.
