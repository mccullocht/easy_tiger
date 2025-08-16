//! Binary vector coding and distance computation.

use simsimd::BinarySimilarity;

use crate::vectors::{F32VectorCoder, QueryVectorDistance, VectorDistance};

#[derive(Debug, Copy, Clone)]
pub struct BinaryQuantizedVectorCoder;

impl F32VectorCoder for BinaryQuantizedVectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        for (c, o) in vector.chunks(8).zip(out.iter_mut()) {
            *o = c
                .iter()
                .enumerate()
                .filter_map(|(i, d)| if *d > 0.0 { Some(1u8 << i) } else { None })
                .fold(0, |a, b| a | b)
        }
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions.div_ceil(8)
    }
}

/// Computes a score from two bitmaps using hamming distance.
#[derive(Debug, Copy, Clone)]
pub struct HammingDistance;

impl VectorDistance for HammingDistance {
    fn distance(&self, a: &[u8], b: &[u8]) -> f64 {
        BinarySimilarity::hamming(a, b).expect("same dimensionality")
    }
}

/// Quantize the input vector into an i8 vector and compare it to a binary quantized vector.
///
/// This is ~7x slower than raw hamming distance computation, which is less than it would be to use
/// a transposed bitmap and compute hamming distance 8x.
///
/// This assumes vectors are l2 normalized like for dot product distance.
#[derive(Debug, Clone)]
pub struct I1DotProductQueryDistance {
    quantized: Vec<i8>,
    quantized_scale: f64,
    i1_value: i8,
}

impl I1DotProductQueryDistance {
    pub fn new(query: &[f32]) -> Self {
        // XXX share the scaled-uniform implementation of this
        let max_abs = query
            .iter()
            .copied()
            .map(f32::abs)
            .max_by(f32::total_cmp)
            .unwrap() as f64;
        let scale = (f64::from(i8::MAX) / max_abs) as f32;
        let inv_scale = max_abs / f64::from(i8::MAX);
        // XXX share the scaled-uniform implementation of this too.
        let quantized = query
            .iter()
            .copied()
            .map(|d| (d * scale).round() as i8)
            .collect();
        // Map values in the i1 vector into i8 space assuming an l2 norm of 1.
        // This also assumes the same quantized_scale for the i1 vector.
        let i1_value = (((1.0 / query.len() as f64).sqrt() as f32) * scale).round() as i8;
        Self {
            quantized,
            quantized_scale: inv_scale,
            i1_value,
        }
    }

    #[allow(dead_code)]
    fn dot_unnormalized_i32_scalar(&self, vector: &[u8], start_dim: usize) -> i32 {
        let decode_table = [-self.i1_value, self.i1_value];
        self.quantized[start_dim..]
            .chunks(8)
            .zip(vector[(start_dim / 8)..].iter())
            .flat_map(|(q, d)| {
                q.iter()
                    .zip(
                        [
                            *d & 0x1,
                            (*d >> 1) & 0x1,
                            (*d >> 2) & 0x1,
                            (*d >> 3) & 0x1,
                            (*d >> 4) & 0x1,
                            (*d >> 5) & 0x1,
                            (*d >> 6) & 0x1,
                            (*d >> 7) & 0x1,
                        ]
                        .into_iter()
                        .map(|x| decode_table[x as usize]),
                    )
                    .map(|(q, d)| *q as i32 * d as i32)
            })
            .sum::<i32>()
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn dot_unnormalized_i32(&self, vector: &[u8]) -> i32 {
        self.dot_unnormalized_i32_scalar(vector, 0)
    }

    #[cfg(target_arch = "aarch64")]
    fn dot_unnormalized_i32(&self, vector: &[u8]) -> i32 {
        let tail_split = self.quantized.len() & !15;
        let partial_dot = unsafe {
            use std::arch::aarch64::{
                vaddq_s32, vaddvq_s32, vandq_u8, vcombine_u8, vdup_n_u8, vdupq_n_s32, vdupq_n_u8,
                vget_low_s8, vld1q_s8, vld1q_u8, vminq_u8, vmlal_high_s8, vmull_s8, vpaddlq_s16,
                vqtbl1q_s8,
            };

            let doc_mask =
                vld1q_u8([1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128].as_ptr());
            let v = self.i1_value;
            let doc_tbl_mask = vld1q_s8([-v, v, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].as_ptr());
            let mut dot = vdupq_n_s32(0);
            for i in (0..tail_split).step_by(16) {
                let qv = vld1q_s8(self.quantized.as_ptr().add(i));
                // Broadcast the next two byte values from doc to two different 8 byte values, then
                // mask for each bit and take min(1) to get 0 or 1 in a lane for each bit. This is
                // fed to a table decoder to get a representative positive or negative value.
                // TODO: if we can cheaply produce a vector of [-1,1] we can multiply by i8_value
                // and skip the table decode which requires an additional mask and is slow.
                let dv = vqtbl1q_s8(
                    doc_tbl_mask,
                    vminq_u8(
                        vandq_u8(
                            vcombine_u8(vdup_n_u8(vector[i / 8]), vdup_n_u8(vector[(i / 8) + 1])),
                            doc_mask,
                        ),
                        vdupq_n_u8(1),
                    ),
                );

                // Multiply i8 entries and widen to i16, then sum pairs to widen to i32.
                // 127 * 127 * 2 < 2^15-1 so this can be done without overflow/underflow.
                dot = vaddq_s32(
                    dot,
                    vpaddlq_s16(vmlal_high_s8(
                        vmull_s8(vget_low_s8(qv), vget_low_s8(dv)),
                        qv,
                        dv,
                    )),
                );
            }

            vaddvq_s32(dot)
        };
        partial_dot + self.dot_unnormalized_i32_scalar(vector, tail_split)
    }
}

impl QueryVectorDistance for I1DotProductQueryDistance {
    fn distance(&self, vector: &[u8]) -> f64 {
        // NB: we assume both the query and doc vectors have an l2 norm of 1.0
        let dot =
            self.dot_unnormalized_i32(vector) as f64 * self.quantized_scale * self.quantized_scale;
        (-dot + 1.0) / 2.0
    }
}
