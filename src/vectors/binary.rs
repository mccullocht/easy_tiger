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
    fn dot_unnormalized_i32_scalar(&self, vector: &[u8]) -> i32 {
        let decode_table = [-self.i1_value, self.i1_value];
        self.quantized
            .chunks(8)
            .zip(vector.iter())
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
        self.dot_unnormalized_i32_scalar(vector)
    }

    // XXX this is ~12x slower than just doing hamming distance.
    // XXX if I can get the number down to 8x it is competitive with replacing the transposed
    // hamming distance method, if I can get it below that it is superior.
    #[cfg(target_arch = "aarch64")]
    fn dot_unnormalized_i32(&self, vector: &[u8]) -> i32 {
        // XXX fix tail split
        unsafe {
            use std::arch::aarch64::{vaddvq_s32, vdup_n_u8, vdupq_n_s32, vld1_s8};

            let dshift_mask = vld1_s8([0, -1, -2, -3, -4, -5, -6, -7].as_ptr());
            let dmask = vdup_n_u8(0x1);
            let dtbl_mask = vld1_s8([-self.i1_value, self.i1_value, 0, 0, 0, 0, 0, 0].as_ptr());
            let mut dot = vdupq_n_s32(0);
            for i in (0..self.quantized.len()).step_by(8) {
                use std::arch::aarch64::{
                    vaddq_s32, vand_u8, vdup_n_u8, vmull_s8, vpaddlq_s16, vreinterpret_s8_u8,
                    vreinterpret_u8_s8, vshl_u8, vtbl1_u8,
                };

                let qv = vld1_s8(self.quantized.as_ptr().add(i));
                // Broadcast the current doc value, then shift and mask to isolate each bit.
                // Table decode 0 into a negative value and 1 into a positive value.
                let dv = vreinterpret_s8_u8(vtbl1_u8(
                    vand_u8(vshl_u8(vdup_n_u8(vector[i / 8]), dshift_mask), dmask),
                    vreinterpret_u8_s8(dtbl_mask),
                ));
                // XXX vdot_s32 would almost certainly be faster here (3 instr -> 1)
                // XXX operating on 128 bits could eliminate many instructions:
                // * -1 load (query)
                // * -1 shl, -1 and, -1 vtbl, +1 vcombine (doc)
                // * 1 mul, 1 addp, 1 addq (dot)
                // 6 instructions in the middle of the loop.
                dot = vaddq_s32(dot, vpaddlq_s16(vmull_s8(qv, dv)));
            }

            vaddvq_s32(dot)
        }
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
