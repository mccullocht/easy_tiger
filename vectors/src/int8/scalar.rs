//! Platform-independent kernel implementations written in safe rust.
//!
//! Most of these implementations are marked inline as optimized kernels could be built on them,
//! either by using them to process a scalar tail or by allowing them to auto-vectorize with
//! specific architecture target features.

#[inline(always)]
pub fn decode_to_f32(vector: &[i8], out: &mut [f32]) {
    for (i, o) in vector.iter().zip(out.iter_mut()) {
        *o = *i as f32;
    }
}
