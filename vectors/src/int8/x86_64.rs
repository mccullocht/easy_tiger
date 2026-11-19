//! x86_64 specific kernel implementations.

use super::scalar;

/// AVX2-specific kernel implementations.
pub mod avx2 {
    #[target_feature(enable = "avx2")]
    #[inline]
    pub unsafe fn decode_to_f32(vector: &[i8], out: &mut [f32]) {
        super::scalar::decode_to_f32(vector, out)
    }
}

/// AVX512-specific kernel implementations.
pub mod avx512 {
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub unsafe fn decode_to_f32(vector: &[i8], out: &mut [f32]) {
        super::scalar::decode_to_f32(vector, out);
    }
}
