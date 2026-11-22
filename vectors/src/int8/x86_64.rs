//! x86_64 specific kernel implementations.

use super::scalar;

/// AVX2-specific kernel implementations.
pub mod avx2 {
    use std::arch::x86_64::{
        __m128i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_loadu_si128, _mm_shuffle_epi32,
        _mm_unpackhi_epi64, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi8_epi16,
        _mm256_extracti128_si256, _mm256_madd_epi16, _mm256_setzero_si256,
    };

    #[target_feature(enable = "avx2")]
    #[inline]
    pub unsafe fn decode_to_f32(vector: &[i8], out: &mut [f32]) {
        super::scalar::decode_to_f32(vector, out)
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub unsafe fn inner_product(a: &[i8], b: &[i8]) -> i32 {
        let (ahead, atail) = a.as_chunks::<16>();
        let (bhead, btail) = b.as_chunks::<16>();

        let ip = unsafe {
            let ipv = ahead
                .iter()
                .zip(bhead.iter())
                .fold(_mm256_setzero_si256(), |acc, (a, b)| {
                    let a = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr() as *const __m128i));
                    let b = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.as_ptr() as *const __m128i));
                    _mm256_add_epi32(acc, _mm256_madd_epi16(a, b))
                });

            let lo = _mm256_castsi256_si128(ipv);
            let hi = _mm256_extracti128_si256(ipv, 1);
            let s = _mm_add_epi32(lo, hi);
            let s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, s));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
            _mm_cvtsi128_si32(s)
        };

        ip + super::scalar::inner_product(atail, btail)
    }
}

/// AVX512-specific kernel implementations.
pub mod avx512 {
    use std::arch::x86_64::{
        __m512i, _mm512_dpbusd_epi32, _mm512_loadu_si512, _mm512_reduce_add_epi32,
        _mm512_set1_epi8, _mm512_setzero_si512, _mm512_slli_epi32, _mm512_sub_epi32,
        _mm512_xor_si512,
    };

    #[target_feature(enable = "avx512f")]
    #[inline]
    pub unsafe fn decode_to_f32(vector: &[i8], out: &mut [f32]) {
        super::scalar::decode_to_f32(vector, out);
    }

    #[target_feature(enable = "avx512f,avx512vnni")]
    #[inline]
    pub unsafe fn inner_product(a: &[i8], b: &[i8]) -> i32 {
        let (ahead, atail) = a.as_chunks::<64>();
        let (bhead, btail) = b.as_chunks::<64>();

        // VNNI only implements u8xi8, so transform `a` into logical u8 and sum `b` to correct the
        // bias at the end of the computation.
        let ip = unsafe {
            let sign_flip = _mm512_set1_epi8(-128);
            let one = _mm512_set1_epi8(1);

            let (ipb, sumb) = ahead.iter().zip(bhead.iter()).fold(
                (_mm512_setzero_si512(), _mm512_setzero_si512()),
                |acc, (a, b)| {
                    let a = _mm512_loadu_si512(a.as_ptr() as *const __m512i);
                    let b = _mm512_loadu_si512(b.as_ptr() as *const __m512i);
                    (
                        _mm512_dpbusd_epi32(acc.0, _mm512_xor_si512(a, sign_flip), b),
                        _mm512_dpbusd_epi32(acc.1, one, b),
                    )
                },
            );

            let ipub = _mm512_sub_epi32(ipb, _mm512_slli_epi32(sumb, 7));
            _mm512_reduce_add_epi32(ipub)
        };

        ip + super::scalar::inner_product(atail, btail)
    }
}
