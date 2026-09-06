//! x86_64 specific implementations of kernel functions.

/// Avx512-specific implementations of kernel functions.
pub mod avx512 {
    use std::arch::x86_64::{
        __m128i, _mm_lddqu_si128, _mm512_add_epi32, _mm512_add_epi64, _mm512_and_si512,
        _mm512_broadcast_i32x4, _mm512_loadu_epi8, _mm512_loadu_epi64, _mm512_mullo_epi32,
        _mm512_popcnt_epi32, _mm512_popcnt_epi64, _mm512_reduce_add_epi32, _mm512_reduce_add_epi64,
        _mm512_set_epi32, _mm512_set1_epi32, _mm512_set1_epi64, _mm512_xor_si512,
    };

    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    #[inline]
    pub unsafe fn bitstring_inner_product<const S: bool>(a: &[u8], b: &[u8]) -> u32 {
        assert_eq!(a.len(), b.len());
        let (ahead, atail) = a.as_chunks::<64>();
        let (bhead, btail) = b.as_chunks::<64>();
        let ip = unsafe {
            let mut ip = _mm512_set1_epi64(0);
            for (a, b) in ahead.iter().zip(bhead.iter()) {
                let a = _mm512_loadu_epi64(a.as_ptr() as *const i64);
                let b = _mm512_loadu_epi64(b.as_ptr() as *const i64);
                let d = if S {
                    _mm512_xor_si512(a, b)
                } else {
                    _mm512_and_si512(a, b)
                };
                ip = _mm512_add_epi64(ip, _mm512_popcnt_epi64(d));
            }

            _mm512_reduce_add_epi64(ip) as u32
        };

        if atail.is_empty() {
            ip
        } else {
            ip + crate::kernels::scalar::bitstring_inner_product_tail::<S>(atail, btail)
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vpopcntdq")]
    #[inline]
    pub unsafe fn turbo_4x1_inner_product(a: &[u8], b: &[u8]) -> u32 {
        let (ahead, atail) = a.as_chunks::<64>();
        let (bhead, btail) = b.split_at(ahead.len() * 16);

        // Each 64 byte query chunk holds the four 128-bit query bitplanes for the 128 dimensions
        // covered by the matching 16 doc bytes, so one 512-bit load covers all four planes and the
        // doc value can be broadcast to each 128-bit lane.
        //
        // Popcounts are accumulated per 32-bit lane and only weighted by the plane significance at
        // the end: each iteration adds at most 32 to a lane, so this can absorb millions of
        // iterations before overflow.
        let mut ip = unsafe {
            let mut acc = _mm512_set1_epi32(0);
            for (a, b) in ahead.iter().zip(bhead.as_chunks::<16>().0) {
                let a = _mm512_loadu_epi8(a.as_ptr() as *const i8);
                let b = _mm512_broadcast_i32x4(_mm_lddqu_si128(b.as_ptr() as *const __m128i));
                acc = _mm512_add_epi32(acc, _mm512_popcnt_epi32(_mm512_and_si512(a, b)));
            }

            // Weight each 128-bit lane (one bitplane) by its significance before reducing.
            let weights = _mm512_set_epi32(8, 8, 8, 8, 4, 4, 4, 4, 2, 2, 2, 2, 1, 1, 1, 1);
            _mm512_reduce_add_epi32(_mm512_mullo_epi32(acc, weights)) as u32
        };

        if !atail.is_empty() {
            ip += crate::kernels::scalar::turbo_4x1_inner_product_tail(atail, btail);
        }

        ip
    }
}
