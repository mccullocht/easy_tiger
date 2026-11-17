use std::arch::x86_64::{
    __m128i, __m256i, _mm_loadu_si128, _mm_set1_epi8, _mm_xor_si128, _mm256_loadu_si256,
    _mm512_and_si512, _mm512_broadcast_i32x4, _mm512_castsi128_si512, _mm512_castsi256_si512,
    _mm512_dpbusd_epi32, _mm512_loadu_epi8, _mm512_maskz_loadu_epi8, _mm512_permutexvar_epi64,
    _mm512_reduce_add_epi32, _mm512_set_epi64, _mm512_set1_epi8, _mm512_setzero_si512,
    _mm512_shuffle_epi8, _mm512_srlv_epi64,
};

use super::scalar;

/// `vpdpbusd` computes unsigned x signed products, so the codebook is biased into unsigned range by
/// flipping the sign bit (`t + 128`). Each kernel returns `Σ q·(t + 128)`, so subtracting
/// `128·Σ q` yields the signed inner product.
const BIAS: i32 = 128;

/// Load `table` broadcast to all 128-bit lanes with each entry biased into unsigned range.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn load_biased_table(table: &[i8; 16]) -> std::arch::x86_64::__m512i {
    unsafe {
        _mm512_broadcast_i32x4(_mm_xor_si128(
            _mm_loadu_si128(table.as_ptr() as *const __m128i),
            _mm_set1_epi8(i8::MIN),
        ))
    }
}

/// Inner product of an i8 query and a 2-bit turbo packed doc decoded through `table`.
///
/// `query_sum` must be the sum of all entries in `query`.
///
/// # Safety
/// Requires `avx512f`, `avx512bw` and `avx512vnni`. `query` must contain at least 64 entries for
/// every complete 16 byte block in `doc`.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub(super) unsafe fn lut_dot2(query: &[i8], query_sum: i32, table: &[i8; 16], doc: &[u8]) -> i32 {
    let (blocks, tail) = doc.as_chunks::<16>();
    let head = blocks.len() * 64;
    debug_assert!(query.len() >= head);
    let ip = unsafe {
        let lut = load_biased_table(table);
        // 128-bit lane k holds dimensions [16k, 16k + 16) which are stored at bit offset 2k.
        let shift = _mm512_set_epi64(6, 6, 4, 4, 2, 2, 0, 0);
        let mask = _mm512_set1_epi8(0x3);
        let mut acc = _mm512_setzero_si512();
        for (i, b) in blocks.iter().enumerate() {
            let q = _mm512_loadu_epi8(query.as_ptr().add(i * 64));
            let d = _mm512_broadcast_i32x4(_mm_loadu_si128(b.as_ptr() as *const __m128i));
            let d = _mm512_and_si512(_mm512_srlv_epi64(d, shift), mask);
            acc = _mm512_dpbusd_epi32(acc, _mm512_shuffle_epi8(lut, d), q);
        }
        _mm512_reduce_add_epi32(acc)
    };
    ip.wrapping_add(scalar::lut_dot::<2>(&query[head..], table, BIAS, tail))
        .wrapping_sub(query_sum.wrapping_mul(BIAS))
}

/// Inner product of an i8 query and a 4-bit turbo packed doc decoded through `table`.
///
/// `query_sum` must be the sum of all entries in `query`.
///
/// # Safety
/// Requires `avx512f`, `avx512bw` and `avx512vnni`. `query` must contain at least 32 entries for
/// every complete 16 byte block in `doc`.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub(super) unsafe fn lut_dot4(query: &[i8], query_sum: i32, table: &[i8; 16], doc: &[u8]) -> i32 {
    let (blocks, tail) = doc.as_chunks::<16>();
    let head = blocks.len() * 32;
    debug_assert!(query.len() >= head);
    let ip = unsafe {
        let lut = load_biased_table(table);
        // Expand two 16 byte blocks [b0, b1] into 128-bit lanes [b0, b0, b1, b1], then shift odd
        // lanes to select the high nibble. Lanes then hold dimensions in query order.
        let perm = _mm512_set_epi64(3, 2, 3, 2, 1, 0, 1, 0);
        let shift = _mm512_set_epi64(4, 4, 0, 0, 4, 4, 0, 0);
        let mask = _mm512_set1_epi8(0xf);
        let mut acc = _mm512_setzero_si512();
        let (pairs, rem) = blocks.as_chunks::<2>();
        for (i, p) in pairs.iter().enumerate() {
            let q = _mm512_loadu_epi8(query.as_ptr().add(i * 64));
            let d = _mm512_castsi256_si512(_mm256_loadu_si256(p.as_ptr() as *const __m256i));
            let d = _mm512_permutexvar_epi64(perm, d);
            let d = _mm512_and_si512(_mm512_srlv_epi64(d, shift), mask);
            acc = _mm512_dpbusd_epi32(acc, _mm512_shuffle_epi8(lut, d), q);
        }
        if let Some(b) = rem.first() {
            // Upper half of the query is masked to zero so the undefined upper doc lanes do not
            // contribute.
            let q = _mm512_maskz_loadu_epi8(0xffff_ffff, query.as_ptr().add(pairs.len() * 64));
            let d = _mm512_castsi128_si512(_mm_loadu_si128(b.as_ptr() as *const __m128i));
            let d = _mm512_permutexvar_epi64(perm, d);
            let d = _mm512_and_si512(_mm512_srlv_epi64(d, shift), mask);
            acc = _mm512_dpbusd_epi32(acc, _mm512_shuffle_epi8(lut, d), q);
        }
        _mm512_reduce_add_epi32(acc)
    };
    ip.wrapping_add(scalar::lut_dot::<4>(&query[head..], table, BIAS, tail))
        .wrapping_sub(query_sum.wrapping_mul(BIAS))
}
