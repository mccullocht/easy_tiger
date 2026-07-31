use super::{Kernel, scalar::Scalar};

use std::arch::x86_64::{
    __m512i, _mm512_add_epi64, _mm512_loadu_epi8, _mm512_popcnt_epi64, _mm512_reduce_add_epi64,
    _mm512_set1_epi8, _mm512_set1_epi64, _mm512_slli_epi64, _mm512_srli_epi64, _mm512_sub_epi64,
    _mm512_ternarylogic_epi64, _mm512_xor_si512,
};

pub struct Avx512;

impl Avx512 {
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn tau_unsafe(v: &[f32]) -> f32 {
        Scalar::tau(v)
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn quantize_unsafe(v: &[f32], tau: f32, out: &mut [u8]) -> (f32, f32, u32) {
        Scalar::quantize(v, tau, out)
    }

    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    #[inline]
    unsafe fn symmetric_distance_unsafe(a: &[u8], b: &[u8]) -> i32 {
        let (ac, ar) = a.as_chunks::<128>();
        let (bc, br) = b.as_chunks::<128>();
        let dist = unsafe {
            // XXX maybe switch to 32 and avoid the cast at the end???
            let mut w = _mm512_set1_epi64(0);
            let mut m = _mm512_set1_epi64(0);
            let mut s = _mm512_set1_epi64(0);
            for (a, b) in ac.iter().zip(bc.iter()) {
                let (a_s, a_m) = Self::bitplane_split1024(a);
                let (b_s, b_m) = Self::bitplane_split1024(b);

                let smm = _mm512_xor_si512(a_s, b_s);
                s = _mm512_add_epi64(
                    s,
                    _mm512_popcnt_epi64(_mm512_ternarylogic_epi64::<0x40>(a_m, b_m, smm)),
                );
                s = _mm512_sub_epi64(
                    s,
                    _mm512_popcnt_epi64(_mm512_ternarylogic_epi64::<0x80>(a_m, b_m, smm)),
                );
                m = _mm512_add_epi64(
                    m,
                    _mm512_popcnt_epi64(_mm512_ternarylogic_epi64::<0x14>(a_m, b_m, smm)),
                );
                m = _mm512_sub_epi64(
                    m,
                    _mm512_popcnt_epi64(_mm512_ternarylogic_epi64::<0x28>(a_m, b_m, smm)),
                );
                w = _mm512_add_epi64(
                    w,
                    _mm512_popcnt_epi64(_mm512_ternarylogic_epi64::<0x01>(a_m, b_m, smm)),
                );
                w = _mm512_sub_epi64(
                    w,
                    _mm512_popcnt_epi64(_mm512_ternarylogic_epi64::<0x02>(a_m, b_m, smm)),
                );
            }
            _mm512_reduce_add_epi64(w) as i32
                + _mm512_reduce_add_epi64(m) as i32 * 2
                + _mm512_reduce_add_epi64(s) as i32 * 4
        };
        dist + Scalar::symmetric_distance(ar, br)
    }

    #[target_feature(enable = "avx512f,avx512vnni")]
    #[inline]
    unsafe fn asymmetric_distance_unsafe(q: &[i8], d: &[u8], weak: i8, strong: i8) -> i32 {
        Scalar::asymmetric_distance(q, d, weak, strong)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn bitplane_split1024(v: &[u8; 128]) -> (__m512i, __m512i) {
        unsafe {
            let a = _mm512_loadu_epi8(v.as_ptr() as *const i8);
            let b = _mm512_loadu_epi8(v.as_ptr().add(64) as *const i8);
            let m = _mm512_set1_epi8(0x55);
            let sgn = _mm512_ternarylogic_epi64::<0xCA>(m, _mm512_srli_epi64::<1>(a), b);
            let mag = _mm512_ternarylogic_epi64::<0xCA>(m, a, _mm512_slli_epi64::<1>(b));
            (sgn, mag)
        }
    }
}

impl Kernel for Avx512 {
    #[inline]
    fn tau(v: &[f32]) -> f32 {
        unsafe { Self::tau_unsafe(v) }
    }

    #[inline]
    fn quantize(v: &[f32], tau: f32, out: &mut [u8]) -> (f32, f32, u32) {
        unsafe { Self::quantize_unsafe(v, tau, out) }
    }

    #[inline]
    fn symmetric_distance(a: &[u8], b: &[u8]) -> i32 {
        unsafe { Self::symmetric_distance_unsafe(a, b) }
    }

    #[inline]
    fn asymmetric_distance(q: &[i8], d: &[u8], weak: i8, strong: i8) -> i32 {
        unsafe { Self::asymmetric_distance_unsafe(q, d, weak, strong) }
    }
}
