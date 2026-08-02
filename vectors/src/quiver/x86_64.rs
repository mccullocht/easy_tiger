use super::{Kernel, scalar::Scalar};

use std::arch::x86_64::{
    __m128i, __m512, __m512i, _CMP_GT_OQ, _mm_and_si128, _mm_loadu_si128, _mm_movm_epi8,
    _mm_or_epi32, _mm_set_epi8, _mm_set1_epi8, _mm_slli_epi32, _mm_storeu_epi8, _mm512_abs_ps,
    _mm512_add_epi32, _mm512_add_ps, _mm512_and_si512, _mm512_broadcast_i32x4, _mm512_cmp_ps_mask,
    _mm512_dpbusd_epi32, _mm512_loadu_epi8, _mm512_loadu_ps, _mm512_maskz_mov_ps,
    _mm512_popcnt_epi32, _mm512_reduce_add_epi32, _mm512_reduce_add_ps, _mm512_set_epi64,
    _mm512_set1_epi8, _mm512_set1_epi32, _mm512_set1_ps, _mm512_shuffle_epi8, _mm512_slli_epi32,
    _mm512_srli_epi32, _mm512_srlv_epi64, _mm512_sub_epi32, _mm512_ternarylogic_epi32,
    _mm512_xor_si512,
};

struct QuantizationState {
    tau: __m512,
    zero: __m512,

    weak_sum: __m512,
    strong_sum: __m512,
    strong_count: u32,
}

impl QuantizationState {
    fn new(tau: f32) -> Self {
        unsafe {
            Self {
                tau: _mm512_set1_ps(tau),
                zero: _mm512_set1_ps(0.0),
                weak_sum: _mm512_set1_ps(0.0),
                strong_sum: _mm512_set1_ps(0.0),
                strong_count: 0,
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    #[inline]
    unsafe fn quantize16(&mut self, v: &[f32; 16]) -> __m128i {
        unsafe {
            let v = _mm512_loadu_ps(v.as_ptr());
            let sgn_mask = _mm512_cmp_ps_mask::<{ _CMP_GT_OQ }>(v, self.zero);
            let v = _mm512_abs_ps(v);
            let mag_mask = _mm512_cmp_ps_mask::<{ _CMP_GT_OQ }>(v, self.tau);

            self.strong_count += mag_mask.count_ones();
            self.strong_sum = _mm512_add_ps(self.strong_sum, _mm512_maskz_mov_ps(mag_mask, v));
            self.weak_sum = _mm512_add_ps(self.weak_sum, _mm512_maskz_mov_ps(!mag_mask, v));

            let sgn = _mm_and_si128(_mm_movm_epi8(sgn_mask), _mm_set1_epi8(2));
            let mag = _mm_and_si128(_mm_movm_epi8(mag_mask), _mm_set1_epi8(1));
            _mm_or_epi32(sgn, mag)
        }
    }

    #[inline]
    fn header_sums(&self) -> (f32, f32, u32) {
        unsafe {
            (
                _mm512_reduce_add_ps(self.weak_sum),
                _mm512_reduce_add_ps(self.strong_sum),
                self.strong_count,
            )
        }
    }
}

pub struct Avx512;

impl Avx512 {
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn tau_unsafe(v: &[f32]) -> f32 {
        Scalar.tau(v)
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    #[inline]
    unsafe fn quantize_unsafe(v: &[f32], tau: f32, out: &mut [u8]) -> (f32, f32, u32) {
        let (vc, vr) = v.as_chunks::<64>();
        // Split `out` by `vc.len()` rather than chunking it independently by 16 bytes: the
        // packed size of the tail `vr` (`vr.len().div_ceil(4)` bytes) can be as large as 16
        // bytes, which would otherwise misalign `oc`/`or` against `vc`/`vr` and leave `or` too
        // small (or empty) to hold the scalar fallback's output.
        let (oc, or) = out.split_at_mut(vc.len() * 16);
        let oc = oc.as_chunks_mut::<16>().0;
        let (mut weak_sum, mut strong_sum, mut strong_count) = unsafe {
            let mut state = QuantizationState::new(tau);
            for (v, o) in vc.iter().zip(oc.iter_mut()) {
                let v16 = v.as_chunks::<16>().0;
                let mut q = state.quantize16(&v16[0]);
                q = _mm_or_epi32(q, _mm_slli_epi32::<2>(state.quantize16(&v16[1])));
                q = _mm_or_epi32(q, _mm_slli_epi32::<4>(state.quantize16(&v16[2])));
                q = _mm_or_epi32(q, _mm_slli_epi32::<6>(state.quantize16(&v16[3])));
                _mm_storeu_epi8(o.as_mut_ptr() as *mut i8, q);
            }
            state.header_sums()
        };
        if !vr.is_empty() {
            let (w, s, c) = Scalar.quantize(vr, tau, or);
            weak_sum += w;
            strong_sum += s;
            strong_count += c;
        }
        (weak_sum, strong_sum, strong_count)
    }

    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    #[inline]
    unsafe fn symmetric_distance_unsafe(a: &[u8], b: &[u8]) -> i32 {
        let (ac, ar) = a.as_chunks::<128>();
        let (bc, br) = b.as_chunks::<128>();
        let mut dist = unsafe {
            let mut w = _mm512_set1_epi32(0);
            let mut m = _mm512_set1_epi32(0);
            let mut s = _mm512_set1_epi32(0);
            for (a, b) in ac.iter().zip(bc.iter()) {
                let (a_s, a_m) = Self::bitplane_split1024(a);
                let (b_s, b_m) = Self::bitplane_split1024(b);

                let smm = _mm512_xor_si512(a_s, b_s);
                s = _mm512_add_epi32(
                    s,
                    _mm512_popcnt_epi32(_mm512_ternarylogic_epi32::<0x40>(a_m, b_m, smm)),
                );
                s = _mm512_sub_epi32(
                    s,
                    _mm512_popcnt_epi32(_mm512_ternarylogic_epi32::<0x80>(a_m, b_m, smm)),
                );
                m = _mm512_add_epi32(
                    m,
                    _mm512_popcnt_epi32(_mm512_ternarylogic_epi32::<0x14>(a_m, b_m, smm)),
                );
                m = _mm512_sub_epi32(
                    m,
                    _mm512_popcnt_epi32(_mm512_ternarylogic_epi32::<0x28>(a_m, b_m, smm)),
                );
                w = _mm512_add_epi32(
                    w,
                    _mm512_popcnt_epi32(_mm512_ternarylogic_epi32::<0x01>(a_m, b_m, smm)),
                );
                w = _mm512_sub_epi32(
                    w,
                    _mm512_popcnt_epi32(_mm512_ternarylogic_epi32::<0x02>(a_m, b_m, smm)),
                );
            }
            _mm512_reduce_add_epi32(w) as i32
                + _mm512_reduce_add_epi32(m) as i32 * 2
                + _mm512_reduce_add_epi32(s) as i32 * 4
        };
        if !ar.is_empty() {
            dist += Scalar.symmetric_distance(ar, br)
        }
        dist
    }

    #[target_feature(enable = "avx512f,avx512vnni,avx512bw")]
    #[inline]
    unsafe fn asymmetric_distance_unsafe(q: &[i8], d: &[u8], weak: i8, strong: i8) -> i32 {
        let (qc, qr) = q.as_chunks::<64>();
        let (dc, dr) = d.as_chunks::<16>();
        let dist = unsafe {
            let shift_mask = _mm512_set_epi64(6, 6, 4, 4, 2, 2, 0, 0);
            let value_mask = _mm512_set1_epi8(3);
            let shuffle_mask = _mm512_broadcast_i32x4(_mm_set_epi8(
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, strong, weak, -strong, -weak,
            ));

            // vpbusd performs unsigned x signed dot, but we need full signed. Transform the query
            // input into unsigned (xor 0x80) and compute sum(d) to unbias the result.
            let mut dot = _mm512_set1_epi32(0);
            let mut sumd = _mm512_set1_epi32(0);
            for (q, d) in qc.iter().zip(dc.iter()) {
                let qv = _mm512_xor_si512(
                    _mm512_loadu_epi8(q.as_ptr() as *const i8),
                    _mm512_set1_epi8(-128),
                );
                let mut dv = _mm512_broadcast_i32x4(_mm_loadu_si128(d.as_ptr() as *const __m128i));
                dv = _mm512_and_si512(_mm512_srlv_epi64(dv, shift_mask), value_mask);
                dv = _mm512_shuffle_epi8(shuffle_mask, dv);

                dot = _mm512_dpbusd_epi32(dot, qv, dv);
                sumd = _mm512_dpbusd_epi32(sumd, _mm512_set1_epi8(1), dv);
            }
            _mm512_reduce_add_epi32(dot) - 128 * _mm512_reduce_add_epi32(sumd)
        };
        dist + Scalar.asymmetric_distance(qr, dr, weak, strong)
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn bitplane_split1024(v: &[u8; 128]) -> (__m512i, __m512i) {
        unsafe {
            let a = _mm512_loadu_epi8(v.as_ptr() as *const i8);
            let b = _mm512_loadu_epi8(v.as_ptr().add(64) as *const i8);
            let m = _mm512_set1_epi8(0x55);
            let sgn = _mm512_ternarylogic_epi32::<0xCA>(m, _mm512_srli_epi32::<1>(a), b);
            let mag = _mm512_ternarylogic_epi32::<0xCA>(m, a, _mm512_slli_epi32::<1>(b));
            (sgn, mag)
        }
    }
}

impl Kernel for Avx512 {
    #[inline]
    fn tau(&self, v: &[f32]) -> f32 {
        unsafe { Self::tau_unsafe(v) }
    }

    #[inline]
    fn quantize(&self, v: &[f32], tau: f32, out: &mut [u8]) -> (f32, f32, u32) {
        unsafe { Self::quantize_unsafe(v, tau, out) }
    }

    #[inline]
    fn symmetric_distance(&self, a: &[u8], b: &[u8]) -> i32 {
        unsafe { Self::symmetric_distance_unsafe(a, b) }
    }

    #[inline]
    fn asymmetric_distance(&self, q: &[i8], d: &[u8], weak: i8, strong: i8) -> i32 {
        unsafe { Self::asymmetric_distance_unsafe(q, d, weak, strong) }
    }
}
