#![allow(unsafe_op_in_unsafe_fn)]

#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot(q: &[u8], d: &[u8]) -> f64 {
    use std::arch::x86_64::{
        _mm_cvtss_f32, _mm_hadd_ps, _mm512_add_ps, _mm512_castps512_ps128, _mm512_fmadd_ps,
        _mm512_maskz_loadu_ps, _mm512_set1_ps, _mm512_shuffle_f32x4,
    };
    let mut dot = _mm512_set1_ps(0.0);
    for i in (0..q.len()).step_by(64) {
        let rem = (q.len() - i).min(64) / 4;
        let mask = u16::MAX >> (16 - rem);
        let qv = _mm512_maskz_loadu_ps(mask, q.as_ptr().add(i) as *const f32);
        let dv = _mm512_maskz_loadu_ps(mask, d.as_ptr().add(i) as *const f32);
        dot = _mm512_fmadd_ps(qv, dv, dot);
    }

    let x = _mm512_add_ps(dot, _mm512_shuffle_f32x4(dot, dot, 0b00001110));
    let r = _mm512_castps512_ps128(_mm512_add_ps(x, _mm512_shuffle_f32x4(x, x, 0b00000001)));
    let r = _mm_hadd_ps(r, r);
    _mm_cvtss_f32(_mm_hadd_ps(r, r)).into()
}

#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn l2sq(q: &[u8], d: &[u8]) -> f64 {
    use std::arch::x86_64::{
        _mm_cvtss_f32, _mm_hadd_ps, _mm512_add_ps, _mm512_castps512_ps128, _mm512_fmadd_ps,
        _mm512_maskz_loadu_ps, _mm512_set1_ps, _mm512_shuffle_f32x4, _mm512_sub_ps,
    };
    let mut sum = _mm512_set1_ps(0.0);
    for i in (0..q.len()).step_by(64) {
        let rem = (q.len() - i).min(64) / 4;
        let mask = u16::MAX >> (16 - rem);
        let qv = _mm512_maskz_loadu_ps(mask, q.as_ptr().add(i) as *const f32);
        let dv = _mm512_maskz_loadu_ps(mask, d.as_ptr().add(i) as *const f32);
        let diff = _mm512_sub_ps(qv, dv);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    let x = _mm512_add_ps(sum, _mm512_shuffle_f32x4(sum, sum, 0b00_00_11_10));
    let r = _mm512_castps512_ps128(_mm512_add_ps(x, _mm512_shuffle_f32x4(x, x, 0b00_00_00_01)));
    let r = _mm_hadd_ps(r, r);
    _mm_cvtss_f32(_mm_hadd_ps(r, r)).into()
}
