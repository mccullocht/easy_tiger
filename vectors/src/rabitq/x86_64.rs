pub mod avx512 {
    use std::arch::x86_64::{
        _mm512_abs_ps, _mm512_fnmadd_ps, _mm512_maskz_loadu_ps, _mm512_reduce_add_ps,
        _mm512_set1_ps,
    };

    #[target_feature(enable = "avx512f")]
    #[inline]
    pub unsafe fn l1_norm_scaled(v: &[f32], scale: f32) -> f32 {
        unsafe {
            let scale = _mm512_set1_ps(scale);
            let mut sum = _mm512_set1_ps(0.0);
            for c in v.chunks(16) {
                let m = u16::MAX >> (16 - c.len());
                let c = _mm512_abs_ps(_mm512_maskz_loadu_ps(m, c.as_ptr()));
                sum = _mm512_fnmadd_ps(c, scale, sum);
            }
            _mm512_reduce_add_ps(sum) / (v.len() as f32).sqrt()
        }
    }
}
