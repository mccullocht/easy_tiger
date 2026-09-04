pub mod avx512 {
    use std::arch::x86_64::{
        __m512i, _mm_storeu_epi8, _mm512_abs_ps, _mm512_add_epi32, _mm512_castps_si512,
        _mm512_cvtepi32_epi8, _mm512_fnmadd_ps, _mm512_loadu_ps, _mm512_maskz_loadu_ps,
        _mm512_or_epi32, _mm512_reduce_add_epi32, _mm512_reduce_add_ps, _mm512_set1_epi32,
        _mm512_set1_ps, _mm512_setzero_si512, _mm512_slli_epi32, _mm512_srli_epi32,
    };

    #[target_feature(enable = "avx512f")]
    #[inline]
    pub unsafe fn quantize_and_pack(v: &[f32], out: &mut [u8]) -> u32 {
        let (vhead, vtail) = v.as_chunks::<128>();
        let (ohead, otail) = out.split_at_mut(vhead.len() * 16);
        let ohead = ohead.as_chunks_mut::<16>().0;

        let csum: u32 = unsafe {
            let mut csum = _mm512_set1_epi32(0);
            for (v, o) in vhead.iter().zip(ohead.iter_mut()) {
                let mut p = _mm512_setzero_si512();
                (csum, p) = pack_group::<0>(v.as_ptr(), csum, p);
                (csum, p) = pack_group::<1>(v.as_ptr(), csum, p);
                (csum, p) = pack_group::<2>(v.as_ptr(), csum, p);
                (csum, p) = pack_group::<3>(v.as_ptr(), csum, p);
                (csum, p) = pack_group::<4>(v.as_ptr(), csum, p);
                (csum, p) = pack_group::<5>(v.as_ptr(), csum, p);
                (csum, p) = pack_group::<6>(v.as_ptr(), csum, p);
                (csum, p) = pack_group::<7>(v.as_ptr(), csum, p);
                _mm_storeu_epi8(o.as_mut_ptr() as *mut i8, _mm512_cvtepi32_epi8(p));
            }
            _mm512_reduce_add_epi32(csum) as u32
        };

        if vhead.is_empty() {
            csum
        } else {
            csum + crate::rabitq::scalar::quantize_and_pack(vtail, otail)
        }
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn pack_group<const N: u32>(
        v: *const f32,
        csum: __m512i,
        p: __m512i,
    ) -> (__m512i, __m512i) {
        unsafe {
            let v = _mm512_castps_si512(_mm512_loadu_ps(v.add(N as usize * 16)));
            let b = _mm512_srli_epi32::<31>(v);
            (
                _mm512_add_epi32(csum, b),
                _mm512_or_epi32(p, _mm512_slli_epi32::<N>(b)),
            )
        }
    }

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
