pub mod avx512 {
    use std::arch::x86_64::{
        __m512, __m512i, _mm_loadu_epi8, _mm_storeu_epi8, _mm512_abs_ps, _mm512_add_epi32,
        _mm512_add_ps, _mm512_castps_si512, _mm512_castsi512_ps, _mm512_cvtepi8_epi32,
        _mm512_cvtepi32_epi8, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_maskz_loadu_ps,
        _mm512_popcnt_epi32, _mm512_reduce_add_epi32, _mm512_reduce_add_ps,
        _mm512_set1_epi32, _mm512_set1_ps, _mm512_setzero_si512, _mm512_slli_epi32,
        _mm512_srli_epi32, _mm512_storeu_ps, _mm512_ternarylogic_epi32,
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
                p = pack_group::<0, 31>(v.as_ptr(), p);
                p = pack_group::<1, 30>(v.as_ptr(), p);
                p = pack_group::<2, 29>(v.as_ptr(), p);
                p = pack_group::<3, 28>(v.as_ptr(), p);
                p = pack_group::<4, 27>(v.as_ptr(), p);
                p = pack_group::<5, 26>(v.as_ptr(), p);
                p = pack_group::<6, 25>(v.as_ptr(), p);
                p = pack_group::<7, 24>(v.as_ptr(), p);
                csum = _mm512_add_epi32(csum, _mm512_popcnt_epi32(p));
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
    unsafe fn pack_group<const N: u32, const SHIFT: u32>(v: *const f32, p: __m512i) -> __m512i {
        unsafe {
            let v = _mm512_castps_si512(_mm512_loadu_ps(v.add(N as usize * 16)));
            let shifted = _mm512_srli_epi32::<SHIFT>(v);
            let mask = _mm512_set1_epi32(1i32 << N);
            // (shifted & mask) | p: keep bit N of the sign-shifted lane, OR it into the accumulator.
            _mm512_ternarylogic_epi32::<0xEC>(shifted, p, mask)
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
                sum = _mm512_fmadd_ps(c, scale, sum);
            }
            _mm512_reduce_add_ps(sum) / (v.len() as f32).sqrt()
        }
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    pub unsafe fn decode(v: &[u8], magnitude: f32, center: Option<&[f32]>, out: &mut [f32]) {
        let (vhead, vtail) = v.as_chunks::<16>();
        let (ohead, otail) = out.as_chunks_mut::<128>();

        unsafe {
            let mag = _mm512_set1_ps(magnitude);
            if let Some(center) = center {
                let (chead, ctail) = center.as_chunks::<128>();
                for ((v, c), o) in vhead.iter().zip(chead.iter()).zip(ohead.iter_mut()) {
                    let v = _mm512_cvtepi8_epi32(_mm_loadu_epi8(v.as_ptr() as *const i8));
                    unpack_group_centered::<0, 31>(v, mag, c.as_ptr(), o.as_mut_ptr());
                    unpack_group_centered::<1, 30>(v, mag, c.as_ptr(), o.as_mut_ptr());
                    unpack_group_centered::<2, 29>(v, mag, c.as_ptr(), o.as_mut_ptr());
                    unpack_group_centered::<3, 28>(v, mag, c.as_ptr(), o.as_mut_ptr());
                    unpack_group_centered::<4, 27>(v, mag, c.as_ptr(), o.as_mut_ptr());
                    unpack_group_centered::<5, 26>(v, mag, c.as_ptr(), o.as_mut_ptr());
                    unpack_group_centered::<6, 25>(v, mag, c.as_ptr(), o.as_mut_ptr());
                    unpack_group_centered::<7, 24>(v, mag, c.as_ptr(), o.as_mut_ptr());
                }
                crate::rabitq::scalar::decode(vtail, magnitude, Some(ctail), otail);
            } else {
                for (v, o) in vhead.iter().zip(ohead.iter_mut()) {
                    let v = _mm512_cvtepi8_epi32(_mm_loadu_epi8(v.as_ptr() as *const i8));
                    unpack_group::<0, 31>(v, mag, o.as_mut_ptr());
                    unpack_group::<1, 30>(v, mag, o.as_mut_ptr());
                    unpack_group::<2, 29>(v, mag, o.as_mut_ptr());
                    unpack_group::<3, 28>(v, mag, o.as_mut_ptr());
                    unpack_group::<4, 27>(v, mag, o.as_mut_ptr());
                    unpack_group::<5, 26>(v, mag, o.as_mut_ptr());
                    unpack_group::<6, 25>(v, mag, o.as_mut_ptr());
                    unpack_group::<7, 24>(v, mag, o.as_mut_ptr());
                }
                crate::rabitq::scalar::decode(vtail, magnitude, None, otail);
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn unpack_group_ps<const SHIFT: u32>(v: __m512i, m: __m512) -> __m512 {
        let t = _mm512_slli_epi32::<SHIFT>(v);
        let signmask = _mm512_set1_epi32(i32::MIN);
        // select(c=signmask): take the sign bit from t, everything else from m.
        let r = _mm512_ternarylogic_epi32::<0xE4>(t, _mm512_castps_si512(m), signmask);
        _mm512_castsi512_ps(r)
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn unpack_group<const N: u32, const SHIFT: u32>(v: __m512i, m: __m512, o: *mut f32) {
        unsafe {
            _mm512_storeu_ps(o.add(N as usize * 16), unpack_group_ps::<SHIFT>(v, m));
        }
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn unpack_group_centered<const N: u32, const SHIFT: u32>(
        v: __m512i,
        m: __m512,
        c: *const f32,
        o: *mut f32,
    ) {
        unsafe {
            _mm512_storeu_ps(
                o.add(N as usize * 16),
                _mm512_add_ps(
                    _mm512_loadu_ps(c.add(N as usize * 16)),
                    unpack_group_ps::<SHIFT>(v, m),
                ),
            );
        }
    }
}
