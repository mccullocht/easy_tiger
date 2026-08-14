use std::arch::x86_64::{
    __m512, __m512i, __mmask16, _mm512_add_ps, _mm512_castps_si512, _mm512_castsi512_ps,
    _mm512_loadu_ps, _mm512_loadu_si512, _mm512_mask_sub_ps, _mm512_mul_ps, _mm512_permutexvar_ps,
    _mm512_set1_ps, _mm512_set_epi32, _mm512_storeu_ps, _mm512_sub_ps, _mm512_xor_si512,
};

/// Walsh-Hadamard Transform vector `v` with `signs` random sign flips, using AVX-512F.
///
/// # Safety
///
/// The `avx512f` CPU feature must be available, e.g. verified with
/// `is_x86_feature_detected!("avx512f")` before calling this function.
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_walsh_hadamard_transform<const F: bool>(v: &mut [f32], signs: &[u32]) {
    assert!(
        v.len().is_power_of_two(),
        "Hadamard transform requires power of 2 length"
    );
    assert_eq!(v.len(), signs.len());
    if v.len() < 64 {
        return super::scalar::walsh_hadamard_transform::<F>(v, signs);
    }

    // Perform the early strides of the block transformation together in 64 dimension chunks
    // in an effort to improve locality. v.len() is a power of 2 and there are at least 64
    // entries, so there will be no tail entries.
    let blocks = v.as_chunks_mut::<64>().0;
    let sblocks = signs.as_chunks::<64>().0;
    if F {
        for (b, s) in blocks.iter_mut().zip(sblocks.iter()) {
            unsafe { wht_fixed_block64::<true>(b, s) };
        }
    } else {
        for (b, s) in blocks.iter_mut().zip(sblocks.iter()) {
            unsafe { wht_fixed_block64::<false>(b, s) };
        }
    }
    // Continue butterfly transformation at block size and beyond.
    unsafe { wht_block_from64::<F>(v, signs) };
}

/// Continue the butterfly transformation for strides 64 and beyond, fusing the final stride
/// with normalization (and the backward sign flip) so the whole block is only touched once
/// more after `wht_fixed_block64`.
#[target_feature(enable = "avx512f")]
unsafe fn wht_block_from64<const F: bool>(block: &mut [f32], signs: &[u32]) {
    unsafe {
        let n = block.len();
        assert!(
            n.is_power_of_two(),
            "Hadamard transform requires power of 2 length"
        );
        let scale = 1.0 / (n as f32).sqrt();
        let scalev = _mm512_set1_ps(scale);

        if n == 64 {
            // The base 64-wide transform already completed every stride; there is no further
            // butterfly stage to fuse the normalization into, so just scale (and sign-flip for
            // backward) in place.
            for i in (0..n).step_by(16) {
                let x = _mm512_loadu_ps(block.as_ptr().add(i));
                let mut v = _mm512_mul_ps(x, scalev);
                if !F {
                    let s = _mm512_loadu_si512(signs.as_ptr().add(i) as *const __m512i);
                    v = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(v), s));
                }
                _mm512_storeu_ps(block.as_mut_ptr().add(i), v);
            }
            return;
        }

        let mut h = 64;
        while h < n / 2 {
            for i in (0..n).step_by(h * 2) {
                for j in (0..h).step_by(16) {
                    let x_off = i + j;
                    let y_off = i + j + h;
                    let x = _mm512_loadu_ps(block.as_ptr().add(x_off));
                    let y = _mm512_loadu_ps(block.as_ptr().add(y_off));
                    _mm512_storeu_ps(block.as_mut_ptr().add(x_off), _mm512_add_ps(x, y));
                    _mm512_storeu_ps(block.as_mut_ptr().add(y_off), _mm512_sub_ps(x, y));
                }
            }
            h *= 2;
        }

        for i in (0..n).step_by(h * 2) {
            for j in (0..h).step_by(16) {
                let x_off = i + j;
                let y_off = i + j + h;
                let x = _mm512_loadu_ps(block.as_ptr().add(x_off));
                let y = _mm512_loadu_ps(block.as_ptr().add(y_off));

                let mut a = _mm512_mul_ps(_mm512_add_ps(x, y), scalev);
                let mut b = _mm512_mul_ps(_mm512_sub_ps(x, y), scalev);

                if !F {
                    let sx = _mm512_loadu_si512(signs.as_ptr().add(x_off) as *const __m512i);
                    let sy = _mm512_loadu_si512(signs.as_ptr().add(y_off) as *const __m512i);
                    a = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a), sx));
                    b = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(b), sy));
                }

                _mm512_storeu_ps(block.as_mut_ptr().add(x_off), a);
                _mm512_storeu_ps(block.as_mut_ptr().add(y_off), b);
            }
        }
    }
}

/// Initial base Walsh-Hadamard Transform over a fixed size block.
///
/// This includes the sign flips that are needed before the operation begins if `F` is true.
#[target_feature(enable = "avx512f")]
unsafe fn wht_fixed_block64<const F: bool>(block: &mut [f32; 64], signs: &[u32; 64]) {
    unsafe {
        // Lane-swap index vectors for the intra-register butterfly stages (h = 1, 2, 4, 8):
        // idx_h reads lane `i ^ h` of the register.
        let idx1 = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
        let idx2 = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
        let idx4 = _mm512_set_epi32(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
        let idx8 = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);

        let mut r0 = load16::<F>(block.as_ptr(), signs.as_ptr(), 0);
        let mut r1 = load16::<F>(block.as_ptr(), signs.as_ptr(), 16);
        let mut r2 = load16::<F>(block.as_ptr(), signs.as_ptr(), 32);
        let mut r3 = load16::<F>(block.as_ptr(), signs.as_ptr(), 48);

        // Perform butterfly rotation steps within each register for strides 1, 2, 4, and 8.
        r0 = butterfly16(r0, idx1, idx2, idx4, idx8);
        r1 = butterfly16(r1, idx1, idx2, idx4, idx8);
        r2 = butterfly16(r2, idx1, idx2, idx4, idx8);
        r3 = butterfly16(r3, idx1, idx2, idx4, idx8);

        // Stride 16: butterfly across register pairs (r0, r1) and (r2, r3).
        let (n0, n1) = (_mm512_add_ps(r0, r1), _mm512_sub_ps(r0, r1));
        let (n2, n3) = (_mm512_add_ps(r2, r3), _mm512_sub_ps(r2, r3));

        // Stride 32: butterfly across register pairs (n0, n2) and (n1, n3).
        let (f0, f2) = (_mm512_add_ps(n0, n2), _mm512_sub_ps(n0, n2));
        let (f1, f3) = (_mm512_add_ps(n1, n3), _mm512_sub_ps(n1, n3));

        _mm512_storeu_ps(block.as_mut_ptr(), f0);
        _mm512_storeu_ps(block.as_mut_ptr().add(16), f1);
        _mm512_storeu_ps(block.as_mut_ptr().add(32), f2);
        _mm512_storeu_ps(block.as_mut_ptr().add(48), f3);
    }
}

/// Load 16 entries at a time, optionally flipping signs if F is true.
#[target_feature(enable = "avx512f")]
unsafe fn load16<const F: bool>(b: *const f32, s: *const u32, off: usize) -> __m512 {
    unsafe {
        let r = _mm512_loadu_ps(b.add(off));
        if F {
            let sv = _mm512_loadu_si512(s.add(off) as *const __m512i);
            _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(r), sv))
        } else {
            r
        }
    }
}

/// Butterfly rotation across all 16 lanes of a register for strides 1, 2, 4, and 8.
#[target_feature(enable = "avx512f")]
unsafe fn butterfly16(
    x: __m512,
    idx1: __m512i,
    idx2: __m512i,
    idx4: __m512i,
    idx8: __m512i,
) -> __m512 {
    unsafe {
        let x = butterfly_lanes(x, idx1, MASK1);
        let x = butterfly_lanes(x, idx2, MASK2);
        let x = butterfly_lanes(x, idx4, MASK4);
        butterfly_lanes(x, idx8, MASK8)
    }
}

/// One lane-stride butterfly stage: swap each lane `i` with lane `i ^ h` (via `idx`), then
/// combine so that lanes with bit `h` clear get `x + swapped` and lanes with bit `h` set get
/// `swapped - x` (`mask` selects the latter, matching the scalar `block[i] = x - y` half of the
/// butterfly).
#[target_feature(enable = "avx512f")]
unsafe fn butterfly_lanes(x: __m512, idx: __m512i, mask: __mmask16) -> __m512 {
    let swapped = _mm512_permutexvar_ps(idx, x);
    let sum = _mm512_add_ps(x, swapped);
    _mm512_mask_sub_ps(sum, mask, swapped, x)
}

const MASK1: __mmask16 = lane_mask(1);
const MASK2: __mmask16 = lane_mask(2);
const MASK4: __mmask16 = lane_mask(4);
const MASK8: __mmask16 = lane_mask(8);

/// Mask with bit `i` set wherever lane index `i` has bit `h` set, i.e. the "second half" of
/// each swapped pair for stride `h`.
const fn lane_mask(h: usize) -> __mmask16 {
    let mut m: u16 = 0;
    let mut i = 0;
    while i < 16 {
        if i & h != 0 {
            m |= 1 << i;
        }
        i += 1;
    }
    m
}
