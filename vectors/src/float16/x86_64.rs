use std::arch::x86_64::{
    __m128i, __m256, _MM_FROUND_TO_NEAREST_INT, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps,
    _mm_loadu_si128, _mm_shuffle_ps, _mm_storeu_si128, _mm256_castps256_ps128, _mm256_cvtph_ps,
    _mm256_cvtps_ph, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps,
    _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_ps,
};

/// Load 8 packed `f16` values (16 bytes), zero-filling missing entries.
///
/// # Safety
///
/// The `f16c` CPU feature must be available.
#[inline]
#[target_feature(enable = "avx,f16c")]
unsafe fn load_f16x8_tail(v: &[u8]) -> __m256 {
    debug_assert!(v.len() <= 16);
    let mut r = [0u8; 16];
    r[..v.len()].copy_from_slice(v);
    unsafe { _mm256_cvtph_ps(_mm_loadu_si128(r.as_ptr().cast())) }
}

/// Load 8 packed `f32` values (32 bytes), zero-filling missing entries.
#[inline]
#[target_feature(enable = "avx")]
unsafe fn load_f32x8_tail(v: &[f32]) -> __m256 {
    debug_assert!(v.len() <= 8);
    let mut r = [0.0f32; 8];
    r[..v.len()].copy_from_slice(v);
    unsafe { _mm256_loadu_ps(r.as_ptr()) }
}

/// Horizontally sum the 8 lanes of `v`.
#[inline]
#[target_feature(enable = "avx")]
unsafe fn reduce_f32x8(v: __m256) -> f32 {
    let x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    // Equivalent to `_mm_shuffle_ps(x, x, _MM_SHUFFLE(0, 0, 3, 2))`; `_MM_SHUFFLE` is not yet
    // stabilized so the immediate is computed by hand.
    let y = _mm_shuffle_ps::<0b00_00_11_10>(x, x);
    let z = _mm_add_ps(x, y);
    _mm_cvtss_f32(_mm_hadd_ps(z, z))
}

/// Serialize `v` to half precision floats in `out`, optionally scaling each value by `scale`.
///
/// # Safety
///
/// The `avx` and `f16c` CPU features must be available, e.g. verified with
/// `std::arch::is_x86_feature_detected!("avx")` and
/// `std::arch::is_x86_feature_detected!("f16c")` before calling this function.
#[target_feature(enable = "avx,f16c")]
pub unsafe fn serialize_f16(v: &[f32], scale: Option<f32>, out: &mut [u8]) {
    assert_eq!(out.len(), v.len() * 2);
    let tail_split = v.len() & !7;
    for i in (0..tail_split).step_by(8) {
        unsafe {
            let mut vs = _mm256_loadu_ps(v.as_ptr().add(i));
            if let Some(scale) = scale {
                vs = _mm256_mul_ps(vs, _mm256_set1_ps(scale));
            }
            let vh = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(vs);
            _mm_storeu_si128(out.as_mut_ptr().add(i * 2).cast(), vh);
        }
    }

    if tail_split != v.len() {
        let mut vt = [0.0f32; 8];
        vt[..v.len() - tail_split].copy_from_slice(&v[tail_split..]);
        unsafe {
            let mut vs = _mm256_loadu_ps(vt.as_ptr());
            if let Some(scale) = scale {
                vs = _mm256_mul_ps(vs, _mm256_set1_ps(scale));
            }
            let vh = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(vs);
            let mut vo = [0u8; 16];
            _mm_storeu_si128(vo.as_mut_ptr().cast(), vh);
            out[tail_split * 2..].copy_from_slice(&vo[..(v.len() - tail_split) * 2]);
        }
    }
}

/// Deserialize half precision floats in `v` to `out`.
///
/// # Safety
///
/// The `avx` and `f16c` CPU features must be available, e.g. verified with
/// `std::arch::is_x86_feature_detected!("avx")` and
/// `std::arch::is_x86_feature_detected!("f16c")` before calling this function.
#[target_feature(enable = "avx,f16c")]
pub unsafe fn deserialize_f16(v: &[u8], out: &mut [f32]) {
    let len = out.len();
    assert_eq!(v.len(), len * 2);
    let tail_split = len & !7;
    for i in (0..tail_split).step_by(8) {
        unsafe {
            let vh = _mm_loadu_si128(v.as_ptr().add(i * 2).cast::<__m128i>());
            let vs = _mm256_cvtph_ps(vh);
            _mm256_storeu_ps(out.as_mut_ptr().add(i), vs);
        }
    }

    if tail_split < len {
        let mut tail_in = [0u8; 16];
        tail_in[..v.len() - tail_split * 2].copy_from_slice(&v[tail_split * 2..]);
        unsafe {
            let vh = _mm_loadu_si128(tail_in.as_ptr().cast());
            let vs = _mm256_cvtph_ps(vh);
            let mut tail_out = [0.0f32; 8];
            _mm256_storeu_ps(tail_out.as_mut_ptr(), vs);
            out[tail_split..].copy_from_slice(&tail_out[..len - tail_split]);
        }
    }
}

/// Dot product of two half precision vectors.
///
/// # Safety
///
/// The `avx`, `f16c`, and `fma` CPU features must be available, e.g. verified with
/// `std::arch::is_x86_feature_detected!("avx")`, `std::arch::is_x86_feature_detected!("f16c")`,
/// and `std::arch::is_x86_feature_detected!("fma")` before calling this function.
#[target_feature(enable = "avx,f16c,fma")]
pub unsafe fn dot_f16_f16(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len());
    let len = a.len() / 2;
    let tail_split = len & !7;
    let mut dotv = _mm256_set1_ps(0.0);
    for i in (0..tail_split).step_by(8) {
        unsafe {
            let av = _mm256_cvtph_ps(_mm_loadu_si128(a.as_ptr().add(i * 2).cast()));
            let bv = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i * 2).cast()));
            dotv = _mm256_fmadd_ps(av, bv, dotv);
        }
    }

    if tail_split < len {
        unsafe {
            let av = load_f16x8_tail(&a[tail_split * 2..len * 2]);
            let bv = load_f16x8_tail(&b[tail_split * 2..len * 2]);
            dotv = _mm256_fmadd_ps(av, bv, dotv);
        }
    }

    unsafe { reduce_f32x8(dotv) }
}

/// Dot product of a single precision vector and a half precision vector.
///
/// # Safety
///
/// The `avx`, `f16c`, and `fma` CPU features must be available, e.g. verified with
/// `std::arch::is_x86_feature_detected!("avx")`, `std::arch::is_x86_feature_detected!("f16c")`,
/// and `std::arch::is_x86_feature_detected!("fma")` before calling this function.
#[target_feature(enable = "avx,f16c,fma")]
pub unsafe fn dot_f32_f16(a: &[f32], b: &[u8]) -> f32 {
    assert_eq!(b.len(), a.len() * 2);
    let len = a.len();
    let tail_split = len & !7;
    let mut dotv = _mm256_set1_ps(0.0);
    for i in (0..tail_split).step_by(8) {
        unsafe {
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i * 2).cast()));
            dotv = _mm256_fmadd_ps(av, bv, dotv);
        }
    }

    if tail_split < len {
        unsafe {
            let av = load_f32x8_tail(&a[tail_split..]);
            let bv = load_f16x8_tail(&b[tail_split * 2..len * 2]);
            dotv = _mm256_fmadd_ps(av, bv, dotv);
        }
    }

    unsafe { reduce_f32x8(dotv) }
}

/// Squared euclidean (l2) distance between two half precision vectors.
///
/// # Safety
///
/// The `avx`, `f16c`, and `fma` CPU features must be available, e.g. verified with
/// `std::arch::is_x86_feature_detected!("avx")`, `std::arch::is_x86_feature_detected!("f16c")`,
/// and `std::arch::is_x86_feature_detected!("fma")` before calling this function.
#[target_feature(enable = "avx,f16c,fma")]
pub unsafe fn l2_f16_f16(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len());
    let len = a.len() / 2;
    let tail_split = len & !7;
    let mut sumv = _mm256_set1_ps(0.0);
    for i in (0..tail_split).step_by(8) {
        unsafe {
            let av = _mm256_cvtph_ps(_mm_loadu_si128(a.as_ptr().add(i * 2).cast()));
            let bv = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i * 2).cast()));
            let diff = _mm256_sub_ps(av, bv);
            sumv = _mm256_fmadd_ps(diff, diff, sumv);
        }
    }

    if tail_split < len {
        unsafe {
            let av = load_f16x8_tail(&a[tail_split * 2..len * 2]);
            let bv = load_f16x8_tail(&b[tail_split * 2..len * 2]);
            let diff = _mm256_sub_ps(av, bv);
            sumv = _mm256_fmadd_ps(diff, diff, sumv);
        }
    }

    unsafe { reduce_f32x8(sumv) }
}

/// Squared euclidean (l2) distance between a single precision vector and a half precision
/// vector.
///
/// # Safety
///
/// The `avx`, `f16c`, and `fma` CPU features must be available, e.g. verified with
/// `std::arch::is_x86_feature_detected!("avx")`, `std::arch::is_x86_feature_detected!("f16c")`,
/// and `std::arch::is_x86_feature_detected!("fma")` before calling this function.
#[target_feature(enable = "avx,f16c,fma")]
pub unsafe fn l2_f32_f16(a: &[f32], b: &[u8]) -> f32 {
    assert_eq!(b.len(), a.len() * 2);
    let len = a.len();
    let tail_split = len & !7;
    let mut sumv = _mm256_set1_ps(0.0);
    for i in (0..tail_split).step_by(8) {
        unsafe {
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i * 2).cast()));
            let diff = _mm256_sub_ps(av, bv);
            sumv = _mm256_fmadd_ps(diff, diff, sumv);
        }
    }

    if tail_split < len {
        unsafe {
            let av = load_f32x8_tail(&a[tail_split..]);
            let bv = load_f16x8_tail(&b[tail_split * 2..len * 2]);
            let diff = _mm256_sub_ps(av, bv);
            sumv = _mm256_fmadd_ps(diff, diff, sumv);
        }
    }

    unsafe { reduce_f32x8(sumv) }
}
