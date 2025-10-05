//! Dense vector scoring traits and implementations.

use std::borrow::Cow;

pub(crate) enum Acceleration {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl Default for Acceleration {
    #[cfg(target_arch = "aarch64")]
    fn default() -> Self {
        Acceleration::Neon
    }

    #[cfg(target_arch = "x86_64")]
    fn default() -> Self {
        use std::is_x86_feature_detected;
        if is_x86_feature_detected!("avx512f") {
            Acceleration::Avx512
        } else {
            Acceleration::Scalar
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    fn default() -> Self {
        Acceleration::Scalar
    }
}

#[inline(always)]
#[cfg(target_arch = "aarch64")]
unsafe fn load_f32x4_le(p: *const u8) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64;
    if cfg!(target_endian = "big") {
        aarch64::vreinterpretq_f32_u8(aarch64::vrev32q_u8(aarch64::vld1q_u8(p)))
    } else {
        aarch64::vld1q_f32(p as *const f32)
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn f32_le_iter<'b>(b: &'b [u8]) -> impl ExactSizeIterator<Item = f32> + 'b {
    b.chunks_exact(4)
        .map(|f| f32::from_le_bytes(f.try_into().expect("4 bytes")))
}

// TODO: byte swapped load on big endian archs.

#[inline(always)]
pub(crate) fn l2sq_f32(q: &[f32], d: &[f32]) -> f64 {
    simsimd::SpatialSimilarity::l2sq(q, d).expect("same dimensions")
}

pub(crate) fn l2sq_f32_bytes(q: &[u8], d: &[u8]) -> f64 {
    assert_eq!(q.len(), d.len());
    assert_eq!(q.len() % 4, 0);
    match Acceleration::default() {
        Acceleration::Scalar => f32_le_iter(q)
            .zip(f32_le_iter(d))
            .map(|(q, d)| {
                let delta = q - d;
                delta * delta
            })
            .sum::<f32>() as f64,
        #[cfg(target_arch = "aarch64")]
        Acceleration::Neon => unsafe {
            use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vsubq_f32};
            let suffix_start = q.len() & !15;
            let mut l2sqv = vdupq_n_f32(0.0);
            for i in (0..suffix_start).step_by(16) {
                let dv = vsubq_f32(
                    load_f32x4_le(q.as_ptr().add(i)),
                    load_f32x4_le(d.as_ptr().add(i)),
                );
                l2sqv = vfmaq_f32(l2sqv, dv, dv);
            }
            let mut l2sq = vaddvq_f32(l2sqv);
            for i in (suffix_start..q.len()).step_by(4) {
                let delta = std::ptr::read_unaligned(q.as_ptr().add(i) as *const f32)
                    - std::ptr::read_unaligned(d.as_ptr().add(i) as *const f32);
                l2sq += delta * delta;
            }
            l2sq as f64
        },
        #[cfg(target_arch = "x86_64")]
        Acceleration::Avx512 => unsafe { l2sq_f32_bytes_avx512(q, d) },
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2sq_f32_bytes_avx512(q: &[u8], d: &[u8]) -> f64 {
    use core::arch::x86_64::{
        _mm512_add_ps, _mm512_castps512_ps128, _mm512_fmadd_ps, _mm512_maskz_loadu_ps,
        _mm512_set1_ps, _mm512_shuffle_f32x4, _mm512_sub_ps, _mm_cvtss_f32, _mm_hadd_ps,
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

pub(crate) fn l2(q: &[f32], d: &[f32]) -> f64 {
    l2sq_f32(q, d).sqrt()
}

#[inline(always)]
pub(crate) fn dot_f32(q: &[f32], d: &[f32]) -> f64 {
    simsimd::SpatialSimilarity::dot(q, d).expect("same dimensions")
}

pub(crate) fn dot_f32_bytes(q: &[u8], d: &[u8]) -> f64 {
    assert_eq!(q.len(), d.len());
    assert_eq!(q.len() % 4, 0);
    match Acceleration::default() {
        Acceleration::Scalar => f32_le_iter(q)
            .zip(f32_le_iter(d))
            .map(|(q, d)| q * d)
            .sum::<f32>() as f64,
        #[cfg(target_arch = "aarch64")]
        Acceleration::Neon => unsafe {
            use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32};
            let suffix_start = q.len() & !15;
            let mut dotv = vdupq_n_f32(0.0);
            for i in (0..suffix_start).step_by(16) {
                dotv = vfmaq_f32(
                    dotv,
                    load_f32x4_le(q.as_ptr().add(i)),
                    load_f32x4_le(d.as_ptr().add(i)),
                );
            }
            let mut dot = vaddvq_f32(dotv);
            for i in (suffix_start..q.len()).step_by(4) {
                dot += std::ptr::read_unaligned(q.as_ptr().add(i) as *const f32)
                    * std::ptr::read_unaligned(d.as_ptr().add(i) as *const f32);
            }
            dot as f64
        },
        #[cfg(target_arch = "x86_64")]
        Acceleration::Avx512 => unsafe { dot_f32_bytes_avx512f(q, d) },
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_f32_bytes_avx512f(q: &[u8], d: &[u8]) -> f64 {
    use core::arch::x86_64::{
        _mm512_add_ps, _mm512_castps512_ps128, _mm512_fmadd_ps, _mm512_maskz_loadu_ps,
        _mm512_set1_ps, _mm512_shuffle_f32x4, _mm_cvtss_f32, _mm_hadd_ps,
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

/// Normalize the contents of vector in l2 space.
///
/// May return the input vector if it is already normalized.
pub fn l2_normalize<'a>(vector: impl Into<Cow<'a, [f32]>>) -> Cow<'a, [f32]> {
    let mut vector: Cow<'a, [f32]> = vector.into();
    let norm = dot_f32(&vector, &vector).sqrt() as f32;
    if norm != 1.0 {
        let norm_inv = norm.recip();
        for d in vector.to_mut().iter_mut() {
            *d *= norm_inv;
        }
    }
    vector
}
