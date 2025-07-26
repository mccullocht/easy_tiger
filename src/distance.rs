//! Dense vector scoring traits and implementations.

use std::borrow::Cow;

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
    #[cfg(target_arch = "aarch64")]
    unsafe {
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
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        f32_le_iter(q)
            .zip(f32_le_iter(d))
            .map(|(q, d)| {
                let delta = q - d;
                delta * delta
            })
            .sum::<f32>() as f64
    }
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
    #[cfg(target_arch = "aarch64")]
    unsafe {
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
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        f32_le_iter(q)
            .zip(f32_le_iter(d))
            .map(|(q, d)| q * d)
            .sum::<f32>() as f64
    }
}

/// Normalize the contents of vector in l2 space.
///
/// May return the input vector if it is already normalized.
pub fn l2_normalize<'a>(vector: &'a [f32]) -> Cow<'a, [f32]> {
    let norm = dot_f32(&vector, &vector).sqrt() as f32;
    if norm == 1.0 {
        vector.into()
    } else {
        let norm_inv = norm.recip();
        vector
            .iter()
            .map(|d| *d * norm_inv)
            .collect::<Vec<_>>()
            .into()
    }
}
