use std::arch::aarch64::{
    float16x4_t, float16x8_t, float32x4_t, vaddq_f32, vaddvq_f32, vcvt_f16_f32, vcvt_f32_f16,
    vcvt_high_f32_f16, vdupq_n_f32, vfmaq_f32, vfmlalq_high_f16, vfmlalq_low_f16, vget_high_f16,
    vget_low_f16, vld1_u16, vld1q_f32, vld1q_u16, vmulq_n_f32, vreinterpret_f16_u16,
    vreinterpret_u8_f16, vreinterpretq_f16_u16, vst1_u8, vst1q_f32, vsubq_f16, vsubq_f32,
};

/// Load 8 packed `f16` values (16 bytes) from `v`, reinterpreting bytes loaded as `u16` since
/// the direct `f16` load intrinsics are not yet stabilized.
#[inline]
#[target_feature(enable = "fp16")]
unsafe fn load_f16x8(v: &[u8]) -> float16x8_t {
    debug_assert!(v.len() >= 16);
    unsafe { vreinterpretq_f16_u16(vld1q_u16(v.as_ptr().cast())) }
}

/// Load 4 packed `f16` values (8 bytes) from `v`, reinterpreting bytes loaded as `u16` since the
/// direct `f16` load intrinsics are not yet stabilized.
#[inline]
#[target_feature(enable = "fp16")]
unsafe fn load_f16x4(v: &[u8]) -> float16x4_t {
    debug_assert!(v.len() >= 8);
    unsafe { vreinterpret_f16_u16(vld1_u16(v.as_ptr().cast())) }
}

/// It's faster to fill out a full 8 value tail entry than it is to convert and compute one
/// element at a time.
#[inline]
#[target_feature(enable = "fp16")]
unsafe fn load_tail_f16x8(v: &[u8]) -> float16x8_t {
    debug_assert!(v.len() <= 16);
    let mut tail = [0u8; 16];
    tail[..v.len()].copy_from_slice(v);
    unsafe { load_f16x8(&tail) }
}

/// It's faster to fill out a full 4 value tail entry than it is to convert and compute one
/// element at a time.
#[inline]
#[target_feature(enable = "fp16")]
unsafe fn load_tail_f16x4(v: &[u8]) -> float16x4_t {
    debug_assert!(v.len() <= 8);
    let mut tail = [0u8; 8];
    tail[..v.len()].copy_from_slice(v);
    unsafe { load_f16x4(&tail) }
}

/// Load a partial value into a vector register to cover cases comparing to f16 where we will
/// also have to load a partial value into a vector register.
#[inline]
fn load_tail_f32x4(v: &[f32]) -> float32x4_t {
    debug_assert!(v.len() <= 4);
    let mut tail = [0.0f32; 4];
    tail[..v.len()].copy_from_slice(v);
    unsafe { vld1q_f32(tail.as_ptr()) }
}

/// Serialize `v` to half precision floats in `out`, optionally scaling each value by `scale`.
///
/// # Safety
///
/// The `fp16` CPU feature must be available, e.g. verified with
/// `std::arch::is_aarch64_feature_detected!("fp16")` before calling this function.
#[target_feature(enable = "fp16")]
pub unsafe fn serialize_f16(v: &[f32], scale: Option<f32>, out: &mut [u8]) {
    assert_eq!(out.len(), v.len() * 2);
    let tail_split = v.len() & !3;
    for i in (0..tail_split).step_by(4) {
        unsafe {
            let mut in_ = vld1q_f32(v.as_ptr().add(i));
            if let Some(scale) = scale {
                in_ = vmulq_n_f32(in_, scale);
            }
            vst1_u8(
                out.as_mut_ptr().add(i * 2),
                vreinterpret_u8_f16(vcvt_f16_f32(in_)),
            );
        }
    }

    if tail_split < v.len() {
        let mut tail_in = [0.0f32; 4];
        tail_in[..v.len() - tail_split].copy_from_slice(&v[tail_split..]);
        unsafe {
            let mut in_ = vld1q_f32(tail_in.as_ptr());
            if let Some(scale) = scale {
                in_ = vmulq_n_f32(in_, scale);
            }
            let mut tail_out = [0u8; 8];
            vst1_u8(
                tail_out.as_mut_ptr(),
                vreinterpret_u8_f16(vcvt_f16_f32(in_)),
            );
            out[tail_split * 2..].copy_from_slice(&tail_out[..(v.len() - tail_split) * 2]);
        }
    }
}

/// Deserialize half precision floats in `v` to `out`.
///
/// # Safety
///
/// The `fp16` CPU feature must be available, e.g. verified with
/// `std::arch::is_aarch64_feature_detected!("fp16")` before calling this function.
#[target_feature(enable = "fp16")]
pub unsafe fn deserialize_f16(v: &[u8], out: &mut [f32]) {
    let len = out.len();
    assert_eq!(v.len(), len * 2);
    let tail_split = len & !7;
    for i in (0..tail_split).step_by(8) {
        unsafe {
            let in_ = load_f16x8(&v[i * 2..]);
            vst1q_f32(out.as_mut_ptr().add(i), vcvt_f32_f16(vget_low_f16(in_)));
            vst1q_f32(
                out.as_mut_ptr().add(i + 4),
                vcvt_f32_f16(vget_high_f16(in_)),
            );
        }
    }

    if tail_split < len {
        unsafe {
            let in_ = load_tail_f16x8(&v[tail_split * 2..len * 2]);
            let mut tail_out = [0.0f32; 8];
            vst1q_f32(tail_out.as_mut_ptr(), vcvt_f32_f16(vget_low_f16(in_)));
            vst1q_f32(
                tail_out.as_mut_ptr().add(4),
                vcvt_f32_f16(vget_high_f16(in_)),
            );
            out[tail_split..].copy_from_slice(&tail_out[..len - tail_split]);
        }
    }
}

/// Dot product of two half precision vectors.
///
/// # Safety
///
/// The `fp16` and `fhm` CPU features must be available, e.g. verified with
/// `std::arch::is_aarch64_feature_detected!("fp16")` and
/// `std::arch::is_aarch64_feature_detected!("fhm")` before calling this function.
#[target_feature(enable = "fp16,fhm")]
pub unsafe fn dot_f16_f16(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len());
    let len = a.len() / 2;

    let mut dot0 = vdupq_n_f32(0.0);
    let mut dot1 = vdupq_n_f32(0.0);
    let mut dot2 = vdupq_n_f32(0.0);
    let mut dot3 = vdupq_n_f32(0.0);

    let len16 = len & !15;
    for i in (0..len16).step_by(16) {
        unsafe {
            let av16 = load_f16x8(&a[i * 2..]);
            let bv16 = load_f16x8(&b[i * 2..]);
            dot0 = vfmlalq_low_f16(dot0, av16, bv16);
            dot1 = vfmlalq_high_f16(dot1, av16, bv16);

            let av16 = load_f16x8(&a[(i + 8) * 2..]);
            let bv16 = load_f16x8(&b[(i + 8) * 2..]);
            dot2 = vfmlalq_low_f16(dot2, av16, bv16);
            dot3 = vfmlalq_high_f16(dot3, av16, bv16);
        }
    }

    dot0 = vaddq_f32(vaddq_f32(dot0, dot1), vaddq_f32(dot2, dot3));
    let len8 = len & !7;
    if len16 < len8 {
        unsafe {
            let av16 = load_f16x8(&a[len16 * 2..]);
            let bv16 = load_f16x8(&b[len16 * 2..]);
            dot0 = vfmlalq_low_f16(dot0, av16, bv16);
            dot0 = vfmlalq_high_f16(dot0, av16, bv16);
        }
    }

    if len8 < len {
        unsafe {
            let av16 = load_tail_f16x8(&a[len8 * 2..len * 2]);
            let bv16 = load_tail_f16x8(&b[len8 * 2..len * 2]);
            dot0 = vfmlalq_low_f16(dot0, av16, bv16);
            dot0 = vfmlalq_high_f16(dot0, av16, bv16);
        }
    }

    vaddvq_f32(dot0)
}

/// Dot product of a single precision vector and a half precision vector.
///
/// # Safety
///
/// The `fp16` CPU feature must be available, e.g. verified with
/// `std::arch::is_aarch64_feature_detected!("fp16")` before calling this function.
#[target_feature(enable = "fp16")]
pub unsafe fn dot_f32_f16(a: &[f32], b: &[u8]) -> f32 {
    assert_eq!(b.len(), a.len() * 2);
    let len = a.len();

    let mut dot0 = vdupq_n_f32(0.0);
    let mut dot1 = vdupq_n_f32(0.0);
    let mut dot2 = vdupq_n_f32(0.0);
    let mut dot3 = vdupq_n_f32(0.0);

    let len16 = len & !15;
    for i in (0..len16).step_by(16) {
        unsafe {
            let bv16 = load_f16x8(&b[i * 2..]);
            dot0 = vfmaq_f32(
                dot0,
                vld1q_f32(a.as_ptr().add(i)),
                vcvt_f32_f16(vget_low_f16(bv16)),
            );
            dot1 = vfmaq_f32(
                dot1,
                vld1q_f32(a.as_ptr().add(i + 4)),
                vcvt_high_f32_f16(bv16),
            );

            let bv16 = load_f16x8(&b[(i + 8) * 2..]);
            dot2 = vfmaq_f32(
                dot2,
                vld1q_f32(a.as_ptr().add(i + 8)),
                vcvt_f32_f16(vget_low_f16(bv16)),
            );
            dot3 = vfmaq_f32(
                dot3,
                vld1q_f32(a.as_ptr().add(i + 12)),
                vcvt_high_f32_f16(bv16),
            );
        }
    }

    dot0 = vaddq_f32(vaddq_f32(dot0, dot1), vaddq_f32(dot2, dot3));
    let len4 = len & !3;
    for i in (len16..len4).step_by(4) {
        unsafe {
            let av = vld1q_f32(a.as_ptr().add(i));
            let bv = vcvt_f32_f16(load_f16x4(&b[i * 2..]));
            dot0 = vfmaq_f32(dot0, av, bv);
        }
    }

    if len4 < len {
        unsafe {
            let av = load_tail_f32x4(&a[len4..]);
            let bv = vcvt_f32_f16(load_tail_f16x4(&b[len4 * 2..len * 2]));
            dot0 = vfmaq_f32(dot0, av, bv);
        }
    }

    vaddvq_f32(dot0)
}

/// Squared euclidean (l2) distance between two half precision vectors.
///
/// # Safety
///
/// The `fp16` and `fhm` CPU features must be available, e.g. verified with
/// `std::arch::is_aarch64_feature_detected!("fp16")` and
/// `std::arch::is_aarch64_feature_detected!("fhm")` before calling this function.
#[target_feature(enable = "fp16,fhm")]
pub unsafe fn l2_f16_f16(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len());
    let len = a.len() / 2;

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    let len16 = len & !15;
    for i in (0..len16).step_by(16) {
        unsafe {
            let dv = vsubq_f16(load_f16x8(&a[i * 2..]), load_f16x8(&b[i * 2..]));
            sum0 = vfmlalq_low_f16(sum0, dv, dv);
            sum1 = vfmlalq_high_f16(sum1, dv, dv);

            let dv = vsubq_f16(load_f16x8(&a[(i + 8) * 2..]), load_f16x8(&b[(i + 8) * 2..]));
            sum2 = vfmlalq_low_f16(sum2, dv, dv);
            sum3 = vfmlalq_high_f16(sum3, dv, dv);
        }
    }

    sum0 = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    let len8 = len & !7;
    if len16 < len8 {
        unsafe {
            let dv = vsubq_f16(load_f16x8(&a[len16 * 2..]), load_f16x8(&b[len16 * 2..]));
            sum0 = vfmlalq_low_f16(sum0, dv, dv);
            sum0 = vfmlalq_high_f16(sum0, dv, dv);
        }
    }

    if len8 < len {
        unsafe {
            let dv = vsubq_f16(
                load_tail_f16x8(&a[len8 * 2..len * 2]),
                load_tail_f16x8(&b[len8 * 2..len * 2]),
            );
            sum0 = vfmlalq_low_f16(sum0, dv, dv);
            sum0 = vfmlalq_high_f16(sum0, dv, dv);
        }
    }

    vaddvq_f32(sum0)
}

/// Squared euclidean (l2) distance between a single precision vector and a half precision
/// vector.
///
/// # Safety
///
/// The `fp16` CPU feature must be available, e.g. verified with
/// `std::arch::is_aarch64_feature_detected!("fp16")` before calling this function.
#[target_feature(enable = "fp16")]
pub unsafe fn l2_f32_f16(a: &[f32], b: &[u8]) -> f32 {
    assert_eq!(b.len(), a.len() * 2);
    let len = a.len();

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    let len16 = len & !15;
    for i in (0..len16).step_by(16) {
        unsafe {
            let bv16 = load_f16x8(&b[i * 2..]);
            let mut dv = vsubq_f32(
                vld1q_f32(a.as_ptr().add(i)),
                vcvt_f32_f16(vget_low_f16(bv16)),
            );
            sum0 = vfmaq_f32(sum0, dv, dv);
            dv = vsubq_f32(vld1q_f32(a.as_ptr().add(i + 4)), vcvt_high_f32_f16(bv16));
            sum1 = vfmaq_f32(sum1, dv, dv);

            let bv16 = load_f16x8(&b[(i + 8) * 2..]);
            dv = vsubq_f32(
                vld1q_f32(a.as_ptr().add(i + 8)),
                vcvt_f32_f16(vget_low_f16(bv16)),
            );
            sum2 = vfmaq_f32(sum2, dv, dv);
            dv = vsubq_f32(vld1q_f32(a.as_ptr().add(i + 12)), vcvt_high_f32_f16(bv16));
            sum3 = vfmaq_f32(sum3, dv, dv);
        }
    }

    sum0 = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    let len4 = len & !3;
    for i in (len16..len4).step_by(4) {
        unsafe {
            let dv = vsubq_f32(
                vld1q_f32(a.as_ptr().add(i)),
                vcvt_f32_f16(load_f16x4(&b[i * 2..])),
            );
            sum0 = vfmaq_f32(sum0, dv, dv);
        }
    }

    if len4 < len {
        unsafe {
            let dv = vsubq_f32(
                load_tail_f32x4(&a[len4..]),
                vcvt_f32_f16(load_tail_f16x4(&b[len4 * 2..len * 2])),
            );
            sum0 = vfmaq_f32(sum0, dv, dv);
        }
    }

    vaddvq_f32(sum0)
}
