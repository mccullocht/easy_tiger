#![allow(unsafe_op_in_unsafe_fn)]

#[inline]
pub unsafe fn dot(a: &[u8], b: &[u8]) -> f64 {
    use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32};
    let len64 = a.len() & !63;
    let mut dot0 = vdupq_n_f32(0.0);
    let mut dot1 = vdupq_n_f32(0.0);
    let mut dot2 = vdupq_n_f32(0.0);
    let mut dot3 = vdupq_n_f32(0.0);
    for i in (0..len64).step_by(64) {
        dot0 = vfmaq_f32(
            dot0,
            load_f32x4_le(a.as_ptr().add(i)),
            load_f32x4_le(b.as_ptr().add(i)),
        );
        dot1 = vfmaq_f32(
            dot1,
            load_f32x4_le(a.as_ptr().add(i + 16)),
            load_f32x4_le(b.as_ptr().add(i + 16)),
        );
        dot2 = vfmaq_f32(
            dot2,
            load_f32x4_le(a.as_ptr().add(i + 32)),
            load_f32x4_le(b.as_ptr().add(i + 32)),
        );
        dot3 = vfmaq_f32(
            dot3,
            load_f32x4_le(a.as_ptr().add(i + 48)),
            load_f32x4_le(b.as_ptr().add(i + 48)),
        );
    }

    dot0 = vaddq_f32(vaddq_f32(dot0, dot1), vaddq_f32(dot2, dot3));
    let len16 = a.len() & !15;
    for i in (len64..len16).step_by(16) {
        dot0 = vfmaq_f32(
            dot0,
            load_f32x4_le(a.as_ptr().add(i)),
            load_f32x4_le(b.as_ptr().add(i)),
        );
    }

    let mut dot = vaddvq_f32(dot0);
    for i in (len16..a.len()).step_by(4) {
        dot += std::ptr::read_unaligned(a.as_ptr().add(i) as *const f32)
            * std::ptr::read_unaligned(b.as_ptr().add(i) as *const f32);
    }
    dot.into()
}

#[inline]
pub unsafe fn l2sq(q: &[u8], d: &[u8]) -> f64 {
    use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vsubq_f32};

    let len64 = q.len() & !63;
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);
    for i in (0..len64).step_by(64) {
        let mut diff = vsubq_f32(
            load_f32x4_le(q.as_ptr().add(i)),
            load_f32x4_le(d.as_ptr().add(i)),
        );
        sum0 = vfmaq_f32(sum0, diff, diff);

        diff = vsubq_f32(
            load_f32x4_le(q.as_ptr().add(i + 16)),
            load_f32x4_le(d.as_ptr().add(i + 16)),
        );
        sum1 = vfmaq_f32(sum1, diff, diff);

        diff = vsubq_f32(
            load_f32x4_le(q.as_ptr().add(i + 32)),
            load_f32x4_le(d.as_ptr().add(i + 32)),
        );
        sum2 = vfmaq_f32(sum2, diff, diff);

        diff = vsubq_f32(
            load_f32x4_le(q.as_ptr().add(i + 48)),
            load_f32x4_le(d.as_ptr().add(i + 48)),
        );
        sum3 = vfmaq_f32(sum3, diff, diff);
    }

    sum0 = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    let len16 = q.len() & !15;
    for i in (len64..len16).step_by(16) {
        let diff = vsubq_f32(
            load_f32x4_le(q.as_ptr().add(i)),
            load_f32x4_le(d.as_ptr().add(i)),
        );
        sum0 = vfmaq_f32(sum0, diff, diff);
    }

    let mut sum = vaddvq_f32(sum0);
    for i in (len16..q.len()).step_by(4) {
        let diff = std::ptr::read_unaligned(q.as_ptr().add(i) as *const f32)
            - std::ptr::read_unaligned(d.as_ptr().add(i) as *const f32);
        sum = diff.mul_add(diff, sum);
    }

    sum.into()
}

#[inline(always)]
unsafe fn load_f32x4_le(p: *const u8) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64;
    if cfg!(target_endian = "big") {
        aarch64::vreinterpretq_f32_u8(aarch64::vrev32q_u8(aarch64::vld1q_u8(p)))
    } else {
        aarch64::vld1q_f32(p as *const f32)
    }
}
