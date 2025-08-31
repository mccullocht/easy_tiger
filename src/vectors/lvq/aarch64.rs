//! aarch64 implementations of lvq routines.

use std::arch::aarch64::{
    float32x4_t, uint32x4_t, uint8x16_t, vaddlvq_u8, vaddq_f32, vaddq_f64, vaddvq_f32, vaddvq_u16,
    vaddvq_u32, vaddvq_u64, vand_u16, vand_u32, vand_u8, vcombine_u16, vcombine_u32, vcombine_u8,
    vcvt_f64_f32, vcvt_high_f64_f32, vcvtaq_u32_f32, vcvtq_f32_u32, vdivq_f32, vdup_n_u16,
    vdup_n_u32, vdup_n_u8, vdupq_n_f32, vdupq_n_f64, vextq_f64, vfmaq_f32, vfmaq_f64, vget_low_f32,
    vget_low_u16, vgetq_lane_f64, vld1_s16, vld1_s32, vld1_s8, vld1_u8, vld1q_f32, vld1q_s16,
    vld1q_s32, vld1q_s64, vld1q_s8, vld1q_u8, vmaxq_f32, vmaxvq_f32, vminq_f32, vminvq_f32,
    vmovl_high_u16, vmovl_u16, vmovl_u8, vmovn_high_u16, vmovn_high_u32, vmovn_u16, vmovn_u32,
    vmulq_f32, vmulq_f64, vpaddlq_u16, vpaddlq_u32, vpaddlq_u8, vqtbl1q_u8, vreinterpretq_u32_u8,
    vreinterpretq_u8_u16, vreinterpretq_u8_u32, vrndaq_f32, vshl_u16, vshl_u32, vshl_u8, vshlq_u16,
    vshlq_u32, vshlq_u64, vshlq_u8, vst1q_u8, vsubq_f32, vsubq_f64,
};

use crate::vectors::lvq::{PrimaryVector, TwoLevelVector};

use super::{packing, VectorStats, LAMBDA, MINIMUM_MSE_GRID};

pub fn compute_vector_stats(vector: &[f32]) -> VectorStats {
    let tail_split = vector.len() & !3;
    let (min, max, mean, mean_sq, dot) = if tail_split > 0 {
        unsafe {
            let mut min = vdupq_n_f32(f32::MAX);
            let mut max = vdupq_n_f32(f32::MIN);
            let mut mean = vdupq_n_f32(0.0);
            let mut mean_sq = vdupq_n_f32(0.0);
            let mut dot = vdupq_n_f32(0.0);
            for i in (0..tail_split).step_by(4) {
                let x = vld1q_f32(vector.as_ptr().add(i));
                min = vminq_f32(min, x);
                max = vmaxq_f32(max, x);
                let delta = vsubq_f32(x, mean);
                mean = vaddq_f32(mean, vdivq_f32(delta, vdupq_n_f32(((i / 4) + 1) as f32)));
                let delta2 = vsubq_f32(x, mean);
                mean_sq = vfmaq_f32(mean_sq, delta, delta2);
                dot = vfmaq_f32(dot, x, x);
            }

            (
                vminvq_f32(min),
                vmaxvq_f32(max),
                vaddvq_f32(mean) / 4.0,
                reduce_variance(mean, mean_sq, tail_split),
                vaddvq_f32(dot),
            )
        }
    } else {
        (f32::MAX, f32::MIN, 0.0, 0.0, 0.0)
    };
    let (min, max, mean, mean_sq, dot) = vector.iter().copied().enumerate().skip(tail_split).fold(
        (min, max, mean, mean_sq, dot),
        |mut stats, (i, x)| {
            stats.0 = x.min(stats.0);
            stats.1 = x.max(stats.1);
            stats.4 += x * x;
            let delta = x - stats.2;
            stats.2 += delta / (i + 1) as f32;
            stats.3 += delta * (x - stats.2);
            stats
        },
    );
    VectorStats {
        min,
        max,
        mean,
        std_dev: (mean_sq / vector.len() as f32).sqrt(),
        l2_norm_sq: dot,
    }
}

fn reduce_variance(means: float32x4_t, m2: float32x4_t, n: usize) -> f32 {
    let (means2, m2) = unsafe {
        let means0 = vcvt_f64_f32(vget_low_f32(means));
        let means1 = vcvt_high_f64_f32(means);
        let var0 = vcvt_f64_f32(vget_low_f32(m2));
        let var1 = vcvt_high_f64_f32(m2);

        let mean = vmulq_f64(vaddq_f64(means0, means1), vdupq_n_f64(0.5));

        let delta = vsubq_f64(means0, means1);
        let delta_sq = vmulq_f64(delta, delta);
        let weight = vdupq_n_f64(n as f64 / 8.0);
        let m2 = vfmaq_f64(vaddq_f64(var0, var1), delta_sq, weight);

        (mean, m2)
    };

    unsafe {
        let delta = vsubq_f64(means2, vextq_f64::<1>(means2, means2));
        let delta_sq = vmulq_f64(delta, delta);
        let weight = vdupq_n_f64(n as f64 / 4.0);
        let m2 = vfmaq_f64(vaddq_f64(m2, vextq_f64::<1>(m2, m2)), delta_sq, weight);
        vgetq_lane_f64::<0>(m2) as f32
    }
}

pub fn optimize_interval(vector: &[f32], stats: &VectorStats, bits: usize) -> (f32, f32) {
    let norm_sq: f64 = stats.l2_norm_sq.into();
    let mut loss = compute_loss(vector, (stats.min, stats.max), norm_sq, bits);

    let scale = (1.0 - LAMBDA) / norm_sq as f32;
    let mut lower =
        (MINIMUM_MSE_GRID[bits - 1].0 * stats.std_dev + stats.mean).clamp(stats.min, stats.max);
    let mut upper =
        (MINIMUM_MSE_GRID[bits - 1].1 * stats.std_dev + stats.mean).clamp(stats.min, stats.max);

    let points_incl = ((1 << bits) - 1) as f32;
    for _ in 0..5 {
        let step_inv = points_incl / (upper - lower);
        // calculate the grid points for coordinate descent.
        let tail_split = vector.len() & !3;
        let (mut daa, mut dab, mut dbb, mut dax, mut dbx) = if tail_split > 0 {
            unsafe {
                let lower = vdupq_n_f32(lower);
                let upper = vdupq_n_f32(upper);
                let step_inv = vdupq_n_f32(step_inv);
                let mut daa = vdupq_n_f32(0.0);
                let mut dab = vdupq_n_f32(0.0);
                let mut dbb = vdupq_n_f32(0.0);
                let mut dax = vdupq_n_f32(0.0);
                let mut dbx = vdupq_n_f32(0.0);
                for i in (0..tail_split).step_by(4) {
                    let xi = vld1q_f32(vector.as_ptr().add(i));
                    let k = vrndaq_f32(vmulq_f32(
                        vsubq_f32(vminq_f32(vmaxq_f32(xi, lower), upper), lower),
                        step_inv,
                    ));
                    let s = vdivq_f32(k, vdupq_n_f32(points_incl));
                    let si = vsubq_f32(vdupq_n_f32(1.0), s);
                    daa = vfmaq_f32(daa, si, si);
                    dab = vfmaq_f32(dab, si, s);
                    dbb = vfmaq_f32(dbb, s, s);
                    dax = vfmaq_f32(dax, xi, si);
                    dbx = vfmaq_f32(dbx, xi, s);
                }

                (
                    vaddvq_f32(daa),
                    vaddvq_f32(dab),
                    vaddvq_f32(dbb),
                    vaddvq_f32(dax),
                    vaddvq_f32(dbx),
                )
            }
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0)
        };

        for xi in vector.iter().copied().skip(tail_split) {
            let k = ((xi.clamp(lower, upper) - lower) * step_inv).round();
            let s = k / points_incl;
            daa += (1.0 - s) * (1.0 - s);
            dab += (1.0 - s) * s;
            dbb += s * s;
            dax += xi * (1.0 - s);
            dbx += xi * s;
        }
        let m0 = scale * dax * dax + LAMBDA * daa;
        let m1 = scale * dax * dbx + LAMBDA * dab;
        let m2 = scale * dbx * dbx + LAMBDA * dbb;
        let det = m0 * m2 - m1 * m1;
        // if the determinant is zero we can't update the interval
        if det == 0.0 {
            break;
        }

        let lower_candidate = (m2 * dax - m1 * dbx) / det;
        let upper_candidate = (m0 * dbx - m1 * dax) / det;
        if (lower - lower_candidate).abs() < 1e-8 && (upper - upper_candidate).abs() < 1e-8 {
            break;
        }
        let loss_candidate =
            compute_loss(vector, (lower_candidate, upper_candidate), norm_sq, bits);
        if loss_candidate > loss {
            break;
        }
        lower = lower_candidate;
        upper = upper_candidate;
        loss = loss_candidate;
    }
    (lower, upper)
}

pub fn compute_loss(vector: &[f32], interval: (f32, f32), norm_sq: f64, bits: usize) -> f64 {
    let a = interval.0;
    let b = interval.1;
    let step = (b - a) / ((1 << bits) - 1) as f32;
    let step_inv = step.recip();

    let tail_split = vector.len() & !3;
    let (mut xe, mut e) = if tail_split > 0 {
        unsafe {
            let a = vdupq_n_f32(a);
            let b = vdupq_n_f32(b);
            let step = vdupq_n_f32(step);
            let step_inv = vdupq_n_f32(step_inv);
            let mut xe = vdupq_n_f32(0.0);
            let mut e = vdupq_n_f32(0.0);
            for i in (0..tail_split).step_by(4) {
                let xi = vld1q_f32(vector.as_ptr().add(i));
                let mut xiq = vmaxq_f32(xi, a);
                xiq = vminq_f32(xiq, b);
                xiq = vsubq_f32(xiq, a);
                xiq = vmulq_f32(xiq, step_inv);
                xiq = vrndaq_f32(xiq);
                xiq = vfmaq_f32(a, xiq, step);
                let diff = vsubq_f32(xi, xiq);
                xe = vfmaq_f32(xe, xi, diff);
                e = vfmaq_f32(e, diff, diff);
            }
            (vaddvq_f32(xe), vaddvq_f32(e))
        }
    } else {
        (0.0, 0.0)
    };

    for xi in vector.iter().skip(tail_split).copied() {
        let xiq = a + step * ((xi.clamp(a, b) - a) * step_inv).round();
        let diff = xi - xiq;
        xe += xi * diff;
        e += diff * diff;
    }
    (1.0 - LAMBDA as f64) * xe as f64 * xe as f64 / norm_sq + LAMBDA as f64 * e as f64
}

pub fn lvq1_quantize_and_pack<const B: usize>(
    v: &[f32],
    lower: f32,
    upper: f32,
    out: &mut [u8],
) -> u32 {
    let delta = (upper - lower) / ((1 << B) - 1) as f32;

    let tail_split = v.len() & !15;
    let (head, tail) = out.split_at_mut(packing::byte_len(tail_split, B));
    let component_sum = if tail_split > 0 {
        unsafe {
            let lowerv = vdupq_n_f32(lower);
            let upperv = vdupq_n_f32(upper);
            let deltav = vdupq_n_f32(delta.recip());
            let mut component_sumv = 0u32;
            for i in (0..tail_split).step_by(16) {
                // Load and quantize 16 values.
                let qa = quantize4(vld1q_f32(v.as_ptr().add(i)), lowerv, upperv, deltav);
                let qb = quantize4(vld1q_f32(v.as_ptr().add(i + 4)), lowerv, upperv, deltav);
                let qc = quantize4(vld1q_f32(v.as_ptr().add(i + 8)), lowerv, upperv, deltav);
                let qd = quantize4(vld1q_f32(v.as_ptr().add(i + 12)), lowerv, upperv, deltav);

                // Reduce to a single byte per dimension.
                let qabcd = pack_to_byte(qa, qb, qc, qd);
                component_sumv += u32::from(vaddlvq_u8(qabcd));

                match B {
                    1 => pack1(i, qabcd, head),
                    2 => pack2(i, qabcd, head),
                    4 => pack4(i, qabcd, head),
                    8 => pack8(i, qabcd, head),
                    _ => unimplemented!(),
                };
            }
            component_sumv
        }
    } else {
        0
    };

    if tail_split < v.len() {
        component_sum
            + super::scalar::lvq1_quantize_and_pack::<B>(&v[tail_split..], lower, upper, tail)
    } else {
        component_sum
    }
}

pub fn lvq2_quantize_and_pack<const B1: usize, const B2: usize>(
    v: &[f32],
    lower: f32,
    upper: f32,
    primary: &mut [u8],
    residual_interval: f32,
    residual: &mut [u8],
) -> u32 {
    let delta = (upper - lower) / ((1 << B1) - 1) as f32;
    let res_lower = -residual_interval / 2.0;
    let res_upper = residual_interval / 2.0;
    let res_delta = residual_interval / ((1 << B2) - 1) as f32;

    let tail_split = v.len() & !15;
    let (head_primary, tail_primary) = primary.split_at_mut(packing::byte_len(tail_split, B1));
    let (head_residual, tail_residual) = residual.split_at_mut(packing::byte_len(tail_split, B2));
    let component_sum = if tail_split > 0 {
        unsafe {
            let lowerv = vdupq_n_f32(lower);
            let upperv = vdupq_n_f32(upper);
            let deltav = vdupq_n_f32(delta);
            let delta_inv = vdupq_n_f32(delta.recip());
            let res_lowerv = vdupq_n_f32(res_lower);
            let res_upperv = vdupq_n_f32(res_upper);
            let res_delta_inv = vdupq_n_f32(res_delta.recip());
            let mut component_sumv = 0u32;
            for i in (0..tail_split).step_by(16) {
                // Load and quantize 16 values, primary and residual
                let a = vld1q_f32(v.as_ptr().add(i));
                let qa = quantize4(a, lowerv, upperv, delta_inv);
                let ra = quantize_residual4(
                    a,
                    qa,
                    lowerv,
                    deltav,
                    res_lowerv,
                    res_upperv,
                    res_delta_inv,
                );

                let b = vld1q_f32(v.as_ptr().add(i + 4));
                let qb = quantize4(b, lowerv, upperv, delta_inv);
                let rb = quantize_residual4(
                    b,
                    qb,
                    lowerv,
                    deltav,
                    res_lowerv,
                    res_upperv,
                    res_delta_inv,
                );

                let c = vld1q_f32(v.as_ptr().add(i + 8));
                let qc = quantize4(c, lowerv, upperv, delta_inv);
                let rc = quantize_residual4(
                    c,
                    qc,
                    lowerv,
                    deltav,
                    res_lowerv,
                    res_upperv,
                    res_delta_inv,
                );

                let d = vld1q_f32(v.as_ptr().add(i + 12));
                let qd = quantize4(d, lowerv, upperv, delta_inv);
                let rd = quantize_residual4(
                    d,
                    qd,
                    lowerv,
                    deltav,
                    res_lowerv,
                    res_upperv,
                    res_delta_inv,
                );

                // Reduce to a single byte per dimension, sum, and pack.
                let qabcd = pack_to_byte(qa, qb, qc, qd);
                component_sumv += u32::from(vaddlvq_u8(qabcd));
                match B1 {
                    1 => pack1(i, qabcd, head_primary),
                    2 => pack2(i, qabcd, head_primary),
                    4 => pack4(i, qabcd, head_primary),
                    8 => pack8(i, qabcd, head_primary),
                    _ => unimplemented!(),
                };

                // Reduce to a single byte per dimension and pack.
                let rabcd = pack_to_byte(ra, rb, rc, rd);
                match B2 {
                    1 => pack1(i, rabcd, head_residual),
                    2 => pack2(i, rabcd, head_residual),
                    4 => pack4(i, rabcd, head_residual),
                    8 => pack8(i, rabcd, head_residual),
                    _ => unimplemented!(),
                };
            }
            component_sumv
        }
    } else {
        0
    };

    if tail_split < v.len() {
        component_sum
            + super::scalar::lvq2_quantize_and_pack::<B1, B2>(
                &v[tail_split..],
                lower,
                upper,
                tail_primary,
                residual_interval,
                tail_residual,
            )
    } else {
        component_sum
    }
}

#[inline(always)]
unsafe fn quantize4(
    v: float32x4_t,
    lower: float32x4_t,
    upper: float32x4_t,
    delta_inv: float32x4_t,
) -> uint32x4_t {
    vcvtaq_u32_f32(vmulq_f32(vsubq_f32(vminq_f32(v, upper), lower), delta_inv))
}

#[inline(always)]
unsafe fn quantize_residual4(
    v: float32x4_t,
    q: uint32x4_t,
    lower: float32x4_t,
    delta: float32x4_t,
    res_lower: float32x4_t,
    res_upper: float32x4_t,
    res_delta: float32x4_t,
) -> uint32x4_t {
    let q = vfmaq_f32(lower, vcvtq_f32_u32(q), delta);
    let res = vsubq_f32(v, q);
    vcvtaq_u32_f32(vmulq_f32(
        vsubq_f32(vminq_f32(res, res_upper), res_lower),
        res_delta,
    ))
}

#[inline(always)]
unsafe fn pack_to_byte(a: uint32x4_t, b: uint32x4_t, c: uint32x4_t, d: uint32x4_t) -> uint8x16_t {
    let ab = vmovn_high_u32(vmovn_u32(a), b);
    let cd = vmovn_high_u32(vmovn_u32(c), d);
    vmovn_high_u16(vmovn_u16(ab), cd)
}

/// Pack 16 scalar quantized entries into 1 bit per dimension (2 bytes) and write to out.
#[inline(always)]
unsafe fn pack1(start_dim: usize, qabcd: uint8x16_t, out: &mut [u8]) {
    // pack 2 dimensions in a single lane and widen to 16
    let qp2abcd = vpaddlq_u8(vshlq_u8(
        qabcd,
        vld1q_s8([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1].as_ptr()),
    ));
    let v = vaddvq_u16(vshlq_u16(
        qp2abcd,
        vld1q_s16([0, 2, 4, 6, 8, 10, 12, 14].as_ptr()),
    ));
    std::ptr::write_unaligned(out.as_mut_ptr().add(start_dim / 8) as *mut u16, v.to_le());
}

/// Pack 16 scalar quantized entries into 2 bits per dimension (4 bytes) and write to out.
#[inline(always)]
unsafe fn pack2(start_dim: usize, qabcd: uint8x16_t, out: &mut [u8]) {
    // pack 2 dimensions in a single lane with pair add and widen
    let qp2abcd = vpaddlq_u8(vshlq_u8(
        qabcd,
        vld1q_s8([0, 2, 4, 8, 0, 2, 4, 8, 0, 2, 4, 8, 0, 2, 4, 8].as_ptr()),
    ));
    // pack 4 dimensions in a single lane (low byte) with pair add and widen.
    let qp4abcd = vpaddlq_u16(qp2abcd);
    // shift each entry into a different byte and sum across.
    let v = vaddvq_u32(vshlq_u32(qp4abcd, vld1q_s32([0, 8, 16, 24].as_ptr())));
    std::ptr::write_unaligned(out.as_mut_ptr().add(start_dim / 4) as *mut u32, v.to_le());
}

/// Pack 16 scalar quantized entries into 4 bits per dimension (8 bytes) and write to out.
#[inline(always)]
unsafe fn pack4(start_dim: usize, qabcd: uint8x16_t, out: &mut [u8]) {
    // pack 2 dimensions in a single lane with pair add and widen to 16
    let qp2abcd = vpaddlq_u8(vshlq_u8(
        qabcd,
        vld1q_s8([0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4].as_ptr()),
    ));
    // pack 4 dimensions in a single lane with pair add and widen to 32.
    let qp4abcd = vpaddlq_u16(vshlq_u16(
        qp2abcd,
        vld1q_s16([0, 8, 0, 8, 0, 8, 0, 8].as_ptr()),
    ));
    // pack 8 dimensions in a single lane with pair add and widen to 32.
    let qp8abcd = vpaddlq_u32(vshlq_u32(qp4abcd, vld1q_s32([0, 16, 0, 16].as_ptr())));
    let v = vaddvq_u64(vshlq_u64(qp8abcd, vld1q_s64([0, 32].as_ptr())));
    std::ptr::write_unaligned(out.as_mut_ptr().add(start_dim / 2) as *mut u64, v.to_le());
}

/// Pack 16 scalar quantized entries into 8 bits per dimension (16 bytes) and write to out.
#[inline(always)]
unsafe fn pack8(start_dim: usize, qabcd: uint8x16_t, out: &mut [u8]) {
    vst1q_u8(out.as_mut_ptr().add(start_dim), qabcd);
}

struct LVQ1F32Converter {
    delta: float32x4_t,
    lower: float32x4_t,
}

impl LVQ1F32Converter {
    #[inline(always)]
    unsafe fn from_vector<const B: usize>(vector: &PrimaryVector<'_, B>) -> Self {
        Self {
            delta: vdupq_n_f32(vector.delta),
            lower: vdupq_n_f32(vector.header.lower),
        }
    }

    #[inline(always)]
    unsafe fn unpacked_to_f32(&self, unpacked: uint32x4_t) -> float32x4_t {
        vfmaq_f32(self.lower, self.delta, vcvtq_f32_u32(unpacked))
    }
}

struct LVQ2F32Converter {
    l1: LVQ1F32Converter,
    delta: float32x4_t,
    lower: float32x4_t,
}

impl LVQ2F32Converter {
    #[inline(always)]
    unsafe fn from_vector<const B1: usize, const B2: usize>(
        vector: &TwoLevelVector<'_, B1, B2>,
    ) -> Self {
        Self {
            l1: LVQ1F32Converter::from_vector(&vector.primary),
            delta: vdupq_n_f32(vector.delta),
            lower: vdupq_n_f32(vector.lower),
        }
    }

    #[inline(always)]
    unsafe fn unpacked_to_f32(&self, unpacked: (uint32x4_t, uint32x4_t)) -> float32x4_t {
        let l1 = self.l1.unpacked_to_f32(unpacked.0);
        let l2 = vfmaq_f32(self.lower, self.delta, vcvtq_f32_u32(unpacked.1));
        vaddq_f32(l1, l2)
    }
}

/// Perform unnormalized dot product between an `f32` query and a primary lvq vector.
pub fn lvq1_f32_dot_unnormalized<const B: usize>(query: &[f32], doc: &PrimaryVector<'_, B>) -> f64 {
    let tail_split = query.len() & !7;
    let (query_head, query_tail) = query.split_at(tail_split);
    let (doc_head, doc_tail) = doc.vector.split_at(packing::byte_len(tail_split, B));

    let pdot = if !query_head.is_empty() {
        unsafe {
            let converter = LVQ1F32Converter::from_vector(doc);
            let mut dot = vdupq_n_f32(0.0);
            for i in (0..tail_split).step_by(8) {
                let q = (
                    vld1q_f32(query_head.as_ptr().add(i)),
                    vld1q_f32(query_head.as_ptr().add(i + 4)),
                );
                let d = unpack::<B>(i, doc_head);
                dot = vfmaq_f32(dot, q.0, converter.unpacked_to_f32(d.0));
                dot = vfmaq_f32(dot, q.1, converter.unpacked_to_f32(d.1));
            }
            vaddvq_f32(dot)
        }
    } else {
        0.0
    };

    (pdot
        + query_tail
            .iter()
            .zip(
                packing::unpack_iter::<B>(doc_tail)
                    .map(|q| q as f32 * doc.delta + doc.header.lower),
            )
            .map(|(q, d)| *q * d)
            .sum::<f32>())
    .into()
}

pub fn lvq2_dot_unnormalized<const B1: usize, const B2: usize>(
    a: &TwoLevelVector<'_, B1, B2>,
    b: &TwoLevelVector<'_, B1, B2>,
) -> f64 {
    let dim = if B1 > B2 {
        (a.primary.vector.len() * 8).div_ceil(B1)
    } else {
        (a.vector.len() * 8).div_ceil(B2)
    };
    let tail_split = dim & !7;

    let (a_l1_head, _) = a.primary.vector.split_at(packing::byte_len(tail_split, B1));
    let (a_l2_head, _) = a.vector.split_at(packing::byte_len(tail_split, B2));
    let (b_l1_head, _) = b.primary.vector.split_at(packing::byte_len(tail_split, B1));
    let (b_l2_head, _) = b.vector.split_at(packing::byte_len(tail_split, B2));

    let pdot = if !a_l1_head.is_empty() {
        unsafe {
            let a_converter = LVQ2F32Converter::from_vector(a);
            let b_converter = LVQ2F32Converter::from_vector(b);

            let mut dot = vdupq_n_f32(0.0);
            for i in (0..tail_split).step_by(8) {
                let a = unpack_lvq2::<B1, B2>(i, a_l1_head, a_l2_head);
                let b = unpack_lvq2::<B1, B2>(i, b_l1_head, b_l2_head);
                dot = vfmaq_f32(
                    dot,
                    a_converter.unpacked_to_f32(a.0),
                    b_converter.unpacked_to_f32(b.0),
                );
                dot = vfmaq_f32(
                    dot,
                    a_converter.unpacked_to_f32(a.1),
                    b_converter.unpacked_to_f32(b.1),
                );
            }
            vaddvq_f32(dot)
        }
    } else {
        0.0
    };

    if tail_split < dim {
        pdot + a
            .f32_iter()
            .skip(tail_split)
            .zip(b.f32_iter().skip(tail_split))
            .map(|(a, b)| a * b)
            .sum::<f32>()
    } else {
        pdot
    }
    .into()
}

pub fn lvq2_f32_dot_unnormalized<const B1: usize, const B2: usize>(
    query: &[f32],
    doc: &TwoLevelVector<'_, B1, B2>,
) -> f64 {
    let tail_split = query.len() & !7;

    let (doc_l1_head, _) = doc
        .primary
        .vector
        .split_at(packing::byte_len(tail_split, B1));
    let (doc_l2_head, _) = doc.vector.split_at(packing::byte_len(tail_split, B2));

    let pdot = if !doc_l1_head.is_empty() {
        unsafe {
            let converter = LVQ2F32Converter::from_vector(doc);
            let mut dot = vdupq_n_f32(0.0);
            for i in (0..tail_split).step_by(8) {
                let q = (
                    vld1q_f32(query.as_ptr().add(i)),
                    vld1q_f32(query.as_ptr().add(i + 4)),
                );
                let d = unpack_lvq2::<B1, B2>(i, doc_l1_head, doc_l2_head);
                dot = vfmaq_f32(dot, q.0, converter.unpacked_to_f32(d.0));
                dot = vfmaq_f32(dot, q.1, converter.unpacked_to_f32(d.1));
            }
            vaddvq_f32(dot)
        }
    } else {
        0.0
    };

    if tail_split < query.len() {
        pdot + query
            .iter()
            .skip(tail_split)
            .zip(doc.f32_iter().skip(tail_split))
            .map(|(q, d)| *q * d)
            .sum::<f32>()
    } else {
        pdot
    }
    .into()
}

// Unpack 8 values from a vector with N-bit dimensions starting at `start_dim`
#[inline(always)]
unsafe fn unpack_lvq2<const B1: usize, const B2: usize>(
    start_dim: usize,
    l1: &[u8],
    l2: &[u8],
) -> ((uint32x4_t, uint32x4_t), (uint32x4_t, uint32x4_t)) {
    let l1 = unpack::<B1>(start_dim, l1);
    let l2 = unpack::<B2>(start_dim, l2);
    ((l1.0, l2.0), (l1.1, l2.1))
}

// Unpack 8 values from a vector with N-bit dimensions starting at `start_dim`
#[inline(always)]
unsafe fn unpack<const N: usize>(start_dim: usize, vector: &[u8]) -> (uint32x4_t, uint32x4_t) {
    match N {
        1 => unpack1(start_dim, vector),
        2 => unpack2(start_dim, vector),
        4 => unpack4(start_dim, vector),
        8 => unpack8(start_dim, vector),
        _ => unimplemented!(),
    }
}

// Unpack 8 values from a vector with 1-bit dimensions starting at `start_dim`.
#[inline(always)]
unsafe fn unpack1(start_dim: usize, vector: &[u8]) -> (uint32x4_t, uint32x4_t) {
    let mut dp = vdup_n_u8(vector[start_dim / 8]);
    dp = vand_u8(
        vshl_u8(dp, vld1_s8([0, -1, -2, -3, -4, -5, -6, -7].as_ptr())),
        vdup_n_u8(1),
    );
    let d = vcombine_u8(dp, dp);
    (
        vreinterpretq_u32_u8(vqtbl1q_u8(
            d,
            vld1q_u8([0, 16, 16, 16, 1, 16, 16, 16, 2, 16, 16, 16, 3, 16, 16, 16].as_ptr()),
        )),
        vreinterpretq_u32_u8(vqtbl1q_u8(
            d,
            vld1q_u8([4, 16, 16, 16, 5, 16, 16, 16, 6, 16, 16, 16, 7, 16, 16, 16].as_ptr()),
        )),
    )
}

// Unpack 8 values from a vector with 2-bit dimensions starting at `start_dim`.
#[inline(always)]
unsafe fn unpack2(start_dim: usize, vector: &[u8]) -> (uint32x4_t, uint32x4_t) {
    let mut dp = vdup_n_u16(u16::from_le_bytes(std::ptr::read_unaligned(
        vector.as_ptr().add(start_dim / 4) as *const [u8; 2],
    )));
    // packed [0, 4, 1, 5, 2, 6, 3, 7]
    dp = vand_u16(
        vshl_u16(dp, vld1_s16([0, -2, -4, -6].as_ptr())),
        vdup_n_u16(0x3333),
    );
    let d = vreinterpretq_u8_u16(vcombine_u16(dp, dp));
    (
        vreinterpretq_u32_u8(vqtbl1q_u8(
            d,
            vld1q_u8([0, 16, 16, 16, 4, 16, 16, 16, 1, 16, 16, 16, 5, 16, 16, 16].as_ptr()),
        )),
        vreinterpretq_u32_u8(vqtbl1q_u8(
            d,
            vld1q_u8([2, 16, 16, 16, 6, 16, 16, 16, 3, 16, 16, 16, 7, 16, 16, 16].as_ptr()),
        )),
    )
}

// Unpack 8 values from a vector with 4-bit dimensions starting at `start_dim`.
#[inline(always)]
unsafe fn unpack4(start_dim: usize, vector: &[u8]) -> (uint32x4_t, uint32x4_t) {
    let mut dp = vdup_n_u32(u32::from_le_bytes(std::ptr::read_unaligned(
        vector.as_ptr().add(start_dim / 2) as *const [u8; 4],
    )));
    // now the value are packed in bytes in dim order [0, 2, 4, 8, 1, 3, 5, 7]
    dp = vand_u32(
        vshl_u32(dp, vld1_s32([0, -4].as_ptr())),
        vdup_n_u32(0x0f0f0f0f),
    );
    let d = vreinterpretq_u8_u32(vcombine_u32(dp, dp));
    (
        vreinterpretq_u32_u8(vqtbl1q_u8(
            d,
            vld1q_u8([0, 16, 16, 16, 4, 16, 16, 16, 1, 16, 16, 16, 5, 16, 16, 16].as_ptr()),
        )),
        vreinterpretq_u32_u8(vqtbl1q_u8(
            d,
            vld1q_u8([2, 16, 16, 16, 6, 16, 16, 16, 3, 16, 16, 16, 7, 16, 16, 16].as_ptr()),
        )),
    )
}

// Unpack 8 values from a vector with 8-bit dimensions starting at `start_dim`.
#[inline(always)]
unsafe fn unpack8(start_dim: usize, vector: &[u8]) -> (uint32x4_t, uint32x4_t) {
    let d = vmovl_u8(vld1_u8(vector.as_ptr().add(start_dim)));
    (vmovl_u16(vget_low_u16(d)), vmovl_high_u16(d))
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;

    #[test]
    fn compute_vector_stats() {
        let vector = [
            -0.3395960498933277f32,
            -0.107044758804973,
            -0.9136713822456826,
            0.7929940644049387,
            -0.9663694282674806,
            -0.06159472946522837,
            -0.5468361488246924,
            -0.5161968350093071,
            -0.8445820026353674,
            0.7554848919515382,
            0.9397280985327834,
            -0.811219062040905,
            -0.11134847738907583,
            0.020033662822942944,
            0.9628870565403511,
            -0.5074461455585713,
        ];
        let scalar_stats = crate::vectors::lvq::scalar::compute_vector_stats(&vector);
        let aarch64_stats = super::compute_vector_stats(&vector);
        assert_eq!(scalar_stats.min, aarch64_stats.min);
        assert_eq!(scalar_stats.max, aarch64_stats.max);
        assert_abs_diff_eq!(scalar_stats.mean, aarch64_stats.mean, epsilon = 0.00001);
        assert_abs_diff_eq!(
            scalar_stats.std_dev,
            aarch64_stats.std_dev,
            epsilon = 0.00001
        );
        assert_abs_diff_eq!(
            scalar_stats.l2_norm_sq,
            aarch64_stats.l2_norm_sq,
            epsilon = 0.00001
        );
    }

    #[test]
    fn compute_vector_stats1() {
        let vector = [-0.30671382, 0.76678455, 0.21469967, -0.5214135];
        let scalar_stats = crate::vectors::lvq::scalar::compute_vector_stats(&vector);
        let aarch64_stats = super::compute_vector_stats(&vector);
        assert_eq!(scalar_stats.min, aarch64_stats.min);
        assert_eq!(scalar_stats.max, aarch64_stats.max);
        assert_abs_diff_eq!(scalar_stats.mean, aarch64_stats.mean, epsilon = 0.00001);
        assert_abs_diff_eq!(
            scalar_stats.std_dev,
            aarch64_stats.std_dev,
            epsilon = 0.00001
        );
        assert_abs_diff_eq!(
            scalar_stats.l2_norm_sq,
            aarch64_stats.l2_norm_sq,
            epsilon = 0.00001
        );
    }
}
