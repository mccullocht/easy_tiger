//! aarch64 implementations of lvq routines.

#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::aarch64::{
    float32x4_t, uint8x16_t, uint8x16x4_t, uint32x4_t, vaddlvq_u8, vaddlvq_u16, vaddq_f32,
    vaddq_f64, vaddq_u16, vaddvq_f32, vandq_u8, vandq_u32, vcntq_u8, vcvt_f64_f32,
    vcvt_high_f64_f32, vcvtaq_u32_f32, vcvtq_f32_u32, vdivq_f32, vdupq_n_f32, vdupq_n_f64,
    vdupq_n_s8, vdupq_n_u8, vdupq_n_u16, vdupq_n_u32, vextq_f64, vfmaq_f32, vfmaq_f64,
    vget_low_f32, vgetq_lane_f64, vld1q_f32, vld1q_u8, vmaxq_f32, vmaxvq_f32, vminq_f32,
    vminvq_f32, vmulq_f32, vmulq_f64, vorrq_u8, vpaddlq_u8, vqtbl1q_u8, vqtbl4q_u8,
    vreinterpretq_u8_u32, vreinterpretq_u32_u8, vrndaq_f32, vshlq_u8, vshrq_n_u32, vst1q_f32,
    vst1q_u8, vsubq_f32, vsubq_f64,
};

use crate::lvq::{
    RESIDUAL_BITS, ResidualDotComponents, TURBO_BLOCK_SIZE, TurboPrimaryVector,
    TurboResidualVector, VectorDecodeTerms, VectorEncodeTerms, scalar,
};

use super::{LAMBDA, MINIMUM_MSE_GRID, VectorStats, packing};

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

pub fn optimize_interval_neon(vector: &[f32], stats: &VectorStats, bits: usize) -> (f32, f32) {
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

fn compute_loss(vector: &[f32], interval: (f32, f32), norm_sq: f64, bits: usize) -> f64 {
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

pub fn primary_quantize_and_pack<const B: usize>(
    vector: &[f32],
    terms: VectorEncodeTerms,
    out: &mut [u8],
) -> u32 {
    let tail_split = vector.len() & !(packing::block_dim(B) - 1);
    assert!(tail_split.is_multiple_of(16));
    let (vector_head, vector_tail) = vector.split_at(tail_split);
    let (out_head, out_tail) = out.split_at_mut(packing::byte_len(tail_split, B));
    let mut component_sum = 0u32;
    if !vector_head.is_empty() {
        unsafe {
            let terms = NeonVectorEncodeTerms::from_terms(&terms);
            let shuffle_mask = vld1q_u8(
                [
                    0u8, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
                ]
                .as_ptr(),
            );
            let mut block = 0usize;
            let mut shift = 0i8;
            let mut d = vdupq_n_u8(0);
            for i in (0..tail_split).step_by(16) {
                let qa = quantize4(vld1q_f32(vector_head.as_ptr().add(i)), &terms);
                let qb = quantize4(vld1q_f32(vector_head.as_ptr().add(i + 4)), &terms);
                let qc = quantize4(vld1q_f32(vector_head.as_ptr().add(i + 8)), &terms);
                let qd = quantize4(vld1q_f32(vector_head.as_ptr().add(i + 12)), &terms);

                let qabcd = vqtbl4q_u8(
                    uint8x16x4_t(
                        vreinterpretq_u8_u32(qa),
                        vreinterpretq_u8_u32(qb),
                        vreinterpretq_u8_u32(qc),
                        vreinterpretq_u8_u32(qd),
                    ),
                    shuffle_mask,
                );
                component_sum += u32::from(vaddlvq_u8(qabcd));

                d = vorrq_u8(d, vshlq_u8(qabcd, vdupq_n_s8(shift)));
                shift += B as i8;
                if shift == 8 {
                    vst1q_u8(out_head.as_mut_ptr().add(block * 16), d);
                    d = vdupq_n_u8(0);
                    shift = 0;
                    block += 1;
                }
            }
        }
    }

    if !vector_tail.is_empty() {
        component_sum += scalar::primary_quantize_and_pack::<B>(vector_tail, terms, out_tail);
    }
    component_sum
}

pub fn primary_decode<const B: usize>(vector: TurboPrimaryVector<'_, B>, out: &mut [f32]) {
    let (tail_split, in_head, in_tail) = vector.split_tail(out.len());
    let (out_head, out_tail) = out.split_at_mut(tail_split);

    if !in_head.rep.data.is_empty() {
        unsafe {
            let lower = vdupq_n_f32(vector.rep.terms.lower);
            let delta = vdupq_n_f32(vector.rep.terms.delta);
            let mut expander = TLVQExpander32::<B>::new(in_head.rep.data.as_ptr());
            for i in (0..tail_split).step_by(16) {
                let [d0, d1, d2, d3] = expander.next();
                vst1q_f32(out_head.as_mut_ptr().add(i), vfmaq_f32(lower, d0, delta));
                vst1q_f32(
                    out_head.as_mut_ptr().add(i + 4),
                    vfmaq_f32(lower, d1, delta),
                );
                vst1q_f32(
                    out_head.as_mut_ptr().add(i + 8),
                    vfmaq_f32(lower, d2, delta),
                );
                vst1q_f32(
                    out_head.as_mut_ptr().add(i + 12),
                    vfmaq_f32(lower, d3, delta),
                );
            }
        }
    }

    if !in_tail.rep.data.is_empty() {
        scalar::primary_decode::<B>(in_tail, out_tail);
    }
}

pub fn residual_quantize_and_pack<const B: usize>(
    vector: &[f32],
    primary_terms: VectorEncodeTerms,
    residual_terms: VectorEncodeTerms,
    primary_delta: f32,
    primary_out: &mut [u8],
    residual_out: &mut [u8],
) -> (u32, u32) {
    let tail_split = vector.len() & !(packing::block_dim(B) - 1);
    assert!(tail_split.is_multiple_of(16));
    let (vector_head, vector_tail) = vector.split_at(tail_split);
    let (primary_out_head, primary_out_tail) =
        primary_out.split_at_mut(packing::byte_len(tail_split, B));
    let (residual_out_head, residual_out_tail) = residual_out.split_at_mut(tail_split);
    let mut primary_component_sum = 0u32;
    let mut residual_component_sum = 0u32;
    if !vector_head.is_empty() {
        unsafe {
            let primary_terms = NeonVectorEncodeTerms::from_terms(&primary_terms);
            let primary_delta = vdupq_n_f32(primary_delta);
            let residual_terms = NeonVectorEncodeTerms::from_terms(&residual_terms);

            let shuffle_mask = vld1q_u8(
                [
                    0u8, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
                ]
                .as_ptr(),
            );
            let mut block = 0usize;
            let mut shift = 0i8;
            let mut d = vdupq_n_u8(0);
            for i in (0..tail_split).step_by(16) {
                let (pa, ra) = quantize4_residual(
                    vld1q_f32(vector_head.as_ptr().add(i)),
                    &primary_terms,
                    primary_delta,
                    &residual_terms,
                );
                let (pb, rb) = quantize4_residual(
                    vld1q_f32(vector_head.as_ptr().add(i + 4)),
                    &primary_terms,
                    primary_delta,
                    &residual_terms,
                );
                let (pc, rc) = quantize4_residual(
                    vld1q_f32(vector_head.as_ptr().add(i + 8)),
                    &primary_terms,
                    primary_delta,
                    &residual_terms,
                );
                let (pd, rd) = quantize4_residual(
                    vld1q_f32(vector_head.as_ptr().add(i + 12)),
                    &primary_terms,
                    primary_delta,
                    &residual_terms,
                );

                let pabcd = vqtbl4q_u8(
                    uint8x16x4_t(
                        vreinterpretq_u8_u32(pa),
                        vreinterpretq_u8_u32(pb),
                        vreinterpretq_u8_u32(pc),
                        vreinterpretq_u8_u32(pd),
                    ),
                    shuffle_mask,
                );
                primary_component_sum += u32::from(vaddlvq_u8(pabcd));

                d = vorrq_u8(d, vshlq_u8(pabcd, vdupq_n_s8(shift)));
                shift += B as i8;
                if shift == 8 {
                    vst1q_u8(primary_out_head.as_mut_ptr().add(block * 16), d);
                    d = vdupq_n_u8(0);
                    shift = 0;
                    block += 1;
                }

                let rabcd = vqtbl4q_u8(
                    uint8x16x4_t(
                        vreinterpretq_u8_u32(ra),
                        vreinterpretq_u8_u32(rb),
                        vreinterpretq_u8_u32(rc),
                        vreinterpretq_u8_u32(rd),
                    ),
                    shuffle_mask,
                );
                residual_component_sum += u32::from(vaddlvq_u8(rabcd));
                vst1q_u8(residual_out_head.as_mut_ptr().add(i), rabcd);
            }
        }
    }

    if !vector_tail.is_empty() {
        let (tail_primary_sum, tail_residual_sum) = scalar::residual_quantize_and_pack::<B>(
            vector_tail,
            primary_terms,
            residual_terms,
            primary_delta,
            primary_out_tail,
            residual_out_tail,
        );
        primary_component_sum += tail_primary_sum;
        residual_component_sum += tail_residual_sum;
    }
    (primary_component_sum, residual_component_sum)
}

pub fn residual_decode<const B: usize>(vector: &TurboResidualVector<'_, B>, out: &mut [f32]) {
    let (tail_split, in_head, in_tail) = vector.split_tail(out.len());
    let (out_head, out_tail) = out.split_at_mut(tail_split);

    if !in_head.primary.data.is_empty() {
        unsafe {
            let primary_terms = NeonVectorDecodeTerms::from_terms(&in_head.primary.terms);
            let residual_terms = NeonVectorDecodeTerms::from_terms(&in_head.residual.terms);
            let mut primary_expander = TLVQExpander32::<B>::new(in_head.primary.data.as_ptr());
            let mut residual_expander =
                TLVQExpander32::<RESIDUAL_BITS>::new(in_head.residual.data.as_ptr());
            for i in (0..tail_split).step_by(16) {
                let [pa, pb, pc, pd] = primary_expander.next();
                let [ra, rb, rc, rd] = residual_expander.next();
                vst1q_f32(
                    out_head.as_mut_ptr().add(i),
                    dequantize4_residual(pa, ra, &primary_terms, &residual_terms),
                );
                vst1q_f32(
                    out_head.as_mut_ptr().add(i + 4),
                    dequantize4_residual(pb, rb, &primary_terms, &residual_terms),
                );
                vst1q_f32(
                    out_head.as_mut_ptr().add(i + 8),
                    dequantize4_residual(pc, rc, &primary_terms, &residual_terms),
                );
                vst1q_f32(
                    out_head.as_mut_ptr().add(i + 12),
                    dequantize4_residual(pd, rd, &primary_terms, &residual_terms),
                );
            }
        }
    }

    if !in_tail.primary.data.is_empty() {
        scalar::residual_decode::<B>(&in_tail, out_tail);
    }
}

#[inline(always)]
unsafe fn dequantize4_residual(
    primary: float32x4_t,
    residual: float32x4_t,
    primary_terms: &NeonVectorDecodeTerms,
    residual_terms: &NeonVectorDecodeTerms,
) -> float32x4_t {
    vaddq_f32(
        vfmaq_f32(primary_terms.lower, primary, primary_terms.delta),
        vfmaq_f32(residual_terms.lower, residual, residual_terms.delta),
    )
}

struct NeonVectorEncodeTerms {
    lower: float32x4_t,
    upper: float32x4_t,
    delta_inv: float32x4_t,
}

impl NeonVectorEncodeTerms {
    #[inline(always)]
    unsafe fn new(lower: f32, upper: f32, delta_inv: f32) -> Self {
        Self {
            lower: vdupq_n_f32(lower),
            upper: vdupq_n_f32(upper),
            delta_inv: vdupq_n_f32(delta_inv),
        }
    }

    #[inline(always)]
    unsafe fn from_terms(terms: &VectorEncodeTerms) -> Self {
        Self::new(terms.lower, terms.upper, terms.delta_inv)
    }
}

struct NeonVectorDecodeTerms {
    lower: float32x4_t,
    delta: float32x4_t,
}

impl NeonVectorDecodeTerms {
    #[inline(always)]
    unsafe fn from_terms(terms: &VectorDecodeTerms) -> Self {
        Self {
            lower: vdupq_n_f32(terms.lower),
            delta: vdupq_n_f32(terms.delta),
        }
    }
}

#[inline(always)]
unsafe fn quantize4(v: float32x4_t, terms: &NeonVectorEncodeTerms) -> uint32x4_t {
    vcvtaq_u32_f32(vmulq_f32(
        vsubq_f32(vminq_f32(v, terms.upper), terms.lower),
        terms.delta_inv,
    ))
}

#[inline(always)]
unsafe fn quantize4_residual(
    v: float32x4_t,
    primary_terms: &NeonVectorEncodeTerms,
    primary_delta: float32x4_t,
    residual_terms: &NeonVectorEncodeTerms,
) -> (uint32x4_t, uint32x4_t) {
    let primary = quantize4(v, primary_terms);
    let dq = vfmaq_f32(primary_terms.lower, vcvtq_f32_u32(primary), primary_delta);
    (primary, quantize4(vsubq_f32(v, dq), residual_terms))
}

unsafe extern "C" {
    // Symmetric dot product 2 bit
    unsafe fn et_lvq_dot_u2(a: *const u8, b: *const u8, len: usize) -> u32;
    // Symmetric dot product 4 bit
    unsafe fn et_lvq_dot_u4(a: *const u8, b: *const u8, len: usize) -> u32;
    // Symmetric dot product 8 bit
    unsafe fn et_lvq_dot_u8(a: *const u8, b: *const u8, len: usize) -> u32;
    // Asymmetric dot product 8 bit-1 bit. len must be a multiple of 128.
    unsafe fn et_lvq_dot_u8_u1(q: *const u8, d: *const u8, len: usize) -> u32;
    // Asymmetric dot product 8 bit-2 bit. len must be a multiple of 64.
    unsafe fn et_lvq_dot_u8_u2(q: *const u8, d: *const u8, len: usize) -> u32;
    // Asymmetric dot product 8 bit-4 bit. len must be a multiple of 32.
    unsafe fn et_lvq_dot_u8_u4(q: *const u8, d: *const u8, len: usize) -> u32;

    // This function requires that len is a multiple of 128.
    unsafe fn et_residual_dot_u1_u8(
        ap: *const u8,
        ar: *const u8,
        bp: *const u8,
        br: *const u8,
        len: usize,
    ) -> ResidualDotComponents;

    // This function requires that len is a multiple of 64.
    unsafe fn et_residual_dot_u2_u8(
        ap: *const u8,
        ar: *const u8,
        bp: *const u8,
        br: *const u8,
        len: usize,
    ) -> ResidualDotComponents;

    // This function requires that len is a multiple of 32.
    unsafe fn et_residual_dot_u4_u8(
        ap: *const u8,
        ar: *const u8,
        bp: *const u8,
        br: *const u8,
        len: usize,
    ) -> ResidualDotComponents;

    // This function requires that len is a multiple of 16.
    unsafe fn et_residual_dot_u8_u8(
        ap: *const u8,
        ar: *const u8,
        bp: *const u8,
        br: *const u8,
        len: usize,
    ) -> ResidualDotComponents;
}

#[inline]
pub fn dot_u8<const B: usize>(a: &[u8], b: &[u8]) -> u32 {
    match B {
        1 => unsafe {
            let mut dot0 = vdupq_n_u16(0);
            let mut dot1 = vdupq_n_u16(0);
            let len32 = a.len() & !31;
            for i in (0..len32).step_by(32) {
                let mut av = vld1q_u8(a.as_ptr().add(i));
                let mut bv = vld1q_u8(b.as_ptr().add(i));
                dot0 = vaddq_u16(dot0, vpaddlq_u8(vcntq_u8(vandq_u8(av, bv))));

                av = vld1q_u8(a.as_ptr().add(i + 16));
                bv = vld1q_u8(b.as_ptr().add(i + 16));
                dot1 = vaddq_u16(dot1, vpaddlq_u8(vcntq_u8(vandq_u8(av, bv))));
            }

            dot0 = vaddq_u16(dot0, dot1);
            let len16 = a.len() & !15;
            if len32 < len16 {
                let av = vld1q_u8(a.as_ptr().add(len32));
                let bv = vld1q_u8(b.as_ptr().add(len32));
                dot0 = vaddq_u16(dot0, vpaddlq_u8(vcntq_u8(vandq_u8(av, bv))));
            }

            let mut dot = vaddlvq_u16(dot0);
            for i in len16..a.len() {
                dot += (a[i] & b[i]).count_ones();
            }
            dot
        },
        2 => unsafe { et_lvq_dot_u2(a.as_ptr(), b.as_ptr(), a.len()) },
        4 => unsafe { et_lvq_dot_u4(a.as_ptr(), b.as_ptr(), a.len()) },
        8 => unsafe { et_lvq_dot_u8(a.as_ptr(), b.as_ptr(), a.len()) },
        _ => unimplemented!(),
    }
}

struct TLVQExpander32<const B: usize> {
    start: *const u8,
    next_block: usize,

    buf: uint32x4_t,
    mask: uint32x4_t,
    shuffle_mask: uint8x16_t,

    shuffle_mask8: [uint8x16_t; 4],
}

impl<const B: usize> TLVQExpander32<B> {
    const SHUFFLE_MASK: [u8; 16] = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];

    const SHUFFLE_MASKS8: [[u8; 16]; 4] = [
        [0, 16, 16, 16, 1, 16, 16, 16, 2, 16, 16, 16, 3, 16, 16, 16],
        [4, 16, 16, 16, 5, 16, 16, 16, 6, 16, 16, 16, 7, 16, 16, 16],
        [8, 16, 16, 16, 9, 16, 16, 16, 10, 16, 16, 16, 11, 16, 16, 16],
        [
            12, 16, 16, 16, 13, 16, 16, 16, 14, 16, 16, 16, 15, 16, 16, 16,
        ],
    ];

    unsafe fn new(start: *const u8) -> Self {
        let (shuffle_mask, shuffle_mask8) = unsafe {
            match B {
                8 => (
                    vdupq_n_u8(0),
                    [
                        vld1q_u8(Self::SHUFFLE_MASKS8[0].as_ptr()),
                        vld1q_u8(Self::SHUFFLE_MASKS8[1].as_ptr()),
                        vld1q_u8(Self::SHUFFLE_MASKS8[2].as_ptr()),
                        vld1q_u8(Self::SHUFFLE_MASKS8[3].as_ptr()),
                    ],
                ),
                _ => (vld1q_u8(Self::SHUFFLE_MASK.as_ptr()), [vdupq_n_u8(0); 4]),
            }
        };
        unsafe {
            Self {
                start,
                next_block: 0,
                buf: vdupq_n_u32(0),
                mask: vdupq_n_u32(u32::from(u8::MAX) >> (8 - B)),
                shuffle_mask,
                shuffle_mask8,
            }
        }
    }

    unsafe fn next(&mut self) -> [float32x4_t; 4] {
        let group = match B {
            8 => {
                let d = vld1q_u8(self.start.add(self.next_block * TURBO_BLOCK_SIZE));
                [
                    vreinterpretq_u32_u8(vqtbl1q_u8(d, self.shuffle_mask8[0])),
                    vreinterpretq_u32_u8(vqtbl1q_u8(d, self.shuffle_mask8[1])),
                    vreinterpretq_u32_u8(vqtbl1q_u8(d, self.shuffle_mask8[2])),
                    vreinterpretq_u32_u8(vqtbl1q_u8(d, self.shuffle_mask8[3])),
                ]
            }
            _ => {
                let interval = 8 / B;
                self.buf = if self.next_block.is_multiple_of(interval) {
                    let x = vld1q_u8(
                        self.start
                            .add((self.next_block / interval) * TURBO_BLOCK_SIZE),
                    );
                    vreinterpretq_u32_u8(vqtbl1q_u8(x, self.shuffle_mask))
                } else {
                    shr_u32::<B>(self.buf)
                };
                [
                    vandq_u32(self.buf, self.mask),
                    vandq_u32(vshrq_n_u32::<8>(self.buf), self.mask),
                    vandq_u32(vshrq_n_u32::<16>(self.buf), self.mask),
                    vandq_u32(vshrq_n_u32::<24>(self.buf), self.mask),
                ]
            }
        };
        self.next_block += 1;
        [
            vcvtq_f32_u32(group[0]),
            vcvtq_f32_u32(group[1]),
            vcvtq_f32_u32(group[2]),
            vcvtq_f32_u32(group[3]),
        ]
    }
}

#[inline]
pub fn primary_query8_dot_unnormalized<const B: usize>(
    query: &[u8],
    doc: &TurboPrimaryVector<'_, B>,
) -> u32 {
    if B == 8 {
        return unsafe { et_lvq_dot_u8(query.as_ptr(), doc.rep.data.as_ptr(), query.len()) };
    }

    let (tail_split, doc_head, doc_tail) = doc.split_tail(query.len());
    let (query_head, query_tail) = query.split_at(tail_split);
    let mut dot = if !query_head.is_empty() {
        match B {
            1 => unsafe {
                et_lvq_dot_u8_u1(
                    query_head.as_ptr(),
                    doc_head.rep.data.as_ptr(),
                    query_head.len(),
                )
            },
            2 => unsafe {
                et_lvq_dot_u8_u2(
                    query_head.as_ptr(),
                    doc_head.rep.data.as_ptr(),
                    query_head.len(),
                )
            },
            4 => unsafe {
                et_lvq_dot_u8_u4(
                    query_head.as_ptr(),
                    doc_head.rep.data.as_ptr(),
                    query_head.len(),
                )
            },
            _ => unimplemented!(),
        }
    } else {
        0
    };

    if !query_tail.is_empty() {
        dot += scalar::primary_query8_dot_unnormalized::<B>(query_tail, &doc_tail);
    }
    dot
}

#[inline]
pub fn residual_dot_unnormalized<const B: usize>(
    query: (&[u8], &[u8]),
    doc: (&[u8], &[u8]),
) -> ResidualDotComponents {
    let (_, query_head, query_tail) = TurboResidualVector::<B>::split_vector_tail(query);
    let (_, doc_head, doc_tail) = TurboResidualVector::<B>::split_vector_tail(doc);

    let mut dot = if !query_head.0.is_empty() {
        match B {
            1 => unsafe {
                et_residual_dot_u1_u8(
                    query_head.0.as_ptr(),
                    query_head.1.as_ptr(),
                    doc_head.0.as_ptr(),
                    doc_head.1.as_ptr(),
                    query_head.1.len(),
                )
            },
            2 => unsafe {
                et_residual_dot_u2_u8(
                    query_head.0.as_ptr(),
                    query_head.1.as_ptr(),
                    doc_head.0.as_ptr(),
                    doc_head.1.as_ptr(),
                    query_head.1.len(),
                )
            },
            4 => unsafe {
                et_residual_dot_u4_u8(
                    query_head.0.as_ptr(),
                    query_head.1.as_ptr(),
                    doc_head.0.as_ptr(),
                    doc_head.1.as_ptr(),
                    query_head.1.len(),
                )
            },
            8 => unsafe {
                et_residual_dot_u8_u8(
                    query_head.0.as_ptr(),
                    query_head.1.as_ptr(),
                    doc_head.0.as_ptr(),
                    doc_head.1.as_ptr(),
                    query_head.1.len(),
                )
            },
            _ => scalar::residual_dot_unnormalized::<B>(query_head, doc_head),
        }
    } else {
        ResidualDotComponents::default()
    };

    if !query_tail.0.is_empty() {
        dot += scalar::residual_dot_unnormalized::<B>(query_tail, doc_tail);
    }

    dot
}

#[inline(always)]
unsafe fn shr_u32<const N: usize>(v: uint32x4_t) -> uint32x4_t {
    match N {
        1 => vshrq_n_u32::<1>(v),
        2 => vshrq_n_u32::<2>(v),
        4 => vshrq_n_u32::<4>(v),
        8 => vshrq_n_u32::<8>(v),
        _ => unreachable!(),
    }
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
        let scalar_stats = crate::lvq::scalar::compute_vector_stats(&vector);
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
        let scalar_stats = crate::lvq::scalar::compute_vector_stats(&vector);
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
