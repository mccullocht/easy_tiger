//! Scalar implementations of lvq routines suitable for use on any platform.

#![allow(dead_code)]

use super::{LAMBDA, MINIMUM_MSE_GRID, PrimaryVector, TwoLevelVector, VectorStats};

pub fn compute_vector_stats(vector: &[f32]) -> VectorStats {
    let (min, max, mean, variance, dot) = vector.iter().copied().enumerate().fold(
        (f32::MAX, f32::MIN, 0.0, 0.0, 0.0),
        |mut stats, (i, x)| {
            stats.0 = x.min(stats.0);
            stats.1 = x.max(stats.1);
            let delta = x - stats.2;
            stats.2 += delta / (i + 1) as f32;
            stats.3 += delta * (x - stats.2);
            stats.4 += x * x;
            stats
        },
    );
    VectorStats {
        min,
        max,
        mean,
        std_dev: (variance / vector.len() as f32).sqrt(),
        l2_norm_sq: dot,
    }
}

pub fn optimize_interval_scalar(vector: &[f32], stats: &VectorStats, bits: usize) -> (f32, f32) {
    let norm_sq = stats.l2_norm_sq;
    let mut loss = compute_loss(vector, (stats.min, stats.max), norm_sq.into(), bits);

    let scale = (1.0 - LAMBDA) / norm_sq;
    let mut lower =
        (MINIMUM_MSE_GRID[bits - 1].0 * stats.std_dev + stats.mean).clamp(stats.min, stats.max);
    let mut upper =
        (MINIMUM_MSE_GRID[bits - 1].1 * stats.std_dev + stats.mean).clamp(stats.min, stats.max);

    let points_incl = ((1 << bits) - 1) as f32;
    for _ in 0..5 {
        let step_inv = points_incl / (upper - lower);
        // calculate the grid points for coordinate descent.
        let mut daa = 0.0;
        let mut dab = 0.0;
        let mut dbb = 0.0;
        let mut dax = 0.0;
        let mut dbx = 0.0;
        for xi in vector.iter().copied() {
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
        let loss_candidate = compute_loss(
            vector,
            (lower_candidate, upper_candidate),
            norm_sq.into(),
            bits,
        );
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
    let a: f64 = interval.0.into();
    let b: f64 = interval.1.into();
    let step = (b - a) / ((1 << bits) - 1) as f64;
    let step_inv = step.recip();
    let mut xe = 0.0;
    let mut e = 0.0;
    for xi in vector.iter().copied().map(f64::from) {
        let xiq = a + step * ((xi.clamp(a, b) - a) * step_inv).round();
        let diff = xi - xiq;
        xe += xi * diff;
        e += diff * diff;
    }
    (1.0 - LAMBDA as f64) * xe * xe / norm_sq + LAMBDA as f64 * e
}

pub fn lvq1_quantize_and_pack<const B: usize>(
    v: &[f32],
    lower: f32,
    upper: f32,
    out: &mut [u8],
) -> u32 {
    let delta_inv = ((1 << B) - 1) as f32 / (upper - lower);
    let mut component_sum = 0u32;
    super::packing::pack_iter::<B>(
        v.iter().copied().map(|x| {
            let q = ((x.clamp(lower, upper) - lower) * delta_inv).round() as u8;
            component_sum += u32::from(q);
            q
        }),
        out,
    );
    component_sum
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
    let delta_inv = ((1 << B1) - 1) as f32 / (upper - lower);
    let res_lower = -residual_interval / 2.0;
    let res_upper = residual_interval / 2.0;
    let res_delta_inv = ((1 << B2) - 1) as f32 / residual_interval;
    let mut component_sum = 0u32;
    super::packing::pack_iter2::<B1, B2>(
        v.iter().copied().map(|x| {
            let q = ((x.clamp(lower, upper) - lower) * delta_inv).round() as u8;
            component_sum += u32::from(q);
            // After producing the primary value, calculate the error produced and quantize that
            // value based on the delta between primary items.
            let res = x - (q as f32).mul_add(delta, lower);
            let r = ((res.clamp(res_lower, res_upper) - res_lower) * res_delta_inv).round() as u16;
            (q, r)
        }),
        primary,
        residual,
    );
    component_sum
}

#[inline]
pub fn dot_u8<const B: usize>(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .copied()
        .zip(b.iter().copied())
        .map(|(a, b)| match B {
            1 => (a & b).count_ones(),
            2 => {
                let a = [a & 0x3, (a >> 2) & 0x3, (a >> 4) & 0x3, (a >> 6) & 0x3];
                let b = [b & 0x3, (b >> 2) & 0x3, (b >> 4) & 0x3, (b >> 6) & 0x3];
                (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]).into()
            }
            4 => {
                let a = [a & 0xf, a >> 4];
                let b = [b & 0xf, b >> 4];
                (a[0] as u16 * b[0] as u16 + a[1] as u16 * b[1] as u16).into()
            }
            8 => a as u32 * b as u32,
            _ => unimplemented!(),
        })
        .sum::<u32>()
}

#[inline]
pub fn lvq1_f32_dot_unnormalized<const B: usize>(query: &[f32], doc: &PrimaryVector<'_, B>) -> f64 {
    query
        .iter()
        .zip(
            super::packing::unpack_iter::<B>(doc.vector)
                .map(|q| q as f32 * doc.delta + doc.header.lower),
        )
        .map(|(q, d)| *q * d)
        .sum::<f32>()
        .into()
}

#[inline]
pub fn lvq2_dot_unnormalized<const B1: usize, const B2: usize>(
    a: &TwoLevelVector<'_, B1, B2>,
    b: &TwoLevelVector<'_, B1, B2>,
) -> f64 {
    a.f32_iter()
        .zip(b.f32_iter())
        .map(|(a, b)| a * b)
        .sum::<f32>()
        .into()
}

#[inline]
pub fn lvq2_f32_dot_unnormalized<const B1: usize, const B2: usize>(
    query: &[f32],
    doc: &TwoLevelVector<'_, B1, B2>,
) -> f64 {
    query
        .iter()
        .zip(doc.f32_iter())
        .map(|(q, d)| *q * d)
        .sum::<f32>()
        .into()
}
