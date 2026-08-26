//! Scalar implementations of lvq routines suitable for use on any platform.

#![allow(dead_code)]

use super::{
    LAMBDA, MINIMUM_MSE_GRID, QuantizationStats, TurboPrimaryVector, VectorEncodeTerms,
    VectorStats,
    packing::{TurboPacker, TurboUnpacker},
};

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

pub fn primary_quantize_and_pack<const B: usize>(
    vector: &[f32],
    terms: VectorEncodeTerms,
    out: &mut [u8],
) -> QuantizationStats {
    let mut packer = TurboPacker::<B>::new(out);
    vector
        .iter()
        .map(|&v| {
            let q = ((v.clamp(terms.lower, terms.upper) - terms.lower) * terms.delta_inv).round();
            let r = v - q.mul_add(terms.delta, terms.lower);
            packer.push(q as u8);
            (q as u32, v, r)
        })
        .fold(QuantizationStats::default(), |stats, (q, v, r)| {
            stats.add_component(q, v, r)
        })
}

pub fn primary_decode<const B: usize>(vector: TurboPrimaryVector<'_, B>, out: &mut [f32]) {
    for (q, o) in TurboUnpacker::<B>::new(vector.rep.data).zip(out.iter_mut()) {
        *o = (q as f32).mul_add(vector.rep.terms.delta, vector.rep.terms.lower);
    }
}

#[inline]
pub fn dot_u8<const B: usize>(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .copied()
        .zip(b.iter().copied())
        .map(|(a, b)| match B {
            1 => (a & b).count_ones(),
            2 => {
                let a = (a & 0x3, (a >> 2) & 0x3, (a >> 4) & 0x3, a >> 6);
                let b = (b & 0x3, (b >> 2) & 0x3, (b >> 4) & 0x3, b >> 6);
                (a.0 * b.0 + a.1 * b.1 + a.2 * b.2 + a.3 * b.3).into()
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

/// Compute the unnormalized dot product of a bitplane-split 4-bit query with a 1-bit document.
///
/// `query` must be the output of `packing::bitplane_split4()`; `doc` is the packed 1-bit document
/// data covering the same number of dimensions (`query.len() * 2`).
#[inline]
pub fn query4_doc1_bitplane_dot(query: &[u8], doc: &[u8]) -> u32 {
    let (qhead, qtail) = query.as_chunks::<64>();
    let (dhead, dtail) = doc.split_at(qhead.len() * 16);
    let dhead = dhead.as_chunks::<16>().0;
    let mut pdot = [0u32; 4];
    for (q, d) in qhead.iter().zip(dhead.iter()) {
        let qc = q.as_chunks::<16>().0;
        let q0 = u128::from_le_bytes(qc[0]);
        let q1 = u128::from_le_bytes(qc[1]);
        let q2 = u128::from_le_bytes(qc[2]);
        let q3 = u128::from_le_bytes(qc[3]);
        let d = u128::from_le_bytes(*d);
        pdot[0] += (q0 & d).count_ones();
        pdot[1] += (q1 & d).count_ones();
        pdot[2] += (q2 & d).count_ones();
        pdot[3] += (q3 & d).count_ones();
    }

    if !qtail.is_empty() {
        let mut qit = qtail.chunks(qtail.len() / 4);
        let q = [
            qit.next().unwrap(),
            qit.next().unwrap(),
            qit.next().unwrap(),
            qit.next().unwrap(),
        ];

        for (i, &d) in dtail.iter().enumerate() {
            pdot[0] += (q[0][i] & d).count_ones();
            pdot[1] += (q[1][i] & d).count_ones();
            pdot[2] += (q[2][i] & d).count_ones();
            pdot[3] += (q[3][i] & d).count_ones();
        }
    }

    pdot[0] + pdot[1] * 2 + pdot[2] * 4 + pdot[3] * 8
}

#[inline]
pub fn primary_query8_dot_unnormalized<const B: usize>(
    query: &[u8],
    doc: &TurboPrimaryVector<'_, B>,
) -> u32 {
    query
        .iter()
        .zip(TurboUnpacker::<B>::new(doc.rep.data))
        .map(|(&q, d)| q as u32 * d as u32)
        .sum::<u32>()
}
