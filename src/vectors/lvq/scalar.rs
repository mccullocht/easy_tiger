//! Scalar implementations of lvq routines suitable for use on any platform.

#![allow(dead_code)]

use super::{VectorStats, LAMBDA, MINIMUM_MSE_GRID};

pub fn compute_vector_stats(vector: &[f32]) -> VectorStats {
    let (min, max, mean, variance, dot) = vector.iter().copied().enumerate().fold(
        (f32::MAX, f32::MIN, 0.0, 0.0, 0.0),
        |mut stats, (i, x)| {
            stats.0 = x.min(stats.0);
            stats.1 = x.max(stats.1);
            let x: f64 = x.into();
            let delta = x - stats.2;
            stats.2 += delta / (i + 1) as f64;
            stats.3 += delta * (x - stats.2);
            stats.4 += x * x;
            stats
        },
    );
    VectorStats {
        min,
        max,
        mean,
        std_dev: (variance / vector.len() as f64).sqrt(),
        l2_norm_sq: dot,
    }
}

pub fn optimize_interval(vector: &[f32], stats: &VectorStats, bits: usize) -> (f32, f32) {
    let norm_sq: f64 = stats.l2_norm_sq;
    let mut loss = compute_loss(vector, (stats.min, stats.max), norm_sq, bits);

    let scale = (1.0 - LAMBDA as f64) / norm_sq;
    let mut lower: f64 = (MINIMUM_MSE_GRID[bits - 1].0 as f64 * stats.std_dev + stats.mean)
        .clamp(stats.min.into(), stats.max.into());
    let mut upper: f64 = (MINIMUM_MSE_GRID[bits - 1].1 as f64 * stats.std_dev + stats.mean)
        .clamp(stats.min.into(), stats.max.into());

    let points_incl = ((1 << bits) - 1) as f64;
    for _ in 0..5 {
        let step_inv = points_incl / (upper - lower);
        // calculate the grid points for coordinate descent.
        let mut daa = 0.0;
        let mut dab = 0.0;
        let mut dbb = 0.0;
        let mut dax = 0.0;
        let mut dbx = 0.0;
        for xi in vector.iter().copied().map(f64::from) {
            let k = ((xi.clamp(lower, upper) - lower) * step_inv).round();
            let s = k / points_incl;
            daa += (1.0 - s) * (1.0 - s);
            dab += (1.0 - s) * s;
            dbb += s * s;
            dax += xi * (1.0 - s);
            dbx += xi * s;
        }
        let m0 = scale * dax * dax + LAMBDA as f64 * daa;
        let m1 = scale * dax * dbx + LAMBDA as f64 * dab;
        let m2 = scale * dbx * dbx + LAMBDA as f64 * dbb;
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
            (lower_candidate as f32, upper_candidate as f32),
            norm_sq,
            bits,
        );
        if loss_candidate > loss {
            break;
        }
        lower = lower_candidate;
        upper = upper_candidate;
        loss = loss_candidate;
    }
    (lower as f32, upper as f32)
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
