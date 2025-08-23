//! Scalar implementations of lvq routines suitable for use on any platform.

#![allow(dead_code)]

use super::LAMBDA;

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
    (1.0 - LAMBDA) * xe * xe / norm_sq + LAMBDA * e
}
