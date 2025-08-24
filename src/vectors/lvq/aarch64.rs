//! aarch64 implementations of lvq routines.

use std::arch::aarch64::{
    float32x4_t, vaddq_f32, vaddq_f64, vaddvq_f32, vcvt_f64_f32, vcvt_high_f64_f32, vdivq_f32,
    vdivq_f64, vdupq_n_f32, vdupq_n_f64, vextq_f64, vfmaq_f32, vfmaq_f64, vget_low_f32,
    vgetq_lane_f64, vld1q_f32, vmaxq_f32, vmaxvq_f32, vminq_f32, vminvq_f32, vmulq_f32, vmulq_f64,
    vrndaq_f32, vsubq_f32, vsubq_f64,
};

use super::{VectorStats, LAMBDA};

fn reduce_variance(means: float32x4_t, vars: float32x4_t, n: usize) -> f64 {
    let (means2, vars2) = unsafe {
        let means0 = vcvt_f64_f32(vget_low_f32(means));
        let means1 = vcvt_high_f64_f32(means);
        let var0 = vcvt_f64_f32(vget_low_f32(vars));
        let var1 = vcvt_high_f64_f32(vars);

        let mean = vmulq_f64(vaddq_f64(means0, means1), vdupq_n_f64(0.5));

        let delta = vsubq_f64(means0, means1);
        let delta_sq = vmulq_f64(delta, delta);
        // XXX this is just 2 in eery case.
        let weight = vdupq_n_f64(((n / 4) * (n / 4) / (n / 2)) as f64);

        let m2 = vfmaq_f64(vaddq_f64(var0, var1), delta_sq, weight);
        let var = vdivq_f64(m2, vdupq_n_f64((n / 2) as f64));

        (mean, var)
    };

    unsafe {
        let delta = vsubq_f64(means2, vextq_f64::<1>(means2, means2));
        let delta_sq = vmulq_f64(delta, delta);
        let weight = vdupq_n_f64(((n / 2) * (n / 2) / n) as f64);
        let m2 = vfmaq_f64(
            vaddq_f64(vars2, vextq_f64::<1>(vars2, vars2)),
            delta_sq,
            weight,
        );
        vgetq_lane_f64::<0>(m2) / n as f64
    }
}

pub fn compute_vector_stats(vector: &[f32]) -> VectorStats {
    let tail_split = vector.len() & !3;
    let (min, max, mean, variance, dot) = if tail_split > 0 {
        unsafe {
            let mut min = vdupq_n_f32(f32::MAX);
            let mut max = vdupq_n_f32(f32::MIN);
            let mut mean = vdupq_n_f32(0.0);
            let mut variance = vdupq_n_f32(0.0);
            let mut dot = vdupq_n_f32(0.0);
            for i in (0..tail_split).step_by(4) {
                let x = vld1q_f32(vector.as_ptr().add(i));
                min = vminq_f32(min, x);
                max = vmaxq_f32(max, x);
                let delta = vsubq_f32(x, mean);
                mean = vaddq_f32(mean, vdivq_f32(delta, vdupq_n_f32(((i / 4) + 1) as f32)));
                let delta2 = vsubq_f32(x, mean);
                variance = vfmaq_f32(variance, delta, delta2);
                dot = vfmaq_f32(dot, x, x);
            }

            (
                vminvq_f32(min),
                vmaxvq_f32(max),
                vaddvq_f32(mean) as f64 / 4.0,
                reduce_variance(mean, variance, tail_split),
                vaddvq_f32(dot),
            )
        }
    } else {
        (0.0, 0.0, 0.0, 0.0, 0.0)
    };
    // XXX must finish tail
    /*
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
    */
    VectorStats {
        min,
        max,
        mean,
        std_dev: (variance / vector.len() as f64).sqrt(),
        l2_norm_sq: dot.into(),
    }
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
    (1.0 - LAMBDA) * xe as f64 * xe as f64 / norm_sq + LAMBDA * e as f64
}
