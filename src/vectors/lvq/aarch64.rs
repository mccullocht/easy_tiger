//! aarch64 implementations of lvq routines.

use std::arch::aarch64::{
    float32x4_t, vaddq_f32, vaddq_f64, vaddvq_f32, vcvt_f64_f32, vcvt_high_f64_f32, vdivq_f32,
    vdupq_n_f32, vdupq_n_f64, vextq_f64, vfmaq_f32, vfmaq_f64, vget_low_f32, vgetq_lane_f64,
    vld1q_f32, vmaxq_f32, vmaxvq_f32, vminq_f32, vminvq_f32, vmulq_f32, vmulq_f64, vrndaq_f32,
    vsubq_f32, vsubq_f64,
};

use super::{VectorStats, LAMBDA, MINIMUM_MSE_GRID};

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
