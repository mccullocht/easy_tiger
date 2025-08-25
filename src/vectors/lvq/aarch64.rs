//! aarch64 implementations of lvq routines.

use std::arch::aarch64::{
    float32x4_t, vaddq_f32, vaddq_f64, vaddvq_f32, vcvt_f64_f32, vcvt_high_f64_f32, vdivq_f32,
    vdupq_n_f32, vdupq_n_f64, vextq_f64, vfmaq_f32, vfmaq_f64, vget_low_f32, vgetq_lane_f64,
    vld1q_f32, vmaxq_f32, vmaxvq_f32, vminq_f32, vminvq_f32, vmulq_f32, vmulq_f64, vrndaq_f32,
    vsubq_f32, vsubq_f64,
};

use super::{VectorStats, LAMBDA};

// XXX this could compute both mean and variance
fn reduce_variance(means: float32x4_t, m2: float32x4_t, n: usize) -> f64 {
    let (means2, m2) = unsafe {
        // XXX switch to f32 here; widen once done.
        let means0 = vcvt_f64_f32(vget_low_f32(means));
        let means1 = vcvt_high_f64_f32(means);
        let var0 = vcvt_f64_f32(vget_low_f32(m2));
        let var1 = vcvt_high_f64_f32(m2);

        let mean = vmulq_f64(vaddq_f64(means0, means1), vdupq_n_f64(0.5));

        let delta = vsubq_f64(means0, means1);
        let delta_sq = vmulq_f64(delta, delta);
        let weight = vdupq_n_f64(((n / 4) * (n / 4) / (n / 2)) as f64);
        let m2 = vfmaq_f64(vaddq_f64(var0, var1), delta_sq, weight);

        (mean, m2)
    };

    unsafe {
        let delta = vsubq_f64(means2, vextq_f64::<1>(means2, means2));
        let delta_sq = vmulq_f64(delta, delta);
        let weight = vdupq_n_f64(((n / 2) * (n / 2) / n) as f64);
        let m2 = vfmaq_f64(vaddq_f64(m2, vextq_f64::<1>(m2, m2)), delta_sq, weight);
        vgetq_lane_f64::<0>(m2)
    }
}

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
                vaddvq_f32(mean) as f64 / 4.0,
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
            let x: f64 = x.into();
            let delta = x - stats.2;
            stats.2 += delta / (i + 1) as f64;
            stats.3 += delta * (x - stats.2);
            stats
        },
    );
    VectorStats {
        min,
        max,
        mean,
        std_dev: (mean_sq / vector.len() as f64).sqrt(),
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
}
