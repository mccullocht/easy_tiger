use std::arch::x86_64::{
    __m128i, _mm256_add_ps, _mm256_castps256_ps128, _mm256_cvtepu8_epi16, _mm256_extractf32x4_ps,
    _mm256_fmadd_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_sub_ps, _mm512_add_ps,
    _mm512_castps512_ps256, _mm512_cvtepu16_epi32, _mm512_cvtepu32_ps, _mm512_div_ps,
    _mm512_extractf32x8_ps, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_mask_mul_ps,
    _mm512_mask_sub_ps, _mm512_maskz_add_ps, _mm512_maskz_fmadd_ps, _mm512_maskz_loadu_ps,
    _mm512_maskz_mov_ps, _mm512_max_ps, _mm512_min_ps, _mm512_mul_ps, _mm512_reduce_add_ps,
    _mm512_reduce_max_ps, _mm512_reduce_min_ps, _mm512_roundscale_ps, _mm512_set1_ps,
    _mm512_sub_ps, _mm_add_ps, _mm_and_si128, _mm_cmpgt_epi8, _mm_cvtps_pd, _mm_cvtsd_f64,
    _mm_fmadd_pd, _mm_fmadd_ps, _mm_hadd_pd, _mm_hadd_ps, _mm_hsub_pd, _mm_hsub_ps, _mm_load_epi64,
    _mm_loadu_epi64, _mm_loadu_si128, _mm_maskz_loadu_epi8, _mm_movemask_epi8, _mm_mul_pd,
    _mm_mul_ps, _mm_set1_epi16, _mm_set1_epi64x, _mm_set1_epi8, _mm_set1_pd, _mm_set1_ps,
    _mm_shuffle_epi8, _mm_sub_ps, _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT,
};

use crate::vectors::lvq::PrimaryVector;

use super::{VectorStats, LAMBDA, MINIMUM_MSE_GRID};

#[target_feature(enable = "avx512f,avx,fma")]
pub unsafe fn compute_vector_stats_avx512(vector: &[f32]) -> VectorStats {
    let mut minv = _mm512_set1_ps(f32::MAX);
    let mut maxv = _mm512_set1_ps(f32::MIN);
    let mut meanv = _mm512_set1_ps(0.0);
    let mut variancev = _mm512_set1_ps(0.0);
    let mut dotv = _mm512_set1_ps(0.0);
    let (head, tail) = vector.as_chunks::<16>();
    for (i, c) in head.iter().enumerate() {
        let x = _mm512_loadu_ps(c.as_ptr());
        minv = _mm512_min_ps(minv, x);
        maxv = _mm512_max_ps(maxv, x);
        let delta = _mm512_sub_ps(x, meanv);
        meanv = _mm512_add_ps(meanv, _mm512_div_ps(delta, _mm512_set1_ps((i + 1) as f32)));
        let delta2 = _mm512_sub_ps(x, meanv);
        variancev = _mm512_fmadd_ps(delta, delta2, variancev);
        dotv = _mm512_fmadd_ps(x, x, dotv);
    }

    let base = head.len() * 16;
    let mut min = _mm512_reduce_min_ps(minv);
    let mut max = _mm512_reduce_max_ps(maxv);
    let mut mean = _mm512_reduce_add_ps(meanv) / 16.0;
    let mut variance = {
        let (means_8, m2_8) = {
            let means = (
                _mm512_castps512_ps256(meanv),
                _mm512_extractf32x8_ps(meanv, 1),
            );
            let m2s = (
                _mm512_castps512_ps256(variancev),
                _mm512_extractf32x8_ps(variancev, 1),
            );

            let mean = _mm256_mul_ps(_mm256_add_ps(means.0, means.1), _mm256_set1_ps(0.5));
            let delta = _mm256_sub_ps(means.0, means.1);
            let delta_sq = _mm256_mul_ps(delta, delta);
            let m2 = _mm256_fmadd_ps(
                delta_sq,
                _mm256_set1_ps(base as f32 / 16.0),
                _mm256_add_ps(m2s.0, m2s.1),
            );

            (mean, m2)
        };

        let (means_4, m2_4) = {
            let means = (
                _mm256_castps256_ps128(means_8),
                _mm256_extractf32x4_ps(means_8, 1),
            );
            let m2s = (
                _mm256_castps256_ps128(m2_8),
                _mm256_extractf32x4_ps(m2_8, 1),
            );

            let mean = _mm_mul_ps(_mm_add_ps(means.0, means.1), _mm_set1_ps(0.5));
            let delta = _mm_sub_ps(means.0, means.1);
            let delta_sq = _mm_mul_ps(delta, delta);
            let m2 = _mm_fmadd_ps(
                delta_sq,
                _mm_set1_ps(base as f32 / 8.0),
                _mm_add_ps(m2s.0, m2s.1),
            );

            (mean, m2)
        };

        let (means_2, m2_2) = {
            let mean = _mm_mul_ps(_mm_hadd_ps(means_4, means_4), _mm_set1_ps(0.5));
            let delta = _mm_hsub_ps(mean, mean);
            let delta_sq = _mm_mul_ps(delta, delta);
            let m2 = _mm_fmadd_ps(
                delta_sq,
                _mm_set1_ps(base as f32 / 8.0),
                _mm_hadd_ps(m2_4, m2_4),
            );

            (_mm_cvtps_pd(mean), _mm_cvtps_pd(m2))
        };

        let delta = _mm_hsub_pd(means_2, means_2);
        let delta_sq = _mm_mul_pd(delta, delta);
        _mm_cvtsd_f64(_mm_fmadd_pd(
            delta_sq,
            _mm_set1_pd(base as f64 / 4.0),
            _mm_hadd_pd(m2_2, m2_2),
        )) as f32
    };
    let mut dot = _mm512_reduce_add_ps(dotv);

    for (i, x) in tail.iter().enumerate() {
        min = x.min(min);
        max = x.max(max);
        let delta = *x - mean;
        mean += delta / (base + i + 1) as f32;
        variance += delta * (x - mean);
        dot += x * x;
    }

    VectorStats {
        min,
        max,
        mean,
        std_dev: (variance / vector.len() as f32).sqrt(),
        l2_norm_sq: dot,
    }
}

#[target_feature(enable = "avx512f")]
pub unsafe fn optimize_interval_avx512(
    vector: &[f32],
    stats: &VectorStats,
    bits: usize,
) -> (f32, f32) {
    let norm_sq = stats.l2_norm_sq;
    let mut loss = unsafe { compute_loss(vector, (stats.min, stats.max), norm_sq.into(), bits) };

    let scale = (1.0 - LAMBDA) / norm_sq;
    let mut lower =
        (MINIMUM_MSE_GRID[bits - 1].0 * stats.std_dev + stats.mean).clamp(stats.min, stats.max);
    let mut upper =
        (MINIMUM_MSE_GRID[bits - 1].1 * stats.std_dev + stats.mean).clamp(stats.min, stats.max);

    let points_incl = ((1 << bits) - 1) as f32;
    for _ in 0..5 {
        let lowerv = _mm512_set1_ps(lower);
        let upperv = _mm512_set1_ps(upper);
        let step_inv = _mm512_set1_ps(points_incl / (upper - lower));
        let points_incl_inv = _mm512_set1_ps(1.0 / points_incl);
        // calculate the grid points for coordinate descent.
        let mut daav = _mm512_set1_ps(0.0);
        let mut dabv = _mm512_set1_ps(0.0);
        let mut dbbv = _mm512_set1_ps(0.0);
        let mut daxv = _mm512_set1_ps(0.0);
        let mut dbxv = _mm512_set1_ps(0.0);
        for c in vector.chunks(16) {
            let mask = u16::MAX >> (16 - c.len());
            let xv = _mm512_maskz_loadu_ps(mask, c.as_ptr());
            let mut xq = _mm512_max_ps(xv, lowerv);
            xq = _mm512_min_ps(xq, upperv);
            xq = _mm512_sub_ps(xq, lowerv);
            xq = _mm512_mul_ps(xq, step_inv);
            xq = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(xq);
            let s = _mm512_mask_mul_ps(_mm512_set1_ps(0.0), mask, xq, points_incl_inv);
            let s1 = _mm512_mask_sub_ps(_mm512_set1_ps(0.0), mask, _mm512_set1_ps(1.0), s);
            daav = _mm512_fmadd_ps(s1, s1, daav);
            dabv = _mm512_fmadd_ps(s1, s, dabv);
            dbbv = _mm512_fmadd_ps(s, s, dbbv);
            daxv = _mm512_fmadd_ps(xv, s1, daxv);
            dbxv = _mm512_fmadd_ps(xv, s, dbxv);
        }

        let daa = _mm512_reduce_add_ps(daav);
        let dab = _mm512_reduce_add_ps(dabv);
        let dbb = _mm512_reduce_add_ps(dbbv);
        let dax = _mm512_reduce_add_ps(daxv);
        let dbx = _mm512_reduce_add_ps(dbxv);

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
        let loss_candidate = unsafe {
            compute_loss(
                vector,
                (lower_candidate, upper_candidate),
                norm_sq.into(),
                bits,
            )
        };
        if loss_candidate > loss {
            break;
        }
        lower = lower_candidate;
        upper = upper_candidate;
        loss = loss_candidate;
    }
    (lower, upper)
}

#[target_feature(enable = "avx512f")]
unsafe fn compute_loss(vector: &[f32], interval: (f32, f32), norm_sq: f64, bits: usize) -> f64 {
    let a: f64 = interval.0.into();
    let b: f64 = interval.1.into();
    let step = (b - a) / ((1 << bits) - 1) as f64;
    let step_inv = step.recip();

    let av = _mm512_set1_ps(a as f32);
    let bv = _mm512_set1_ps(b as f32);
    let stepv = _mm512_set1_ps(step as f32);
    let step_invv = _mm512_set1_ps(step_inv as f32);
    let mut xev = _mm512_set1_ps(0.0);
    let mut ev = _mm512_set1_ps(0.0);
    for c in vector.chunks(16) {
        let mask = u16::MAX >> (16 - c.len());
        let xi = _mm512_maskz_loadu_ps(mask, c.as_ptr());
        let mut xiq = _mm512_max_ps(xi, av);
        xiq = _mm512_min_ps(xiq, bv);
        xiq = _mm512_sub_ps(xiq, av);
        xiq = _mm512_mul_ps(xiq, step_invv);
        xiq = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(xiq);
        xiq = _mm512_fmadd_ps(stepv, xiq, av);
        let diff = _mm512_mask_sub_ps(_mm512_set1_ps(0.0), mask, xi, xiq);
        xev = _mm512_fmadd_ps(xi, diff, xev);
        ev = _mm512_fmadd_ps(diff, diff, ev);
    }

    let xe = _mm512_reduce_add_ps(xev) as f64;
    let e = _mm512_reduce_add_ps(ev) as f64;
    (1.0 - LAMBDA as f64) * xe * xe / norm_sq + LAMBDA as f64 * e
}

#[target_feature(enable = "avx512vnni,avx512bw,avx512vl,avx512vpopcntdq,avx512f")]
pub fn dot_u8<const B: usize>(a: &[u8], b: &[u8]) -> u32 {
    super::scalar::dot_u8::<B>(a, b)
}

#[target_feature(enable = "avx512vnni,avx512bw,avx512vl,avx512f")]
pub unsafe fn lvq1_f32_dot_unnormalized<const B: usize>(
    query: &[f32],
    doc: &PrimaryVector<'_, B>,
) -> f64 {
    match B {
        1 => {
            let delta = _mm512_set1_ps(doc.delta);
            let lower = _mm512_set1_ps(doc.header.lower);
            let mut dot = _mm512_set1_ps(0.0);
            for (q, d) in query.chunks(16).zip(doc.vector.chunks(2)) {
                let mask = u16::MAX >> (16 - q.len());
                let qv = _mm512_maskz_loadu_ps(mask, q.as_ptr());
                // Load 16 bits and shuffle such that the first byte occupies the low part of the
                // register and the second byte occupies the high part of the register.
                let rd = if d.len() == 2 {
                    u16::from_le_bytes(d.try_into().unwrap())
                } else {
                    d[0] as u16
                };
                let mut dqv = _mm_shuffle_epi8(
                    _mm_set1_epi16(rd as i16),
                    _mm_load_epi64([0, 0x0101010101010101].as_ptr()),
                );
                // Mask each byte down to the representative bit and compare to zero.
                dqv = _mm_and_si128(dqv, _mm_set1_epi64x(0x0807060504030201));
                dqv = _mm_cmpgt_epi8(dqv, _mm_set1_epi8(0));
                let dmask = _mm_movemask_epi8(dqv) as u16;
                // Mask out delta, then combine it with lower to generate to doc values.
                let dv = _mm512_maskz_add_ps(mask, _mm512_maskz_mov_ps(dmask, delta), lower);
                dot = _mm512_fmadd_ps(qv, dv, dot);
            }
            _mm512_reduce_add_ps(dot).into()
        }
        4 => {
            let delta = _mm512_set1_ps(doc.delta);
            let lower = _mm512_set1_ps(doc.header.lower);
            let mut dot = _mm512_set1_ps(0.0);
            for (q, d) in query.chunks(16).zip(doc.vector.chunks(8)) {
                let mask = u16::MAX >> (16 - q.len());
                let qv = _mm512_maskz_loadu_ps(mask, q.as_ptr());
                // Load 64 bits and shuffle so that all the low nibbles appear in the low lanes and
                // all the upper nibbles appear in the high lanes.
                let rd = if d.len() == 8 {
                    u64::from_le_bytes(d.try_into().unwrap())
                } else {
                    let mut buf = [0u8; 8];
                    buf[..d.len()].copy_from_slice(d);
                    u64::from_le_bytes(buf)
                } as i64;
                let dqv = _mm_and_si128(
                    _mm_shuffle_epi8(
                        _mm_loadu_epi64([rd, rd >> 4].as_ptr()),
                        _mm_loadu_si128(
                            [0u8, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15].as_ptr()
                                as *const __m128i,
                        ),
                    ),
                    _mm_set1_epi8(0xf),
                );
                let dqvf = _mm512_cvtepu32_ps(_mm512_cvtepu16_epi32(_mm256_cvtepu8_epi16(dqv)));
                let dv = _mm512_maskz_fmadd_ps(mask, dqvf, delta, lower);
                dot = _mm512_fmadd_ps(qv, dv, dot);
            }
            _mm512_reduce_add_ps(dot).into()
        }
        8 => {
            let delta = _mm512_set1_ps(doc.delta);
            let lower = _mm512_set1_ps(doc.header.lower);
            let mut dot = _mm512_set1_ps(0.0);
            for (q, d) in query.chunks(16).zip(doc.vector.chunks(16)) {
                let mask = u16::MAX >> (16 - q.len());
                let qv = _mm512_maskz_loadu_ps(mask, q.as_ptr());
                let dqv = _mm512_cvtepu32_ps(_mm512_cvtepu16_epi32(_mm256_cvtepu8_epi16(
                    _mm_maskz_loadu_epi8(mask, d.as_ptr() as *const i8),
                )));
                let dv = _mm512_maskz_fmadd_ps(mask, dqv, delta, lower);
                dot = _mm512_fmadd_ps(qv, dv, dot);
            }
            _mm512_reduce_add_ps(dot).into()
        }
        _ => super::scalar::lvq1_f32_dot_unnormalized::<B>(query, doc),
    }
}
