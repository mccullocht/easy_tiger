#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::{
    __m128i, __m512, __m512i, _MM_FROUND_NO_EXC, _MM_FROUND_TRUNC, _mm_add_pd, _mm_add_ps,
    _mm_and_si128, _mm_andnot_si128, _mm_bsrli_si128, _mm_cmpeq_epi8, _mm_cvtps_pd, _mm_cvtsd_f64,
    _mm_fmadd_pd, _mm_fmadd_ps, _mm_hadd_pd, _mm_hsub_pd, _mm_lddqu_si128, _mm_loadu_epi8,
    _mm_loadu_epi64, _mm_mask_storeu_epi8, _mm_maskz_loadu_epi8, _mm_movehl_ps, _mm_mul_pd,
    _mm_mul_ps, _mm_or_si128, _mm_set1_epi8, _mm_set1_epi16, _mm_set1_epi64x, _mm_set1_pd,
    _mm_set1_ps, _mm_shuffle_epi8, _mm_sllv_epi64, _mm_srli_epi64, _mm_storeu_si128, _mm_sub_pd,
    _mm_sub_ps, _mm_unpacklo_epi8, _mm256_add_ps, _mm256_castps256_ps128, _mm256_cvtepi8_epi16,
    _mm256_cvtepi16_epi8, _mm256_cvtepu8_epi16, _mm256_extractf32x4_ps, _mm256_fmadd_ps,
    _mm256_loadu_epi16, _mm256_maskz_loadu_epi8, _mm256_mul_ps, _mm256_set1_ps, _mm256_sllv_epi16,
    _mm256_sub_ps, _mm512_add_epi32, _mm512_add_ps, _mm512_and_epi32, _mm512_and_si512,
    _mm512_broadcast_i32x4, _mm512_castps512_ps256, _mm512_cmpgt_epu8_mask, _mm512_cvtepi16_epi32,
    _mm512_cvtepi32_epi8, _mm512_cvtepi32_epi16, _mm512_cvtepu8_epi16, _mm512_cvtepu8_epi32,
    _mm512_cvtepu32_ps, _mm512_cvtps_epu32, _mm512_div_ps, _mm512_dpbusd_epi32,
    _mm512_dpwssd_epi32, _mm512_extractf32x8_ps, _mm512_fmadd_ps, _mm512_loadu_epi8,
    _mm512_loadu_ps, _mm512_mask_mul_ps, _mm512_mask_storeu_ps, _mm512_mask_sub_ps,
    _mm512_maskz_cvtps_epu32, _mm512_maskz_expand_epi64, _mm512_maskz_loadu_epi8,
    _mm512_maskz_loadu_epi64, _mm512_maskz_loadu_ps, _mm512_max_ps, _mm512_min_ps,
    _mm512_movm_epi8, _mm512_mul_ps, _mm512_or_si512, _mm512_permutexvar_epi8, _mm512_popcnt_epi32,
    _mm512_reduce_add_epi32, _mm512_reduce_add_ps, _mm512_reduce_max_ps, _mm512_reduce_min_ps,
    _mm512_roundscale_ps, _mm512_set_epi64, _mm512_set1_epi8, _mm512_set1_epi32, _mm512_set1_epi64,
    _mm512_set1_ps, _mm512_setzero_si512, _mm512_shuffle_i64x2, _mm512_sll_epi32, _mm512_sll_epi64,
    _mm512_srli_epi32, _mm512_srli_epi64, _mm512_srlv_epi64, _mm512_storeu_ps, _mm512_sub_ps,
    _mm512_unpackhi_epi8, _mm512_unpacklo_epi8,
};

use crate::lvq::{
    TURBO_BLOCK_SIZE, TurboPrimaryVector, TurboResidualVector, VectorDecodeTerms,
    VectorEncodeTerms, packing,
};

use super::{
    LAMBDA, MINIMUM_MSE_GRID, PrimaryVector, ResidualDotComponents, TwoLevelVector, VectorStats,
};

/// For an input vector `v` where all values are non-negative, round each value with ties (e.g. 0.5)
/// rounding away from zero.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn mm512_round_nonnegative_ties_away_zero_ps(v: __m512) -> __m512 {
    _mm512_roundscale_ps(
        _mm512_add_ps(v, _mm512_set1_ps(0.5)),
        _MM_FROUND_TRUNC | _MM_FROUND_NO_EXC,
    )
}

#[target_feature(enable = "avx512f,avx,fma,avx512dq")]
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
                _mm256_set1_ps(base as f32 / 32.0),
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
                _mm_set1_ps(base as f32 / 16.0),
                _mm_add_ps(m2s.0, m2s.1),
            );

            (mean, m2)
        };

        let (means_2, m2_2) = {
            let means0 = _mm_cvtps_pd(means_4);
            let means1 = _mm_cvtps_pd(_mm_movehl_ps(means_4, means_4));
            let var0 = _mm_cvtps_pd(m2_4);
            let var1 = _mm_cvtps_pd(_mm_movehl_ps(m2_4, m2_4));

            let mean = _mm_mul_pd(_mm_add_pd(means0, means1), _mm_set1_pd(0.5));
            let delta = _mm_sub_pd(means0, means1);
            let delta_sq = _mm_mul_pd(delta, delta);
            let m2 = _mm_fmadd_pd(
                delta_sq,
                _mm_set1_pd(base as f64 / 8.0),
                _mm_add_pd(var0, var1),
            );

            (mean, m2)
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
            xq = mm512_round_nonnegative_ties_away_zero_ps(xq);
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

#[target_feature(enable = "avx512f,avx512vbmi2")]
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
        xiq = mm512_round_nonnegative_ties_away_zero_ps(xiq);
        xiq = _mm512_fmadd_ps(stepv, xiq, av);
        let diff = _mm512_mask_sub_ps(_mm512_set1_ps(0.0), mask, xi, xiq);
        xev = _mm512_fmadd_ps(xi, diff, xev);
        ev = _mm512_fmadd_ps(diff, diff, ev);
    }

    let xe = _mm512_reduce_add_ps(xev) as f64;
    let e = _mm512_reduce_add_ps(ev) as f64;
    (1.0 - LAMBDA as f64) * xe * xe / norm_sq + LAMBDA as f64 * e
}

#[target_feature(enable = "avx512f")]
pub unsafe fn primary_quantize_and_pack_avx512<const B: usize>(
    vector: &[f32],
    terms: VectorEncodeTerms,
    out: &mut [u8],
) -> u32 {
    let tail_split = vector.len() & !(packing::block_dim(B) - 1);
    let (in_head, in_tail) = vector.split_at(tail_split);
    let (out_head, out_tail) = out.split_at_mut(packing::byte_len(tail_split, B));

    let mut component_sum = if !in_head.is_empty() {
        // XXX use terms and common quantization routine.
        let lower = _mm512_set1_ps(terms.lower);
        let upper = _mm512_set1_ps(terms.upper);
        let delta_inv = _mm512_set1_ps(terms.delta_inv);
        let mut qbuf = _mm512_set1_epi32(0);
        let mut component_sum = _mm512_set1_epi32(0);
        let mut shift = 0;
        for i in (0..tail_split).step_by(16) {
            let mut v = _mm512_loadu_ps(in_head.as_ptr().add(i));
            v = _mm512_min_ps(v, upper);
            v = _mm512_max_ps(v, lower);
            v = _mm512_sub_ps(v, lower);
            v = _mm512_mul_ps(v, delta_inv);
            let q = _mm512_cvtps_epu32(mm512_round_nonnegative_ties_away_zero_ps(v));
            component_sum = _mm512_add_epi32(component_sum, q);
            qbuf = _mm512_or_si512(qbuf, _mm512_sll_epi32(q, _mm_set1_epi64x(shift as i64)));
            shift += B;
            if shift == 8 {
                _mm_storeu_si128(
                    out_head.as_mut_ptr().add(i / packing::block_dim(B) * 16) as *mut __m128i,
                    _mm512_cvtepi32_epi8(qbuf),
                );
                qbuf = _mm512_set1_epi32(0);
                shift = 0;
            }
        }

        // tail_split should be a multiple of the number of dimensions per block.
        assert_eq!(shift, 0);

        _mm512_reduce_add_epi32(component_sum) as u32
    } else {
        0
    };

    if !in_tail.is_empty() {
        component_sum += super::scalar::primary_quantize_and_pack::<B>(in_tail, terms, out_tail);
    }

    component_sum
}

#[target_feature(enable = "avx512f,avx512bw,avx2")]
pub unsafe fn primary_decode_avx512<const B: usize>(
    vector: TurboPrimaryVector<'_, B>,
    out: &mut [f32],
) {
    let (tail_split, in_head, in_tail) = vector.split_tail(out.len());
    let (out_head, out_tail) = out.split_at_mut(tail_split);

    if !in_head.rep.data.is_empty() {
        unsafe {
            let load_interval = TURBO_BLOCK_SIZE * 8 / B;
            let lower = _mm512_set1_ps(vector.rep.terms.lower);
            let delta = _mm512_set1_ps(vector.rep.terms.delta);
            let mask = _mm512_set1_epi32(i32::from(u8::MAX >> (8 - B)));
            let mut d = _mm512_set1_epi32(0);
            for i in (0..tail_split).step_by(16) {
                d = if i.is_multiple_of(load_interval) {
                    // Load 16 bytes but expand such that each byte is LSB of a 32-bit integer
                    _mm512_cvtepi16_epi32(_mm256_cvtepu8_epi16(_mm_lddqu_si128(
                        in_head
                            .rep
                            .data
                            .as_ptr()
                            .add(i / load_interval * TURBO_BLOCK_SIZE)
                            as *const __m128i,
                    )))
                } else {
                    match B {
                        1 => _mm512_srli_epi32::<1>(d),
                        2 => _mm512_srli_epi32::<2>(d),
                        4 => _mm512_srli_epi32::<4>(d),
                        // 8 will load on every iteration of the loop.
                        _ => unreachable!(),
                    }
                };
                _mm512_storeu_ps(
                    out_head.as_mut_ptr().add(i),
                    _mm512_fmadd_ps(_mm512_cvtepu32_ps(_mm512_and_si512(d, mask)), delta, lower),
                );
            }
        }
    }

    if !in_tail.rep.data.is_empty() {
        super::scalar::primary_decode::<B>(in_tail, out_tail);
    }
}

#[target_feature(enable = "avx512f")]
pub unsafe fn residual_quantize_and_pack_avx512<const B: usize>(
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

    let (mut primary_component_sum, mut residual_component_sum) = if !vector_head.is_empty() {
        let primary_terms = VectorEncodeTermsAvx512::from_terms(&primary_terms);
        let residual_terms = VectorEncodeTermsAvx512::from_terms(&residual_terms);
        let primary_delta = _mm512_set1_ps(primary_delta);
        let mut primary_component_sum = _mm512_set1_epi32(0);
        let mut residual_component_sum = _mm512_set1_epi32(0);
        let mut pbuf = _mm512_set1_epi32(0);
        let mut block = 0usize;
        let mut shift = 0i64;
        for i in (0..tail_split).step_by(16) {
            let v = _mm512_loadu_ps(vector_head.as_ptr().add(i));
            let p = primary_terms.quantize(v);
            let dq = _mm512_fmadd_ps(primary_delta, _mm512_cvtepu32_ps(p), primary_terms.lower);
            let r = residual_terms.quantize(_mm512_sub_ps(v, dq));

            primary_component_sum = _mm512_add_epi32(primary_component_sum, p);
            residual_component_sum = _mm512_add_epi32(residual_component_sum, r);

            pbuf = _mm512_or_si512(pbuf, _mm512_sll_epi64(p, _mm_set1_epi64x(shift)));
            shift += B as i64;
            if shift == 8 {
                _mm_storeu_si128(
                    primary_out_head.as_mut_ptr().add(block * 16) as *mut __m128i,
                    _mm512_cvtepi32_epi8(pbuf),
                );
                pbuf = _mm512_set1_epi32(0);
                shift = 0;
                block += 1;
            }

            _mm_storeu_si128(
                residual_out_head.as_mut_ptr().add(i) as *mut __m128i,
                _mm512_cvtepi32_epi8(r),
            )
        }

        // tail_split should be a multiple of the number of dimensions per block.
        assert_eq!(shift, 0);

        (
            _mm512_reduce_add_epi32(primary_component_sum) as u32,
            _mm512_reduce_add_epi32(residual_component_sum) as u32,
        )
    } else {
        (0, 0)
    };

    if !vector_tail.is_empty() {
        let (p, r) = super::scalar::residual_quantize_and_pack::<B>(
            vector_tail,
            primary_terms,
            residual_terms,
            primary_delta,
            primary_out_tail,
            residual_out_tail,
        );
        primary_component_sum += p;
        residual_component_sum += r;
    }

    (primary_component_sum, residual_component_sum)
}

#[target_feature(enable = "avx512f")]
pub fn residual_decode_avx512<const B: usize>(
    vector: &TurboResidualVector<'_, B>,
    out: &mut [f32],
) {
    let (tail_split, in_head, in_tail) = vector.split_tail(out.len());
    let (out_head, out_tail) = out.split_at_mut(tail_split);

    if !in_head.primary.data.is_empty() {
        let primary_interval = packing::block_dim(B);
        unsafe {
            let primary_terms = VectorDecodeTermsAvx512::from_terms(&in_head.primary.terms);
            let residual_terms = VectorDecodeTermsAvx512::from_terms(&in_head.residual.terms);
            let mask = _mm512_set1_epi32(i32::from(u8::MAX >> (8 - B)));
            let mut pbuf = _mm512_set1_epi32(0);
            for i in (0..tail_split).step_by(16) {
                if i.is_multiple_of(primary_interval) {
                    pbuf = _mm512_cvtepu8_epi32(_mm_lddqu_si128(
                        in_head
                            .primary
                            .data
                            .as_ptr()
                            .add(i / primary_interval * TURBO_BLOCK_SIZE)
                            as *const __m128i,
                    ));
                } else {
                    pbuf = match B {
                        1 => _mm512_srli_epi32::<1>(pbuf),
                        2 => _mm512_srli_epi32::<2>(pbuf),
                        4 => _mm512_srli_epi32::<4>(pbuf),
                        8 => _mm512_srli_epi32::<8>(pbuf),
                        _ => unreachable!(),
                    };
                }

                let primary = _mm512_and_si512(pbuf, mask);
                let residual = _mm512_cvtepu8_epi32(_mm_lddqu_si128(
                    in_head.residual.data.as_ptr().add(i) as *const __m128i,
                ));

                let decoded = _mm512_add_ps(
                    primary_terms.dequantize(primary),
                    residual_terms.dequantize(residual),
                );
                _mm512_storeu_ps(out_head.as_mut_ptr().add(i), decoded);
            }
        }
    }

    if !in_tail.primary.data.is_empty() {
        super::scalar::residual_decode::<B>(&in_tail, out_tail);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn lvq1_quantize_and_pack_avx512<const B: usize>(
    v: &[f32],
    lower: f32,
    upper: f32,
    out: &mut [u8],
) -> u32 {
    if B == 2 {
        return super::scalar::lvq1_quantize_and_pack::<B>(v, lower, upper, out);
    }

    let delta_inv = _mm512_set1_ps(((1 << B) - 1) as f32 / (upper - lower));
    let lower = _mm512_set1_ps(lower);
    let upper = _mm512_set1_ps(upper);
    let mut component_sum = _mm512_set1_epi32(0);
    let out_chunks = (B * 16) / 8;
    for (c, o) in v.chunks(16).zip(out.chunks_mut(out_chunks)) {
        let mask = u16::MAX >> (16 - c.len());
        let mut v = _mm512_maskz_loadu_ps(mask, c.as_ptr());
        v = _mm512_min_ps(v, upper);
        v = _mm512_max_ps(v, lower);
        v = _mm512_sub_ps(v, lower);
        v = _mm512_mul_ps(v, delta_inv);
        let q = _mm512_maskz_cvtps_epu32(mask, mm512_round_nonnegative_ties_away_zero_ps(v));
        component_sum = _mm512_add_epi32(component_sum, q);
        pack::<B>(q, o);
    }
    _mm512_reduce_add_epi32(component_sum) as u32
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn lvq1_decode_avx512<const B: usize>(v: &PrimaryVector<B>, out: &mut [f32]) {
    let chunk_size = (B * 16).div_ceil(8);
    let lower = _mm512_set1_ps(v.v.terms.lower);
    let delta = _mm512_set1_ps(v.v.terms.delta);
    for (c, o) in v.v.data.chunks(chunk_size).zip(out.chunks_mut(16)) {
        let v = _mm512_fmadd_ps(_mm512_cvtepu32_ps(unpack::<B>(c)), delta, lower);
        _mm512_mask_storeu_ps(o.as_mut_ptr(), u16::MAX >> (16 - o.len()), v);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,avx")]
pub unsafe fn lvq2_quantize_and_pack<const B1: usize, const B2: usize>(
    v: &[f32],
    lower: f32,
    upper: f32,
    primary: &mut [u8],
    residual_interval: f32,
    residual: &mut [u8],
) -> (u32, u32) {
    let p_chunk_size = (B1 * 16) / 8;
    let p_lower = _mm512_set1_ps(lower);
    let p_upper = _mm512_set1_ps(upper);
    let p_delta = _mm512_set1_ps((upper - lower) / ((1 << B1) - 1) as f32);
    let p_delta_inv = _mm512_set1_ps(((1 << B1) - 1) as f32 / (upper - lower));
    let mut p_sum = _mm512_set1_epi32(0);

    let r_chunk_size = (B2 * 16) / 8;
    let r_lower = _mm512_set1_ps(-residual_interval / 2.0);
    let r_upper = _mm512_set1_ps(residual_interval / 2.0);
    let r_delta_inv = _mm512_set1_ps(((1 << B2) - 1) as f32 / residual_interval);
    let mut r_sum = _mm512_set1_epi32(0);

    for (vc, (pc, rc)) in v.chunks(16).zip(
        primary
            .chunks_mut(p_chunk_size)
            .zip(residual.chunks_mut(r_chunk_size)),
    ) {
        let vmask = u16::MAX >> (16 - vc.len());
        let v = _mm512_maskz_loadu_ps(vmask, vc.as_ptr());
        let mut ps = _mm512_min_ps(v, p_upper);
        ps = _mm512_max_ps(ps, p_lower);
        ps = _mm512_sub_ps(ps, p_lower);
        ps = _mm512_mul_ps(ps, p_delta_inv);
        let pi = _mm512_maskz_cvtps_epu32(vmask, mm512_round_nonnegative_ties_away_zero_ps(ps));
        p_sum = _mm512_add_epi32(p_sum, pi);
        pack::<B1>(pi, pc);

        // Compute the residual delta from the dequantized value.
        let mut rs = _mm512_sub_ps(v, _mm512_fmadd_ps(_mm512_cvtepu32_ps(pi), p_delta, p_lower));
        rs = _mm512_min_ps(rs, r_upper);
        rs = _mm512_max_ps(rs, r_lower);
        rs = _mm512_sub_ps(rs, r_lower);
        rs = _mm512_mul_ps(rs, r_delta_inv);
        let ri = _mm512_maskz_cvtps_epu32(vmask, mm512_round_nonnegative_ties_away_zero_ps(rs));
        r_sum = _mm512_add_epi32(r_sum, ri);
        pack::<B2>(ri, rc);
    }

    (
        _mm512_reduce_add_epi32(p_sum) as u32,
        _mm512_reduce_add_epi32(r_sum) as u32,
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn lvq2_decode_avx512<const B1: usize, const B2: usize>(
    v: &TwoLevelVector<B1, B2>,
    out: &mut [f32],
) {
    let p_chunk_size = (B1 * 16).div_ceil(8);
    let p_lower = _mm512_set1_ps(v.primary.v.terms.lower);
    let p_delta = _mm512_set1_ps(v.primary.v.terms.delta);

    let r_chunk_size = (B2 * 16).div_ceil(8);
    let r_lower = _mm512_set1_ps(v.residual.terms.lower);
    let r_delta = _mm512_set1_ps(v.residual.terms.delta);

    for ((p, r), o) in v
        .primary
        .v
        .data
        .chunks(p_chunk_size)
        .zip(v.residual.data.chunks(r_chunk_size))
        .zip(out.chunks_mut(16))
    {
        let pps = _mm512_fmadd_ps(_mm512_cvtepu32_ps(unpack::<B1>(p)), p_delta, p_lower);
        let rps = _mm512_fmadd_ps(_mm512_cvtepu32_ps(unpack::<B2>(r)), r_delta, r_lower);
        let v = _mm512_add_ps(pps, rps);
        _mm512_mask_storeu_ps(o.as_mut_ptr(), u16::MAX >> (16 - o.len()), v);
    }
}

#[inline]
unsafe fn pack<const N: usize>(v: __m512i, out: &mut [u8]) {
    match N {
        1 => pack1(v, out),
        4 => pack4(v, out),
        8 => pack8(v, out),
        _ => unimplemented!(),
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2")]
#[inline]
unsafe fn pack1(v: __m512i, out: &mut [u8]) {
    let mut p = _mm512_cvtepi32_epi16(v);
    p = _mm256_sllv_epi16(
        p,
        _mm256_loadu_epi16([0i16, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7].as_ptr()),
    );
    let mut p = _mm256_cvtepi16_epi8(p);
    // [7, 6, 5, 4, 3, 2, 1, 0] => [7, 6, 5, 4, 73, 62, 51, 40]
    p = _mm_or_si128(p, _mm_srli_epi64::<32>(p));
    // [7, 6, 5, 4, 73, 62, 51, 40] => [7, 6, 75, 64, 753, 642, 7351, 6420]
    p = _mm_or_si128(p, _mm_srli_epi64::<16>(p));
    // [7, 6, 75, 64, 753, 642, 7351, 6420] => [7, 76, 765, 7654, 76543, 765432, 7654321, 76543210]
    p = _mm_or_si128(p, _mm_srli_epi64::<8>(p));
    // lowest byte in each 64 bit lane are the only relevant bits in this reduction.
    p = _mm_shuffle_epi8(p, _mm_set1_epi16(0x08_00));
    _mm_mask_storeu_epi8(out.as_mut_ptr() as *mut i8, u16::MAX >> (16 - out.len()), p);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2")]
#[inline]
unsafe fn pack4(v: __m512i, out: &mut [u8]) {
    let mut p = _mm256_cvtepi16_epi8(_mm512_cvtepi32_epi16(v));
    // place even lanes in the low 64 bits and odd lanes in the high 64 bits.
    p = _mm_shuffle_epi8(
        p,
        _mm_loadu_epi8([0i8, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15].as_ptr()),
    );
    p = _mm_sllv_epi64(p, _mm_loadu_epi64([0i64, 4].as_ptr()));
    p = _mm_or_si128(p, _mm_bsrli_si128::<8>(p));
    _mm_mask_storeu_epi8(out.as_mut_ptr() as *mut i8, u16::MAX >> (16 - out.len()), p);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn pack8(v: __m512i, out: &mut [u8]) {
    _mm_mask_storeu_epi8(
        out.as_mut_ptr() as *mut i8,
        u16::MAX >> (16 - out.len()),
        _mm256_cvtepi16_epi8(_mm512_cvtepi32_epi16(v)),
    );
}

#[target_feature(enable = "avx512vnni,avx512bw,avx512vl,avx512vpopcntdq,avx512f")]
#[inline]
pub unsafe fn dot_u8<const B: usize>(a: &[u8], b: &[u8]) -> u32 {
    match B {
        1 => {
            let mut sum = _mm512_set1_epi32(0);
            for (ac, bc) in a.chunks(64).zip(b.chunks(64)) {
                let mask = u64::MAX >> (64 - ac.len());
                let av = _mm512_maskz_loadu_epi8(mask, ac.as_ptr() as *const i8);
                let bv = _mm512_maskz_loadu_epi8(mask, bc.as_ptr() as *const i8);
                let dot = _mm512_and_epi32(av, bv);
                sum = _mm512_add_epi32(sum, _mm512_popcnt_epi32(dot));
            }
            _mm512_reduce_add_epi32(sum) as u32
        }
        2 => {
            let mut dot0 = _mm512_set1_epi32(0);
            let mut dot1 = _mm512_set1_epi32(0);
            let mut dot2 = _mm512_set1_epi32(0);
            let mut dot3 = _mm512_set1_epi32(0);
            let dibit_mask = _mm512_set1_epi8(0x3);
            for (ac, bc) in a.chunks(64).zip(b.chunks(64)) {
                let mask = u64::MAX >> (64 - ac.len());
                let av = _mm512_maskz_loadu_epi8(mask, ac.as_ptr() as *const i8);
                let bv = _mm512_maskz_loadu_epi8(mask, bc.as_ptr() as *const i8);

                dot0 = _mm512_dpbusd_epi32(
                    dot0,
                    _mm512_and_si512(av, dibit_mask),
                    _mm512_and_si512(bv, dibit_mask),
                );
                dot1 = _mm512_dpbusd_epi32(
                    dot1,
                    _mm512_and_si512(_mm512_srli_epi64::<2>(av), dibit_mask),
                    _mm512_and_si512(_mm512_srli_epi64::<2>(bv), dibit_mask),
                );
                dot2 = _mm512_dpbusd_epi32(
                    dot2,
                    _mm512_and_si512(_mm512_srli_epi64::<4>(av), dibit_mask),
                    _mm512_and_si512(_mm512_srli_epi64::<4>(bv), dibit_mask),
                );
                dot3 = _mm512_dpbusd_epi32(
                    dot3,
                    _mm512_and_si512(_mm512_srli_epi64::<6>(av), dibit_mask),
                    _mm512_and_si512(_mm512_srli_epi64::<6>(bv), dibit_mask),
                );
            }
            dot0 = _mm512_add_epi32(dot0, dot1);
            dot2 = _mm512_add_epi32(dot2, dot3);
            _mm512_reduce_add_epi32(_mm512_add_epi32(dot0, dot2)) as u32
        }
        4 => {
            let mut dot0 = _mm512_set1_epi32(0);
            let mut dot1 = _mm512_set1_epi32(0);
            let nibble_mask = _mm512_set1_epi8(0xf);
            for (ac, bc) in a.chunks(64).zip(b.chunks(64)) {
                let mask = u64::MAX >> (64 - ac.len());
                let av = _mm512_maskz_loadu_epi8(mask, ac.as_ptr() as *const i8);
                let av_even = _mm512_and_si512(av, nibble_mask);
                let av_odd = _mm512_and_si512(_mm512_srli_epi64::<4>(av), nibble_mask);
                let bv = _mm512_maskz_loadu_epi8(mask, bc.as_ptr() as *const i8);
                let bv_even = _mm512_and_si512(bv, nibble_mask);
                let bv_odd = _mm512_and_si512(_mm512_srli_epi64::<4>(bv), nibble_mask);

                dot0 = _mm512_dpbusd_epi32(dot0, av_even, bv_even);
                dot1 = _mm512_dpbusd_epi32(dot1, av_odd, bv_odd);
            }
            _mm512_reduce_add_epi32(_mm512_add_epi32(dot0, dot1)) as u32
        }
        8 => {
            let zero = _mm512_set1_epi8(0);
            let mut dot0 = _mm512_set1_epi32(0);
            let mut dot1 = _mm512_set1_epi32(0);
            for (ac, bc) in a.chunks(64).zip(b.chunks(64)) {
                let mask = u64::MAX >> (64 - ac.len());
                let av = _mm512_maskz_loadu_epi8(mask, ac.as_ptr() as *const i8);
                let av_lo = _mm512_unpacklo_epi8(av, zero);
                let av_hi = _mm512_unpackhi_epi8(av, zero);
                let bv = _mm512_maskz_loadu_epi8(mask, bc.as_ptr() as *const i8);
                let bv_lo = _mm512_unpacklo_epi8(bv, zero);
                let bv_hi = _mm512_unpackhi_epi8(bv, zero);

                dot0 = _mm512_dpwssd_epi32(dot0, av_lo, bv_lo);
                dot1 = _mm512_dpwssd_epi32(dot1, av_hi, bv_hi);
            }
            _mm512_reduce_add_epi32(_mm512_add_epi32(dot0, dot1)) as u32
        }
        _ => unimplemented!(),
    }
}

struct ResidualDotComponents512 {
    ap_dot_bp: __m512i,
    ap_dot_br: __m512i,
    ar_dot_bp: __m512i,
    ar_dot_br: __m512i,
}

impl ResidualDotComponents512 {
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn new() -> Self {
        Self {
            ap_dot_bp: _mm512_set1_epi32(0),
            ap_dot_br: _mm512_set1_epi32(0),
            ar_dot_bp: _mm512_set1_epi32(0),
            ar_dot_br: _mm512_set1_epi32(0),
        }
    }

    // XXX fix name
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn into_lvq2_dot(self) -> ResidualDotComponents {
        ResidualDotComponents {
            ap_dot_bp: _mm512_reduce_add_epi32(self.ap_dot_bp) as u32,
            ap_dot_br: _mm512_reduce_add_epi32(self.ap_dot_br) as u32,
            ar_dot_bp: _mm512_reduce_add_epi32(self.ar_dot_bp) as u32,
            ar_dot_br: _mm512_reduce_add_epi32(self.ar_dot_br) as u32,
        }
    }
}

#[rustfmt::skip]
const EXPAND_U1_MASK: [i8; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7,
];

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
#[inline]
unsafe fn expand_u1(v: __m512i) -> __m512i {
    let expanded = _mm512_permutexvar_epi8(_mm512_loadu_epi8(EXPAND_U1_MASK.as_ptr()), v);
    let masked = _mm512_and_si512(expanded, _mm512_set1_epi64(0x80402010_08040201u64 as i64));
    let cmp = _mm512_movm_epi8(_mm512_cmpgt_epu8_mask(masked, _mm512_set1_epi8(0)));
    _mm512_and_si512(cmp, _mm512_set1_epi8(1))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
pub unsafe fn dot_residual_u8<const B1: usize, const B2: usize>(
    ap: &[u8],
    ar: &[u8],
    bp: &[u8],
    br: &[u8],
) -> ResidualDotComponents {
    match (B1, B2) {
        (1, 8) => {
            let zero = _mm512_set1_epi8(0);
            let mut dot = ResidualDotComponents512::new();
            for ((ap, ar), (bp, br)) in ap
                .chunks(8)
                .zip(ar.chunks(64))
                .zip(bp.chunks(8).zip(br.chunks(64)))
            {
                // We're loading 64 _dimensions_ at a time so load 8 bytes from primary vectors.
                let mask1 = u64::MAX >> (64 - ap.len());
                let mask8 = u64::MAX >> (64 - ar.len());
                let apv = expand_u1(_mm512_maskz_loadu_epi8(mask1, ap.as_ptr() as *const i8));
                let arv = _mm512_maskz_loadu_epi8(mask8, ar.as_ptr() as *const i8);
                let bpv = expand_u1(_mm512_maskz_loadu_epi8(mask1, bp.as_ptr() as *const i8));
                let brv = _mm512_maskz_loadu_epi8(mask8, br.as_ptr() as *const i8);

                // Perform byte doc product for anything involving primary vectors. Note that the
                // second arg is treated as *signed* so it must be the primary/nibble vector.
                dot.ap_dot_bp = _mm512_dpbusd_epi32(dot.ap_dot_bp, apv, bpv);
                dot.ap_dot_br = _mm512_dpbusd_epi32(dot.ap_dot_br, brv, apv);
                dot.ar_dot_bp = _mm512_dpbusd_epi32(dot.ar_dot_bp, arv, bpv);
                // 8 bit dot product must be done with a 16-bit instruction due to signed-ness.
                let arv_lo = _mm512_unpacklo_epi8(arv, zero);
                let brv_lo = _mm512_unpacklo_epi8(brv, zero);
                dot.ar_dot_br = _mm512_dpwssd_epi32(dot.ar_dot_br, arv_lo, brv_lo);
                let arv_hi = _mm512_unpackhi_epi8(arv, zero);
                let brv_hi = _mm512_unpackhi_epi8(brv, zero);
                dot.ar_dot_br = _mm512_dpwssd_epi32(dot.ar_dot_br, arv_hi, brv_hi);
            }
            dot.into_lvq2_dot()
        }
        (4, 4) => {
            let nibble_mask = _mm512_set1_epi8(0xf);
            let mut dot = ResidualDotComponents512::new();
            for ((ap, ar), (bp, br)) in ap
                .chunks(64)
                .zip(ar.chunks(64))
                .zip(bp.chunks(64).zip(br.chunks(64)))
            {
                let mask = u64::MAX >> (64 - ap.len());
                let apv = _mm512_maskz_loadu_epi8(mask, ap.as_ptr() as *const i8);
                let arv = _mm512_maskz_loadu_epi8(mask, ar.as_ptr() as *const i8);
                let bpv = _mm512_maskz_loadu_epi8(mask, bp.as_ptr() as *const i8);
                let brv = _mm512_maskz_loadu_epi8(mask, br.as_ptr() as *const i8);

                let apv_lo = _mm512_and_si512(apv, nibble_mask);
                let bpv_lo = _mm512_and_si512(bpv, nibble_mask);
                dot.ap_dot_bp = _mm512_dpbusd_epi32(dot.ap_dot_bp, apv_lo, bpv_lo);
                let brv_lo = _mm512_and_si512(brv, nibble_mask);
                dot.ap_dot_br = _mm512_dpbusd_epi32(dot.ap_dot_br, apv_lo, brv_lo);
                let arv_lo = _mm512_and_si512(arv, nibble_mask);
                dot.ar_dot_bp = _mm512_dpbusd_epi32(dot.ar_dot_bp, arv_lo, bpv_lo);
                dot.ar_dot_br = _mm512_dpbusd_epi32(dot.ar_dot_br, arv_lo, brv_lo);

                let apv_hi = _mm512_and_si512(_mm512_srli_epi64::<4>(apv), nibble_mask);
                let bpv_hi = _mm512_and_si512(_mm512_srli_epi64::<4>(bpv), nibble_mask);
                dot.ap_dot_bp = _mm512_dpbusd_epi32(dot.ap_dot_bp, apv_hi, bpv_hi);
                let brv_hi = _mm512_and_si512(_mm512_srli_epi64::<4>(brv), nibble_mask);
                dot.ap_dot_br = _mm512_dpbusd_epi32(dot.ap_dot_br, apv_hi, brv_hi);
                let arv_hi = _mm512_and_si512(_mm512_srli_epi64::<4>(arv), nibble_mask);
                dot.ar_dot_bp = _mm512_dpbusd_epi32(dot.ar_dot_bp, arv_hi, bpv_hi);
                dot.ar_dot_br = _mm512_dpbusd_epi32(dot.ar_dot_br, arv_hi, brv_hi);
            }
            dot.into_lvq2_dot()
        }
        (4, 8) => {
            let nibble_mask = _mm512_set1_epi8(0xf);
            let zero = _mm512_set1_epi8(0);
            let mut dot = ResidualDotComponents512::new();
            for ((ap, ar), (bp, br)) in ap
                .chunks(32)
                .zip(ar.chunks(64))
                .zip(bp.chunks(32).zip(br.chunks(64)))
            {
                // We're loading 64 _dimensions_ at a time so load 32 bytes from primary vectors.
                let mask4 = u64::MAX >> (64 - ap.len());
                let mask8 = u64::MAX >> (64 - ar.len());
                // Interleave 64 bit words after load so that unpacklo_epi8 works correctly.
                let apv_raw = _mm512_maskz_expand_epi64(
                    0b01010101,
                    _mm512_maskz_loadu_epi8(mask4, ap.as_ptr() as *const i8),
                );
                let arv = _mm512_maskz_loadu_epi8(mask8, ar.as_ptr() as *const i8);
                let bpv_raw = _mm512_maskz_expand_epi64(
                    0b01010101,
                    _mm512_maskz_loadu_epi8(mask4, bp.as_ptr() as *const i8),
                );
                let brv = _mm512_maskz_loadu_epi8(mask8, br.as_ptr() as *const i8);

                // Unpack primary vectors by interleaving input nibbles.
                let apv = _mm512_unpacklo_epi8(
                    _mm512_and_si512(apv_raw, nibble_mask),
                    _mm512_and_si512(_mm512_srli_epi64::<4>(apv_raw), nibble_mask),
                );
                let bpv = _mm512_unpacklo_epi8(
                    _mm512_and_si512(bpv_raw, nibble_mask),
                    _mm512_and_si512(_mm512_srli_epi64::<4>(bpv_raw), nibble_mask),
                );

                // Perform byte doc product for anything involving primary vectors. Note that the
                // second arg is treated as *signed* so it must be the primary/nibble vector.
                dot.ap_dot_bp = _mm512_dpbusd_epi32(dot.ap_dot_bp, apv, bpv);
                dot.ap_dot_br = _mm512_dpbusd_epi32(dot.ap_dot_br, brv, apv);
                dot.ar_dot_bp = _mm512_dpbusd_epi32(dot.ar_dot_bp, arv, bpv);
                // 8 bit dot product must be done with a 16-bit instruction due to signed-ness.
                let arv_lo = _mm512_unpacklo_epi8(arv, zero);
                let brv_lo = _mm512_unpacklo_epi8(brv, zero);
                dot.ar_dot_br = _mm512_dpwssd_epi32(dot.ar_dot_br, arv_lo, brv_lo);
                let arv_hi = _mm512_unpackhi_epi8(arv, zero);
                let brv_hi = _mm512_unpackhi_epi8(brv, zero);
                dot.ar_dot_br = _mm512_dpwssd_epi32(dot.ar_dot_br, arv_hi, brv_hi);
            }
            dot.into_lvq2_dot()
        }
        (8, 8) => {
            let zero = _mm512_set1_epi8(0);
            let mut dot = ResidualDotComponents512::new();
            for ((ap, ar), (bp, br)) in ap
                .chunks(64)
                .zip(ar.chunks(64))
                .zip(bp.chunks(64).zip(br.chunks(64)))
            {
                let mask = u64::MAX >> (64 - ap.len());
                let apv = _mm512_maskz_loadu_epi8(mask, ap.as_ptr() as *const i8);
                let arv = _mm512_maskz_loadu_epi8(mask, ar.as_ptr() as *const i8);
                let bpv = _mm512_maskz_loadu_epi8(mask, bp.as_ptr() as *const i8);
                let brv = _mm512_maskz_loadu_epi8(mask, br.as_ptr() as *const i8);

                let apv_lo = _mm512_unpacklo_epi8(apv, zero);
                let bpv_lo = _mm512_unpacklo_epi8(bpv, zero);
                dot.ap_dot_bp = _mm512_dpwssd_epi32(dot.ap_dot_bp, apv_lo, bpv_lo);
                let brv_lo = _mm512_unpacklo_epi8(brv, zero);
                dot.ap_dot_br = _mm512_dpwssd_epi32(dot.ap_dot_br, apv_lo, brv_lo);
                let arv_lo = _mm512_unpacklo_epi8(arv, zero);
                dot.ar_dot_bp = _mm512_dpwssd_epi32(dot.ar_dot_bp, arv_lo, bpv_lo);
                dot.ar_dot_br = _mm512_dpwssd_epi32(dot.ar_dot_br, arv_lo, brv_lo);

                let apv_hi = _mm512_unpackhi_epi8(apv, zero);
                let bpv_hi = _mm512_unpackhi_epi8(bpv, zero);
                dot.ap_dot_bp = _mm512_dpwssd_epi32(dot.ap_dot_bp, apv_hi, bpv_hi);
                let brv_hi = _mm512_unpackhi_epi8(brv, zero);
                dot.ap_dot_br = _mm512_dpwssd_epi32(dot.ap_dot_br, apv_hi, brv_hi);
                let arv_hi = _mm512_unpackhi_epi8(arv, zero);
                dot.ar_dot_bp = _mm512_dpwssd_epi32(dot.ar_dot_bp, arv_hi, bpv_hi);
                dot.ar_dot_br = _mm512_dpwssd_epi32(dot.ar_dot_br, arv_hi, brv_hi);
            }
            dot.into_lvq2_dot()
        }
        _ => super::scalar::dot_residual_u8::<B1, B2>(ap, ar, bp, br),
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn lvq2_f32_dot_unnormalized<const B1: usize, const B2: usize>(
    query: &[f32],
    query_sum: f32,
    doc: &TwoLevelVector<'_, B1, B2>,
) -> f64 {
    let p_chunk_size = (B1 * 16).div_ceil(8);
    let r_chunk_size = (B2 * 16).div_ceil(8);

    let mut p_dot = _mm512_set1_ps(0.0);
    let mut r_dot = _mm512_set1_ps(0.0);
    for (q, (p, r)) in query.chunks(16).zip(
        doc.primary
            .v
            .data
            .chunks(p_chunk_size)
            .zip(doc.residual.data.chunks(r_chunk_size)),
    ) {
        let qmask = u16::MAX >> (16 - q.len());
        let qv = _mm512_maskz_loadu_ps(qmask, q.as_ptr());
        p_dot = _mm512_fmadd_ps(qv, _mm512_cvtepu32_ps(unpack::<B1>(p)), p_dot);
        r_dot = _mm512_fmadd_ps(qv, _mm512_cvtepu32_ps(unpack::<B2>(r)), r_dot);
    }
    doc.f32_dot_correction(
        query_sum,
        _mm512_reduce_add_ps(p_dot),
        _mm512_reduce_add_ps(r_dot),
    )
    .into()
}

#[target_feature(enable = "avx512f,avx512vnni,avx512bw")]
#[inline]
pub unsafe fn primary_query8_dot_unnormalized_avx512<const B: usize>(
    query: &[u8],
    doc: &TurboPrimaryVector<'_, B>,
) -> u32 {
    // 1, 2, and 4 bits are specialized. 8 bit can use the symmetrical impl.
    match B {
        1 | 2 | 4 => {}
        8 => return dot_u8::<8>(query, doc.rep.data),
        _ => unimplemented!(),
    }

    let (tail_split, doc_head, doc_tail) = doc.split_tail(query.len());
    let (query_head, query_tail) = query.split_at(tail_split);
    // TODO: unroll these loops to use more accumulator registers.
    let mut dot = match B {
        1 => {
            let mask = _mm512_set1_epi8(0x1);
            let mut dot0 = _mm512_set1_epi32(0);
            let mut dot1 = _mm512_set1_epi32(0);
            for i in (0..tail_split).step_by(128) {
                // Load 128 bits and broadcast to 512 bits, then shift right by 0, 1, 2, 3.
                // This will arrange each dimension into the lowest bit to align with an 8 bit query.
                let mut dv = _mm512_broadcast_i32x4(_mm_lddqu_si128(
                    doc_head.rep.data.as_ptr().add(i / 128 * 16) as *const __m128i,
                ));
                dv = _mm512_srlv_epi64(dv, _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0));

                dot0 = _mm512_dpbusd_epi32(
                    dot0,
                    _mm512_loadu_epi8(query_head.as_ptr().add(i) as *const i8),
                    _mm512_and_si512(dv, mask),
                );
                dot1 = _mm512_dpbusd_epi32(
                    dot1,
                    _mm512_loadu_epi8(query_head.as_ptr().add(i + 64) as *const i8),
                    _mm512_and_si512(_mm512_srli_epi64::<4>(dv), mask),
                );
            }

            _mm512_reduce_add_epi32(_mm512_add_epi32(dot0, dot1)) as u32
        }
        2 => {
            let mask = _mm512_set1_epi8(0x3);
            let mut dot = _mm512_set1_epi32(0);
            for i in (0..tail_split).step_by(64) {
                // Load 128 bits and broadcast to 512 bits, then shift right by 0, 2, 4, 6
                // This will arrange each dimension into the lowest dibit to align with an 8 bit query.
                let mut dv = _mm512_broadcast_i32x4(_mm_lddqu_si128(
                    doc_head.rep.data.as_ptr().add(i / 64 * 16) as *const __m128i,
                ));
                dv = _mm512_srlv_epi64(dv, _mm512_set_epi64(6, 6, 4, 4, 2, 2, 0, 0));

                dot = _mm512_dpbusd_epi32(
                    dot,
                    _mm512_loadu_epi8(query_head.as_ptr().add(i) as *const i8),
                    _mm512_and_si512(dv, mask),
                );
            }

            _mm512_reduce_add_epi32(dot) as u32
        }
        4 => {
            let mask = _mm512_set1_epi8(0xf);
            let mut dot = _mm512_set1_epi32(0);
            for i in (0..tail_split).step_by(64) {
                // Load 128 for 32 dim or 256 bits for 64 dim.
                let load_mask = u8::MAX >> (8 - (tail_split - i).min(64) / 16);
                let mut dv = _mm512_maskz_loadu_epi64(
                    load_mask,
                    doc_head.rep.data.as_ptr().add(i / 2) as *const i64,
                );
                // Shuffle to duplicate each 128 bit block, then shift and mask to unpack.
                dv = _mm512_shuffle_i64x2::<0b0101_0000>(dv, dv);
                dv = _mm512_and_si512(
                    _mm512_srlv_epi64(dv, _mm512_set_epi64(4, 4, 0, 0, 4, 4, 0, 0)),
                    mask,
                );

                dot = _mm512_dpbusd_epi32(
                    dot,
                    _mm512_maskz_loadu_epi8(
                        u64::MAX >> (64 - (tail_split - i).min(64)),
                        query_head.as_ptr().add(i) as *const i8,
                    ),
                    dv,
                );
            }

            _mm512_reduce_add_epi32(dot) as u32
        }
        _ => unreachable!(),
    };

    if !query_tail.is_empty() {
        dot += super::scalar::primary_query8_dot_unnormalized::<B>(query_tail, &doc_tail);
    }
    dot
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
pub unsafe fn residual_dot_unnormalized_avx512<const B: usize>(
    query: (&[u8], &[u8]),
    doc: (&[u8], &[u8]),
) -> ResidualDotComponents {
    let tail_split = query.1.len() & !(packing::block_dim(B) - 1);
    let primary_split = packing::byte_len(tail_split, B);

    let (query_head, query_tail) = split_residual_vector(query, primary_split, tail_split);
    let (doc_head, doc_tail) = split_residual_vector(doc, primary_split, tail_split);

    // XXX have to factor out all of these loading routines, they are often shared with other fns.
    let mut dot = if !query_head.0.is_empty() {
        let mut dot = ResidualDotComponents512::new();
        match B {
            1 => {
                let mask = _mm512_set1_epi8(0x1);
                for i in (0..tail_split).step_by(128) {
                    // Load 128 bits and broadcast to 512 bits, then shift right by 0, 1, 2, 3.
                    // This will arrange each dimension into the lowest bit to align with an 8 bit query.
                    let ap_buf = _mm512_srlv_epi64(
                        _mm512_broadcast_i32x4(_mm_lddqu_si128(
                            query_head.0.as_ptr().add(i / 128 * 16) as *const __m128i,
                        )),
                        _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0),
                    );
                    let bp_buf = _mm512_srlv_epi64(
                        _mm512_broadcast_i32x4(_mm_lddqu_si128(
                            doc_head.0.as_ptr().add(i / 128 * 16) as *const __m128i,
                        )),
                        _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0),
                    );

                    let ap = _mm512_and_si512(ap_buf, mask);
                    let bp = _mm512_and_si512(bp_buf, mask);
                    let ar = _mm512_loadu_epi8(query_head.1.as_ptr().add(i) as *const i8);
                    let br = _mm512_loadu_epi8(doc_head.1.as_ptr().add(i) as *const i8);

                    dot.ap_dot_bp = _mm512_dpbusd_epi32(dot.ap_dot_bp, ap, bp);
                    dot.ap_dot_br = _mm512_dpbusd_epi32(dot.ap_dot_br, br, ap);
                    dot.ar_dot_bp = _mm512_dpbusd_epi32(dot.ar_dot_bp, ar, bp);
                    dot.ar_dot_br = mm512_dot_u8(dot.ar_dot_br, ar, br);

                    let ap = _mm512_and_si512(_mm512_srli_epi64::<4>(ap_buf), mask);
                    let bp = _mm512_and_si512(_mm512_srli_epi64::<4>(bp_buf), mask);
                    let ar = _mm512_loadu_epi8(query_head.1.as_ptr().add(i + 64) as *const i8);
                    let br = _mm512_loadu_epi8(doc_head.1.as_ptr().add(i + 64) as *const i8);

                    dot.ap_dot_bp = _mm512_dpbusd_epi32(dot.ap_dot_bp, ap, bp);
                    dot.ap_dot_br = _mm512_dpbusd_epi32(dot.ap_dot_br, br, ap);
                    dot.ar_dot_bp = _mm512_dpbusd_epi32(dot.ar_dot_bp, ar, bp);
                    dot.ar_dot_br = mm512_dot_u8(dot.ar_dot_br, ar, br);
                }

                dot.into_lvq2_dot()
            }
            2 => {
                let mask = _mm512_set1_epi8(0x3);
                for i in (0..tail_split).step_by(64) {
                    let ap = _mm512_and_si512(
                        mask,
                        _mm512_srlv_epi64(
                            _mm512_broadcast_i32x4(_mm_lddqu_si128(
                                query_head.0.as_ptr().add(i / 64 * 16) as *const __m128i,
                            )),
                            _mm512_set_epi64(6, 6, 4, 4, 2, 2, 0, 0),
                        ),
                    );
                    let bp = _mm512_and_si512(
                        mask,
                        _mm512_srlv_epi64(
                            _mm512_broadcast_i32x4(_mm_lddqu_si128(
                                doc_head.0.as_ptr().add(i / 64 * 16) as *const __m128i,
                            )),
                            _mm512_set_epi64(6, 6, 4, 4, 2, 2, 0, 0),
                        ),
                    );
                    let ar = _mm512_loadu_epi8(query_head.1.as_ptr().add(i) as *const i8);
                    let br = _mm512_loadu_epi8(doc_head.1.as_ptr().add(i) as *const i8);

                    dot.ap_dot_bp = _mm512_dpbusd_epi32(dot.ap_dot_bp, ap, bp);
                    dot.ap_dot_br = _mm512_dpbusd_epi32(dot.ap_dot_br, br, ap);
                    dot.ar_dot_bp = _mm512_dpbusd_epi32(dot.ar_dot_bp, ar, bp);
                    dot.ar_dot_br = mm512_dot_u8(dot.ar_dot_br, ar, br);
                }
                dot.into_lvq2_dot()
            }
            4 => {
                let mask = _mm512_set1_epi8(0xf);
                for i in (0..tail_split).step_by(64) {
                    // Load 128 for 32 dim or 256 bits for 64 dim.
                    let load_mask = u8::MAX >> (8 - (tail_split - i).min(64) / 16);
                    let mut ap = _mm512_maskz_loadu_epi64(
                        load_mask,
                        query_head.0.as_ptr().add(i / 2) as *const i64,
                    );
                    ap = _mm512_and_si512(
                        _mm512_srlv_epi64(
                            _mm512_shuffle_i64x2::<0b0101_0000>(ap, ap),
                            _mm512_set_epi64(4, 4, 0, 0, 4, 4, 0, 0),
                        ),
                        mask,
                    );
                    let mut bp = _mm512_maskz_loadu_epi64(
                        load_mask,
                        doc_head.0.as_ptr().add(i / 2) as *const i64,
                    );
                    bp = _mm512_and_si512(
                        _mm512_srlv_epi64(
                            _mm512_shuffle_i64x2::<0b0101_0000>(bp, bp),
                            _mm512_set_epi64(4, 4, 0, 0, 4, 4, 0, 0),
                        ),
                        mask,
                    );
                    let residual_load_mask = u64::MAX >> (64 - (tail_split - i).min(64));
                    let ar = _mm512_maskz_loadu_epi8(
                        residual_load_mask,
                        query_head.1.as_ptr().add(i) as *const i8,
                    );
                    let br = _mm512_maskz_loadu_epi8(
                        residual_load_mask,
                        doc_head.1.as_ptr().add(i) as *const i8,
                    );

                    dot.ap_dot_bp = _mm512_dpbusd_epi32(dot.ap_dot_bp, ap, bp);
                    dot.ap_dot_br = _mm512_dpbusd_epi32(dot.ap_dot_br, br, ap);
                    dot.ar_dot_bp = _mm512_dpbusd_epi32(dot.ar_dot_bp, ar, bp);
                    dot.ar_dot_br = mm512_dot_u8(dot.ar_dot_br, ar, br);
                }
                dot.into_lvq2_dot()
            }
            8 => {
                // XXX try going to 64? might be faster.
                for i in (0..tail_split).step_by(32) {
                    let load_mask = u32::MAX >> (32 - (tail_split - i).min(32)) as u32;
                    let ap = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(
                        load_mask,
                        query_head.0.as_ptr().add(i) as *const i8,
                    ));
                    let ar = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(
                        load_mask,
                        query_head.1.as_ptr().add(i) as *const i8,
                    ));
                    let bp = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(
                        load_mask,
                        doc_head.0.as_ptr().add(i) as *const i8,
                    ));
                    let br = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(
                        load_mask,
                        doc_head.1.as_ptr().add(i) as *const i8,
                    ));

                    dot.ap_dot_bp = _mm512_dpwssd_epi32(dot.ap_dot_bp, ap, bp);
                    dot.ap_dot_br = _mm512_dpwssd_epi32(dot.ap_dot_br, ap, br);
                    dot.ar_dot_bp = _mm512_dpwssd_epi32(dot.ar_dot_bp, ar, bp);
                    dot.ar_dot_br = _mm512_dpwssd_epi32(dot.ar_dot_br, ar, br);
                }
                dot.into_lvq2_dot()
            }
            _ => unreachable!(),
        }
    } else {
        ResidualDotComponents::default()
    };

    if !query_tail.0.is_empty() {
        dot += super::scalar::residual_dot_unnormalized::<B>(query_tail, doc_tail);
    }
    dot
}

fn split_residual_vector<'a>(
    v: (&'a [u8], &'a [u8]),
    primary_split: usize,
    residual_split: usize,
) -> ((&'a [u8], &'a [u8]), (&'a [u8], &'a [u8])) {
    let primary = v.0.split_at(primary_split);
    let residual = v.1.split_at(residual_split);
    ((primary.0, residual.0), (primary.1, residual.1))
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn unpack<const N: usize>(bytes: &[u8]) -> __m512i {
    match N {
        1 => unpack1(bytes),
        4 => unpack4(bytes),
        8 => unpack8(bytes),
        _ => unimplemented!(),
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx2")]
#[inline]
unsafe fn unpack1(bytes: &[u8]) -> __m512i {
    let mut v = _mm_maskz_loadu_epi8((1 << bytes.len()) - 1, bytes.as_ptr() as *const i8);
    v = _mm_shuffle_epi8(v, _mm_loadu_epi64([0i64, 0x01010101_01010101].as_ptr()));
    v = _mm_and_si128(v, _mm_set1_epi64x(0x80402010_08040201u64 as i64));
    v = _mm_cmpeq_epi8(v, _mm_set1_epi8(0));
    v = _mm_andnot_si128(v, _mm_set1_epi8(1));
    _mm512_cvtepi16_epi32(_mm256_cvtepi8_epi16(v))
}

#[target_feature(enable = "avx512f,avx512bw,avx2,avx512vl")]
#[inline]
unsafe fn unpack4(bytes: &[u8]) -> __m512i {
    let mut v = _mm_maskz_loadu_epi8((1 << bytes.len()) - 1, bytes.as_ptr() as *const i8);
    let nibble_mask = _mm_set1_epi8(0xf);
    let w = _mm_and_si128(_mm_srli_epi64::<4>(v), nibble_mask);
    v = _mm_unpacklo_epi8(_mm_and_si128(v, nibble_mask), w);
    _mm512_cvtepi16_epi32(_mm256_cvtepi8_epi16(v))
}

#[target_feature(enable = "avx512f,avx512bw,avx2")]
#[inline]
unsafe fn unpack8(bytes: &[u8]) -> __m512i {
    let v = _mm_maskz_loadu_epi8(u16::MAX >> (16 - bytes.len()), bytes.as_ptr() as *const i8);
    _mm512_cvtepi16_epi32(_mm256_cvtepu8_epi16(v))
}

struct VectorEncodeTermsAvx512 {
    lower: __m512,
    upper: __m512,
    delta_inv: __m512,
}

impl VectorEncodeTermsAvx512 {
    #[inline(always)]
    unsafe fn from_terms(terms: &VectorEncodeTerms) -> Self {
        Self {
            lower: _mm512_set1_ps(terms.lower),
            upper: _mm512_set1_ps(terms.upper),
            delta_inv: _mm512_set1_ps(terms.delta_inv),
        }
    }

    #[inline(always)]
    unsafe fn quantize(&self, v: __m512) -> __m512i {
        let v = _mm512_min_ps(v, self.upper);
        let v = _mm512_max_ps(v, self.lower);
        let v = _mm512_sub_ps(v, self.lower);
        let v = _mm512_mul_ps(v, self.delta_inv);
        let v = mm512_round_nonnegative_ties_away_zero_ps(v);
        _mm512_cvtps_epu32(v)
    }
}

struct VectorDecodeTermsAvx512 {
    lower: __m512,
    delta: __m512,
}

impl VectorDecodeTermsAvx512 {
    #[inline(always)]
    unsafe fn from_terms(terms: &VectorDecodeTerms) -> Self {
        Self {
            lower: _mm512_set1_ps(terms.lower),
            delta: _mm512_set1_ps(terms.delta),
        }
    }

    #[inline(always)]
    unsafe fn dequantize(&self, v: __m512i) -> __m512 {
        _mm512_fmadd_ps(_mm512_cvtepu32_ps(v), self.delta, self.lower)
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn mm512_dot_u8(dot: __m512i, a: __m512i, b: __m512i) -> __m512i {
    let zero = _mm512_setzero_si512();
    let (a_lo, a_hi) = (_mm512_unpacklo_epi8(a, zero), _mm512_unpackhi_epi8(a, zero));
    let (b_lo, b_hi) = (_mm512_unpacklo_epi8(b, zero), _mm512_unpackhi_epi8(b, zero));
    let dot = _mm512_dpwssd_epi32(dot, a_lo, b_lo);
    _mm512_dpwssd_epi32(dot, a_hi, b_hi)
}
