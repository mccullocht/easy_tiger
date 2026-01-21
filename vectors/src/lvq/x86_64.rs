#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::{
    __m128i, __m512, __m512i, _MM_FROUND_NO_EXC, _MM_FROUND_TRUNC, _mm_add_pd, _mm_add_ps,
    _mm_cvtps_pd, _mm_cvtsd_f64, _mm_fmadd_pd, _mm_fmadd_ps, _mm_hadd_pd, _mm_hsub_pd,
    _mm_lddqu_si128, _mm_movehl_ps, _mm_mul_pd, _mm_mul_ps, _mm_set1_epi64x, _mm_set1_pd,
    _mm_set1_ps, _mm_storeu_si128, _mm_sub_pd, _mm_sub_ps, _mm256_add_ps, _mm256_castps256_ps128,
    _mm256_cvtepu8_epi16, _mm256_extractf32x4_ps, _mm256_fmadd_ps, _mm256_mul_ps, _mm256_set1_ps,
    _mm256_sub_ps, _mm512_add_epi32, _mm512_add_ps, _mm512_and_epi32, _mm512_and_si512,
    _mm512_broadcast_i32x4, _mm512_castps512_ps256, _mm512_cvtepi16_epi32, _mm512_cvtepi32_epi8,
    _mm512_cvtepu8_epi32, _mm512_cvtepu32_ps, _mm512_cvtps_epu32, _mm512_div_ps,
    _mm512_dpbusd_epi32, _mm512_dpwssd_epi32, _mm512_extractf32x8_ps, _mm512_fmadd_ps,
    _mm512_loadu_epi8, _mm512_loadu_ps, _mm512_mask_mul_ps, _mm512_mask_sub_ps,
    _mm512_maskz_loadu_epi8, _mm512_maskz_loadu_epi64, _mm512_maskz_loadu_ps, _mm512_max_ps,
    _mm512_min_ps, _mm512_mul_ps, _mm512_or_si512, _mm512_popcnt_epi32, _mm512_reduce_add_epi32,
    _mm512_reduce_add_ps, _mm512_reduce_max_ps, _mm512_reduce_min_ps, _mm512_roundscale_ps,
    _mm512_set_epi64, _mm512_set1_epi8, _mm512_set1_epi32, _mm512_set1_ps, _mm512_shuffle_i64x2,
    _mm512_sll_epi32, _mm512_sll_epi64, _mm512_srli_epi16, _mm512_srli_epi32, _mm512_srli_epi64,
    _mm512_srlv_epi64, _mm512_storeu_ps, _mm512_sub_ps, _mm512_unpackhi_epi8, _mm512_unpacklo_epi8,
};

use crate::lvq::{
    TURBO_BLOCK_SIZE, TurboPrimaryVector, TurboResidualVector, VectorDecodeTerms,
    VectorEncodeTerms, packing,
};

use super::{LAMBDA, MINIMUM_MSE_GRID, ResidualDotComponents, VectorStats};

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
        let terms = VectorEncodeTermsAvx512::from_terms(&terms);
        let mut qbuf = _mm512_set1_epi32(0);
        let mut component_sum = _mm512_set1_epi32(0);
        let mut shift = 0;
        for i in (0..tail_split).step_by(16) {
            let q = terms.quantize(_mm512_loadu_ps(in_head.as_ptr().add(i)));
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

#[target_feature(enable = "avx512vnni,avx512bw,avx512vl,avx512vpopcntdq,avx512f")]
#[inline]
pub unsafe fn dot_u8_avx512<const B: usize>(a: &[u8], b: &[u8]) -> u32 {
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

#[target_feature(enable = "avx512f,avx512vnni,avx512bw")]
#[inline]
pub unsafe fn primary_query8_dot_unnormalized_avx512<const B: usize>(
    query: &[u8],
    doc: &TurboPrimaryVector<'_, B>,
) -> u32 {
    // 1, 2, and 4 bits are specialized. 8 bit can use the symmetrical impl.
    match B {
        1 | 2 | 4 => {}
        8 => return dot_u8_avx512::<8>(query, doc.rep.data),
        _ => unimplemented!(),
    }

    let (tail_split, doc_head, doc_tail) = doc.split_tail(query.len());
    let (query_head, query_tail) = query.split_at(tail_split);
    // TODO: unroll these loops to use more accumulator registers.
    let mut dot = match B {
        1 => {
            let mut dot0 = _mm512_set1_epi32(0);
            let mut dot1 = _mm512_set1_epi32(0);
            for i in (0..tail_split).step_by(128) {
                let (dv64, dv128) = unpack_u1_avx512(_mm_lddqu_si128(
                    doc_head.rep.data.as_ptr().add(i / 128 * 16) as *const __m128i,
                ));

                dot0 = _mm512_dpbusd_epi32(
                    dot0,
                    _mm512_loadu_epi8(query_head.as_ptr().add(i) as *const i8),
                    dv64,
                );
                dot1 = _mm512_dpbusd_epi32(
                    dot1,
                    _mm512_loadu_epi8(query_head.as_ptr().add(i + 64) as *const i8),
                    dv128,
                );
            }

            _mm512_reduce_add_epi32(_mm512_add_epi32(dot0, dot1)) as u32
        }
        2 => {
            let mut dot = _mm512_set1_epi32(0);
            for i in (0..tail_split).step_by(64) {
                let dv = unpack_u2_avx512(_mm_lddqu_si128(
                    doc_head.rep.data.as_ptr().add(i / 64 * 16) as *const __m128i,
                ));

                dot = _mm512_dpbusd_epi32(
                    dot,
                    _mm512_loadu_epi8(query_head.as_ptr().add(i) as *const i8),
                    dv,
                );
            }

            _mm512_reduce_add_epi32(dot) as u32
        }
        4 => {
            let mut dot = _mm512_set1_epi32(0);
            for i in (0..tail_split).step_by(64) {
                let load_mask = u8::MAX >> (8 - (tail_split - i).min(64) / 16);
                let dv = unpack_u4_avx512(_mm512_maskz_loadu_epi64(
                    load_mask,
                    doc_head.rep.data.as_ptr().add(i / 2) as *const i64,
                ));

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

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn into_components(self) -> ResidualDotComponents {
        ResidualDotComponents {
            ap_dot_bp: _mm512_reduce_add_epi32(self.ap_dot_bp) as u32,
            ap_dot_br: _mm512_reduce_add_epi32(self.ap_dot_br) as u32,
            ar_dot_bp: _mm512_reduce_add_epi32(self.ar_dot_bp) as u32,
            ar_dot_br: _mm512_reduce_add_epi32(self.ar_dot_br) as u32,
        }
    }
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

    let mut dot = if !query_head.0.is_empty() {
        let mut dot = ResidualDotComponents512::new();
        match B {
            1 => {
                for i in (0..tail_split).step_by(128) {
                    let (ap64, ap128) = unpack_u1_avx512(_mm_lddqu_si128(
                        query_head.0.as_ptr().add(i / 128 * 16) as *const __m128i,
                    ));
                    let (bp64, bp128) = unpack_u1_avx512(_mm_lddqu_si128(
                        doc_head.0.as_ptr().add(i / 128 * 16) as *const __m128i,
                    ));

                    let ar = _mm512_loadu_epi8(query_head.1.as_ptr().add(i) as *const i8);
                    let br = _mm512_loadu_epi8(doc_head.1.as_ptr().add(i) as *const i8);
                    dot.ap_dot_bp = _mm512_dpbusd_epi32(dot.ap_dot_bp, ap64, bp64);
                    dot.ap_dot_br = _mm512_dpbusd_epi32(dot.ap_dot_br, br, ap64);
                    dot.ar_dot_bp = _mm512_dpbusd_epi32(dot.ar_dot_bp, ar, bp64);
                    dot.ar_dot_br = mm512_dot_u8(dot.ar_dot_br, ar, br);

                    let ar = _mm512_loadu_epi8(query_head.1.as_ptr().add(i + 64) as *const i8);
                    let br = _mm512_loadu_epi8(doc_head.1.as_ptr().add(i + 64) as *const i8);
                    dot.ap_dot_bp = _mm512_dpbusd_epi32(dot.ap_dot_bp, ap128, bp128);
                    dot.ap_dot_br = _mm512_dpbusd_epi32(dot.ap_dot_br, br, ap128);
                    dot.ar_dot_bp = _mm512_dpbusd_epi32(dot.ar_dot_bp, ar, bp128);
                    dot.ar_dot_br = mm512_dot_u8(dot.ar_dot_br, ar, br);
                }
            }
            2 => {
                for i in (0..tail_split).step_by(64) {
                    let ap = unpack_u2_avx512(_mm_lddqu_si128(
                        query_head.0.as_ptr().add(i / 64 * 16) as *const __m128i,
                    ));
                    let bp = unpack_u2_avx512(_mm_lddqu_si128(
                        doc_head.0.as_ptr().add(i / 64 * 16) as *const __m128i,
                    ));
                    let ar = _mm512_loadu_epi8(query_head.1.as_ptr().add(i) as *const i8);
                    let br = _mm512_loadu_epi8(doc_head.1.as_ptr().add(i) as *const i8);

                    dot.ap_dot_bp = _mm512_dpbusd_epi32(dot.ap_dot_bp, ap, bp);
                    dot.ap_dot_br = _mm512_dpbusd_epi32(dot.ap_dot_br, br, ap);
                    dot.ar_dot_bp = _mm512_dpbusd_epi32(dot.ar_dot_bp, ar, bp);
                    dot.ar_dot_br = mm512_dot_u8(dot.ar_dot_br, ar, br);
                }
            }
            4 => {
                for i in (0..tail_split).step_by(64) {
                    // Load 128 for 32 dim or 256 bits for 64 dim.
                    let load_mask = u8::MAX >> (8 - (tail_split - i).min(64) / 16);
                    let ap = unpack_u4_avx512(_mm512_maskz_loadu_epi64(
                        load_mask,
                        query_head.0.as_ptr().add(i / 2) as *const i64,
                    ));
                    let bp = unpack_u4_avx512(_mm512_maskz_loadu_epi64(
                        load_mask,
                        doc_head.0.as_ptr().add(i / 2) as *const i64,
                    ));
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
            }
            8 => {
                for i in (0..tail_split).step_by(64) {
                    // Load as u64s since the load mask will be for a multiple of 16 bytes.
                    let rem_dw = (tail_split - i).min(64) / 16;
                    let load_mask = u8::MAX >> (8 - rem_dw * 2);
                    let ap = _mm512_maskz_loadu_epi64(
                        load_mask,
                        query_head.0.as_ptr().add(i) as *const i64,
                    );
                    let ar = _mm512_maskz_loadu_epi64(
                        load_mask,
                        query_head.1.as_ptr().add(i) as *const i64,
                    );
                    let bp = _mm512_maskz_loadu_epi64(
                        load_mask,
                        doc_head.0.as_ptr().add(i) as *const i64,
                    );
                    let br = _mm512_maskz_loadu_epi64(
                        load_mask,
                        doc_head.1.as_ptr().add(i) as *const i64,
                    );

                    dot.ap_dot_bp = mm512_dot_u8(dot.ap_dot_bp, ap, bp);
                    dot.ap_dot_br = mm512_dot_u8(dot.ap_dot_br, ap, br);
                    dot.ar_dot_bp = mm512_dot_u8(dot.ar_dot_bp, ar, bp);
                    dot.ar_dot_br = mm512_dot_u8(dot.ar_dot_br, ar, br);
                }
            }
            _ => unreachable!(),
        }
        dot.into_components()
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

#[target_feature(enable = "avx512f,avx512vnni")]
#[inline]
unsafe fn mm512_dot_u8(dot: __m512i, a: __m512i, b: __m512i) -> __m512i {
    // Separate into 16 bit values to perform dot product because avx512vnni doesn't support pure
    // u8 dot product. This unpack does not produce a linear dimension order but it doesn't matter
    // here because our unpacks match and dot product is a summation.
    let (a_lo, a_hi) = (
        _mm512_and_si512(a, _mm512_set1_epi8(-1)),
        _mm512_srli_epi16::<8>(a),
    );
    let (b_lo, b_hi) = (
        _mm512_and_si512(b, _mm512_set1_epi8(-1)),
        _mm512_srli_epi16::<8>(b),
    );
    let dot = _mm512_dpwssd_epi32(dot, a_lo, b_lo);
    _mm512_dpwssd_epi32(dot, a_hi, b_hi)
}

/// Unpack 128 bits of u1 input as 128 bytes with 1 dimension per byte.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn unpack_u1_avx512(v: __m128i) -> (__m512i, __m512i) {
    // Broadcast the 128 bit to all lanes in the 512 bit register.
    let b = _mm512_broadcast_i32x4(v);
    // Shift each 128 bit lane to the right to expose bits [0, 1, 2, 3]
    let shifted = _mm512_srlv_epi64(b, _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0));
    // Shift one copy of the register right by 4 to expose bits [4, 5, 6, 7].
    // Then mask both outputs to the lowest byte.
    let mask = _mm512_set1_epi8(1);
    (
        _mm512_and_si512(shifted, mask),
        _mm512_and_si512(_mm512_srli_epi64::<4>(shifted), mask),
    )
}

/// Unpack 128 bits of u2 input as 64 bytes with 1 dimension per byte.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn unpack_u2_avx512(v: __m128i) -> __m512i {
    // Broadcast the 128 bit to all lanes in the 512 bit register.
    let b = _mm512_broadcast_i32x4(v);
    // Shift each 128 bit lane to the right to expose bits [0, 2, 4, 6]
    let shifted = _mm512_srlv_epi64(b, _mm512_set_epi64(6, 6, 4, 4, 2, 2, 0, 0));
    // Mask each dimensio ndown to 2 bits.
    _mm512_and_si512(shifted, _mm512_set1_epi8(3))
}

/// Unpack 256 bits of u4 input (passed as 512) as 64 bytes with 1 dimension per byte.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn unpack_u4_avx512(v: __m512i) -> __m512i {
    // Broadcast to place the low 128 bits end-to-end as 256, do the same with the next 128.
    let b = _mm512_shuffle_i64x2::<0b0101_0000>(v, v);
    // Shift each 128 bit lane right [0, 4, 0, 4] to align the low nibbles correctly.
    let shifted = _mm512_srlv_epi64(b, _mm512_set_epi64(4, 4, 0, 0, 4, 4, 0, 0));
    // Mask each dimension down to 4 bits.
    _mm512_and_si512(shifted, _mm512_set1_epi8(0xf))
}
