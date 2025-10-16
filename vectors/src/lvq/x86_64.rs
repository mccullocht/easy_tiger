#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::{
    __m512i, _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT, _mm_add_ps, _mm_and_si128,
    _mm_andnot_si128, _mm_broadcastd_epi32, _mm_bsrli_si128, _mm_cmpeq_epi8, _mm_cvtps_pd,
    _mm_cvtsd_f64, _mm_fmadd_pd, _mm_fmadd_ps, _mm_hadd_pd, _mm_hadd_ps, _mm_hsub_pd, _mm_hsub_ps,
    _mm_loadu_epi8, _mm_loadu_epi32, _mm_loadu_epi64, _mm_mask_storeu_epi8, _mm_maskz_loadu_epi8,
    _mm_mul_pd, _mm_mul_ps, _mm_or_si128, _mm_set1_epi8, _mm_set1_epi16, _mm_set1_epi64x,
    _mm_set1_pd, _mm_set1_ps, _mm_shuffle_epi8, _mm_sllv_epi64, _mm_srli_epi64, _mm_srlv_epi32,
    _mm_sub_ps, _mm_unpacklo_epi8, _mm256_add_ps, _mm256_and_si256, _mm256_broadcastsi128_si256,
    _mm256_castps256_ps128, _mm256_cvtepi8_epi16, _mm256_cvtepi16_epi8, _mm256_cvtepu8_epi16,
    _mm256_extractf32x4_ps, _mm256_fmadd_ps, _mm256_loadu_epi8, _mm256_loadu_epi16,
    _mm256_loadu_epi32, _mm256_mask_storeu_epi8, _mm256_mask_storeu_epi16,
    _mm256_maskz_loadu_epi16, _mm256_mul_ps, _mm256_or_si256, _mm256_permutevar8x32_epi32,
    _mm256_permutexvar_epi32, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set1_ps,
    _mm256_shuffle_epi8, _mm256_sllv_epi16, _mm256_srlv_epi16, _mm256_sub_ps, _mm512_add_epi32,
    _mm512_add_ps, _mm512_and_epi32, _mm512_and_si512, _mm512_castps512_ps256,
    _mm512_cvtepi16_epi32, _mm512_cvtepi32_epi16, _mm512_cvtepu16_epi32, _mm512_cvtepu32_ps,
    _mm512_div_ps, _mm512_dpbusd_epi32, _mm512_dpwssd_epi32, _mm512_extractf32x8_ps,
    _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_mask_mul_ps, _mm512_mask_sub_ps, _mm512_maskz_add_ps,
    _mm512_maskz_cvtepu32_ps, _mm512_maskz_cvtps_epu32, _mm512_maskz_loadu_epi8,
    _mm512_maskz_loadu_ps, _mm512_max_ps, _mm512_min_ps, _mm512_mul_ps, _mm512_popcnt_epi32,
    _mm512_reduce_add_epi32, _mm512_reduce_add_ps, _mm512_reduce_max_ps, _mm512_reduce_min_ps,
    _mm512_roundscale_ps, _mm512_set1_epi8, _mm512_set1_epi32, _mm512_set1_ps, _mm512_srli_epi64,
    _mm512_sub_ps, _mm512_unpackhi_epi8, _mm512_unpacklo_epi8,
};

use super::{LAMBDA, MINIMUM_MSE_GRID, PrimaryVector, TwoLevelVector, VectorStats};

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

    let delta = (upper - lower) / ((1 << B) - 1) as f32;
    let delta_inv = _mm512_set1_ps(delta.recip());
    let lower = _mm512_set1_ps(lower);
    let upper = _mm512_set1_ps(upper);
    let mut component_sum = _mm512_set1_epi32(0);
    let out_chunks = (B * 16) / 8;
    for (c, o) in v.chunks(16).zip(out.chunks_mut(out_chunks)) {
        let mask = u16::MAX >> (16 - c.len());
        let mut v = _mm512_maskz_loadu_ps(mask, c.as_ptr());
        // NB: we'll clamp to the lower bound later by converting to unsigned while saturating.
        v = _mm512_min_ps(v, upper);
        v = _mm512_sub_ps(v, lower);
        v = _mm512_mul_ps(v, delta_inv);
        let q = _mm512_maskz_cvtps_epu32(mask, v);
        component_sum = _mm512_add_epi32(component_sum, q);
        pack::<B>(q, o);
    }
    _mm512_reduce_add_epi32(component_sum) as u32
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,avx")]
pub unsafe fn lvq2_quantize_and_pack<const B1: usize, const B2: usize>(
    v: &[f32],
    lower: f32,
    upper: f32,
    primary: &mut [u8],
    residual_interval: f32,
    residual: &mut [u8],
) -> u32 {
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

    for (vc, (pc, rc)) in v.chunks(16).zip(
        primary
            .chunks_mut(p_chunk_size)
            .zip(residual.chunks_mut(r_chunk_size)),
    ) {
        let vmask = u16::MAX >> (16 - vc.len());
        let v = _mm512_maskz_loadu_ps(vmask, vc.as_ptr());
        // NB: we'll clamp to the lower bound later by converting to unsigned while saturating.
        let mut ps = _mm512_min_ps(v, p_upper);
        ps = _mm512_sub_ps(ps, p_lower);
        ps = _mm512_mul_ps(ps, p_delta_inv);
        let pi = _mm512_maskz_cvtps_epu32(vmask, ps);
        p_sum = _mm512_add_epi32(p_sum, pi);
        pack::<B1>(pi, pc);

        // Compute the residual delta from the dequantized value.
        let mut rs = _mm512_sub_ps(v, _mm512_fmadd_ps(_mm512_cvtepu32_ps(pi), p_delta, p_lower));
        rs = _mm512_min_ps(rs, r_upper);
        rs = _mm512_sub_ps(rs, r_lower);
        rs = _mm512_mul_ps(rs, r_delta_inv);
        let ri = _mm512_maskz_cvtps_epu32(vmask, rs);
        pack::<B2>(ri, rc);
    }

    _mm512_reduce_add_epi32(p_sum) as u32
}

#[inline]
unsafe fn pack<const N: usize>(v: __m512i, out: &mut [u8]) {
    match N {
        1 => pack1(v, out),
        2 => todo!("allowed but unused"),
        4 => pack4(v, out),
        8 => pack8(v, out),
        12 => pack12(v, out),
        16 => pack16(v, out),
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2")]
#[inline]
unsafe fn pack12(v: __m512i, out: &mut [u8]) {
    let mut p = _mm512_cvtepi32_epi16(v);
    // shift all the odd lanes left by 4.
    p = _mm256_sllv_epi16(p, _mm256_set1_epi32(0x0004_0000));
    // shuffle the even lanes into byte pattern where they are separated by one byte instead of two.
    let e_shuffle =
        _mm_loadu_epi8([0i8, 1, -1, 4, 5, -1, 8, 9, -1, 12, 13, -1, -1, -1, -1, -1].as_ptr());
    let e = _mm256_shuffle_epi8(p, _mm256_broadcastsi128_si256(e_shuffle));
    // shuffle the odd lanes similarly but leave an empty 1 byte prefix.
    let o_shuffle =
        _mm_loadu_epi8([-1i8, 2, 3, -1, 6, 7, -1, 10, 11, -1, 14, 15, -1, -1, -1, -1].as_ptr());
    let o = _mm256_shuffle_epi8(p, _mm256_broadcastsi128_si256(o_shuffle));

    // Now we have 2 128 bit lanes where the value is packed in the bottom 96 bits of each.
    let mut m = _mm256_or_si256(e, o);
    m = _mm256_permutexvar_epi32(_mm256_loadu_epi32([0i32, 1, 2, 4, 5, 6, 0, 0].as_ptr()), m);
    _mm256_mask_storeu_epi8(out.as_mut_ptr() as *mut i8, u32::MAX >> (32 - out.len()), m);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn pack16(v: __m512i, out: &mut [u8]) {
    _mm256_mask_storeu_epi16(
        out.as_mut_ptr() as *mut i16,
        u16::MAX >> (16 - (out.len() / 2)),
        _mm512_cvtepi32_epi16(v),
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
        4 => {
            let mut dot = _mm512_set1_epi32(0);
            let nibble_mask = _mm512_set1_epi8(0xf);
            for (ac, bc) in a.chunks(64).zip(b.chunks(64)) {
                let mask = u64::MAX >> (64 - ac.len());
                let av = _mm512_maskz_loadu_epi8(mask, ac.as_ptr() as *const i8);
                let av_even = _mm512_and_si512(av, nibble_mask);
                let av_odd = _mm512_and_si512(_mm512_srli_epi64::<4>(av), nibble_mask);
                let bv = _mm512_maskz_loadu_epi8(mask, bc.as_ptr() as *const i8);
                let bv_even = _mm512_and_si512(bv, nibble_mask);
                let bv_odd = _mm512_and_si512(_mm512_srli_epi64::<4>(bv), nibble_mask);

                dot = _mm512_dpbusd_epi32(dot, av_even, bv_even);
                dot = _mm512_dpbusd_epi32(dot, av_odd, bv_odd);
            }
            _mm512_reduce_add_epi32(dot) as u32
        }
        8 => {
            let zero = _mm512_set1_epi8(0);
            let mut dot = _mm512_set1_epi32(0);
            for (ac, bc) in a.chunks(64).zip(b.chunks(64)) {
                let mask = u64::MAX >> (64 - ac.len());
                let av = _mm512_maskz_loadu_epi8(mask, ac.as_ptr() as *const i8);
                let av_lo = _mm512_unpacklo_epi8(av, zero);
                let av_hi = _mm512_unpackhi_epi8(av, zero);
                let bv = _mm512_maskz_loadu_epi8(mask, bc.as_ptr() as *const i8);
                let bv_lo = _mm512_unpacklo_epi8(bv, zero);
                let bv_hi = _mm512_unpackhi_epi8(bv, zero);

                dot = _mm512_dpwssd_epi32(dot, av_lo, bv_lo);
                dot = _mm512_dpwssd_epi32(dot, av_hi, bv_hi);
            }
            _mm512_reduce_add_epi32(dot) as u32
        }
        _ => super::scalar::dot_u8::<B>(a, b),
    }
}

#[target_feature(enable = "avx512vnni,avx512bw,avx512vl,avx512f")]
pub unsafe fn lvq1_f32_dot_unnormalized<const B: usize>(
    query: &[f32],
    doc: &PrimaryVector<'_, B>,
) -> f64 {
    let delta = _mm512_set1_ps(doc.delta);
    let lower = _mm512_set1_ps(doc.header.lower);
    let chunk_size = (B * 16).div_ceil(8);
    let mut dot = _mm512_set1_ps(0.0);
    for (q, d) in query.chunks(16).zip(doc.vector.chunks(chunk_size)) {
        let mask = u16::MAX >> (16 - q.len());
        let qv = _mm512_maskz_loadu_ps(mask, q.as_ptr());
        let qpv = _mm512_maskz_cvtepu32_ps(mask, unpack::<B>(d));
        let dv = _mm512_fmadd_ps(qpv, delta, lower);
        dot = _mm512_fmadd_ps(qv, dv, dot);
    }
    _mm512_reduce_add_ps(dot).into()
}

#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn lvq2_dot_unnormalized<const B1: usize, const B2: usize>(
    a: &TwoLevelVector<'_, B1, B2>,
    b: &TwoLevelVector<'_, B1, B2>,
) -> f64 {
    super::scalar::lvq2_dot_unnormalized::<B1, B2>(a, b)
}

#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn lvq2_f32_dot_unnormalized<const B1: usize, const B2: usize>(
    query: &[f32],
    doc: &TwoLevelVector<'_, B1, B2>,
) -> f64 {
    let p_chunk_size = (B1 * 16).div_ceil(8);
    let p_delta = _mm512_set1_ps(doc.primary.delta);
    let p_lower = _mm512_set1_ps(doc.primary.header.lower);
    let r_chunk_size = (B2 * 16).div_ceil(8);
    let r_delta = _mm512_set1_ps(doc.delta);
    let r_lower = _mm512_set1_ps(doc.lower);

    let mut dot = _mm512_set1_ps(0.0);
    for (q, (p, r)) in query.chunks(16).zip(
        doc.primary
            .vector
            .chunks(p_chunk_size)
            .zip(doc.vector.chunks(r_chunk_size)),
    ) {
        let qmask = u16::MAX >> (16 - q.len());
        let qv = _mm512_maskz_loadu_ps(qmask, q.as_ptr());

        let qpv = unpack::<B1>(p);
        let pv = _mm512_fmadd_ps(_mm512_cvtepu32_ps(qpv), p_delta, p_lower);
        let qrv = unpack::<B2>(r);
        let rv = _mm512_fmadd_ps(_mm512_cvtepu32_ps(qrv), r_delta, r_lower);
        let dv = _mm512_maskz_add_ps(qmask, pv, rv);

        dot = _mm512_fmadd_ps(qv, dv, dot);
    }
    _mm512_reduce_add_ps(dot).into()
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn unpack<const N: usize>(bytes: &[u8]) -> __m512i {
    match N {
        1 => unpack1(bytes),
        2 => unpack2(bytes),
        4 => unpack4(bytes),
        8 => unpack8(bytes),
        12 => unpack12(bytes),
        16 => unpack16(bytes),
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
unsafe fn unpack2(bytes: &[u8]) -> __m512i {
    let mut v = _mm_maskz_loadu_epi8((1 << bytes.len()) - 1, bytes.as_ptr() as *const i8);
    // To read 16 dimensions we won't load any more than 32 bits, so broadcast and we can arrange
    // such that neighboring dimensions appear in every 4th lane (e.g. 0, 4, 8, 12, 1, ...).
    v = _mm_broadcastd_epi32(v);
    v = _mm_srlv_epi32(v, _mm_loadu_epi32([0i32, 2, 4, 6].as_ptr()));
    v = _mm_and_si128(v, _mm_set1_epi8(0x3));
    v = _mm_shuffle_epi8(
        v,
        _mm_loadu_epi8([0i8, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15].as_ptr()),
    );
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

#[target_feature(enable = "avx512f,avx512bw,avx2,avx512vl")]
unsafe fn unpack12(bytes: &[u8]) -> __m512i {
    // This should load no more than 24 bytes. Most instructions operate in 128 bit lanes so we will
    // permute the input to place the low 12 bytes in the first 128-bit lane and the rest in the second.
    let mut v = _mm256_maskz_loadu_epi16(
        u16::MAX >> ((32 - bytes.len()) / 2),
        bytes.as_ptr() as *const i16,
    );
    v = _mm256_permutevar8x32_epi32(v, _mm256_loadu_epi32([0i32, 1, 2, 7, 3, 4, 5, 7].as_ptr()));
    v = _mm256_shuffle_epi8(
        v,
        _mm256_loadu_epi8(
            [
                0i8, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 0, 1, 1, 2, 3, 4, 4, 5, 6, 7,
                7, 8, 9, 10, 10, 11,
            ]
            .as_ptr(),
        ),
    );
    v = _mm256_srlv_epi16(v, _mm256_set1_epi32(0x0004_0000));
    v = _mm256_and_si256(v, _mm256_set1_epi16(0xfff));
    _mm512_cvtepi16_epi32(v)
}

#[target_feature(enable = "avx512f,avx512bw,avx2")]
#[inline]
unsafe fn unpack16(bytes: &[u8]) -> __m512i {
    let v = _mm256_maskz_loadu_epi16(
        u16::MAX >> ((32 - bytes.len()) / 2),
        bytes.as_ptr() as *const i16,
    );
    _mm512_cvtepu16_epi32(v)
}
