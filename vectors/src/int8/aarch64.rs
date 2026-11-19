//! aarch64 kernel implementations.

/// NEON-specific kernel implementations.
pub mod neon {
    use std::arch::aarch64::{vaddvq_s32, vdotq_s32, vdupq_n_s32, vld1q_s8};

    // Compute (anorm, bnorm, ip) of `a` and `b`
    #[target_feature(enable = "dotprod")]
    unsafe fn norms_and_ip(a: &[i8], b: &[i8]) -> (i32, i32, i32) {
        let (ahead, atail) = a.as_chunks::<16>();
        let (bhead, btail) = b.as_chunks::<16>();
        let (anorm4, bnorm4, ip4) = ahead.iter().zip(bhead.iter()).fold(
            (vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)),
            |acc, (a, b)| unsafe {
                let a = vld1q_s8(a.as_ptr());
                let b = vld1q_s8(b.as_ptr());
                (
                    vdotq_s32(acc.0, a, a),
                    vdotq_s32(acc.1, b, b),
                    vdotq_s32(acc.2, a, b),
                )
            },
        );

        atail.iter().zip(btail.iter()).fold(
            (vaddvq_s32(anorm4), vaddvq_s32(bnorm4), vaddvq_s32(ip4)),
            |acc, (&a, &b)| {
                let a = a as i32;
                let b = b as i32;
                (acc.0 + a * a, acc.1 + b * b, acc.2 + a * b)
            },
        )
    }

    // XXX this formulation makes me think we might want to allow precomputing norms.
    // reranking probably doesn't pull enough for it to matter. we'll need special asymmetric
    // support that allows us to skip one of the norm computations.
    // XXX
    // * 3 inner products is the optimal strategy on aarch64.
    // * VNNI works on x86_64 but due to i8xu8 semantics I also need sum(a) and sum(b).
    //   - 5 dots per 64 dims.
    // * If without VNNI it's widen, multiply-accumulate.
    //   - 2 widens, 3 multi-accum per 32 dims.
    // If I keep norms the problems get simpler.
    // * aarch64 reduces to inner product plus two norms.
    // * x86_64 + VNNI is similar but still needs to sign flip one vector and sum the other.
    //     summing can potentially be for the "query" vector and compute once for ADC.
    // * x86_64 - VNNI is still bad but at least avoids additional accumulation steps.
    // XXX try each of the scalar implementations with target_feature and check generated output.
    #[target_feature(enable = "dotprod")]
    unsafe fn l2_distance_sq(a: &[i8], b: &[i8]) -> i32 {
        let (anorm, bnorm, ip) = unsafe { norms_and_ip(a, b) };
        anorm + bnorm - 2 * ip
    }

    #[target_feature(enable = "dotprod")]
    unsafe fn inner_product(a: &[i8], b: &[i8]) -> i32 {
        let (ahead, atail) = a.as_chunks::<16>();
        let (bhead, btail) = b.as_chunks::<16>();
        let ip4 = ahead
            .iter()
            .zip(bhead.iter())
            .fold(vdupq_n_s32(0), |acc, (a, b)| unsafe {
                vdotq_s32(acc, vld1q_s8(a.as_ptr()), vld1q_s8(b.as_ptr()))
            });

        atail
            .iter()
            .zip(btail.iter())
            .fold(vaddvq_s32(ip4), |acc, (&a, &b)| {
                let a = a as i32;
                let b = b as i32;
                acc + a * b
            })
    }

    #[target_feature(enable = "dotprod")]
    unsafe fn cosine_similarity(a: &[i8], b: &[i8]) -> f64 {
        let (anorm, bnorm, ip) = unsafe { norms_and_ip(a, b) };
        ip as f64 / (anorm as f64 * bnorm as f64).sqrt()
    }
}
