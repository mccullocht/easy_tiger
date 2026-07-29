use super::scalar::Scalar;
use super::Kernel;

use std::arch::aarch64::{
    int8x16_t, vaddlvq_s8, vandq_s8, vbslq_s8, vcntq_s8, vdupq_n_u8, veorq_s8, vld1q_s8, vmvnq_s8,
    vorrq_s8, vshlq_n_s8, vshrq_n_s8, vsubq_s8,
};

/// NEON kernel for QuIVer operations.
///
/// This requires the `dotprod` feature.
pub struct Neon;

impl Neon {
    // NB: return i8/s8 because it's more convenient for future combination, this is still all
    // being manipulated as bit patterns up until we popcount.
    #[inline]
    fn bitplane_split256(v: &[u8; 32]) -> (int8x16_t, int8x16_t) {
        unsafe {
            let a = vld1q_s8(v.as_ptr() as *const i8);
            let b = vld1q_s8(v.as_ptr().add(16) as *const i8);
            let m = vdupq_n_u8(0x55);

            let sgn = vbslq_s8(m, vshrq_n_s8::<1>(a), b);
            let mag = vbslq_s8(m, a, vshlq_n_s8::<1>(b));
            (sgn, mag)
        }
    }
}

impl Kernel for Neon {
    fn symmetric_distance(a: &[u8], b: &[u8]) -> i32 {
        // XXX accumulate in SIMD. consider SDOT as part of the strategy.
        let (achunks, arem) = a.as_chunks::<32>();
        let (bchunks, brem) = b.as_chunks::<32>();
        let mut dist = achunks
            .iter()
            .map(Self::bitplane_split256)
            .zip(bchunks.iter().map(Self::bitplane_split256))
            .map(|((a_s, a_m), (b_s, b_m))| {
                unsafe {
                    let s_x = veorq_s8(a_s, b_s); // signs mismatch
                    let s_m = vmvnq_s8(s_x); // signs match
                    let m_x = veorq_s8(a_m, b_m); // magnitudes mismatch
                    let m_s = vandq_s8(a_m, b_m); // both magnitudes strong
                    let m_w = vmvnq_s8(vorrq_s8(a_m, b_m)); // both magnitudes weak

                    let weak = vsubq_s8(vcntq_s8(vandq_s8(m_w, s_m)), vcntq_s8(vandq_s8(m_w, s_x)));
                    let mixed =
                        vsubq_s8(vcntq_s8(vandq_s8(m_x, s_m)), vcntq_s8(vandq_s8(m_x, s_x)));
                    let strong =
                        vsubq_s8(vcntq_s8(vandq_s8(m_s, s_m)), vcntq_s8(vandq_s8(m_s, s_x)));

                    vaddlvq_s8(weak) as i32
                        + vaddlvq_s8(mixed) as i32 * 2
                        + vaddlvq_s8(strong) as i32 * 4
                }
            })
            .sum::<i32>();
        if !arem.is_empty() {
            dist += Scalar::symmetric_distance(arem, brem);
        }
        dist
    }
}
