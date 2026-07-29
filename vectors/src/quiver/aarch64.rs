use super::scalar::Scalar;
use super::Kernel;

use std::arch::aarch64::{
    int8x16_t, vaddlvq_s8, vaddq_s8, vandq_s8, vbslq_s8, vcntq_s8, vdupq_n_s8, vdupq_n_u8,
    veorq_s8, vld1q_s8, vmvnq_s8, vorrq_s8, vshlq_n_s8, vshrq_n_s8, vsubq_s8,
};

unsafe extern "C" {
    /// vdotq_s32() intrinsic is unstable until rust 1.98 so call a C function instead.
    unsafe fn et_quiver_asymmetric_ip(
        query: *const i8,
        len: usize,
        doc: *const u8,
        table: *const i8,
    ) -> i32;
}

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
    #[inline]
    fn symmetric_distance(a: &[u8], b: &[u8]) -> i32 {
        // TODO: consider an SDOT accumulation strategy that is not vulnerable to over or
        // underflow.
        let (achunks, arem) = a.as_chunks::<32>();
        let (bchunks, brem) = b.as_chunks::<32>();
        let (w, m, s) = achunks
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

                    (weak, mixed, strong)
                }
            })
            .fold(
                // NB: accumulating this way may experience overflow.
                unsafe { (vdupq_n_s8(0), vdupq_n_s8(0), vdupq_n_s8(0)) },
                |acc, (w, m, s)| unsafe {
                    (vaddq_s8(acc.0, w), vaddq_s8(acc.1, m), vaddq_s8(acc.2, s))
                },
            );
        let mut dist =
            unsafe { vaddlvq_s8(w) as i32 + vaddlvq_s8(m) as i32 * 2 + vaddlvq_s8(s) as i32 * 4 };
        if !arem.is_empty() {
            dist += Scalar::symmetric_distance(arem, brem);
        }
        dist
    }

    #[inline]
    fn asymmetric_distance(q: &[i8], d: &[u8], weak: i8, strong: i8) -> i32 {
        let decode_table = [-weak, -strong, weak, strong];
        unsafe { et_quiver_asymmetric_ip(q.as_ptr(), q.len(), d.as_ptr(), decode_table.as_ptr()) }
    }
}
