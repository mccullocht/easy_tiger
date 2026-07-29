use super::scalar::Scalar;
use super::Kernel;

use std::arch::aarch64::{
    float32x4_t, int8x16_t, uint32x4_t, uint8x16_t, uint8x16x4_t, vabsq_f32, vaddlvq_s8, vaddq_f32,
    vaddq_s8, vaddq_u32, vaddvq_f32, vaddvq_u32, vandq_s8, vandq_u32, vbslq_f32, vbslq_s8,
    vcgtq_f32, vcntq_s8, vdupq_n_f32, vdupq_n_s8, vdupq_n_u32, vdupq_n_u8, veorq_s8, vld1q_f32,
    vld1q_s8, vld1q_u8, vmvnq_s8, vmvnq_u32, vorrq_s8, vorrq_u32, vorrq_u8, vqtbl4q_u8,
    vreinterpretq_u8_u32, vshlq_n_s8, vshlq_n_u8, vshrq_n_s8, vst1q_u8, vsubq_s8,
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

struct QuantizationState {
    tau: float32x4_t,
    zero: float32x4_t,

    sign_mask: uint32x4_t,
    mag_mask: uint32x4_t,

    weak_sum: float32x4_t,
    strong_sum: float32x4_t,
    strong_count: uint32x4_t,

    shuf_mask: uint8x16_t,
}

impl QuantizationState {
    fn new(tau: f32) -> Self {
        unsafe {
            Self {
                tau: vdupq_n_f32(tau),
                zero: vdupq_n_f32(0.0),
                sign_mask: vdupq_n_u32(2),
                mag_mask: vdupq_n_u32(1),
                weak_sum: vdupq_n_f32(0.0),
                strong_sum: vdupq_n_f32(0.0),
                strong_count: vdupq_n_u32(0),
                shuf_mask: vld1q_u8(
                    [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60].as_ptr(),
                ),
            }
        }
    }

    #[inline]
    fn quantize16(&mut self, v: &[f32; 16]) -> uint8x16_t {
        unsafe {
            let q0 = self.quantize4(vld1q_f32(v.as_ptr()));
            let q1 = self.quantize4(vld1q_f32(v.as_ptr().add(4)));
            let q2 = self.quantize4(vld1q_f32(v.as_ptr().add(8)));
            let q3 = self.quantize4(vld1q_f32(v.as_ptr().add(12)));
            vqtbl4q_u8(
                uint8x16x4_t(
                    vreinterpretq_u8_u32(q0),
                    vreinterpretq_u8_u32(q1),
                    vreinterpretq_u8_u32(q2),
                    vreinterpretq_u8_u32(q3),
                ),
                self.shuf_mask,
            )
        }
    }

    #[inline]
    fn quantize4(&mut self, v: float32x4_t) -> uint32x4_t {
        unsafe {
            let s = vcgtq_f32(v, self.zero);
            let v = vabsq_f32(v);
            let m = vcgtq_f32(v, self.tau);

            self.strong_count = vaddq_u32(self.strong_count, vandq_u32(self.mag_mask, m));
            self.strong_sum = vaddq_f32(self.strong_sum, vbslq_f32(m, v, self.zero));
            self.weak_sum = vaddq_f32(self.weak_sum, vbslq_f32(vmvnq_u32(m), v, self.zero));

            vorrq_u32(vandq_u32(self.sign_mask, s), vandq_u32(self.mag_mask, m))
        }
    }

    #[inline]
    fn header_sums(&self) -> (f32, f32, u32) {
        unsafe {
            (
                vaddvq_f32(self.weak_sum),
                vaddvq_f32(self.strong_sum),
                vaddvq_u32(self.strong_count),
            )
        }
    }
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
    fn tau(v: &[f32]) -> f32 {
        let (c, r) = v.as_chunks::<4>();
        let c_sum = unsafe {
            vaddvq_f32(
                c.iter()
                    .map(|x| vabsq_f32(vld1q_f32(x.as_ptr())))
                    .fold(vdupq_n_f32(0.0), |a, x| vaddq_f32(a, x)),
            )
        };
        let r_sum = r.iter().copied().map(f32::abs).sum::<f32>();
        (c_sum + r_sum) / v.len() as f32
    }

    #[inline]
    fn quantize(v: &[f32], tau: f32, out: &mut [u8]) -> (f32, f32, u32) {
        // Quantize 64 dimensions at a time to pack into a single 128 bit register.
        let (cv, rv) = v.as_chunks::<64>();
        let (co, ro) = out.as_chunks_mut::<16>();
        let mut state = QuantizationState::new(tau);

        unsafe {
            for (v, o) in cv.iter().zip(co.iter_mut()) {
                let parts = v.as_ref().as_chunks::<16>().0;
                let mut q = state.quantize16(&parts[0]);
                q = vorrq_u8(q, vshlq_n_u8::<2>(state.quantize16(&parts[1])));
                q = vorrq_u8(q, vshlq_n_u8::<4>(state.quantize16(&parts[2])));
                q = vorrq_u8(q, vshlq_n_u8::<6>(state.quantize16(&parts[3])));
                vst1q_u8(o.as_mut_ptr(), q);
            }
        }

        let (mut weak_sum, mut strong_sum, mut strong_count) = state.header_sums();
        if !rv.is_empty() {
            let (w, s, c) = Scalar::quantize(rv, tau, ro);
            weak_sum += w;
            strong_sum += s;
            strong_count += c;
        }
        (weak_sum, strong_sum, strong_count)
    }

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
