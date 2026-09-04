pub mod neon {
    use std::arch::aarch64::{
        float32x4x4_t, uint8x16_t, uint8x16x4_t, vabsq_f32, vaddlvq_u16, vaddq_f32, vaddq_u16,
        vaddvq_f32, vandq_u8, vbslq_u32, vcntq_u8, vdupq_n_f32, vdupq_n_u8, vdupq_n_u16,
        vdupq_n_u32, vfmaq_f32, vld1q_f32, vld1q_f32_x4, vld1q_u8, vorrq_u8, vpaddlq_u8,
        vqtbl1q_u8, vqtbl4q_u8, vreinterpretq_f32_u32, vreinterpretq_u8_f32, vreinterpretq_u32_u8,
        vshlq_n_u32, vshrq_n_u8, vshrq_n_u32, vst1q_f32, vst1q_u8,
    };

    #[inline]
    pub fn quantize_and_pack(v: &[f32], out: &mut [u8]) -> u32 {
        let (vhead, vtail) = v.as_chunks::<128>();
        let (ohead, otail) = out.split_at_mut(vhead.len() * 16);
        let ohead = ohead.as_chunks_mut::<16>().0;

        let component_sum = unsafe {
            let mut component_sum = vdupq_n_u16(0);
            for (vc, o) in vhead.iter().zip(ohead.iter_mut()) {
                let g = vc.as_chunks::<16>().0;
                let mut p = vshrq_n_u8::<7>(pack_group(vld1q_f32_x4(g[0].as_ptr())));
                p = vorrq_u8(p, vshrq_n_u8::<6>(pack_group(vld1q_f32_x4(g[1].as_ptr()))));
                p = vorrq_u8(p, vshrq_n_u8::<5>(pack_group(vld1q_f32_x4(g[2].as_ptr()))));
                p = vorrq_u8(p, vshrq_n_u8::<4>(pack_group(vld1q_f32_x4(g[3].as_ptr()))));
                p = vorrq_u8(p, vshrq_n_u8::<3>(pack_group(vld1q_f32_x4(g[4].as_ptr()))));
                p = vorrq_u8(p, vshrq_n_u8::<2>(pack_group(vld1q_f32_x4(g[5].as_ptr()))));
                p = vorrq_u8(p, vshrq_n_u8::<1>(pack_group(vld1q_f32_x4(g[6].as_ptr()))));
                p = vorrq_u8(p, pack_group(vld1q_f32_x4(g[7].as_ptr())));

                component_sum = vaddq_u16(component_sum, vpaddlq_u8(vcntq_u8(p)));
                vst1q_u8(o.as_mut_ptr(), p);
            }
            vaddlvq_u16(component_sum)
        };

        component_sum + crate::rabitq::scalar::quantize_and_pack(vtail, otail)
    }

    #[inline(always)]
    fn pack_group(g: float32x4x4_t) -> uint8x16_t {
        unsafe {
            let g = uint8x16x4_t(
                vreinterpretq_u8_f32(g.0),
                vreinterpretq_u8_f32(g.1),
                vreinterpretq_u8_f32(g.2),
                vreinterpretq_u8_f32(g.3),
            );
            let shuf_mask =
                vld1q_u8([3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63].as_ptr());
            // Keep only the sign bit of each component; the caller shifts it into place.
            vandq_u8(vqtbl4q_u8(g, shuf_mask), vdupq_n_u8(0x80))
        }
    }

    #[inline]
    pub fn l1_norm_scaled(v: &[f32], scale: f32) -> f32 {
        let (head, tail) = v.as_chunks::<16>();
        let sum = unsafe {
            let scale = vdupq_n_f32(scale);
            let mut sum = [vdupq_n_f32(0.0); 4];
            for x in head.iter() {
                sum[0] = vfmaq_f32(sum[0], vabsq_f32(vld1q_f32(x.as_ptr())), scale);
                sum[1] = vfmaq_f32(sum[1], vabsq_f32(vld1q_f32(x.as_ptr().add(4))), scale);
                sum[2] = vfmaq_f32(sum[2], vabsq_f32(vld1q_f32(x.as_ptr().add(8))), scale);
                sum[3] = vfmaq_f32(sum[3], vabsq_f32(vld1q_f32(x.as_ptr().add(12))), scale);
            }
            vaddvq_f32(vaddq_f32(
                vaddq_f32(sum[0], sum[1]),
                vaddq_f32(sum[2], sum[3]),
            ))
        };

        (sum + tail.iter().map(|x| x.abs() * scale).sum::<f32>()) / (v.len() as f32).sqrt()
    }

    #[inline]
    pub fn decode(v: &[u8], magnitude: f32, center: Option<&[f32]>, out: &mut [f32]) {
        let (vhead, vtail) = v.as_chunks::<16>();
        let (ohead, otail) = out.as_chunks_mut::<128>();

        let mag = unsafe { vdupq_n_u32(magnitude.to_bits()) };
        let s =
            unsafe { vld1q_u8([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15].as_ptr()) };
        let m32 = unsafe { vdupq_n_u32(0x80000000) };

        macro_rules! unpack_store4 {
            ($v:expr, $o:expr, $off:expr, $shift:literal) => {
                vst1q_f32(
                    $o.add($off),
                    vreinterpretq_f32_u32(vbslq_u32(m32, vshlq_n_u32::<$shift>($v), mag)),
                );
            };
            ($v:expr, $c:expr, $o:expr, $off:expr, $shift:literal) => {
                vst1q_f32(
                    $o.add($off),
                    vaddq_f32(
                        vreinterpretq_f32_u32(vbslq_u32(m32, vshlq_n_u32::<$shift>($v), mag)),
                        vld1q_f32($c.add($off)),
                    ),
                );
            };
        }

        macro_rules! unpack_store16 {
            ($v:expr, $o:expr, $g:literal) => {
                unpack_store4!($v, $o, $g * 16 + 0, 31);
                unpack_store4!($v, $o, $g * 16 + 4, 23);
                unpack_store4!($v, $o, $g * 16 + 8, 15);
                unpack_store4!($v, $o, $g * 16 + 12, 7);
            };
            ($v:expr, $c:expr, $o:expr, $g:literal) => {
                unpack_store4!($v, $c, $o, $g * 16 + 0, 31);
                unpack_store4!($v, $c, $o, $g * 16 + 4, 23);
                unpack_store4!($v, $c, $o, $g * 16 + 8, 15);
                unpack_store4!($v, $c, $o, $g * 16 + 12, 7);
            };
        }

        if let Some(center) = center {
            let (chead, ctail) = center.as_chunks::<128>();
            for ((v, c), o) in vhead.iter().zip(chead.iter()).zip(ohead.iter_mut()) {
                unsafe {
                    let v = vreinterpretq_u32_u8(vqtbl1q_u8(vld1q_u8(v.as_ptr()), s));
                    unpack_store16!(v, c.as_ptr(), o.as_mut_ptr(), 0);
                    unpack_store16!(vshrq_n_u32::<1>(v), c.as_ptr(), o.as_mut_ptr(), 1);
                    unpack_store16!(vshrq_n_u32::<2>(v), c.as_ptr(), o.as_mut_ptr(), 2);
                    unpack_store16!(vshrq_n_u32::<3>(v), c.as_ptr(), o.as_mut_ptr(), 3);
                    unpack_store16!(vshrq_n_u32::<4>(v), c.as_ptr(), o.as_mut_ptr(), 4);
                    unpack_store16!(vshrq_n_u32::<5>(v), c.as_ptr(), o.as_mut_ptr(), 5);
                    unpack_store16!(vshrq_n_u32::<6>(v), c.as_ptr(), o.as_mut_ptr(), 6);
                    unpack_store16!(vshrq_n_u32::<7>(v), c.as_ptr(), o.as_mut_ptr(), 7);
                }
            }
            crate::rabitq::scalar::decode(vtail, magnitude, Some(ctail), otail);
        } else {
            for (v, o) in vhead.iter().zip(ohead.iter_mut()) {
                unsafe {
                    let v = vreinterpretq_u32_u8(vqtbl1q_u8(vld1q_u8(v.as_ptr()), s));
                    unpack_store16!(v, o.as_mut_ptr(), 0);
                    unpack_store16!(vshrq_n_u32::<1>(v), o.as_mut_ptr(), 1);
                    unpack_store16!(vshrq_n_u32::<2>(v), o.as_mut_ptr(), 2);
                    unpack_store16!(vshrq_n_u32::<3>(v), o.as_mut_ptr(), 3);
                    unpack_store16!(vshrq_n_u32::<4>(v), o.as_mut_ptr(), 4);
                    unpack_store16!(vshrq_n_u32::<5>(v), o.as_mut_ptr(), 5);
                    unpack_store16!(vshrq_n_u32::<6>(v), o.as_mut_ptr(), 6);
                    unpack_store16!(vshrq_n_u32::<7>(v), o.as_mut_ptr(), 7);
                }
            }
            crate::rabitq::scalar::decode(vtail, magnitude, None, otail);
        }
    }
}
