pub mod neon {
    use std::arch::aarch64::{
        float32x4x4_t, uint8x16_t, uint8x16x4_t, vabsq_f32, vaddlvq_u16, vaddq_f32, vaddq_u16,
        vaddvq_f32, vcntq_u8, vdupq_n_f32, vdupq_n_u16, vfmaq_f32, vld1q_f32, vld1q_f32_x4,
        vld1q_u8, vorrq_u8, vpaddlq_u8, vqtbl4q_u8, vreinterpretq_u8_f32, vshrq_n_u8, vst1q_u8,
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
                vld1q_u8([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60].as_ptr());
            vqtbl4q_u8(g, shuf_mask)
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
}
