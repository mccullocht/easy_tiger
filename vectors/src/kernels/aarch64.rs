//! aarch64 specific implementations of kernel functions.

/// Neon-specific implementations of kernel functions.
pub mod neon {
    use std::arch::aarch64::{
        vaddlvq_u16, vaddq_u16, vandq_u8, vcntq_u8, vdupq_n_u16, veorq_u8, vld1q_u8, vld1q_u8_x2,
        vld1q_u8_x4, vpaddlq_u8,
    };

    // XXX this should be adopted by the lvq implementation.
    #[inline]
    pub fn bitstring_inner_product<const S: bool>(a: &[u8], b: &[u8]) -> u32 {
        assert_eq!(a.len(), b.len());
        let (ahead, atail) = a.as_chunks::<32>();
        let (bhead, btail) = b.as_chunks::<32>();
        let ip = unsafe {
            let mut ip = [vdupq_n_u16(0); 2];
            for (a, b) in ahead.iter().zip(bhead.iter()) {
                let a = vld1q_u8_x2(a.as_ptr());
                let b = vld1q_u8_x2(b.as_ptr());

                let d = if S {
                    [veorq_u8(a.0, b.0), veorq_u8(a.1, b.1)]
                } else {
                    [vandq_u8(a.0, b.0), vandq_u8(a.1, b.1)]
                };

                // XXX single accumulator, sum the counts as u8, then widen and sum.
                ip[0] = vaddq_u16(ip[0], vpaddlq_u8(vcntq_u8(d[0])));
                ip[1] = vaddq_u16(ip[1], vpaddlq_u8(vcntq_u8(d[1])));
            }

            vaddlvq_u16(vaddq_u16(ip[0], ip[1]))
        };

        if atail.is_empty() {
            ip
        } else {
            ip + crate::kernels::scalar::bitstring_inner_product_tail::<S>(atail, btail)
        }
    }

    #[inline]
    pub fn turbo_4x1_inner_product(a: &[u8], b: &[u8]) -> u32 {
        let (ahead, atail) = a.as_chunks::<64>();
        let (bhead, btail) = b.split_at(ahead.len() * 16);
        let mut dot = unsafe {
            let mut bpdot = [vdupq_n_u16(0); 4];
            for (i, a) in ahead.iter().enumerate() {
                let av = vld1q_u8_x4(a.as_ptr());
                let bv = vld1q_u8(bhead.as_ptr().add(i * 16));
                bpdot[0] = vaddq_u16(bpdot[0], vpaddlq_u8(vcntq_u8(vandq_u8(av.0, bv))));
                bpdot[1] = vaddq_u16(bpdot[1], vpaddlq_u8(vcntq_u8(vandq_u8(av.1, bv))));
                bpdot[2] = vaddq_u16(bpdot[2], vpaddlq_u8(vcntq_u8(vandq_u8(av.2, bv))));
                bpdot[3] = vaddq_u16(bpdot[3], vpaddlq_u8(vcntq_u8(vandq_u8(av.3, bv))));
            }
            vaddlvq_u16(bpdot[0])
                + vaddlvq_u16(bpdot[1]) * 2
                + vaddlvq_u16(bpdot[2]) * 4
                + vaddlvq_u16(bpdot[3]) * 8
        };

        if !atail.is_empty() {
            dot += crate::kernels::scalar::turbo_4x1_inner_product_tail(atail, btail);
        }

        dot
    }
}
