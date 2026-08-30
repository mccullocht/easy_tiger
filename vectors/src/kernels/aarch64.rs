//! aarch64 specific implementations of kernel functions.

/// Neon-specific implementations of kernel functions.
pub mod neon {
    use std::arch::aarch64::{
        vaddlvq_u16, vaddq_u16, vandq_u8, vcntq_u8, vdupq_n_u16, veorq_u8, vld1q_u8_x2, vpaddlq_u8,
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
}
