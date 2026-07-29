use super::Kernel;

/// Scalar kernel for QuIVer operations implemented in safe Rust.
#[derive(Default)]
pub struct Scalar;

impl Scalar {
    const SGN_MASK: u64 = 0xAAAA_AAAA_AAAA_AAAA;
    const MAG_MASK: u64 = 0x5555_5555_5555_5555;

    const DECODE_TABLE: [[i8; 4]; 4] = [
        [1, 2, -1, -2], // weak, negative
        [2, 4, -2, -4], // strong, negative
        [-1, -1, 1, 2], // weak, positive
        [-2, -4, 2, 4], // strong, positive
    ];

    /// `v` contains vector bits interleaved as sign+mask.
    /// Return two 64-bit integers, one containing all the sign bits and another containing all the
    /// magnitude bits, aligned in a consistent order.
    #[inline]
    fn bitplane_split128(v: &[u8; 16]) -> (u64, u64) {
        let parts = v.as_chunks::<8>().0;
        let a = u64::from_le_bytes(parts[0]);
        let b = u64::from_le_bytes(parts[1]);

        // Dims from 'a' are in the even bits; dims from 'b' are in the odd bits.
        let sgn = ((a & Self::SGN_MASK) >> 1) | (b & Self::SGN_MASK);
        let mag = (a & Self::MAG_MASK) | ((b & Self::MAG_MASK) << 1);
        (sgn, mag)
    }
}

impl Kernel for Scalar {
    #[inline]
    fn tau(v: &[f32]) -> f32 {
        v.iter().copied().map(f32::abs).sum::<f32>() / v.len() as f32
    }

    #[inline]
    fn quantize(v: &[f32], tau: f32, out: &mut [u8]) -> (f32, f32, u32) {
        let mut weak_sum = 0.0f32;
        let mut strong_sum = 0.0f32;
        let mut strong_count = 0u32;
        out.fill(0);
        let mut packer = crate::lvq::packing::TurboPacker::<2>::new(out);
        for &d in v.iter() {
            let q = if d > 0.0 { 2u8 } else { 0u8 }
                | if d.abs() > tau {
                    strong_sum += d.abs();
                    strong_count += 1;
                    1u8
                } else {
                    weak_sum += d.abs();
                    0u8
                };
            packer.push(q);
        }
        (weak_sum, strong_sum, strong_count)
    }

    #[inline]
    fn symmetric_distance(a: &[u8], b: &[u8]) -> i32 {
        let (ac, ar) = a.as_chunks::<16>();
        let (bc, br) = b.as_chunks::<16>();

        let mut dist = ac
            .iter()
            .map(Self::bitplane_split128)
            .zip(bc.iter().map(Self::bitplane_split128))
            .map(|((a_s, a_m), (b_s, b_m))| {
                let s_x = a_s ^ b_s; // signs mismatch
                let m_x = a_m ^ b_m; // magnitudes mismatch
                let m_s = a_m & b_m; // both magnitudes strong
                let m_w = !(a_m | b_m); // both magnitudes weak

                // Use bitmask combinations + popcnt to count each of our 6 states: (all strong, all weak,
                // mixed) x (positive, negative).
                ((m_s & !s_x).count_ones() as i32 - (m_s & s_x).count_ones() as i32) * 4
                    + ((m_w & !s_x).count_ones() as i32 - (m_w & s_x).count_ones() as i32)
                    + ((m_x & !s_x).count_ones() as i32 - (m_x & s_x).count_ones() as i32) * 2
            })
            .sum::<i32>();
        if !ar.is_empty() {
            dist += ar
                .iter()
                .zip(br.iter())
                .map(|(&a, &b)| {
                    Self::DECODE_TABLE[(a & 3) as usize][(b & 3) as usize] as i32
                        + Self::DECODE_TABLE[((a >> 2) & 3) as usize][((b >> 2) & 3) as usize]
                            as i32
                        + Self::DECODE_TABLE[((a >> 4) & 3) as usize][((b >> 4) & 3) as usize]
                            as i32
                        + Self::DECODE_TABLE[(a >> 6) as usize][(b >> 6) as usize] as i32
                })
                .sum::<i32>();
        }
        dist
    }

    #[inline]
    fn asymmetric_distance(q: &[i8], d: &[u8], weak: i8, strong: i8) -> i32 {
        let table = [-weak as i32, -strong as i32, weak as i32, strong as i32];
        q.iter()
            .map(|x| *x as i32)
            .zip(crate::lvq::packing::TurboUnpacker::<2>::new(d).map(|x| table[x as usize]))
            .map(|(q, d)| q * d)
            .sum::<i32>()
    }
}
