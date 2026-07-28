//! QuiVer two-bit training free quantization: https://arxiv.org/html/2605.02171v1

use crate::{F32VectorCoder, QueryVectorDistance, VectorDistance};

#[derive(Default, Debug, Copy, Clone)]
struct MeanComputer {
    count: f32,
    mean: f32,
}

impl MeanComputer {
    fn add(&mut self, value: f32) {
        self.count += 1.0;
        self.mean += (value - self.mean) / self.count;
    }
}

struct Header {
    weak: f32,
    strong: f32,
    strong_count: u32,
}

impl Header {
    const LEN: usize = 12;

    fn split(data: &[u8]) -> ([u8; Self::LEN], &[u8]) {
        let (header, vector) = data.split_at(Self::LEN);
        (header.as_chunks::<{ Self::LEN }>().0[0], vector)
    }

    fn split_mut(data: &mut [u8]) -> (&mut [u8; Self::LEN], &mut [u8]) {
        let (header, vector) = data.split_at_mut(Self::LEN);
        (&mut header.as_chunks_mut::<{ Self::LEN }>().0[0], vector)
    }

    fn decode(raw: [u8; Self::LEN]) -> Self {
        let items = raw.as_ref().as_chunks::<4>().0;
        Self {
            weak: f32::from_le_bytes(items[0]),
            strong: f32::from_le_bytes(items[1]),
            strong_count: u32::from_le_bytes(items[2]),
        }
    }

    fn encode(&self, raw: &mut [u8; Self::LEN]) {
        let items = raw.as_mut().as_chunks_mut::<4>().0;
        items[0] = self.weak.to_le_bytes();
        items[1] = self.strong.to_le_bytes();
        items[2] = self.strong_count.to_le_bytes();
    }

    fn split_and_decode(data: &[u8]) -> (Header, &[u8]) {
        let (h, v) = Self::split(data);
        (Self::decode(h), v)
    }
}

/// Coder for QuiVer format.
///
/// The stored encoding contains 2 bits for every dimension packed into bytes.
///
/// The vector begins with additional terms defined by the `Header` struct.
#[derive(Default)]
pub struct Coder;

impl F32VectorCoder for Coder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let tau = vector.iter().copied().map(f32::abs).sum::<f32>() / vector.len() as f32;
        let mut strong = MeanComputer::default();
        let mut weak = MeanComputer::default();

        let (header_bytes, vector_bytes) = Header::split_mut(out);
        vector_bytes.fill(0);
        let mut packer = super::lvq::packing::TurboPacker::<2>::new(vector_bytes);
        for &v in vector.iter() {
            let q = if v > 0.0 { 2u8 } else { 0u8 }
                | if v.abs() > tau {
                    strong.add(v.abs());
                    1u8
                } else {
                    weak.add(v.abs());
                    0u8
                };
            packer.push(q);
        }
        Header {
            weak: weak.mean,
            strong: strong.mean,
            strong_count: strong.count as u32,
        }
        .encode(header_bytes);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions.div_ceil(4) + Header::LEN
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        let (header_bytes, vector_bytes) = Header::split(encoded);
        let header = Header::decode(header_bytes);
        let decode_table = [-header.weak, -header.strong, header.weak, header.strong];
        for (i, o) in super::lvq::packing::TurboUnpacker::<2>::new(vector_bytes).zip(out.iter_mut())
        {
            *o = decode_table[i as usize];
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        (byte_len - Header::LEN) / 4
    }
}

#[cfg(not(target_arch = "aarch64"))]
const SGN_MASK: u128 = 0x5555_5555_5555_5555_5555_5555_5555_5555;
#[cfg(not(target_arch = "aarch64"))]
const MAG_MASK: u128 = 0xAAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA;

/// Take 256 bits of interleaved (sign,magnitude) input and generate two 128 bit outputs that
/// contain all of the signs and all of the magnitudes in consistent order.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn bitplane_split256(v: [u8; 32]) -> (u128, u128) {
    let parts = v.as_chunks::<16>().0;
    let a = u128::from_le_bytes(parts[0]);
    let b = u128::from_le_bytes(parts[1]);

    // Dims from 'a' are in even bits; dims from 'b' are in odd bits.
    // XXX is this wrong???
    let sgn = (a & SGN_MASK) | ((b & SGN_MASK) << 1);
    let mag = ((a & MAG_MASK) >> 1) | (b & MAG_MASK);
    (sgn, mag)
}

/// Compute raw distance between two vectors with interleaved packed (sign,magnitude) values for
/// 256 bits of input.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn distance256(a: [u8; 32], b: [u8; 32]) -> i32 {
    let (a_s, a_m) = bitplane_split256(a);
    let (b_s, b_m) = bitplane_split256(b);

    let s_x = a_s ^ b_s; // signs mismatch
    let m_x = a_m ^ b_m; // magnitudes mismatch
    let m_s = a_m & b_m; // both magnitudes strong
    let m_w = !(a_m | b_m); // both magnitudes weak

    // Use bitmask combinations + popcnt to count each of our 6 states: (all strong, all weak,
    // mixed) x (positive, negative).
    (m_s & !s_x).count_ones() as i32 * 4
        + (m_s & s_x).count_ones() as i32 * -4
        + (m_w & !s_x).count_ones() as i32 * 1
        + (m_w & s_x).count_ones() as i32 * -1
        + (m_x & !s_x).count_ones() as i32 * 2
        + (m_x & s_x).count_ones() as i32 * -2
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn bitplane_split256(
    v: [u8; 32],
) -> (
    std::arch::aarch64::uint8x16_t,
    std::arch::aarch64::uint8x16_t,
) {
    use std::arch::aarch64::{vbslq_u8, vld1q_u8, vshlq_n_u8, vshrq_n_u8};

    unsafe {
        let a = vld1q_u8(v.as_ptr());
        let b = vld1q_u8(v.as_ptr().add(16));
        let m = vld1q_u8([0x55; 16].as_ptr());

        let sgn = vbslq_u8(m, vshrq_n_u8::<1>(a), b);
        let mag = vbslq_u8(m, a, vshlq_n_u8::<1>(b));
        (sgn, mag)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn distance256(a: [u8; 32], b: [u8; 32]) -> i32 {
    let (a_s, a_m) = bitplane_split256(a);
    let (b_s, b_m) = bitplane_split256(b);

    unsafe {
        use std::arch::aarch64::{
            vaddlvq_s8, vandq_u8, vcntq_u8, veorq_u8, vmvnq_u8, vorrq_u8, vreinterpretq_s8_u8,
            vsubq_s8,
        };

        let s_x = veorq_u8(a_s, b_s); // signs mismatch
        let s_m = vmvnq_u8(s_x); // signs match
        let m_x = veorq_u8(a_m, b_m); // magnitudes mismatch
        let m_s = vandq_u8(a_m, b_m); // both magnitudes strong
        let m_w = vmvnq_u8(vorrq_u8(a_m, b_m)); // both magnitudes weak

        let weak = vsubq_s8(
            vreinterpretq_s8_u8(vcntq_u8(vandq_u8(m_w, s_m))),
            vreinterpretq_s8_u8(vcntq_u8(vandq_u8(m_w, s_x))),
        );
        let mixed = vsubq_s8(
            vreinterpretq_s8_u8(vcntq_u8(vandq_u8(m_x, s_m))),
            vreinterpretq_s8_u8(vcntq_u8(vandq_u8(m_x, s_x))),
        );
        let strong = vsubq_s8(
            vreinterpretq_s8_u8(vcntq_u8(vandq_u8(m_s, s_m))),
            vreinterpretq_s8_u8(vcntq_u8(vandq_u8(m_s, s_x))),
        );

        vaddlvq_s8(weak) as i32 + vaddlvq_s8(mixed) as i32 * 2 + vaddlvq_s8(strong) as i32 * 4
    }
}

/// Symmetric distance computation for QuiVer vectors.
///
/// Distance computation ignores the stored tau value. The score most closely resembles an inner
/// product and is normalized as such since the min/max values are [-D*4, +D*4].
#[derive(Default)]
pub struct Distance;

impl VectorDistance for Distance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let (query_header, query) = Header::split_and_decode(query);
        let (doc_header, doc) = Header::split_and_decode(doc);

        // XXX need to process tails
        let raw_dist = query
            .as_chunks::<32>()
            .0
            .iter()
            .zip(doc.as_chunks::<32>().0.iter())
            .map(|(&q, &d)| distance256(q, d))
            .sum::<i32>();

        // Use strong count to compute a more accurate denominator for cosine similarity.
        let dim = query.len() as u32 * 4;
        let q_mag = query_header.strong_count * 4 + (dim - query_header.strong_count);
        let d_mag = doc_header.strong_count * 4 + (dim - doc_header.strong_count);
        let norm_factor = ((q_mag as u64 * d_mag as u64) as f64).sqrt();

        // Divide raw distance by norm_factor to get value in [-1,+1], then invert and add to get a
        // distance in [0,1].
        (raw_dist as f64 / norm_factor as f64) * -0.5 + 0.5
    }
}

#[inline(always)]
fn quantize_i8(value: f32, scale: f32) -> i8 {
    (value * scale).round() as i8
}

#[derive(Default)]
pub struct QueryDistance {
    query: Vec<i8>,
    scale: f32,
    magnitude: i32,
}

impl QueryDistance {
    pub fn new(query: &[f32]) -> Self {
        let max = query
            .iter()
            .copied()
            .map(f32::abs)
            .max_by(f32::total_cmp)
            .unwrap();
        let scale = 127.0 / max;
        let query = query
            .iter()
            .map(|&d| quantize_i8(d, scale))
            .collect::<Vec<_>>();
        let magnitude = query.iter().map(|&d| d as i32 * d as i32).sum::<i32>();
        Self {
            query,
            scale,
            magnitude,
        }
    }
}

unsafe extern "C" {
    unsafe fn et_quiver_asymmetric_ip(
        query: *const i8,
        len: usize,
        doc: *const u8,
        table: *const i8,
    ) -> i32;
}

impl QueryVectorDistance for QueryDistance {
    fn distance(&self, vector: &[u8]) -> f64 {
        // Read strong/weak value and encode as i8
        let (header, vector) = Header::split_and_decode(vector);
        let strong = quantize_i8(header.strong, self.scale);
        let weak = quantize_i8(header.weak, self.scale);
        let decode_table = [-weak, -strong, weak, strong];

        let raw_dist = unsafe {
            et_quiver_asymmetric_ip(
                self.query.as_ptr(),
                self.query.len(),
                vector.as_ptr(),
                decode_table.as_ptr(),
            )
        };
        let doc_magnitude = (strong as i32 * strong as i32 * header.strong_count as i32)
            + (weak as i32 * weak as i32 * (self.query.len() as i32 - header.strong_count as i32));
        let distance_scale = (self.magnitude as f64 * doc_magnitude as f64).sqrt();

        (raw_dist as f64 / distance_scale) * -0.5 + 0.5
    }
}
