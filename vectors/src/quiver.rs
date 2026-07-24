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
        for (i, ic) in vector.chunks(vector_bytes.len()).enumerate() {
            for (&v, o) in ic.iter().zip(vector_bytes.iter_mut()) {
                let q = if v > 0.0 { 2u8 } else { 0u8 }
                    | if v.abs() > tau {
                        strong.add(v.abs());
                        1u8
                    } else {
                        weak.add(v.abs());
                        0u8
                    };
                *o |= q << (i * 2);
            }
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
        for (&qc, oc) in vector_bytes.iter().zip(out.chunks_mut(4)) {
            for (i, o) in oc.iter_mut().enumerate() {
                *o = decode_table[(qc as usize >> (i * 2)) & 0x3];
            }
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        (byte_len - Header::LEN) / 4
    }
}

/// Compares the low 2 bits of a and b to produce a distance.
const fn distance(a: u8, b: u8) -> i8 {
    let h = (a & 3) ^ (b & 3);
    let r: i8 = if h & 1 == 0 {
        // Agree on magnitude
        if a & 1 == 1 {
            4 // strong
        } else {
            1 // weak
        }
    } else {
        // Disagree on magnitude
        2
    };
    // If the signs differ the result is negative.
    if h & 2 == 0 { r } else { -r }
}

/// Precompute a table of distances where the low nibble is 2 dimensions from one vector and the
/// high nibble is 2 dimensions from the other vector.
const fn distance_table() -> [i8; 256] {
    let mut table = [0i8; 256];
    let mut i = 0;
    while i < table.len() {
        let code = i as u8;
        table[i] = distance(code, code >> 4) + distance(code >> 2, code >> 6);
        i += 1;
    }
    table
}

const DISTANCE_LUT: [i8; 256] = distance_table();

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

        let raw_dist = query
            .iter()
            .zip(doc.iter())
            .map(|(&q, &d)| {
                let lo_key = ((q & 0xf) | (d << 4)) as usize;
                let hi_key = ((q >> 4) | (d & 0xf0)) as usize;
                DISTANCE_LUT[lo_key] as i32 + DISTANCE_LUT[hi_key] as i32
            })
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

impl QueryVectorDistance for QueryDistance {
    fn distance(&self, vector: &[u8]) -> f64 {
        // Read strong/weak value and encode as i8
        let (header, vector) = Header::split_and_decode(vector);
        let strong = quantize_i8(header.strong, self.scale);
        let weak = quantize_i8(header.weak, self.scale);
        let decode_table = [
            -weak, -strong, weak, strong, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        let mut raw_dist = 0i32;
        let stride = self.query.len() / 4;
        // XXX this generates SDOT, which is good, but access to the decode table is dogshit
        // scalar code.
        for (i, c) in vector.as_chunks::<16>().0.iter().enumerate() {
            let q0 = i * 16;
            raw_dist += c
                .iter()
                .zip(&self.query[q0..q0 + 16])
                .map(|(&q, &v)| decode_table[(q & 3) as usize] as i32 * v as i32)
                .sum::<i32>();
            let q1 = i * 16 + stride;
            raw_dist += c
                .iter()
                .zip(&self.query[q1..q1 + 16])
                .map(|(&q, &v)| decode_table[((q >> 2) & 3) as usize] as i32 * v as i32)
                .sum::<i32>();
            let q2 = i * 16 + stride * 2;
            raw_dist += c
                .iter()
                .zip(&self.query[q2..q2 + 16])
                .map(|(&q, &v)| decode_table[((q >> 4) & 3) as usize] as i32 * v as i32)
                .sum::<i32>();
            let q3 = i * 16 + stride * 3;
            raw_dist += c
                .iter()
                .zip(&self.query[q3..q3 + 16])
                .map(|(&q, &v)| decode_table[((q >> 6) & 3) as usize] as i32 * v as i32)
                .sum::<i32>();
        }

        let doc_magnitude = (strong as i32 * strong as i32 * header.strong_count as i32)
            + (weak as i32 * weak as i32 * (self.query.len() as i32 - header.strong_count as i32));
        let distance_scale = (self.magnitude as f64 * doc_magnitude as f64).sqrt();

        (raw_dist as f64 / distance_scale) * -0.5 + 0.5
    }
}
