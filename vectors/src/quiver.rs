//! QuiVer two-bit training free quantization: https://arxiv.org/html/2605.02171v1

// TODO: consider something more geometric for the encoding, possible record a "weak" value that
// preserves the magnitude of the vector.

use crate::{F32VectorCoder, VectorDistance};

/// Coder for QuiVer format.
///
/// The stored encoding contains 2 bits for every dimension packed into bytes.
/// The vector is suffixed with a 32-bit float containing the "strong" boundary value that is used
/// to "decode" the vector.
#[derive(Default)]
pub struct Coder;

impl F32VectorCoder for Coder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let tau = vector.iter().copied().map(f32::abs).sum::<f32>() / vector.len() as f32;
        out[..4].copy_from_slice(&tau.to_le_bytes());
        for (c, o) in vector.chunks(4).skip(1).zip(out.iter_mut()) {
            *o = c
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let q = if v > 0.0 { 2u8 } else { 0u8 } | if v.abs() > tau { 1u8 } else { 0u8 };
                    q << (i * 2)
                })
                .fold(0u8, |x, q| x | q)
        }
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions.div_ceil(4) + std::mem::size_of::<u32>()
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        let tau = f32::from_le_bytes(encoded[..4].try_into().unwrap());
        let weak = tau / 2.0;
        for (&qc, oc) in encoded[4..].iter().zip(out.chunks_mut(4)) {
            for (i, o) in oc.iter_mut().enumerate() {
                let q = (qc >> (i * 2)) & 0x3;
                *o = if q & 1 == 1 { tau } else { weak };
                if q & 2 == 0 {
                    *o = -*o;
                }
            }
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        (byte_len - std::mem::size_of::<u32>()) / 4
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
        // TODO: there has got to be a better way to normalize this.
        let raw_dist = query
            .iter()
            .zip(doc.iter())
            .map(|(&q, &d)| {
                let lo_key = ((q & 0xf) | (d << 4)) as usize;
                let hi_key = ((q >> 4) | (d & 0xf0)) as usize;
                DISTANCE_LUT[lo_key] as i32 + DISTANCE_LUT[hi_key] as i32
            })
            .sum::<i32>();
        // 4 dimensions per byte, maximum value of 4 per dimension.
        // identical vector will have this score, antiparallel vector will have negative of this.
        let norm_factor = query.len() * 4 * 4;
        // Divide raw distance by norm_factor to get value in [-1,+1], then invert and add to get a
        // distance in [0,1].
        (raw_dist as f64 / norm_factor as f64) * -0.5 + 0.5
    }
}
