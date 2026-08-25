//! Implementation of RaBitQ vector quantizer (XXX insert paper reference)
//!
//! One note is that this does not include rotation inline in the quantization transform.
//! Callers are expected to rotate the vectors if component distribution is not Gaussian, and
//! they are expected to rotate the center (or compute the mean from rotated vectors).
use std::borrow::Cow;

use crate::{F32VectorCoder, float32};

#[derive(Debug, Copy, Clone, PartialEq, Default)]
struct Header {
    /// L2 norm of the original vector after centering.
    l2_norm: f32,
    /// Inner product of the quantized vector and normalized original vector.
    correction_term: f32,
    /// Sum of all of the component bits.
    component_sum: u32,
}

impl Header {
    const LEN: usize = 12;

    fn split_mut(bytes: &mut [u8]) -> (&mut [u8; Self::LEN], &mut [u8]) {
        let (hbytes, vbytes) = bytes.split_at_mut(Self::LEN);
        (&mut hbytes.as_chunks_mut::<{ Self::LEN }>().0[0], vbytes)
    }

    fn encode(&self, out: &mut [u8; Self::LEN]) {
        let parts = out.as_chunks_mut::<4>().0;
        parts[0] = self.l2_norm.to_le_bytes();
        parts[1] = self.correction_term.to_le_bytes();
        parts[2] = self.component_sum.to_le_bytes();
    }

    fn decode(raw: &[u8]) -> (Header, &[u8]) {
        let (hbytes, vbytes) = raw.split_at(Self::LEN);
        let parts = hbytes.as_chunks::<4>().0;
        (
            Header {
                l2_norm: f32::from_le_bytes(parts[0]),
                correction_term: f32::from_le_bytes(parts[1]),
                component_sum: u32::from_le_bytes(parts[2]),
            },
            vbytes,
        )
    }
}

#[derive(Debug, Clone, Default)]
pub struct Coder {
    center: Option<Vec<f32>>,
}

impl Coder {
    pub fn new(center: Option<Vec<f32>>) -> Self {
        Self { center }
    }
}

impl F32VectorCoder for Coder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let centered_vector: Cow<'_, [f32]> = if let Some(center) = self.center.as_ref() {
            vector
                .iter()
                .zip(center.iter())
                .map(|(v, c)| *v - *c)
                .collect::<Vec<_>>()
                .into()
        } else {
            vector.into()
        };
        let mut header = Header::default();
        header.l2_norm = float32::l2_norm(&centered_vector);
        let unit_vector = float32::l2_normalize(centered_vector);
        header.correction_term = unit_vector.iter().copied().map(f32::abs).sum::<f32>()
            / (unit_vector.len() as f32).sqrt();

        let (hbytes, vbytes) = Header::split_mut(out);
        header.component_sum = unit_vector
            .chunks(8)
            .zip(vbytes.iter_mut())
            .map(|(i, o)| {
                let b = i
                    .iter()
                    .enumerate()
                    .map(|(i, x)| if x.is_sign_negative() { 1u32 << i } else { 0 })
                    .reduce(|a, b| a | b)
                    .unwrap();
                *o = b as u8;
                b.count_ones()
            })
            .sum::<u32>();

        header.encode(hbytes);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        Header::LEN + dimensions.div_ceil(8)
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        let (_, vector) = Header::decode(encoded);
        let sign_mask = 1u32 << 31;
        let magnitude = 1.0 / (out.len() as f32).sqrt();
        for (&c, o) in vector.iter().zip(out.chunks_mut(8)) {
            let c = c as u32;
            for (i, o) in o.iter_mut().enumerate() {
                *o = f32::from_bits(magnitude.to_bits() ^ ((c << (31 - i)) & sign_mask));
            }
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        (byte_len - Header::LEN) * 8
    }
}

// XXX this might be able to use the turbo packer after all to speed up bitplane split for ADC.

// XXX ADC
// * Quantize the query and bitplane split.
// * Use bitplane split hamming trick up to 4 bits, or maybe just use DOT.
// * Do I have to perform signed quantization of the input vector?

// XXX SDC
// * Just use hamming distance.
// * Euclidean is going to be less accurate unless I store the mean magnitude per dimension of the
//   centered vector.
