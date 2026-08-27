//! Implementation of RaBitQ vector quantizer.
//!   Base paper: https://arxiv.org/pdf/2405.12497
//!
//! One note is that this does not include rotation inline in the quantization transform.
//! Callers are expected to rotate the vectors if component distribution is not Gaussian, and
//! they are expected to rotate the center (or compute the mean from rotated vectors).
use std::borrow::Cow;

use rand::{RngExt, SeedableRng};

use crate::{
    EstimatedDistance, F32VectorCoder, QueryVectorDistance, VectorDistance, VectorSimilarity,
    float32, lvq::packing::TurboPacker,
};

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
        let mut packer = super::lvq::packing::TurboPacker::<1>::new(vbytes);
        header.component_sum = unit_vector
            .iter()
            .map(|x| {
                let s = x.to_bits() >> 31;
                packer.push(s as u8);
                s
            })
            .sum::<u32>();

        header.encode(hbytes);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        Header::LEN + dimensions.div_ceil(8)
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        let (_, vector) = Header::decode(encoded);
        let magnitude = 1.0 / (out.len() as f32).sqrt();
        for (q, o) in super::lvq::packing::TurboUnpacker::<1>::new(vector).zip(out.iter_mut()) {
            *o = f32::from_bits(magnitude.to_bits() ^ ((q as u32) << 31));
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        (byte_len - Header::LEN) * 8
    }
}

/// Symmetric distance is just hamming distance.
///
/// This is OK-ish for angular distance and awful for Euclidean, which could use some additional
/// scaling factors.
// XXX this is hella wrong when centered.
#[derive(Debug, Default)]
pub struct Distance;

impl VectorDistance for Distance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let (_, query) = Header::decode(query);
        let (_, doc) = Header::decode(doc);

        let dim = query.len() * 8;
        let (qhead, qtail) = query.as_chunks::<8>();
        let (dhead, dtail) = doc.as_chunks::<8>();

        let mut h = qhead
            .iter()
            .zip(dhead.iter())
            .map(|(q, d)| (u64::from_ne_bytes(*q) ^ u64::from_ne_bytes(*d)).count_ones())
            .sum::<u32>();
        if !qtail.is_empty() {
            h += qtail
                .iter()
                .zip(dtail.iter())
                .map(|(&q, &d)| (q ^ d).count_ones())
                .sum::<u32>();
        }

        (dim - h as usize) as f64 / dim as f64
    }
}

pub struct QueryDistance {
    similarity: VectorSimilarity,
    query: Vec<u8>,
    l2_norm: f32,
    lower: f32,
    delta: f32,
    component_sum: u32,
    dim_sqrt: f64,
    /// q•d; used to compensate for centering in angular distance. 0 for euclidean.
    center_dot: f32,
}

impl QueryDistance {
    pub fn new(similarity: VectorSimilarity, query: &[f32], center: Option<&[f32]>) -> Self {
        let center_dot = match (similarity, center) {
            (VectorSimilarity::Euclidean, _) | (_, None) => 0.0,
            (_, Some(center)) => query
                .iter()
                .zip(center.iter())
                .map(|(&q, &c)| q * c)
                .sum::<f32>(),
        };
        let query: Cow<'_, [f32]> = match (similarity, center) {
            (VectorSimilarity::Euclidean, Some(center)) => query
                .iter()
                .zip(center.iter())
                .map(|(&q, &c)| q - c)
                .collect::<Vec<_>>()
                .into(),
            _ => query.into(),
        };
        let dim_sqrt = (query.len() as f64).sqrt();
        let (lower, upper, l2_norm_sq) = query
            .iter()
            .copied()
            .fold((f32::MAX, f32::MIN, 0f32), |acc, x| {
                (acc.0.min(x), acc.1.max(x), x.mul_add(x, acc.2))
            });
        let delta_inv = 15.0 / (upper - lower);
        let delta = (upper - lower) / 15.0;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::from_seed([0xfe; 32]);
        let mut query4 = vec![0u8; query.len().div_ceil(2)];
        let mut component_sum = 0u32;
        let mut packer = TurboPacker::<4>::new(&mut query4);
        for &x in query.iter() {
            let q = delta_inv
                .mul_add(x - lower, rng.random_range(0.0..1.0))
                .floor()
                .min(15.0) as u32;
            component_sum += q;
            packer.push(q as u8);
        }
        Self {
            similarity,
            query: super::lvq::packing::bitplane_split4(&query4),
            l2_norm: l2_norm_sq.sqrt(),
            lower,
            delta,
            component_sum,
            dim_sqrt,
            center_dot,
        }
    }

    // XXX inner product of uint8 part is shared with lvq.
    #[inline]
    fn ip(&self, header: Header, doc: &[u8]) -> f64 {
        let (qhead, qtail) = self.query.as_chunks::<64>();
        let (dhead, dtail) = doc.as_chunks::<16>();
        let mut bdot = [0u32; 4];
        for (q, d) in qhead.iter().zip(dhead.iter()) {
            let qp = q.as_chunks::<16>().0;
            let q = [
                u128::from_le_bytes(qp[0]),
                u128::from_le_bytes(qp[1]),
                u128::from_le_bytes(qp[2]),
                u128::from_le_bytes(qp[3]),
            ];
            let d = u128::from_le_bytes(*d);
            bdot[0] += (q[0] & d).count_ones();
            bdot[1] += (q[1] & d).count_ones();
            bdot[2] += (q[2] & d).count_ones();
            bdot[3] += (q[3] & d).count_ones();
        }

        if !qtail.is_empty() {
            let mut qit = qtail.chunks(qtail.len() / 4);
            let q = [
                qit.next().unwrap(),
                qit.next().unwrap(),
                qit.next().unwrap(),
                qit.next().unwrap(),
            ];
            for (i, &d) in dtail.iter().enumerate() {
                bdot[0] += (q[0][i] ^ d).count_ones();
                bdot[1] += (q[1][i] ^ d).count_ones();
                bdot[2] += (q[2][i] ^ d).count_ones();
                bdot[3] += (q[3][i] ^ d).count_ones();
            }
        }

        let ip_uint = bdot[0] + bdot[1] * 2 + bdot[2] * 4 + bdot[3] * 8;
        let ip = self.dim_sqrt * self.lower as f64
            - 2.0 * self.lower as f64 * header.component_sum as f64 / self.dim_sqrt
            + self.delta as f64 * self.component_sum as f64 / self.dim_sqrt
            - 2.0 * self.delta as f64 * ip_uint as f64 / self.dim_sqrt;
        ip / header.correction_term as f64
    }

    fn distance_internal(&self, header: Header, doc: &[u8]) -> f64 {
        let dnorm: f64 = header.l2_norm.into();
        let raw = dnorm * self.ip(header, doc);
        match self.similarity {
            VectorSimilarity::Euclidean => {
                let qnorm: f64 = self.l2_norm.into();
                dnorm.powi(2) + qnorm.powi(2) - 2.0 * raw
            }
            VectorSimilarity::Cosine | VectorSimilarity::Dot => {
                let true_dot = raw + self.center_dot as f64;
                true_dot.mul_add(-0.5, 0.5)
            }
        }
        .into()
    }

    fn error(&self, header: Header) -> f64 {
        let c = (header.correction_term as f64).powi(2);
        ((1.0 - c) / c).sqrt() / self.dim_sqrt
    }
}

impl QueryVectorDistance for QueryDistance {
    fn distance(&self, vector: &[u8]) -> f64 {
        let (header, vector) = Header::decode(vector);
        self.distance_internal(header, vector)
    }

    fn estimated_distance(&self, vector: &[u8]) -> EstimatedDistance {
        let (header, vector) = Header::decode(vector);
        let e = self.error(header) * self.l2_norm as f64 * header.l2_norm as f64;
        EstimatedDistance {
            distance: self.distance_internal(header, vector),
            error: match self.similarity {
                VectorSimilarity::Euclidean => 2.0 * e,
                VectorSimilarity::Cosine | VectorSimilarity::Dot => 0.5 * e,
            },
        }
    }
}

// XXX SDC
// * Just use hamming distance.
// * Euclidean is going to be less accurate unless I store the mean magnitude per dimension of the
//   centered vector.
