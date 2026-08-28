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

// XXX rewrite all of this shit it's boerderline uninteligible.
/// Symmetric distance between two RaBitQ codes.
///
/// Each quantized component is `±1/√D`, so hamming distance `h` gives the inner product of the two
/// quantized unit vectors directly: `⟨ū_q, ū_d⟩ = (D - 2h) / D`. That value is a badly biased
/// estimate of the inner product of the *unquantized* unit vectors -- it is pulled hard toward
/// zero, by as much as 0.2 in the high similarity region that decides ranking.
///
/// The debiasing transform comes from the arcsine law: if two components are jointly Gaussian with
/// correlation `ρ` then `E[sign(x)·sign(y)] = (2/π)·arcsin(ρ)`. Averaged over the dimensions that
/// makes `⟨ū_q, ū_d⟩` an estimate of `(2/π)·arcsin⟨u_q, u_d⟩`, so inverting recovers
/// `⟨u_q, u_d⟩ ≈ sin(π/2 · ⟨ū_q, ū_d⟩)`. The joint Gaussian assumption is the same isotropy
/// assumption that makes the random rotation callers are expected to apply worthwhile.
///
/// The stored l2 norms then recover the distance between the (possibly centered) input vectors;
/// the center cancels in the difference so it needs no further adjustment.
///
/// The one regression is near-orthogonal pairs, where `sin` amplifies the sampling noise by up to
/// π/2 (MAE 0.041 vs 0.032 at D=512). That is the unavoidable cost of being unbiased, and those
/// pairs are never near the result set.
#[derive(Debug)]
pub struct Distance {
    similarity: VectorSimilarity,
}

impl Distance {
    pub fn new(similarity: VectorSimilarity) -> Self {
        Self { similarity }
    }
}

impl VectorDistance for Distance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let (qheader, query) = Header::decode(query);
        let (dheader, doc) = Header::decode(doc);

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

        // Each matching bit contributes 1/D and each mismatch -1/D.
        let quantized_ip = (dim as f64 - 2.0 * h as f64) / dim as f64;
        // Invert the arcsine law to debias the estimate of the unquantized inner product.
        // XXX figure out if this is fucking slow and/or necessary. multiplying corrections hurts.
        let ip = (std::f64::consts::FRAC_PI_2 * quantized_ip).sin();

        let qnorm: f64 = qheader.l2_norm.into();
        let dnorm: f64 = dheader.l2_norm.into();
        let l2_dist = qnorm.powi(2) + dnorm.powi(2) - 2.0 * qnorm * dnorm * ip;
        match self.similarity {
            VectorSimilarity::Euclidean => l2_dist,
            VectorSimilarity::Cosine | VectorSimilarity::Dot => (0.25 * l2_dist).clamp(0.0, 1.0),
        }
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
}

impl QueryDistance {
    pub fn new(similarity: VectorSimilarity, query: &[f32], center: Option<&[f32]>) -> Self {
        let query: Cow<'_, [f32]> = if let Some(center) = center {
            query
                .iter()
                .zip(center.iter())
                .map(|(&q, &c)| q - c)
                .collect::<Vec<_>>()
                .into()
        } else {
            query.into()
        };
        let l2_norm = float32::l2_norm(query.as_ref());
        let query = float32::l2_normalize(query);
        let dim_sqrt = (query.len() as f64).sqrt();
        let (lower, upper) = query
            .iter()
            .copied()
            .fold((f32::MAX, f32::MIN), |acc, x| (acc.0.min(x), acc.1.max(x)));
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
            l2_norm,
            lower,
            delta,
            component_sum,
            dim_sqrt,
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
        let qnorm: f64 = self.l2_norm.into();
        let dnorm: f64 = header.l2_norm.into();
        // L2 Distance with two centered vectors needs no additional parameters or adjustments and
        // can be used to produce the cosine similarity.
        let l2_dist = dnorm.powi(2) + qnorm.powi(2) - 2.0 * qnorm * dnorm * self.ip(header, doc);
        match self.similarity {
            VectorSimilarity::Euclidean => l2_dist,
            VectorSimilarity::Cosine | VectorSimilarity::Dot => (0.25 * l2_dist).clamp(0.0, 1.0),
        }
    }

    fn cos_error(&self, header: Header) -> f64 {
        // XXX should be sqrt(D - 1)???
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
        let l2_error = 2.0 * self.l2_norm as f64 * header.l2_norm as f64 * self.cos_error(header);
        EstimatedDistance {
            distance: self.distance_internal(header, vector),
            error: match self.similarity {
                VectorSimilarity::Euclidean => l2_error,
                // angular distance = l2_dist / 4
                VectorSimilarity::Cosine | VectorSimilarity::Dot => 0.25 * l2_error,
            },
        }
    }
}

// XXX SDC
// * Just use hamming distance.
// * Euclidean is going to be less accurate unless I store the mean magnitude per dimension of the
//   centered vector.

#[cfg(test)]
mod test {
    use super::*;

    fn gauss(rng: &mut rand_xoshiro::Xoshiro256PlusPlus) -> f32 {
        let u1: f32 = rng.random_range(1e-9f32..1.0);
        let u2: f32 = rng.random_range(0.0f32..1.0);
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// The symmetric estimator should be near unbiased across the whole similarity range, and
    /// should beat raw hamming badly on the correlated pairs that decide ranking.
    #[test]
    fn symmetric_debias() {
        const DIM: usize = 512;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0x9e3779b9);
        let coder = Coder::new(None);
        let dist = Distance::new(VectorSimilarity::Euclidean);

        // Sums of signed error, absolute error, raw hamming absolute error, and count, for pairs
        // binned by their true cosine. Errors are normalized by the norms so bins are comparable.
        let mut bins = [(0.0f64, 0.0f64, 0.0f64, 0.0f64); 10];
        for _ in 0..4096 {
            let rho: f32 = rng.random_range(0.0f32..1.0);
            let a = (0..DIM).map(|_| gauss(&mut rng)).collect::<Vec<_>>();
            let b = a
                .iter()
                .map(|x| rho * x + (1.0 - rho * rho).sqrt() * gauss(&mut rng))
                .collect::<Vec<_>>();

            let mut ea = vec![0u8; coder.byte_len(DIM)];
            let mut eb = vec![0u8; coder.byte_len(DIM)];
            coder.encode_to(&a, &mut ea);
            coder.encode_to(&b, &mut eb);
            let (ha, va) = Header::decode(&ea);
            let (hb, vb) = Header::decode(&eb);
            let (anorm, bnorm) = (ha.l2_norm as f64, hb.l2_norm as f64);

            let dot = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| *x as f64 * *y as f64)
                .sum::<f64>();
            let cos = dot / (anorm * bnorm);
            let exact = anorm.powi(2) + bnorm.powi(2) - 2.0 * dot;

            // Raw hamming, i.e. what the estimate would be without the arcsine transform.
            let hamming = va
                .iter()
                .zip(vb.iter())
                .map(|(x, y)| (x ^ y).count_ones())
                .sum::<u32>();
            let raw_ip = (DIM as f64 - 2.0 * hamming as f64) / DIM as f64;
            let raw = anorm.powi(2) + bnorm.powi(2) - 2.0 * anorm * bnorm * raw_ip;

            let scale = 2.0 * anorm * bnorm;
            let bin = &mut bins[((cos.max(0.0) * 10.0) as usize).min(9)];
            bin.0 += (dist.distance(&ea, &eb) - exact) / scale;
            bin.1 += ((dist.distance(&ea, &eb) - exact) / scale).abs();
            bin.2 += ((raw - exact) / scale).abs();
            bin.3 += 1.0;
        }

        for (i, bin) in bins.iter().enumerate() {
            assert!(bin.3 > 0.0, "bin {i} is empty");
            let (bias, mae, raw_mae) = (bin.0 / bin.3, bin.1 / bin.3, bin.2 / bin.3);
            assert!(bias.abs() < 0.02, "bin {i} bias {bias} is too large");
            // Above the near-orthogonal bins the debiased estimate should be clearly better.
            if i >= 2 {
                assert!(
                    mae < raw_mae * 0.75,
                    "bin {i} mae {mae} vs raw hamming mae {raw_mae}"
                );
            }
        }
    }
}
