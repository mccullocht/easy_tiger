//! Implementation of RaBitQ vector quantizer.
//!   Base paper: https://arxiv.org/pdf/2405.12497
//!
//! One note is that this does not include rotation inline in the quantization transform.
//! Callers are expected to rotate the vectors if component distribution is not Gaussian, and
//! they are expected to rotate the center (or compute the mean from rotated vectors).

#[cfg(target_arch = "aarch64")]
mod aarch64;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86_64;

use std::borrow::Cow;

use rand::{RngExt, SeedableRng};

use crate::{
    EstimatedDistance, F32VectorCoder, QueryVectorDistance, VectorDistance, VectorSimilarity,
    float32, packing::TurboPacker,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Kernel {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl Kernel {
    const CANDIDATES: &'static [Self] = &[
        #[cfg(target_arch = "aarch64")]
        Self::Neon,
        #[cfg(target_arch = "x86_64")]
        Self::Avx512,
        Self::Scalar,
    ];

    fn is_available(&self) -> bool {
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Neon => true,
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => {
                use std::arch::is_x86_feature_detected as cpu_feature;
                cpu_feature!("avx512f")
                    && cpu_feature!("avx512bw")
                    && cpu_feature!("avx512vpopcntdq")
            }
            Self::Scalar => true,
        }
    }
}

impl Default for Kernel {
    fn default() -> Self {
        Self::CANDIDATES
            .iter()
            .copied()
            .find(Self::is_available)
            .expect("scalar is always available")
    }
}

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
    k: Kernel,
    center: Option<Vec<f32>>,
}

impl Coder {
    pub fn new(center: Option<Vec<f32>>) -> Self {
        Self {
            k: Kernel::default(),
            center,
        }
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
        let l2_norm = float32::l2_norm(&centered_vector);
        let mut header = Header {
            l2_norm,
            ..Default::default()
        };
        header.correction_term = match self.k {
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => aarch64::neon::l1_norm_scaled(&centered_vector, l2_norm.recip()),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe {
                x86_64::avx512::l1_norm_scaled(&centered_vector, l2_norm.recip())
            },
            Kernel::Scalar => scalar::l1_norm_scaled(&centered_vector, l2_norm.recip()),
        };

        let (hbytes, vbytes) = Header::split_mut(out);
        header.component_sum = match self.k {
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => aarch64::neon::quantize_and_pack(&centered_vector, vbytes),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe {
                x86_64::avx512::quantize_and_pack(&centered_vector, vbytes)
            },
            Kernel::Scalar => scalar::quantize_and_pack(&centered_vector, vbytes),
        };
        header.encode(hbytes);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        Header::LEN + dimensions.div_ceil(8)
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        let (_, vector) = Header::decode(encoded);
        let magnitude = 1.0 / (out.len() as f32).sqrt();
        match self.k {
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => aarch64::neon::decode(vector, magnitude, self.center.as_deref(), out),
            _ => scalar::decode(vector, magnitude, self.center.as_deref(), out),
        };
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        (byte_len - Header::LEN) * 8
    }
}

/// Symmetric distance between two RaBitQ codes.
///
/// Each quantized component is `±1/√D`, so hamming distance `h` gives the inner product of the two
/// quantized unit vectors directly: `⟨ū_q, ū_d⟩ = (D - 2h) / D`. That value is a badly biased
/// estimate of the inner product of the *unquantized* unit vectors -- it is pulled hard toward
/// zero.
///
/// The debiasing transform comes from the arcsine law: if two components are jointly Gaussian with
/// correlation `ρ` then `E[sign(x)·sign(y)] = (2/π)·arcsin(ρ)`. Averaged over the dimensions that
/// makes `⟨ū_q, ū_d⟩` an estimate of `(2/π)·arcsin⟨u_q, u_d⟩`, so inverting recovers
/// `⟨u_q, u_d⟩ ≈ sin(π/2 · ⟨ū_q, ū_d⟩)`. The joint Gaussian assumption is the same isotropy
/// assumption that makes the random rotation callers are expected to apply worthwhile.
///
/// The stored l2 norms then recover the distance between the (possibly centered) input vectors.
#[derive(Debug)]
pub struct Distance {
    k: Kernel,
    similarity: VectorSimilarity,
}

impl Distance {
    pub fn new(similarity: VectorSimilarity) -> Self {
        Self {
            k: Kernel::default(),
            similarity,
        }
    }
}

impl VectorDistance for Distance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let (qheader, query) = Header::decode(query);
        let (dheader, doc) = Header::decode(doc);

        let dim = query.len() * 8;
        let h = match self.k {
            Kernel::Scalar => crate::kernels::scalar::bitstring_inner_product::<true>(query, doc),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => {
                crate::kernels::aarch64::neon::bitstring_inner_product::<true>(query, doc)
            }
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe {
                crate::kernels::x86_64::avx512::bitstring_inner_product::<true>(query, doc)
            },
        };

        // Each matching bit contributes 1/D and each mismatch -1/D.
        let quantized_ip = (dim as f64 - 2.0 * h as f64) / dim as f64;
        // Invert the arcsine law to debias the estimate of the unquantized inner product.
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
    k: Kernel,
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
        let (query, l2_norm) = float32::l2_normalize(query);
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
            k: Kernel::default(),
            similarity,
            query: crate::packing::bitplane_split4(&query4),
            l2_norm,
            lower,
            delta,
            component_sum,
            dim_sqrt,
        }
    }

    #[inline]
    fn ip(&self, header: Header, doc: &[u8]) -> f64 {
        let ip_uint = match self.k {
            Kernel::Scalar => crate::kernels::scalar::turbo_4x1_inner_product(&self.query, doc),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => {
                crate::kernels::aarch64::neon::turbo_4x1_inner_product(&self.query, doc)
            }
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe {
                crate::kernels::x86_64::avx512::turbo_4x1_inner_product(&self.query, doc)
            },
        };
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
        let c = (header.correction_term as f64).powi(2);
        // NB: the denominator should be (dim - 1).sqrt() but in practice this doesn't matter as dim
        // is typically very large.
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

#[cfg(test)]
mod test {
    use super::*;

    fn gauss(rng: &mut rand_xoshiro::Xoshiro256PlusPlus) -> f32 {
        let u1: f32 = rng.random_range(1e-9f32..1.0);
        let u2: f32 = rng.random_range(0.0f32..1.0);
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// A high dimensional vector with iid Gaussian components. RaBitQ assumes callers have applied a
    /// random rotation, which makes the component distribution look like this.
    fn gauss_vec(rng: &mut rand_xoshiro::Xoshiro256PlusPlus, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| gauss(rng)).collect()
    }

    /// Produce a vector whose true cosine with `base` is roughly `rho`.
    fn correlated(rng: &mut rand_xoshiro::Xoshiro256PlusPlus, base: &[f32], rho: f32) -> Vec<f32> {
        base.iter()
            .map(|x| rho * x + (1.0 - rho * rho).sqrt() * gauss(rng))
            .collect()
    }

    fn subtract_center(v: &[f32], center: Option<&[f32]>) -> Vec<f32> {
        match center {
            Some(c) => v.iter().zip(c).map(|(x, y)| x - y).collect(),
            None => v.to_vec(),
        }
    }

    fn squared_l2(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let d = *x as f64 - *y as f64;
                d * d
            })
            .sum()
    }

    /// The distance the codec is trying to estimate, computed with an *exact* inner product. The
    /// gap between this and [`Distance`] / [`QueryDistance`] output is pure quantization error.
    fn exact_distance(
        similarity: VectorSimilarity,
        a: &[f32],
        b: &[f32],
        center: Option<&[f32]>,
    ) -> f64 {
        let l2 = squared_l2(&subtract_center(a, center), &subtract_center(b, center));
        match similarity {
            VectorSimilarity::Euclidean => l2,
            VectorSimilarity::Cosine | VectorSimilarity::Dot => (0.25 * l2).clamp(0.0, 1.0),
        }
    }

    /// A center that is offset from the data distribution so that centering meaningfully changes the
    /// encoded signs and norms rather than being a no-op.
    fn test_center(dim: usize) -> Vec<f32> {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0xce17e5);
        (0..dim).map(|_| 0.35 * gauss(&mut rng) + 0.2).collect()
    }

    fn maybe_normalize(similarity: VectorSimilarity, v: Vec<f32>) -> Vec<f32> {
        // Cosine/Dot encoders assume the input vector is already normalized.
        if similarity == VectorSimilarity::Euclidean {
            v
        } else {
            float32::l2_normalize(v).0.into_owned()
        }
    }

    const DIM: usize = 128;
    const TRIALS: usize = 4096;

    #[test]
    fn encode_decode_roundtrip() {
        for center in [None, Some(test_center(DIM))] {
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0x1234abcd);
            let coder = Coder::new(center.clone());

            assert_eq!(coder.byte_len(DIM), Header::LEN + DIM / 8);
            assert_eq!(coder.dimensions(coder.byte_len(DIM)), DIM);

            let magnitude = 1.0 / (DIM as f32).sqrt();
            for _ in 0..64 {
                let v = gauss_vec(&mut rng, DIM);
                let encoded = coder.encode(&v);
                assert_eq!(encoded.len(), coder.byte_len(DIM));

                let decoded = coder.decode(&encoded);
                assert_eq!(decoded.len(), DIM);

                let centered = subtract_center(&v, center.as_deref());
                let mut component_sum = 0u32;
                for (i, (&d, &c)) in decoded.iter().zip(centered.iter()).enumerate() {
                    // Decode reconstructs the quantized centered unit vector (±1/√D with the sign
                    // of `v - center`) and adds the center back.
                    let quantized = if c.is_sign_negative() {
                        -magnitude
                    } else {
                        magnitude
                    };
                    let want = match center.as_deref() {
                        Some(center) => quantized + center[i],
                        None => quantized,
                    };
                    assert_eq!(d, want, "index {i}: centered {c}");
                    component_sum += c.is_sign_negative() as u32;
                }

                // The header records the number of set (negative) sign bits.
                let (header, _) = Header::decode(&encoded);
                assert_eq!(header.component_sum, component_sum);
                let want_norm = float32::l2_norm(&centered) as f64;
                assert!(
                    (want_norm - header.l2_norm as f64).abs() <= 1e-4 * want_norm.max(1.0),
                    "l2_norm mismatch: want {want_norm} got {}",
                    header.l2_norm
                );
            }
        }
    }

    fn eval_symmetric(similarity: VectorSimilarity, center: Option<Vec<f32>>) {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0x5eed01);
        let coder = Coder::new(center.clone());
        let dist = Distance::new(similarity);

        let mut bias = 0.0f64;
        let mut mae = 0.0f64;
        for _ in 0..TRIALS {
            let rho: f32 = rng.random_range(0.0f32..1.0);
            let a = maybe_normalize(similarity, gauss_vec(&mut rng, DIM));
            let b = maybe_normalize(similarity, correlated(&mut rng, &a, rho));

            let ea = coder.encode(&a);
            let eb = coder.encode(&b);
            let (ha, _) = Header::decode(&ea);
            let (hb, _) = Header::decode(&eb);
            // Normalize errors by the norms so Euclidean and Cosine bins are comparable.
            let scale = 2.0 * ha.l2_norm as f64 * hb.l2_norm as f64;

            let est = dist.distance(&ea, &eb);
            let exact = exact_distance(similarity, &a, &b, center.as_deref());
            bias += (est - exact) / scale;
            mae += ((est - exact) / scale).abs();
        }
        bias /= TRIALS as f64;
        mae /= TRIALS as f64;
        eprintln!(
            "SYM {similarity:?} center={} bias={bias:.4} mae={mae:.4}",
            center.is_some()
        );
        assert!(bias.abs() < 0.03, "bias {bias} too large (mae {mae})");
        assert!(mae < 0.15, "mae {mae} too large (bias {bias})");
    }

    #[test]
    fn symmetric_euclidean() {
        eval_symmetric(VectorSimilarity::Euclidean, None);
    }

    #[test]
    fn symmetric_euclidean_centered() {
        eval_symmetric(VectorSimilarity::Euclidean, Some(test_center(DIM)));
    }

    #[test]
    fn symmetric_cosine() {
        eval_symmetric(VectorSimilarity::Cosine, None);
    }

    #[test]
    fn symmetric_cosine_centered() {
        eval_symmetric(VectorSimilarity::Cosine, Some(test_center(DIM)));
    }

    fn eval_asymmetric(similarity: VectorSimilarity, center: Option<Vec<f32>>) {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0xa57a5717);
        let coder = Coder::new(center.clone());

        let mut bias = 0.0f64;
        let mut mae = 0.0f64;
        let mut covered = 0usize;
        for _ in 0..TRIALS {
            let rho: f32 = rng.random_range(0.0f32..1.0);
            let q = maybe_normalize(similarity, gauss_vec(&mut rng, DIM));
            let d = maybe_normalize(similarity, correlated(&mut rng, &q, rho));

            let qd = QueryDistance::new(similarity, &q, center.as_deref());
            let ed = coder.encode(&d);
            let (hd, _) = Header::decode(&ed);
            let qnorm = float32::l2_norm(subtract_center(&q, center.as_deref())) as f64;
            let scale = 2.0 * qnorm * hd.l2_norm as f64;

            let exact = exact_distance(similarity, &q, &d, center.as_deref());
            let est = qd.distance(&ed);
            bias += (est - exact) / scale;
            mae += ((est - exact) / scale).abs();

            // The estimated_distance error is a ~1 sigma statistical bound; Z=3 should almost
            // always contain the true distance.
            let ed_est = qd.estimated_distance(&ed);
            assert!(ed_est.error > 0.0, "error bound not populated");
            assert!((ed_est.distance - est).abs() < 1e-9);
            if (exact - ed_est.distance).abs() <= 3.0 * ed_est.error + 1e-9 {
                covered += 1;
            }
        }
        bias /= TRIALS as f64;
        mae /= TRIALS as f64;
        let coverage = covered as f64 / TRIALS as f64;
        eprintln!(
            "ASYM {similarity:?} center={} bias={bias:.4} mae={mae:.4} cov={coverage:.4}",
            center.is_some()
        );
        assert!(bias.abs() < 0.03, "bias {bias} too large (mae {mae})");
        assert!(mae < 0.15, "mae {mae} too large (bias {bias})");
        assert!(coverage > 0.95, "Z=3 coverage {coverage} too low");
    }

    #[test]
    fn asymmetric_euclidean() {
        eval_asymmetric(VectorSimilarity::Euclidean, None);
    }

    #[test]
    fn asymmetric_euclidean_centered() {
        eval_asymmetric(VectorSimilarity::Euclidean, Some(test_center(DIM)));
    }

    #[test]
    fn asymmetric_cosine() {
        eval_asymmetric(VectorSimilarity::Cosine, None);
    }

    #[test]
    fn asymmetric_cosine_centered() {
        eval_asymmetric(VectorSimilarity::Cosine, Some(test_center(DIM)));
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
