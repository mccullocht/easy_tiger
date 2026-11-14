//! TurboQuant quantizers: https://arxiv.org/pdf/2504.19874
//!
//! This uses separate implementations for MSE optimization (L2/Euclidean distance) and angular
//! (dot product) distance.
//!
//! TurboQuant assumes that inputs have been randomly rotated so that each coordinate of the unit
//! normalized vector is approximately N(0, 1/D). Rotation is the caller's responsibility, typically
//! via [`crate::prepare_vector`].

// TODO: optimize vector comparison using lookup tables.

pub mod codebook;
mod packing;

use std::borrow::Cow;

use crate::{F32VectorCoder, QueryVectorDistance, VectorSimilarity, float32, rabitq};

pub struct MSECoder<const B: usize, const N: usize> {
    /// Unscaled codebook; scaled by `1/sqrt(D)` for each input vector.
    codebook: [f32; N],
}

impl<const B: usize, const N: usize> MSECoder<B, N> {
    pub fn new(codebook: &[f32; N]) -> Self {
        Self {
            codebook: *codebook,
        }
    }
}

impl<const B: usize, const N: usize> F32VectorCoder for MSECoder<B, N> {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        crate::assert_finite_vector(vector);
        let norm = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(norm.is_finite(), "vector magnitude is not finite");
        out[..4].copy_from_slice(&norm.to_le_bytes());

        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        let codebook = codebook::scale(&self.codebook, vector.len());

        let bits = &mut out[4..];
        bits.fill(0);
        packing::pack::<B>(
            vector
                .iter()
                .map(|v| codebook::select_code(&codebook, *v * inv_norm)),
            bits,
        );
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        std::mem::size_of::<f32>() + (dimensions * B).div_ceil(8)
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        let norm = f32::from_le_bytes(encoded[..4].try_into().unwrap());
        let bits = &encoded[4..];

        // NB: the original dimensionality is not stored, so if D * B is not a multiple of 8 the
        // codebook scale is computed from the padded dimension count and will be slightly off.
        let codebook = codebook::scale(&self.codebook, self.dimensions(encoded.len()));
        for (o, i) in out.iter_mut().zip(packing::unpack::<B>(bits)) {
            *o = codebook[i as usize] * norm;
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        ((byte_len - std::mem::size_of::<f32>()) * 8).div_ceil(B)
    }
}

pub struct MSEQueryDistance<const B: usize, const N: usize> {
    similarity: VectorSimilarity,
    /// The query vector after l2 normalization. Callers are expected to have rotated the query in
    /// the same way as the quantized vectors.
    rquery: Vec<f32>,
    /// L2 norm of the input vector before transform.
    norm: f32,
    /// Codebook used to decode quantized vectors for comparison.
    codebook: [f32; N],
    /// `2 / A` where `A = E[Q(z)·z]` for `z ~ N(0,1)` with this codebook. Corrects for the
    /// inner product bias introduced by MSE-optimal quantizers in the Euclidean distance formula.
    euclidean_scale: f32,
}

/// Computes `A = E[Q(z)·z]` for `z ~ N(0,1)` using the given (unscaled) codebook.
///
/// MSE-optimal quantizers are biased for inner product estimation: `⟨q, Q(x)⟩ ≈ A·⟨q, x⟩`
/// where `A < 1`. The Euclidean correction factor is `2/A`.
fn inner_product_bias<const N: usize>(codebook: &[f32; N]) -> f32 {
    let inv_sqrt_2pi = (0.5f32 / std::f32::consts::PI).sqrt();
    let normal_pdf = |x: f32| -> f32 {
        if x.is_infinite() {
            0.0
        } else {
            inv_sqrt_2pi * (-x * x * 0.5).exp()
        }
    };
    let mut a = 0.0f32;
    for (k, &c) in codebook.iter().enumerate() {
        let t_prev = if k == 0 {
            f32::NEG_INFINITY
        } else {
            (codebook[k - 1] + c) * 0.5
        };
        let t_next = if k == N - 1 {
            f32::INFINITY
        } else {
            (c + codebook[k + 1]) * 0.5
        };
        a += c * (normal_pdf(t_prev) - normal_pdf(t_next));
    }
    a
}

impl<const B: usize, const N: usize> MSEQueryDistance<B, N> {
    pub fn new(similarity: VectorSimilarity, query: Cow<'_, [f32]>, codebook: &[f32; N]) -> Self {
        let mut rquery = query.into_owned();
        let norm = rquery.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        for d in rquery.iter_mut() {
            *d *= inv_norm
        }

        let euclidean_scale = 2.0 / inner_product_bias(codebook);
        let codebook = codebook::scale(codebook, rquery.len());
        Self {
            similarity,
            rquery,
            norm,
            codebook,
            euclidean_scale,
        }
    }
}

impl<const B: usize, const N: usize> QueryVectorDistance for MSEQueryDistance<B, N> {
    fn distance(&self, encoded: &[u8]) -> f64 {
        let norm = f32::from_le_bytes(encoded[..4].try_into().unwrap());
        let bits = &encoded[4..];
        // dot estimates A·⟨q̂, db̂⟩ where A = E[Q(z)·z] < 1 for MSE-optimal codebooks.
        // ‖q − db‖² = ‖q‖² + ‖db‖² − 2·‖q‖·‖db‖·⟨q̂, db̂⟩
        //            ≈ ‖q‖² + ‖db‖² − (2/A)·‖q‖·‖db‖·dot
        let dot: f32 = self
            .rquery
            .iter()
            .zip(packing::unpack::<B>(bits))
            .map(|(&q, d)| q * self.codebook[d as usize])
            .sum();
        mse_distance(self.similarity, self.norm, norm, dot, self.euclidean_scale)
    }
}

/// Compute distance from the query and doc norms and `dot`, the inner product of the normalized
/// query and the quantized normalized doc.
///
/// `euclidean_scale` is `2/A` where `A = E[Q(z)·z]`; see [`inner_product_bias`].
#[inline]
fn mse_distance(
    similarity: VectorSimilarity,
    qnorm: f32,
    dnorm: f32,
    dot: f32,
    euclidean_scale: f32,
) -> f64 {
    // Compute l2 distance even for angular similarity: if the vectors were centered they are no
    // longer unit length, but centering is a translation so the l2 distance is unaffected.
    let l2_dist = (qnorm * qnorm + dnorm * dnorm - euclidean_scale * qnorm * dnorm * dot) as f64;
    let d = match similarity {
        VectorSimilarity::Euclidean => l2_dist,
        // For unit vectors ‖q − d‖² = 2 − 2⟨q, d⟩, so this maps angular distance into [0,1].
        VectorSimilarity::Dot => (0.25 * l2_dist).clamp(0.0, 1.0),
    };
    crate::sanitize_distance(d, similarity.angular())
}

/// 1-bit MSE coder specialized to use the RaBitQ sign bit kernels.
///
/// With a symmetric 2 entry codebook `[-c, c]` the code for each dimension is just the sign bit, so
/// the vector is stored as a RaBitQ style sign code (1 for negative, turbo packed) and each
/// component decodes to `±c/√D`. Encoded format:
/// * l2 norm of the input (f32)
/// * number of negative components (u32), needed to compute inner products against a [`Query4`].
/// * turbo packed sign bits.
///
/// [`Query4`]: crate::rabitq::Query4
pub struct MSE1Coder {
    k: rabitq::Kernel,
    /// Positive (unscaled) codebook entry.
    centroid: f32,
}

impl MSE1Coder {
    const HEADER_LEN: usize = 8;

    pub fn new(codebook: &[f32; 2]) -> Self {
        assert_eq!(
            codebook[0], -codebook[1],
            "1-bit codebook must be symmetric"
        );
        Self {
            k: rabitq::Kernel::default(),
            centroid: codebook[1],
        }
    }

    fn split(encoded: &[u8]) -> (f32, u32, &[u8]) {
        let (header, bits) = encoded.split_at(Self::HEADER_LEN);
        let header = header.as_chunks::<4>().0;
        (
            f32::from_le_bytes(header[0]),
            u32::from_le_bytes(header[1]),
            bits,
        )
    }
}

impl F32VectorCoder for MSE1Coder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        crate::assert_finite_vector(vector);
        let norm = float32::l2_norm(vector);
        assert!(norm.is_finite(), "vector magnitude is not finite");

        // Normalization does not change the sign so there's no need to normalize before coding.
        let (header, bits) = out.split_at_mut(Self::HEADER_LEN);
        bits.fill(0);
        let negative = self.k.quantize_and_pack(vector, bits);
        header[..4].copy_from_slice(&norm.to_le_bytes());
        header[4..].copy_from_slice(&negative.to_le_bytes());
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        Self::HEADER_LEN + dimensions.div_ceil(8)
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        let (norm, _, bits) = Self::split(encoded);
        let magnitude = self.centroid / (out.len() as f32).sqrt() * norm;
        self.k.decode(bits, magnitude, out);
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        (byte_len - Self::HEADER_LEN) * 8
    }
}

/// Query distance for [`MSE1Coder`].
///
/// The query is quantized to 4 bits so that the inner product against the doc sign code can be
/// computed with bitwise AND + popcount rather than a float dot product.
pub struct MSE1QueryDistance {
    similarity: VectorSimilarity,
    query: rabitq::Query4,
    /// L2 norm of the input vector before transform.
    norm: f32,
    /// Positive (unscaled) codebook entry.
    centroid: f32,
    /// See [`MSEQueryDistance::euclidean_scale`].
    euclidean_scale: f32,
}

impl MSE1QueryDistance {
    pub fn new(similarity: VectorSimilarity, query: Cow<'_, [f32]>, codebook: &[f32; 2]) -> Self {
        assert_eq!(
            codebook[0], -codebook[1],
            "1-bit codebook must be symmetric"
        );
        let (query, norm) = float32::l2_normalize(query);
        Self {
            similarity,
            query: rabitq::Query4::new(&query),
            norm,
            centroid: codebook[1],
            euclidean_scale: 2.0 / inner_product_bias(codebook),
        }
    }
}

impl QueryVectorDistance for MSE1QueryDistance {
    fn distance(&self, encoded: &[u8]) -> f64 {
        let (norm, negative, bits) = MSE1Coder::split(encoded);
        // Query4::dot() treats each doc component as ±1/√D; scale to ±c/√D.
        let dot = self.centroid * self.query.dot(negative, bits) as f32;
        mse_distance(self.similarity, self.norm, norm, dot, self.euclidean_scale)
    }
}

#[cfg(test)]
mod tests {
    use super::codebook;

    // This test vector contains randomly generated numbers in [-1,1] but is not l2 normalized.
    // It has 19 elements -- long enough to trigger SIMD optimizations but with some remainder to
    // test scalar tail paths.
    const TEST_VECTOR: [f32; 19] = [
        -0.921, -0.061, 0.659, 0.67, 0.573, 0.431, 0.646, 0.001, -0.2, -0.428, 0.73, -0.704,
        -0.273, 0.539, -0.731, 0.436, 0.913, 0.694, 0.202,
    ];

    macro_rules! mse_tests {
        ($mod_name:ident, $B:literal, $centroids:expr, $coder:ty, $query_distance:ty, $header_len:literal) => {
            mod $mod_name {
                use approx::assert_abs_diff_eq;

                use crate::{F32VectorCoder, QueryVectorDistance, VectorSimilarity};

                use super::{TEST_VECTOR, codebook};
                #[allow(unused_imports)]
                use crate::turbo_quant::{
                    MSE1Coder, MSE1QueryDistance, MSECoder, MSEQueryDistance,
                };

                const DIM: usize = TEST_VECTOR.len();

                fn coder() -> $coder {
                    <$coder>::new($centroids)
                }

                fn query_distance(similarity: VectorSimilarity) -> $query_distance {
                    <$query_distance>::new(similarity, TEST_VECTOR.as_slice().into(), $centroids)
                }

                #[test]
                fn byte_len() {
                    let c = coder();
                    assert_eq!(c.byte_len(DIM), $header_len + (DIM * $B).div_ceil(8));
                }

                #[test]
                fn null_vector_roundtrip() {
                    let v = vec![0.0f32; DIM];
                    let c = coder();
                    let mut decoded = vec![0.0f32; DIM];
                    c.decode_to(&c.encode(&v), &mut decoded);
                    assert_abs_diff_eq!(decoded.as_slice(), v.as_slice());
                }

                // The L2 norm of the input is stored verbatim in the first 4 bytes of the
                // encoded buffer.
                #[test]
                fn stored_norm() {
                    let c = coder();
                    let encoded = c.encode(&TEST_VECTOR);
                    let stored_norm = f32::from_le_bytes(encoded[..4].try_into().unwrap());
                    let orig_norm: f32 = TEST_VECTOR.iter().map(|x| x * x).sum::<f32>().sqrt();
                    assert_abs_diff_eq!(stored_norm, orig_norm, epsilon = 1e-5);
                }

                // Euclidean distance from a vector to itself should be near 0.
                #[test]
                fn euclidean_self_distance() {
                    let encoded = coder().encode(&TEST_VECTOR);
                    let dist =
                        query_distance(VectorSimilarity::Euclidean).distance(&encoded) as f32;
                    let norm_sq: f32 = TEST_VECTOR.iter().map(|x| x * x).sum();
                    // Allow tolerance scaling with quantization error; even at 1-bit the
                    // self-distance should be well under half the squared norm.
                    assert!(
                        dist < norm_sq * 0.5,
                        "euclidean self-distance {dist} too large (norm_sq={norm_sq})"
                    );
                }

                // Dot distance from a normalized vector to itself should be near 0.
                #[test]
                fn dot_self_distance() {
                    let encoded = coder().encode(&TEST_VECTOR);
                    let dist = query_distance(VectorSimilarity::Dot).distance(&encoded);
                    assert!(dist < 0.1, "dot self-distance {dist} too large");
                }

                // A vector in the same direction as the query (scaled) should be closer than one
                // pointing in the opposite direction, for all similarity functions.
                #[test]
                fn distance_ordering() {
                    let near: Vec<f32> = TEST_VECTOR.iter().map(|&x| x * 1.1).collect();
                    let far: Vec<f32> = TEST_VECTOR.iter().map(|&x| -x * 2.0).collect();
                    let c = coder();
                    let near_enc = c.encode(&near);
                    let far_enc = c.encode(&far);

                    for similarity in VectorSimilarity::all() {
                        let qd = query_distance(similarity);
                        let near_dist = qd.distance(&near_enc);
                        let far_dist = qd.distance(&far_enc);
                        assert!(
                            near_dist < far_dist,
                            "{similarity:?}: near_dist={near_dist} should be < far_dist={far_dist}"
                        );
                    }
                }
            }
        };
    }

    mse_tests!(
        mse1,
        1,
        &codebook::CENTROIDS_1,
        MSE1Coder,
        MSE1QueryDistance,
        8
    );

    // Dot distance on centered vectors should still estimate the angular distance between the
    // original unit vectors, as centering is a translation that cancels in l2 distance.
    mod centered_dot {
        use rand::{RngExt, SeedableRng};

        use crate::{F32VectorCoding, VectorSimilarity, prepare_vector};

        const DIM: usize = 300;

        fn random_unit_vector(rng: &mut rand_xoshiro::Xoshiro256PlusPlus) -> Vec<f32> {
            let v = (0..DIM)
                .map(|_| rng.random_range(-1.0f32..1.0))
                .collect::<Vec<_>>();
            prepare_vector(v, None, true, None)
        }

        #[test]
        fn centered_dot() {
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0xce47e2);
            // Like real embeddings the data shares a mean direction rather than being centered
            // on the origin; use that mean direction as the center.
            let center = random_unit_vector(&mut rng);
            let pairs = (0..32)
                .map(|_| {
                    let mut gen_vector = || {
                        let noise = random_unit_vector(&mut rng);
                        let v = center.iter().zip(&noise).map(|(c, n)| c + n);
                        prepare_vector(v.collect::<Vec<_>>(), None, true, None)
                    };
                    (gen_vector(), gen_vector())
                })
                .collect::<Vec<_>>();

            for coding in [
                F32VectorCoding::TurboQuantM1,
                F32VectorCoding::TurboQuantM2,
                F32VectorCoding::TurboQuantM3,
                F32VectorCoding::TurboQuantM4,
                F32VectorCoding::TurboQuantM8,
            ] {
                let coder = coding.coder();
                for (q, d) in pairs.iter() {
                    let expected = 0.25
                        * q.iter()
                            .zip(d)
                            .map(|(a, b)| ((a - b) as f64).powi(2))
                            .sum::<f64>();
                    let qd = coding.query_distance_asymmetric(
                        VectorSimilarity::Dot,
                        prepare_vector(q, None, false, Some(&center)),
                    );
                    let encoded = coder.encode(&prepare_vector(d, None, false, Some(&center)));
                    let actual = qd.distance(&encoded);
                    assert!(
                        (actual - expected).abs() < 0.05,
                        "{coding}: actual={actual} expected={expected}"
                    );
                }
            }
        }
    }

    mod mse1_simd {
        use approx::assert_abs_diff_eq;
        use rand::{RngExt, SeedableRng};

        use crate::turbo_quant::{
            MSE1Coder, MSE1QueryDistance, codebook::CENTROIDS_1, inner_product_bias,
        };
        use crate::{F32VectorCoder, QueryVectorDistance, VectorSimilarity};

        // Long enough to cover several SIMD blocks with a remainder.
        const DIM: usize = 300;

        fn random_vector(rng: &mut rand_xoshiro::Xoshiro256PlusPlus) -> Vec<f32> {
            (0..DIM).map(|_| rng.random_range(-1.0f32..1.0)).collect()
        }

        #[test]
        fn decode_matches_scalar() {
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0x7a11);
            let coder = MSE1Coder::new(&CENTROIDS_1);
            for _ in 0..16 {
                let v = random_vector(&mut rng);
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                let magnitude = CENTROIDS_1[1] / (DIM as f32).sqrt() * norm;
                let expected = v
                    .iter()
                    .map(|&x| if x < 0.0 { -magnitude } else { magnitude })
                    .collect::<Vec<_>>();
                let encoded = coder.encode(&v);
                assert_eq!(encoded.len(), coder.byte_len(DIM));
                assert_eq!(coder.dimensions(encoded.len()), DIM.next_multiple_of(8));
                let mut decoded = vec![0.0f32; DIM];
                coder.decode_to(&encoded, &mut decoded);
                assert_abs_diff_eq!(decoded.as_slice(), expected.as_slice(), epsilon = 1e-6);
            }
        }

        // The 4-bit query estimate should be close to the exact float inner product of the query
        // and the decoded doc. Dot similarity expects unit vectors.
        #[test]
        fn query_distance_matches_float() {
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0xd07);
            let coder = MSE1Coder::new(&CENTROIDS_1);
            for _ in 0..16 {
                let q = crate::prepare_vector(random_vector(&mut rng), None, true, None);
                let d = crate::prepare_vector(random_vector(&mut rng), None, true, None);
                let qnorm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
                let encoded = coder.encode(&d);
                let mut decoded = vec![0.0f32; DIM];
                coder.decode_to(&encoded, &mut decoded);
                let dnorm = decoded.iter().map(|x| x * x).sum::<f32>().sqrt();
                // ⟨q̂, Q(d̂)⟩ / A where Q(d̂) has components ±c/√D.
                let dot = q.iter().zip(&decoded).map(|(a, b)| a * b).sum::<f32>() / qnorm
                    * (CENTROIDS_1[1] / dnorm)
                    / inner_product_bias(&CENTROIDS_1);

                let qd = MSE1QueryDistance::new(VectorSimilarity::Dot, q.into(), &CENTROIDS_1);
                let expected = dot.mul_add(-0.5, 0.5) as f64;
                assert_abs_diff_eq!(qd.distance(&encoded), expected, epsilon = 0.01);
            }
        }
    }

    mse_tests!(mse2, 2, &codebook::CENTROIDS_2, MSECoder<2, 4>, MSEQueryDistance<2, 4>, 4);
    mse_tests!(mse3, 3, &codebook::CENTROIDS_3, MSECoder<3, 8>, MSEQueryDistance<3, 8>, 4);
    mse_tests!(mse4, 4, &codebook::CENTROIDS_4, MSECoder<4, 16>, MSEQueryDistance<4, 16>, 4);
    mse_tests!(mse8, 8, &codebook::CENTROIDS_8, MSECoder<8, 256>, MSEQueryDistance<8, 256>, 4);
}
