//! TurboQuant quantizers: https://arxiv.org/pdf/2504.19874
//!
//! This uses separate implementations for MSE optimization (L2/Euclidean distance) and angular
//! (dot product) distance.
//!
//! TurboQuant assumes that inputs have been randomly rotated so that each coordinate of the unit
//! normalized vector is approximately N(0, 1/D). Rotation is the caller's responsibility, typically
//! via [`crate::prepare_vector`].

#[cfg(target_arch = "aarch64")]
mod aarch64;
pub mod codebook;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86_64;

use std::borrow::Cow;

use crate::packing::{TURBO_BLOCK_SIZE, TurboPacker, TurboUnpacker, block_dim};
use crate::{F32VectorCoder, QueryVectorDistance, VectorSimilarity, float32, rabitq};

pub struct MSECoder<const B: usize, const N: usize> {
    /// Unscaled codebook; scaled by `1/sqrt(D)` for each input vector.
    codebook: [f32; N],
    /// Unscaled decision thresholds for `codebook`. Rather than scaling these by `1/sqrt(D)`,
    /// input values are scaled by `sqrt(D)` when encoding.
    thresholds: [f32; N],
}

impl<const B: usize, const N: usize> MSECoder<B, N> {
    pub fn new(codebook: &[f32; N]) -> Self {
        Self {
            codebook: *codebook,
            thresholds: codebook::thresholds(codebook),
        }
    }
}

impl<const B: usize, const N: usize> F32VectorCoder for MSECoder<B, N> {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        crate::assert_finite_vector(vector);
        let norm = float32::l2_norm(vector);
        assert!(norm.is_finite(), "vector magnitude is not finite");
        out[..4].copy_from_slice(&norm.to_le_bytes());

        // Normalize and scale by sqrt(D) in one step so values may be compared against the
        // unscaled thresholds.
        let scale = if norm > 0.0 {
            (vector.len() as f32).sqrt() / norm
        } else {
            0.0
        };

        let bits = &mut out[4..];
        bits.fill(0);
        // Select all the codes for a block before packing them so that selection and packing
        // may each be vectorized. Each byte j of a block holds codes j, j + 16, j + 32, ...
        let (blocks, tail) = bits.as_chunks_mut::<TURBO_BLOCK_SIZE>();
        let per_block = TURBO_BLOCK_SIZE * 8 / B;
        let mut codes = [0u8; TURBO_BLOCK_SIZE * 8];
        for (block, values) in blocks.iter_mut().zip(vector.chunks(per_block)) {
            codebook::select_codes(&self.thresholds, values, scale, &mut codes);
            codes[values.len()..per_block].fill(0);
            for (j, b) in block.iter_mut().enumerate() {
                *b = (0..8 / B).fold(0, |acc, k| {
                    acc | (codes[j + k * TURBO_BLOCK_SIZE] << (k * B))
                });
            }
        }
        let mut packer = TurboPacker::<B>::new(tail);
        for v in vector.iter().skip(blocks.len() * per_block) {
            packer.push(codebook::select_code(&self.thresholds, *v * scale));
        }
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
        for (o, i) in out.iter_mut().zip(TurboUnpacker::<B>::new(bits)) {
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
        let dot = if B == 8 {
            dot8(&self.rquery, &self.codebook, bits)
        } else {
            self.rquery
                .iter()
                .zip(TurboUnpacker::<B>::new(bits))
                .map(|(&q, d)| q * self.codebook[d as usize])
                .sum()
        };
        mse_distance(self.similarity, self.norm, norm, dot, self.euclidean_scale)
    }
}

/// Inner product of `query` and an 8-bit turbo packed doc decoded through `codebook`.
///
/// With 8 bits per dimension turbo packing is the identity, so each byte is the code for the
/// corresponding dimension. Multiple independent accumulators break the loop carried dependency
/// on the sum; the table lookups themselves remain scalar loads.
#[inline]
fn dot8<const N: usize>(query: &[f32], codebook: &[f32; N], codes: &[u8]) -> f32 {
    const LANES: usize = 8;
    let len = query.len().min(codes.len());
    let (qc, qr) = query[..len].as_chunks::<LANES>();
    let (cc, cr) = codes[..len].as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];
    for (q, c) in qc.iter().zip(cc) {
        for i in 0..LANES {
            acc[i] += q[i] * codebook[c[i] as usize];
        }
    }
    let tail = qr
        .iter()
        .zip(cr)
        .map(|(&q, &c)| q * codebook[c as usize])
        .sum::<f32>();
    acc.iter().sum::<f32>() + tail
}

/// Kernels for computing the inner product of an i8 query and a turbo packed doc decoded through an
/// i8 lookup table.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum Kernel {
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
            Self::Neon => std::arch::is_aarch64_feature_detected!("dotprod"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => {
                use std::arch::is_x86_feature_detected as cpu_feature;
                cpu_feature!("avx512f") && cpu_feature!("avx512bw") && cpu_feature!("avx512vnni")
            }
            Self::Scalar => true,
        }
    }

    /// Compute `Σ query[i] * table[doc[i]]` where `doc` is a turbo packed `B` bit vector.
    ///
    /// `query_sum` must be the sum of all entries in `query`. Any doc dimensions beyond the end of
    /// `query` (padding) are ignored.
    ///
    /// # Panics
    /// If `query` is shorter than the number of dimensions covered by full blocks in `doc`.
    fn lut_dot<const B: usize>(
        self,
        query: &[i8],
        query_sum: i32,
        table: &[i8; 16],
        doc: &[u8],
    ) -> i32 {
        const { assert!(B == 2 || B == 4, "lut_dot only supports 2 or 4 bits") };
        assert!(
            doc.len() / TURBO_BLOCK_SIZE * block_dim(B) <= query.len(),
            "doc has more dimensions than query"
        );
        let _ = query_sum;
        match self {
            Self::Scalar => scalar::lut_dot::<B>(query, table, 0, doc),
            // SAFETY: feature availability was checked when the kernel was selected and query
            // length was checked above.
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe {
                if B == 2 {
                    aarch64::lut_dot2(query, table, doc)
                } else {
                    aarch64::lut_dot4(query, table, doc)
                }
            },
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => unsafe {
                if B == 2 {
                    x86_64::lut_dot2(query, query_sum, table, doc)
                } else {
                    x86_64::lut_dot4(query, query_sum, table, doc)
                }
            },
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

/// Query distance for [`MSECoder`] with 2 or 4 bits per dimension.
///
/// Both the normalized query and the codebook are quantized to i8. Each doc code is decoded
/// through a 16 entry table lookup into the quantized codebook and multiplied against the query
/// with an integer dot product, then the result is scaled back into float space.
///
/// Quantizing the codebook adds error that is small relative to the quantization error of the
/// codebook itself (~0.006 vs ~0.1 RMS at 4 bits).
pub struct MSELutQueryDistance<const B: usize, const N: usize> {
    similarity: VectorSimilarity,
    kernel: Kernel,
    /// The l2 normalized query quantized to i8.
    query: Vec<i8>,
    /// Sum of all entries in `query`.
    query_sum: i32,
    /// Codebook quantized to i8, padded to 16 entries.
    table: [i8; 16],
    /// L2 norm of the input vector before transform.
    norm: f32,
    /// Scales the integer dot product back to the inner product of the normalized query and the
    /// quantized normalized doc: `1 / (query_scale * codebook_scale * sqrt(D))`.
    dot_scale: f32,
    /// See [`MSEQueryDistance::euclidean_scale`].
    euclidean_scale: f32,
}

impl<const B: usize, const N: usize> MSELutQueryDistance<B, N> {
    pub fn new(similarity: VectorSimilarity, query: Cow<'_, [f32]>, codebook: &[f32; N]) -> Self {
        Self::with_kernel(Kernel::default(), similarity, query, codebook)
    }

    fn with_kernel(
        kernel: Kernel,
        similarity: VectorSimilarity,
        query: Cow<'_, [f32]>,
        codebook: &[f32; N],
    ) -> Self {
        const {
            assert!(
                B == 2 || B == 4,
                "lut query distance only supports 2 or 4 bits"
            );
            assert!(N == 1 << B, "codebook must have 2^B entries");
        };

        let (query, norm) = float32::l2_normalize(query);
        let qmax = query.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let qscale = if qmax > 0.0 { 127.0 / qmax } else { 0.0 };
        let dim = query.len();
        // The final full block of an encoded doc may contain padding codes if D * B is not a
        // multiple of the block size. Zero pad the query to cover every packed code so that SIMD
        // kernels can consume full blocks; zero entries do not contribute to the inner product.
        let padded_dim = crate::packing::byte_len(dim, B) * 8 / B;
        let query = query
            .iter()
            .map(|&x| (x * qscale).round() as i8)
            .chain(std::iter::repeat_n(0, padded_dim - dim))
            .collect::<Vec<_>>();
        let query_sum = query.iter().map(|&x| x as i32).sum();

        let cmax = codebook.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(cmax > 0.0 && cmax.is_finite(), "invalid codebook");
        let cscale = 127.0 / cmax;
        let mut table = [0i8; 16];
        for (t, &c) in table.iter_mut().zip(codebook) {
            *t = (c * cscale).round() as i8;
        }

        let dot_scale = if qscale > 0.0 {
            1.0 / (qscale * cscale * (dim as f32).sqrt())
        } else {
            0.0
        };
        Self {
            similarity,
            kernel,
            query,
            query_sum,
            table,
            norm,
            dot_scale,
            euclidean_scale: 2.0 / inner_product_bias(codebook),
        }
    }
}

impl<const B: usize, const N: usize> QueryVectorDistance for MSELutQueryDistance<B, N> {
    fn distance(&self, encoded: &[u8]) -> f64 {
        let norm = f32::from_le_bytes(encoded[..4].try_into().unwrap());
        let bits = &encoded[4..];
        let dot = self
            .kernel
            .lut_dot::<B>(&self.query, self.query_sum, &self.table, bits) as f32
            * self.dot_scale;
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
                    MSE1Coder, MSE1QueryDistance, MSECoder, MSELutQueryDistance, MSEQueryDistance,
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

    mse_tests!(mse2, 2, &codebook::CENTROIDS_2, MSECoder<2, 4>, MSELutQueryDistance<2, 4>, 4);
    mse_tests!(mse4, 4, &codebook::CENTROIDS_4, MSECoder<4, 16>, MSELutQueryDistance<4, 16>, 4);
    mse_tests!(mse2_float, 2, &codebook::CENTROIDS_2, MSECoder<2, 4>, MSEQueryDistance<2, 4>, 4);
    mse_tests!(mse4_float, 4, &codebook::CENTROIDS_4, MSECoder<4, 16>, MSEQueryDistance<4, 16>, 4);

    mod lut {
        use rand::{RngExt, SeedableRng};

        use crate::packing::byte_len;
        use crate::turbo_quant::{
            Kernel, MSECoder, MSELutQueryDistance, MSEQueryDistance, codebook,
        };
        use crate::{F32VectorCoder, QueryVectorDistance, VectorSimilarity};

        fn kernels() -> impl Iterator<Item = Kernel> {
            Kernel::CANDIDATES
                .iter()
                .copied()
                .filter(|k| *k != Kernel::Scalar && k.is_available())
        }

        fn kernel_matches_scalar<const B: usize>(seed: u64) {
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
            // Lengths cover empty, tail only, odd and even block counts, and block + tail.
            let dims = (0..=260).chain([511, 512, 1000, 1024, 1031]);
            for dim in dims {
                // Use extreme query values to exercise any intermediate overflow.
                // Queries are padded to cover every packed code; see MSELutQueryDistance.
                let query = (0..byte_len(dim, B) * 8 / B)
                    .map(|_| rng.random_range(i8::MIN..=i8::MAX))
                    .collect::<Vec<_>>();
                let query_sum = query.iter().map(|&x| x as i32).sum::<i32>();
                let table: [i8; 16] = std::array::from_fn(|_| rng.random_range(i8::MIN..=i8::MAX));
                // Codes beyond 2^B entries can't be produced by the packer, so mask them out.
                let mut table = table;
                table[(1 << B)..].fill(0);
                let doc = (0..byte_len(dim, B))
                    .map(|_| rng.random::<u8>())
                    .collect::<Vec<_>>();
                let expected = Kernel::Scalar.lut_dot::<B>(&query, query_sum, &table, &doc);
                for k in kernels() {
                    assert_eq!(
                        k.lut_dot::<B>(&query, query_sum, &table, &doc),
                        expected,
                        "{k:?} B={B} dim={dim}"
                    );
                }
            }
        }

        #[test]
        fn kernel_matches_scalar2() {
            kernel_matches_scalar::<2>(0x1a2);
        }

        #[test]
        fn kernel_matches_scalar4() {
            kernel_matches_scalar::<4>(0x1a4);
        }

        // Scalar LUT dot against a simple reference over the packed codes.
        #[test]
        fn scalar_matches_reference() {
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0x5ca1);
            for dim in [1, 15, 64, 100, 300] {
                let query = (0..dim)
                    .map(|_| rng.random_range(-127i8..=127))
                    .collect::<Vec<_>>();
                let codes = (0..dim)
                    .map(|_| rng.random_range(0u8..16))
                    .collect::<Vec<_>>();
                let table: [i8; 16] = std::array::from_fn(|i| (i as i32 * 16 - 120) as i8);
                let mut doc = vec![0u8; byte_len(dim, 4)];
                let mut packer = crate::packing::TurboPacker::<4>::new(&mut doc);
                for &c in &codes {
                    packer.push(c);
                }
                let expected = query
                    .iter()
                    .zip(&codes)
                    .map(|(&q, &c)| q as i32 * table[c as usize] as i32)
                    .sum::<i32>();
                let query_sum = query.iter().map(|&x| x as i32).sum::<i32>();
                assert_eq!(
                    Kernel::Scalar.lut_dot::<4>(&query, query_sum, &table, &doc),
                    expected
                );
            }
        }

        // The i8 LUT distance should closely track the float codebook distance.
        fn matches_float<const B: usize, const N: usize>(codebook: &[f32; N]) {
            const DIM: usize = 300;
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0xf10a7);
            let coder = MSECoder::<B, N>::new(codebook);
            let mut random_vector = || {
                let v: Vec<f32> = (0..DIM).map(|_| rng.random_range(-1.0f32..1.0)).collect();
                crate::prepare_vector(v, None, true, None)
            };
            for _ in 0..16 {
                let q = random_vector();
                let encoded = coder.encode(&random_vector());
                for similarity in VectorSimilarity::all() {
                    let expected =
                        MSEQueryDistance::<B, N>::new(similarity, q.as_slice().into(), codebook)
                            .distance(&encoded);
                    for k in Kernel::CANDIDATES
                        .iter()
                        .copied()
                        .filter(Kernel::is_available)
                    {
                        let actual = MSELutQueryDistance::<B, N>::with_kernel(
                            k,
                            similarity,
                            q.as_slice().into(),
                            codebook,
                        )
                        .distance(&encoded);
                        assert!(
                            (actual - expected).abs() < 0.01,
                            "{k:?} {similarity:?} B={B}: actual={actual} expected={expected}"
                        );
                    }
                }
            }
        }

        #[test]
        fn matches_float2() {
            matches_float::<2, 4>(&codebook::CENTROIDS_2);
        }

        #[test]
        fn matches_float4() {
            matches_float::<4, 16>(&codebook::CENTROIDS_4);
        }

        // The 8-bit fast path should match decoding through the generic turbo unpacker.
        #[test]
        fn dot8_matches_unpacker() {
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0xd078);
            let codebook = codebook::scale(&codebook::CENTROIDS_8, 300);
            for dim in (0..=40).chain([255, 300, 1031]) {
                let query = (0..dim)
                    .map(|_| rng.random_range(-1.0f32..1.0))
                    .collect::<Vec<_>>();
                let doc = (0..byte_len(dim, 8))
                    .map(|_| rng.random::<u8>())
                    .collect::<Vec<_>>();
                let expected = query
                    .iter()
                    .zip(crate::packing::TurboUnpacker::<8>::new(&doc))
                    .map(|(&q, c)| q * codebook[c as usize])
                    .sum::<f32>();
                let actual = crate::turbo_quant::dot8(&query, &codebook, &doc);
                assert!(
                    (actual - expected).abs() <= 1e-5 * (dim as f32).max(1.0),
                    "dim={dim} actual={actual} expected={expected}"
                );
            }
        }

        #[test]
        fn zero_query() {
            let coder = MSECoder::<4, 16>::new(&codebook::CENTROIDS_4);
            let encoded = coder.encode(&[0.5f32; 40]);
            let q = vec![0.0f32; 40];
            let d = MSELutQueryDistance::<4, 16>::new(
                VectorSimilarity::Euclidean,
                q.into(),
                &codebook::CENTROIDS_4,
            )
            .distance(&encoded);
            assert!(d.is_finite(), "{d}");
        }
    }
    mse_tests!(mse8, 8, &codebook::CENTROIDS_8, MSECoder<8, 256>, MSEQueryDistance<8, 256>, 4);
}
