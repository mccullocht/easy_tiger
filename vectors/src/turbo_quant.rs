//! TurboQuant quantizers: https://arxiv.org/pdf/2504.19874
//!
//! This uses separate implementations for MSE optimization (L2/Euclidean distance) and angular
//! (dot product) distance.

// TODO: optimize vector comparison using lookup tables.

pub mod codebook;
mod packing;

use crate::{F32VectorCoder, QueryVectorDistance, VectorSimilarity, rotate::Rotator};

pub struct MSECoder<const B: usize, const N: usize> {
    dim: usize,
    rotator: Rotator,
    codebook: [f32; N],
}

impl<const B: usize, const N: usize> MSECoder<B, N> {
    pub fn new(dim: usize, seed: u64, codebook: &[f32; N]) -> Self {
        Self {
            dim,
            rotator: Rotator::new(dim, seed),
            codebook: codebook::scale(codebook, dim),
        }
    }
}

impl<const B: usize, const N: usize> F32VectorCoder for MSECoder<B, N> {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        assert_eq!(vector.len(), self.dim);
        let norm = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        out[..4].copy_from_slice(&norm.to_le_bytes());

        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        let normalized: Vec<f32> = vector.iter().map(|&x| x * inv_norm).collect();
        let transformed = self.rotator.forward(&normalized);

        let bits = &mut out[4..];
        bits.fill(0);
        packing::pack::<B>(
            transformed
                .iter()
                .map(|v| codebook::select_code(&self.codebook, *v)),
            bits,
        );
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        std::mem::size_of::<f32>() + (dimensions * B).div_ceil(8)
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        let norm = f32::from_le_bytes(encoded[..4].try_into().unwrap());
        let bits = &encoded[4..];

        let quantized: Vec<f32> = packing::unpack::<B>(bits)
            .map(|i| self.codebook[i as usize])
            .collect();
        let inverted = self.rotator.backward(&quantized);
        for (o, v) in out.iter_mut().zip(inverted.iter()) {
            *o = v * norm;
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        ((byte_len - std::mem::size_of::<f32>()) * 8).div_ceil(B)
    }
}

pub struct MSEQueryDistance<const B: usize, const N: usize> {
    similarity: VectorSimilarity,
    /// The query vector after l2 normalization and rotation. This places the query in the
    /// same space as the quantized vectors which makes computation cheaper.
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
    pub fn new(
        similarity: VectorSimilarity,
        mut query: Vec<f32>,
        seed: u64,
        codebook: &[f32; N],
    ) -> Self {
        let norm = query.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        for d in query.iter_mut() {
            *d = *d * inv_norm
        }

        let rotator = Rotator::new(query.len(), seed);
        let rquery = rotator.forward(&query);
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
        match self.similarity {
            VectorSimilarity::Euclidean => {
                (self.norm * self.norm + norm * norm
                    - self.euclidean_scale * self.norm * norm * dot) as f64
            }
            // The query and the doc were already normalized so the dot product is sufficient.
            VectorSimilarity::Cosine | VectorSimilarity::Dot => dot.mul_add(-0.5, 0.5).into(),
        }
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

    const MSE_SEED: u64 = 42;

    macro_rules! mse_tests {
        ($mod_name:ident, $B:literal, $N:literal, $centroids:expr) => {
            mod $mod_name {
                use approx::assert_abs_diff_eq;

                use crate::{F32VectorCoder, QueryVectorDistance, VectorSimilarity};

                use super::{MSE_SEED, TEST_VECTOR, codebook};
                use crate::turbo_quant::{MSECoder, MSEQueryDistance};

                const DIM: usize = TEST_VECTOR.len();

                fn coder() -> MSECoder<$B, $N> {
                    MSECoder::<$B, $N>::new(DIM, MSE_SEED, $centroids)
                }

                fn query_distance(similarity: VectorSimilarity) -> MSEQueryDistance<$B, $N> {
                    MSEQueryDistance::<$B, $N>::new(
                        similarity,
                        TEST_VECTOR.to_vec(),
                        MSE_SEED,
                        $centroids,
                    )
                }

                #[test]
                fn byte_len() {
                    let c = coder();
                    assert_eq!(c.byte_len(DIM), 4 + (DIM * $B).div_ceil(8));
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

                // Cosine distance from a normalized vector to itself should be near 0.
                #[test]
                fn cosine_self_distance() {
                    let encoded = coder().encode(&TEST_VECTOR);
                    let dist = query_distance(VectorSimilarity::Cosine).distance(&encoded);
                    assert!(dist < 0.1, "cosine self-distance {dist} too large");
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

    mse_tests!(mse1, 1, 2, &codebook::CENTROIDS_1);
    mse_tests!(mse2, 2, 4, &codebook::CENTROIDS_2);
    mse_tests!(mse3, 3, 8, &codebook::CENTROIDS_3);
    mse_tests!(mse4, 4, 16, &codebook::CENTROIDS_4);
    mse_tests!(mse8, 8, 256, &codebook::CENTROIDS_8);
}
