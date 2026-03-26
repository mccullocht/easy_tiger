//! TurboQuant quantizers: https://arxiv.org/pdf/2504.19874
//!
//! This uses separate implementations for MSE optimization (L2/Euclidean distance) and angular
//! (dot product) distance.

use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, RwLock},
};

use ndarray::{Array2, ArrayView1};
use ndarray_linalg::QR;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use rand::SeedableRng;

use crate::{F32VectorCoder, QueryVectorDistance};

struct JLTrans {
    q: Array2<f32>,
    q_inv: Array2<f32>,
}

impl JLTrans {
    fn new(d: usize, seed: u64) -> Self {
        // Generate a random orthogonal matrix using QR decomposition of a random matrix.
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
        let r: Array2<f32> = Array2::random_using((d, d), StandardNormal, &mut rng);
        let q = r.qr().map(|(q, _r)| q).expect("QR decomposition failed");
        let q_inv = q.t().to_owned();
        JLTrans { q, q_inv }
    }

    fn transform(&self, v: &[f32]) -> Vec<f32> {
        self.q.dot(&ArrayView1::from(v)).to_vec()
    }

    fn invert(&self, v: &[f32]) -> Vec<f32> {
        self.q_inv.dot(&ArrayView1::from(v)).to_vec()
    }
}

static JL_CACHE: LazyLock<RwLock<HashMap<(usize, u64), Arc<JLTrans>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

fn cached_jl_trans(d: usize, seed: u64) -> Arc<JLTrans> {
    {
        let m = JL_CACHE.read().expect("not poisoned");
        if let Some(jlt) = m.get(&(d, seed)) {
            return Arc::clone(jlt);
        }
    }

    let jlt = Arc::new(JLTrans::new(d, seed));
    let mut m = JL_CACHE.write().expect("not poisoned");
    Arc::clone(m.entry((d, seed)).or_insert(jlt))
}

fn codebook_bit(dim: usize) -> [f32; 2] {
    let v = (2.0 / (std::f64::consts::PI * dim as f64)).sqrt() as f32;
    [-v, v]
}

/// TurboQuant coder for 1-bit quantization that minimizes MSE. This is most suitable for l2
/// distance where minimizing MSE is equivalent to minimizing distance distortion.
pub struct MSE1Coder {
    dim: usize,
    t: Arc<JLTrans>,
    codebook: [f32; 2],
}

impl MSE1Coder {
    pub fn new(dim: usize, seed: u64) -> Self {
        let t = cached_jl_trans(dim, seed);
        let codebook = codebook_bit(dim);
        MSE1Coder { dim, t, codebook }
    }
}

impl F32VectorCoder for MSE1Coder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        assert_eq!(vector.len(), self.dim);
        let norm = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        out[..4].copy_from_slice(&norm.to_le_bytes());

        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        let normalized: Vec<f32> = vector.iter().map(|&x| x * inv_norm).collect();
        let transformed = self.t.transform(&normalized);

        let bits = &mut out[4..];
        bits.fill(0);
        for (i, &v) in transformed.iter().enumerate() {
            if v > 0.0 {
                bits[i / 8] |= 1 << (i % 8);
            }
        }
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        std::mem::size_of::<f32>() + dimensions.div_ceil(8)
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        assert_eq!(out.len(), self.dim);
        let norm = f32::from_le_bytes(encoded[..4].try_into().unwrap());
        let bits = &encoded[4..];

        let quantized: Vec<f32> = (0..self.dim)
            .map(|i| self.codebook[((bits[i / 8] >> (i % 8)) & 1) as usize])
            .collect();

        let inverted = self.t.invert(&quantized);
        for (o, v) in out.iter_mut().zip(inverted.iter()) {
            *o = v * norm;
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        (byte_len - std::mem::size_of::<f32>()) * 8
    }
}

// XXX what do I need to perform asymmetric l2 distance?
// - simplest: transform the vector, use the codebook to generate f32 vectors, compute l2 dist.
// - more complex: LUT it
//   * for each byte of input generate l2 dist to a sequence of 8 codebook entries.
//     256 entries * 4 bytes = 1KB per 8 dim. 256KB for 2048 dims. Sounds bad.
//   * for each nibble of input generate l2 dist to a sequence of 4 codebook entries.
//     16 entries * 4 bytes = 64 per 4 dim. 32KB for 2048 dims. Less bad!
//   * nibble but SQ the codebook entries. 8 KB for 2048 dims.
//   * LUT it just shuffle + add for each entry, but we probably also have to widen. multiply by 1
//     and just DOT it lol.

pub struct MSE1QueryDistance {
    /// The query vector after l2 normalization and JL transformation. This places the query in the
    /// same space as the quantized vectors which makes computation cheaper.
    tquery: Vec<f32>,
    /// L2 norm of the input vector before JLT.
    norm: f32,
    /// Codebook used to decode quantized vectors for comparison.
    codebook: [f32; 2],
}

impl MSE1QueryDistance {
    pub fn new(query: Vec<f32>) -> Self {
        let norm = query.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        let normalized: Vec<f32> = query.iter().map(|&x| x * inv_norm).collect();

        // XXX seed should be passed.
        let jlt = cached_jl_trans(query.len(), 42);
        let tquery = jlt.transform(&normalized);
        let codebook = codebook_bit(query.len());
        MSE1QueryDistance {
            tquery,
            norm,
            codebook,
        }
    }
}

impl QueryVectorDistance for MSE1QueryDistance {
    fn distance(&self, encoded: &[u8]) -> f64 {
        let norm = f32::from_le_bytes(encoded[..4].try_into().unwrap());
        let bits = &encoded[4..];
        // Compute dot product between the JL-transformed query and the quantized db vector.
        // Each codebook entry is ±√(2/πd), chosen to minimize MSE of reconstruction.
        // However, ⟨JL(q̂), codebook⟩ ≈ (2/π)·⟨q̂, db̂⟩ — biased low by 2/π.
        // Applying the asymmetric form with the exact stored norms:
        //   ‖q − db‖² = ‖q‖² + ‖db‖² − 2·⟨q, db⟩
        //              ≈ ‖q‖² + ‖db‖² − π·‖q‖·‖db‖·⟨JL(q̂), codebook⟩
        let dot: f32 = self
            .tquery
            .iter()
            .enumerate()
            .map(|(i, &q)| q * self.codebook[((bits[i / 8] >> (i % 8)) & 1) as usize])
            .sum();
        (self.norm * self.norm + norm * norm - std::f32::consts::PI * self.norm * norm * dot) as f64
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::F32VectorCoder;

    use super::MSE1Coder;

    // This test vector contains randomly generated numbers in [-1,1] but is not l2 normalized.
    // It has 19 elements -- long enough to trigger SIMD optimizations but with some remainder to
    // test scalar tail paths.
    const TEST_VECTOR: [f32; 19] = [
        -0.921, -0.061, 0.659, 0.67, 0.573, 0.431, 0.646, 0.001, -0.2, -0.428, 0.73, -0.704,
        -0.273, 0.539, -0.731, 0.436, 0.913, 0.694, 0.202,
    ];

    #[test]
    fn mse1_null_vector_decode() {
        let coder = MSE1Coder::new(256, 42);
        let vector = vec![0.0f32; 256];
        let decoded = coder.decode(&coder.encode(&vector));
        assert_abs_diff_eq!(decoded.as_slice(), vector.as_slice());
    }

    #[test]
    fn mse1_coding() {
        let coder = MSE1Coder::new(TEST_VECTOR.len(), 42);
        let encoded = coder.encode(&TEST_VECTOR);
        assert_eq!(encoded.len(), coder.byte_len(TEST_VECTOR.len()));
        let encoded_norm = f32::from_le_bytes(encoded[..4].try_into().unwrap());
        assert_abs_diff_eq!(
            encoded_norm,
            TEST_VECTOR.iter().map(|&x| x * x).sum::<f32>().sqrt(),
            epsilon = 0.00001
        );
        let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
        coder.decode_to(&encoded, &mut decoded);
        assert_abs_diff_eq!(
            decoded.as_slice(),
            [
                -0.3280949,
                -0.17939623,
                0.49887112,
                0.25178882,
                0.19527599,
                -0.16641621,
                0.74232966,
                0.08569759,
                -0.39014056,
                -0.5225581,
                0.15493955,
                -0.5223572,
                0.40732047,
                0.47973183,
                -0.65380883,
                -0.14245099,
                0.8079555,
                0.8074604,
                0.32286346
            ]
            .as_ref(),
            epsilon = 0.00001
        );
    }
}
