//! TurboQuant quantizers: https://arxiv.org/pdf/2504.19874
//!
//! This uses separate implementations for MSE optimization (L2/Euclidean distance) and angular
//! (dot product) distance.

mod codebook;
mod rotate;

use crate::{F32VectorCoder, QueryVectorDistance, VectorSimilarity};

fn codebook_bit(dim: usize) -> [f32; 2] {
    let v = (2.0 / (std::f64::consts::PI * dim as f64)).sqrt() as f32;
    [-v, v]
}

/// TurboQuant coder for 1-bit quantization that minimizes MSE. This is most suitable for l2
/// distance where minimizing MSE is equivalent to minimizing distance distortion.
pub struct MSE1Coder {
    dim: usize,
    rotator: rotate::Rotator,
    codebook: [f32; 2],
}

impl MSE1Coder {
    pub fn new(dim: usize, seed: u64) -> Self {
        let rotator = rotate::Rotator::new(dim, seed);
        let codebook = codebook::mse1(dim);
        MSE1Coder {
            dim,
            rotator,
            codebook,
        }
    }
}

impl F32VectorCoder for MSE1Coder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        assert_eq!(vector.len(), self.dim);
        let norm = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        out[..4].copy_from_slice(&norm.to_le_bytes());

        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        let normalized: Vec<f32> = vector.iter().map(|&x| x * inv_norm).collect();
        let transformed = self.rotator.forward(&normalized);

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

        let inverted = self.rotator.backward(&quantized);
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
    /// The query vector after l2 normalization and rotation. This places the query in the
    /// same space as the quantized vectors which makes computation cheaper.
    rquery: Vec<f32>,
    /// L2 norm of the input vector before JLT.
    norm: f32,
    /// Codebook used to decode quantized vectors for comparison.
    codebook: [f32; 2],
}

impl MSE1QueryDistance {
    pub fn new(query: Vec<f32>, seed: u64) -> Self {
        let norm = query.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        let normalized: Vec<f32> = query.iter().map(|&x| x * inv_norm).collect();

        let rotator = rotate::Rotator::new(query.len(), seed);
        let rquery = rotator.forward(&normalized);
        let codebook = codebook::mse1(rquery.len());
        MSE1QueryDistance {
            rquery,
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
            .rquery
            .iter()
            .enumerate()
            .map(|(i, &q)| q * self.codebook[((bits[i / 8] >> (i % 8)) & 1) as usize])
            .sum();
        (self.norm * self.norm + norm * norm - std::f32::consts::PI * self.norm * norm * dot) as f64
    }
}

// sqrt(pi/2) is used for scaling the decoded qjl vector.
const SQRT_PI_FRAC_2: f32 = 1.25331414;

/// Produces a 1-bit quantization optimized for inner product (angular) similarity.
pub struct Prod1Coder {
    dim: usize,
    rotator: rotate::Rotator,
}

impl Prod1Coder {
    pub fn new(dim: usize, seed: u64) -> Self {
        let rotator = rotate::Rotator::new(dim, seed);
        Self { dim, rotator }
    }
}

impl F32VectorCoder for Prod1Coder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        assert_eq!(vector.len(), self.dim);
        let norm = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        out[..4].copy_from_slice(&norm.to_le_bytes());

        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        let normalized: Vec<f32> = vector.iter().map(|&x| x * inv_norm).collect();
        let transformed = self.rotator.forward(&normalized);

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

        let codebook = [-1.0f32, 1.0f32];
        let quantized: Vec<f32> = (0..self.dim)
            .map(|i| codebook[((bits[i / 8] >> (i % 8)) & 1) as usize])
            .collect();

        let inverted = self.rotator.backward(&quantized);
        let scale = SQRT_PI_FRAC_2 * norm / (self.dim as f32).sqrt();
        for (o, v) in out.iter_mut().zip(inverted.iter()) {
            *o = v * scale;
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        (byte_len - std::mem::size_of::<f32>()) * 8
    }
}

pub struct Prod1QueryDistance {
    rquery: Vec<f32>,
    scale: f32,
}

impl Prod1QueryDistance {
    pub fn new(mut query: Vec<f32>, seed: u64) -> Self {
        let norm = query.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        for d in query.iter_mut() {
            *d *= inv_norm;
        }

        let rotator = rotate::Rotator::new(query.len(), seed);
        let rquery = rotator.forward(&query);
        let scale = (std::f32::consts::PI / (2.0 * query.len() as f32)).sqrt() * norm;

        Self { rquery, scale }
    }
}

impl QueryVectorDistance for Prod1QueryDistance {
    fn distance(&self, encoded: &[u8]) -> f64 {
        let norm = f32::from_le_bytes(encoded[..4].try_into().unwrap());
        let codebook = [-1.0f32, 1.0f32];
        let bits = &encoded[4..];
        let dot_unnormalized: f32 = self
            .rquery
            .iter()
            .enumerate()
            .map(|(i, &q)| q * codebook[((bits[i / 8] >> (i % 8)) & 1) as usize])
            .sum();
        let dot = dot_unnormalized * self.scale * norm;
        dot.mul_add(-0.5, 0.5).into()
    }
}

/// Produces a 2-bit quantization suitable for either l2 or angular similarity.
///
/// The vector is split into two parts: a 1-bit representation optimized for MSE, and a 1-bit
/// representation of the residuals optimized for inner product. These parts can be combined
/// to decode the vector or for asymmetric distance computation.
pub struct Prod2Coder {
    dim: usize,
    mse: rotate::Rotator,
    qjl: rotate::Rotator,
}

impl Prod2Coder {
    pub fn new(dim: usize, mse_seed: u64, qjl_seed: u64) -> Self {
        let mse = rotate::Rotator::new(dim, mse_seed);
        let qjl = rotate::Rotator::new(dim, qjl_seed);
        Self { dim, mse, qjl }
    }
}

impl F32VectorCoder for Prod2Coder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        assert_eq!(vector.len(), self.dim);
        let norm = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        out[..4].copy_from_slice(&norm.to_le_bytes());
        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        let mut scratch: Vec<f32> = vector.iter().map(|&x| x * inv_norm).collect();
        let mut mse_vec = self.mse.forward(&scratch);

        let bitvec_len = self.dim.div_ceil(8);
        let mse_bytes = &mut out[8..(8 + bitvec_len)];
        mse_bytes.fill(0);

        // Compute the MSE 1-bit sign vector of mse_vec and replace it's contents with the
        // dequantized value. This vector must be inverted before computing the residual.
        let codebook = codebook_bit(self.dim);
        for (i, v) in mse_vec.iter_mut().enumerate() {
            if *v > 0.0 {
                mse_bytes[i / 8] |= 1 << (i % 8);
                *v = codebook[1];
            } else {
                *v = codebook[0];
            }
        }
        let mse_reconstructed = self.mse.backward(&mse_vec);

        // Compute the residual vector for QJL encoding. Compute the dot product as we will need
        // to store the l2 norm of the residual.
        let mut residual_dot = 0.0f32;
        for (r, &rec) in scratch.iter_mut().zip(mse_reconstructed.iter()) {
            *r -= rec;
            residual_dot += *r * *r;
        }

        let residual_norm = residual_dot.sqrt();
        out[4..8].copy_from_slice(&residual_norm.to_le_bytes());
        let qjl_vec = self.qjl.forward(&scratch);
        let qjl_bytes = &mut out[(8 + bitvec_len)..];
        qjl_bytes.fill(0);
        for (i, &v) in qjl_vec.iter().enumerate() {
            if v > 0.0 {
                qjl_bytes[i / 8] |= 1 << (i % 8);
            }
        }
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        (std::mem::size_of::<f32>() + dimensions.div_ceil(8)) * 2
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        assert_eq!(out.len(), self.dim);
        let norm = f32::from_le_bytes(encoded[..4].try_into().unwrap());
        let residual_norm = f32::from_le_bytes(encoded[4..8].try_into().unwrap());

        let bitvec_len = self.dim.div_ceil(8);
        let mse_bits = &encoded[8..(8 + bitvec_len)];
        let mse_codebook = codebook_bit(self.dim);
        for (i, o) in out.iter_mut().enumerate() {
            *o = mse_codebook[((mse_bits[i / 8] >> (i % 8)) & 1) as usize];
        }
        let mse_vec = self.mse.backward(out);

        let qjl_bits = &encoded[(8 + bitvec_len)..];
        let qjl_codebook = [-1.0f32, 1.0f32];
        let qjl_scale = SQRT_PI_FRAC_2 * residual_norm / (self.dim as f32).sqrt();
        for (i, o) in out.iter_mut().enumerate() {
            *o = qjl_codebook[((qjl_bits[i / 8] >> (i % 8)) & 1) as usize];
        }
        let qjl_vec = self.qjl.backward(out);

        for ((mse, qjl), o) in mse_vec.iter().zip(qjl_vec.iter()).zip(out.iter_mut()) {
            *o = (mse + qjl * qjl_scale) * norm;
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        (byte_len - std::mem::size_of::<f32>() * 2) * 4
    }
}

pub struct Prod2QueryDistance {
    similarity: VectorSimilarity,
    mse_query: Vec<f32>,
    qjl_query: Vec<f32>,
    norm: f32,
    scale: f32,
}

impl Prod2QueryDistance {
    pub fn new(
        similarity: VectorSimilarity,
        mut query: Vec<f32>,
        mse_seed: u64,
        qjl_seed: u64,
    ) -> Self {
        let norm = query.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        for d in query.iter_mut() {
            *d *= inv_norm;
        }

        let mse = rotate::Rotator::new(query.len(), mse_seed);
        let mse_query = mse.forward(&query);
        let qjl = rotate::Rotator::new(query.len(), qjl_seed);
        let qjl_query = qjl.forward(&query);
        let scale = SQRT_PI_FRAC_2 / (query.len() as f32).sqrt();

        Self {
            similarity,
            mse_query,
            qjl_query,
            norm,
            scale,
        }
    }

    fn dim(&self) -> usize {
        self.mse_query.len()
    }

    fn decode_vector<'a>(&self, encoded: &'a [u8]) -> (f32, f32, &'a [u8], &'a [u8]) {
        let norm = f32::from_le_bytes(encoded[..4].try_into().unwrap());
        let residual_norm = f32::from_le_bytes(encoded[4..8].try_into().unwrap());

        let bitvec_len = self.dim().div_ceil(8);
        let mse_bits = &encoded[8..(8 + bitvec_len)];
        let qjl_bits = &encoded[(8 + bitvec_len)..];
        (norm, residual_norm, mse_bits, qjl_bits)
    }

    fn dot_unnormalized(&self, mse_bits: &[u8], qjl_bits: &[u8], residual_norm: f32) -> f32 {
        let mse_codebook = codebook_bit(self.dim());
        let qjl_codebook = [-1.0f32, 1.0f32];

        let mut mse_dot = 0.0f32;
        let mut qjl_dot = 0.0f32;
        for (i, (&qm, &qq)) in self.mse_query.iter().zip(self.qjl_query.iter()).enumerate() {
            let dm = mse_codebook[((mse_bits[i / 8] >> (i % 8)) & 1) as usize];
            mse_dot += qm * dm;

            let dq = qjl_codebook[((qjl_bits[i / 8] >> (i % 8)) & 1) as usize];
            qjl_dot += qq * dq;
        }

        // Scale and the residual norm can be factored out and applied at the end.
        // Result estimates ⟨y_norm, x_norm⟩ (cosine similarity).
        mse_dot + qjl_dot * self.scale * residual_norm
    }
}

impl QueryVectorDistance for Prod2QueryDistance {
    fn distance(&self, encoded: &[u8]) -> f64 {
        let (norm, residual_norm, mse_bits, qjl_bits) = self.decode_vector(encoded);
        // dot_cos estimates ⟨y_norm, x_norm⟩ (cosine similarity).
        let dot_cos = self.dot_unnormalized(mse_bits, qjl_bits, residual_norm);
        match self.similarity {
            VectorSimilarity::Euclidean => self.norm * self.norm + norm * norm - 2.0 * dot_cos,
            VectorSimilarity::Dot | VectorSimilarity::Cosine => dot_cos.mul_add(-0.5, 0.5),
        }
        .into()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::{
        F32VectorCoder,
        turbo_quant::{Prod1Coder, Prod2Coder},
    };

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

    #[test]
    fn prod1_coding() {
        let coder = Prod1Coder::new(TEST_VECTOR.len(), 42);
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
        // Check that cosine similarity between decoded and original is reasonable for 1-bit.
        let dot: f32 = decoded.iter().zip(TEST_VECTOR.iter()).map(|(d, &v)| d * v).sum();
        let decoded_norm: f32 = decoded.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let input_norm: f32 = TEST_VECTOR.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(dot / (decoded_norm * input_norm) > 0.5);
    }

    #[test]
    fn prod2_coding() {
        let coder = Prod2Coder::new(TEST_VECTOR.len(), 42, 43);
        let encoded = coder.encode(&TEST_VECTOR);
        assert_eq!(encoded.len(), coder.byte_len(TEST_VECTOR.len()));
        let encoded_norm = f32::from_le_bytes(encoded[..4].try_into().unwrap());
        let encoded_residual_norm = f32::from_le_bytes(encoded[4..8].try_into().unwrap());
        assert_abs_diff_eq!(
            encoded_norm,
            TEST_VECTOR.iter().map(|&x| x * x).sum::<f32>().sqrt(),
            epsilon = 0.00001
        );
        assert!(encoded_residual_norm < 1.0);
        let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
        coder.decode_to(&encoded, &mut decoded);
        let dot: f32 = decoded.iter().zip(TEST_VECTOR.iter()).map(|(d, &v)| d * v).sum();
        let decoded_norm: f32 = decoded.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let input_norm: f32 = TEST_VECTOR.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(dot / (decoded_norm * input_norm) > 0.5);
    }
}
