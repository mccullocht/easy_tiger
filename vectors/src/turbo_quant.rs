//! TurboQuant quantizers: https://arxiv.org/pdf/2504.19874
//!
//! This uses separate implementations for MSE optimization (L2/Euclidean distance) and angular
//! (dot product) distance.

// TODO: optimize vector comparison using lookup tables.

pub mod codebook;
mod packing;
mod rotate;

use std::ops::Range;

use crate::{F32VectorCoder, QueryVectorDistance, VectorSimilarity};

pub struct MSECoder<const B: usize, const N: usize> {
    dim: usize,
    rotator: rotate::Rotator,
    codebook: [f32; N],
}

impl<const B: usize, const N: usize> MSECoder<B, N> {
    pub fn new(dim: usize, seed: u64, codebook: &[f32; N]) -> Self {
        Self {
            dim,
            rotator: rotate::Rotator::new(dim, seed),
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

        let rotator = rotate::Rotator::new(query.len(), seed);
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
    similarity: VectorSimilarity,
    rquery: Vec<f32>,
    /// L2 norm of the query vector before normalization and rotation.
    norm: f32,
    /// `sqrt(π / (2d)) * query_norm` — scales the raw dot product to estimate `⟨q, db⟩`.
    scale: f32,
}

impl Prod1QueryDistance {
    pub fn new(similarity: VectorSimilarity, mut query: Vec<f32>, seed: u64) -> Self {
        let norm = query.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        for d in query.iter_mut() {
            *d *= inv_norm;
        }

        let rotator = rotate::Rotator::new(query.len(), seed);
        let rquery = rotator.forward(&query);
        let scale = (std::f32::consts::PI / (2.0 * query.len() as f32)).sqrt() * norm;

        Self {
            similarity,
            rquery,
            norm,
            scale,
        }
    }
}

impl QueryVectorDistance for Prod1QueryDistance {
    fn distance(&self, encoded: &[u8]) -> f64 {
        let db_norm = f32::from_le_bytes(encoded[..4].try_into().unwrap());
        let codebook = [-1.0f32, 1.0f32];
        let bits = &encoded[4..];
        let dot_unnormalized: f32 = self
            .rquery
            .iter()
            .enumerate()
            .map(|(i, &q)| q * codebook[((bits[i / 8] >> (i % 8)) & 1) as usize])
            .sum();
        // dot_unnormalized * scale * db_norm ≈ ⟨q, db⟩
        match self.similarity {
            VectorSimilarity::Euclidean => {
                let dot = dot_unnormalized * self.scale * db_norm;
                (self.norm * self.norm + db_norm * db_norm - 2.0 * dot) as f64
            }
            VectorSimilarity::Dot | VectorSimilarity::Cosine => {
                let dot = dot_unnormalized * self.scale * db_norm;
                dot.mul_add(-0.5, 0.5).into()
            }
        }
    }
}

fn prod_vec_bounds(dim: usize, mse_bits: usize) -> (Range<usize>, Range<usize>) {
    let mse_bitvec_len = (dim * mse_bits).div_ceil(8);
    let mse_range = 8..(8 + mse_bitvec_len);
    let qjl_range = (8 + mse_bitvec_len)..(8 + mse_bitvec_len + dim.div_ceil(8));
    (mse_range, qjl_range)
}

struct ProdVector<'a> {
    norm: f32,
    residual_norm: f32,
    mse_vec: &'a [u8],
    qjl_vec: &'a [u8],
}

impl<'a> ProdVector<'a> {
    fn from_encoded<const B: usize>(encoded: &'a [u8], dim: usize) -> Self {
        let norm = f32::from_le_bytes(encoded[..4].try_into().unwrap());
        let residual_norm = f32::from_le_bytes(encoded[4..8].try_into().unwrap());

        let (mse_range, qjl_range) = prod_vec_bounds(dim, B);
        let mse_vec = &encoded[mse_range];
        let qjl_vec = &encoded[qjl_range];
        Self {
            norm,
            residual_norm,
            mse_vec,
            qjl_vec,
        }
    }
}

const QJL_CODEBOOK: [f32; 2] = [-1.0f32, 1.0f32];

pub struct ProdCoder<const B: usize, const N: usize> {
    dim: usize,
    mse: rotate::Rotator,
    mse_codebook: [f32; N],
    qjl: rotate::Rotator,
}

impl<const B: usize, const N: usize> ProdCoder<B, N> {
    pub fn new(dim: usize, mse_seed: u64, qjl_seed: u64, codebook: &[f32; N]) -> Self {
        let mse = rotate::Rotator::new(dim, mse_seed);
        let qjl = rotate::Rotator::new(dim, qjl_seed);
        Self {
            dim,
            mse,
            mse_codebook: codebook::scale::<N>(codebook, dim),
            qjl,
        }
    }
}

impl<const B: usize, const N: usize> F32VectorCoder for ProdCoder<B, N> {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        assert_eq!(vector.len(), self.dim);
        let norm = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        out[..4].copy_from_slice(&norm.to_le_bytes());
        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        let mut scratch: Vec<f32> = vector.iter().map(|&x| x * inv_norm).collect();
        let mut mse_vec = self.mse.forward(&scratch);

        let (mse_bounds, qjl_bounds) = prod_vec_bounds(self.dim, B);
        let mse_bytes = &mut out[mse_bounds];
        mse_bytes.fill(0);
        packing::pack::<B>(
            mse_vec.iter_mut().map(|v| {
                let code = codebook::select_code(&self.mse_codebook, *v);
                *v = self.mse_codebook[code as usize];
                code as u8
            }),
            mse_bytes,
        );
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
        let qjl_bytes = &mut out[qjl_bounds];
        qjl_bytes.fill(0);
        packing::pack::<1>(
            qjl_vec.iter().map(|&v| if v > 0.0 { 1 } else { 0 }),
            qjl_bytes,
        );
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        std::mem::size_of::<f32>() * 2 + (dimensions * B).div_ceil(8) + dimensions.div_ceil(8)
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        assert_eq!(out.len(), self.dim);
        let vec = ProdVector::from_encoded::<B>(encoded, self.dim);

        for (c, o) in packing::unpack::<B>(vec.mse_vec).zip(out.iter_mut()) {
            *o = self.mse_codebook[c as usize];
        }
        let mse_vec = self.mse.backward(out);

        let qjl_scale = SQRT_PI_FRAC_2 * vec.residual_norm / (self.dim as f32).sqrt();
        for (c, o) in packing::unpack::<1>(vec.qjl_vec).zip(out.iter_mut()) {
            *o = QJL_CODEBOOK[c as usize];
        }
        let qjl_vec = self.qjl.backward(out);

        for ((mse, qjl), o) in mse_vec.iter().zip(qjl_vec.iter()).zip(out.iter_mut()) {
            *o = (mse + qjl * qjl_scale) * vec.norm;
        }
    }

    fn dimensions(&self, _byte_len: usize) -> usize {
        self.dim
    }
}

pub struct ProdQueryDistance<const B: usize, const N: usize> {
    similarity: VectorSimilarity,
    mse_query: Vec<f32>,
    mse_codebook: [f32; N],
    qjl_query: Vec<f32>,
    norm: f32,
    scale: f32,
}

impl<const B: usize, const N: usize> ProdQueryDistance<B, N> {
    pub fn new(
        similarity: VectorSimilarity,
        mut query: Vec<f32>,
        mse_seed: u64,
        qjl_seed: u64,
        mse_codebook: &[f32; N],
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
            mse_codebook: codebook::scale(mse_codebook, query.len()),
            qjl_query,
            norm,
            scale,
        }
    }

    fn dim(&self) -> usize {
        self.mse_query.len()
    }
}

impl<const B: usize, const N: usize> QueryVectorDistance for ProdQueryDistance<B, N> {
    fn distance(&self, encoded: &[u8]) -> f64 {
        let vec = ProdVector::from_encoded::<B>(encoded, self.dim());
        let mse_dot: f32 = self
            .mse_query
            .iter()
            .zip(packing::unpack::<B>(vec.mse_vec))
            .map(|(&q, c)| q * self.mse_codebook[c as usize])
            .sum();
        let qjl_dot: f32 = self
            .qjl_query
            .iter()
            .zip(packing::unpack::<1>(vec.qjl_vec))
            .map(|(&q, c)| q * QJL_CODEBOOK[c as usize])
            .sum();
        let dot = mse_dot + qjl_dot * self.scale * vec.residual_norm;
        match self.similarity {
            VectorSimilarity::Euclidean => self.norm * self.norm + vec.norm * vec.norm - 2.0 * dot,
            VectorSimilarity::Dot | VectorSimilarity::Cosine => dot.mul_add(-0.5, 0.5),
        }
        .into()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::{F32VectorCoder, turbo_quant::Prod1Coder};

    // This test vector contains randomly generated numbers in [-1,1] but is not l2 normalized.
    // It has 19 elements -- long enough to trigger SIMD optimizations but with some remainder to
    // test scalar tail paths.
    const TEST_VECTOR: [f32; 19] = [
        -0.921, -0.061, 0.659, 0.67, 0.573, 0.431, 0.646, 0.001, -0.2, -0.428, 0.73, -0.704,
        -0.273, 0.539, -0.731, 0.436, 0.913, 0.694, 0.202,
    ];

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
        let dot: f32 = decoded
            .iter()
            .zip(TEST_VECTOR.iter())
            .map(|(d, &v)| d * v)
            .sum();
        let decoded_norm: f32 = decoded.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let input_norm: f32 = TEST_VECTOR.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(dot / (decoded_norm * input_norm) > 0.5);
    }
}
