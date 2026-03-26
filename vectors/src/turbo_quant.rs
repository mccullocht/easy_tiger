use ndarray::{Array2, ArrayView1};
use ndarray_linalg::QR;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use rand::SeedableRng;

use crate::F32VectorCoder;

/// Codebook for 1-bit quantization; each entry must be divided by sqrt(D).
const CODEBOOK_TMPL_1: [f32; 2] = [
    (-2.0 / std::f64::consts::PI) as f32,
    (2.0 / std::f64::consts::PI) as f32,
];

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

/// TurboQuant coder for 1-bit quantization that minimizes MSE. This is most suitable for l2
/// distance where minimizing MSE is equivalent to minimizing distance distortion.
pub struct MSE1Coder {
    dim: usize,
    t: JLTrans,
    codebook: Vec<f32>,
}

impl MSE1Coder {
    pub fn new(dim: usize, seed: u64) -> Self {
        let t = JLTrans::new(dim, seed);
        let codebook = CODEBOOK_TMPL_1
            .iter()
            .map(|&x| x / (dim as f32).sqrt())
            .collect();
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
