//! Support for `i8` component vectors.
//!
//! `i8` vectors are not fully supported because transforms almost always need to be performed in
//! floating point space. Instead this offers a subset of the functionality provided for other
//! formats:
//! * A format extension that stores the l2 norm with the vector. This is necessary to maximize the
//!   efficiency of vector comparisons on most CPUs.
//! * Symmetric and asymmetric distance functions that operate on the `i8` format.
//! * Efficient decode-to-float operations so that you can transform and re-quantize `i8` vectors.

#![allow(dead_code)] // XXX DO NOT MERGE

#[cfg(target_arch = "aarch64")]
mod aarch64;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86_64;

use std::borrow::Cow;

use super::{F32VectorCoder, QueryVectorDistance, VectorDistance, VectorSimilarity};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Kernel {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl Kernel {
    const CANDIDATES: &'static [Self] = &[
        #[cfg(target_arch = "aarch64")]
        Self::Neon,
        #[cfg(target_arch = "x86_64")]
        Self::Avx512,
        #[cfg(target_arch = "x86_64")]
        Self::Avx2,
        Self::Scalar,
    ];

    /// Returns true if the specificed kernel is available for use on this cost.
    fn is_available(&self) -> bool {
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Neon => std::arch::is_aarch64_feature_detected!("dotprod"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => {
                std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512vnni")
            }
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => std::arch::is_x86_feature_detected!("avx2"),
            Self::Scalar => true,
        }
    }

    /// Return all of the non-Scalar Kernels that are available on this host.
    #[cfg(test)]
    fn accelerated() -> impl Iterator<Item = Self> {
        Self::CANDIDATES
            .iter()
            .copied()
            .filter(|&k| k.is_available() && k != Self::Scalar)
    }
}

impl Default for Kernel {
    fn default() -> Self {
        Self::CANDIDATES
            .iter()
            .copied()
            .find(Self::is_available)
            .expect("Scalar is always available")
    }
}

/// Vector coder for the `i8` format.
///
/// While this does implement [`F32VectorCoder`] the `encode()` path is not fully implemented since
/// the operation is not well defined, and the purpose of this format is to support `i8` ingestion.
#[derive(Debug, Copy, Clone, Default)]
pub struct VectorCoder(Kernel);

impl VectorCoder {
    const HEADER_LEN: usize = std::mem::size_of::<f32>();

    /// Encode `vector` in the `i8` format.
    ///
    /// This format extends the raw vector with additional information.
    pub fn encode(&self, vector: &[i8]) -> Vec<u8> {
        let mut out = vec![0u8; self.byte_len(vector.len())];
        self.encode_to(vector, &mut out);
        out
    }

    /// Encode `vector` to `out` in the `i8` format.
    ///
    /// `out` must be exactly `VectorCoder::encoded_len(vector.len())` bytes long.
    pub fn encode_to(&self, vector: &[i8], out: &mut [u8]) {
        // Cast ip to f64 to avoid rounding if the product is > 2^23
        let ip = inner_product(self.0, vector, vector);
        let l2_norm = (ip as f64).sqrt() as f32;

        let (header, vector_bytes) = out.split_at_mut(Self::HEADER_LEN);
        header.copy_from_slice(&l2_norm.to_le_bytes());
        vector_bytes.copy_from_slice(bytemuck::cast_slice(vector));
    }

    /// Convert an input `i8` vector to `f32` output the same way as a cast.
    ///
    /// This is equivalent to looping and casting each element but may be accelerated based on the
    /// local architecture.
    #[inline]
    pub fn convert_slice_to_f32(&self, vector: &[i8], out: &mut [f32]) {
        assert_eq!(vector.len(), out.len());
        match self.0 {
            Kernel::Scalar => scalar::decode_to_f32(vector, out),
            // NEON is on by default on aarch64, and it does a fine job auto-vectorizing.
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => scalar::decode_to_f32(vector, out),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => unsafe { x86_64::avx2::decode_to_f32(vector, out) },
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe { x86_64::avx512::decode_to_f32(vector, out) },
        }
    }
}

impl F32VectorCoder for VectorCoder {
    fn encode_to(&self, _vector: &[f32], _out: &mut [u8]) {
        panic!("conversion from f32 not supported for i8 vectors")
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions + Self::HEADER_LEN
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        let vector_len = self.dimensions(encoded.len());
        assert_eq!(vector_len, out.len());
        self.convert_slice_to_f32(bytemuck::cast_slice(&encoded[Self::HEADER_LEN..]), out)
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        byte_len.saturating_sub(Self::HEADER_LEN)
    }
}

/// Compute inner product between `a` and `b`.
#[inline]
fn inner_product(k: Kernel, a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len());
    match k {
        #[cfg(target_arch = "aarch64")]
        Kernel::Neon => unsafe { aarch64::neon::inner_product(a, b) },
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => unsafe { x86_64::avx2::inner_product(a, b) },
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx512 => unsafe { x86_64::avx512::inner_product(a, b) },
        Kernel::Scalar => scalar::inner_product(a, b),
    }
}

#[inline]
fn decode_vector(encoded: &[u8]) -> (f32, &[i8]) {
    assert!(encoded.len() >= VectorCoder::HEADER_LEN);
    let (header, vector) = encoded.split_at(VectorCoder::HEADER_LEN);
    (
        f32::from_le_bytes(header.try_into().unwrap()),
        bytemuck::cast_slice(vector),
    )
}

#[inline]
fn compute_distance(similarity: VectorSimilarity, ip: i32, norms: (f32, f32)) -> f64 {
    match similarity {
        VectorSimilarity::Euclidean => {
            ((norms.0 as f64).powi(2) + (norms.1 as f64).powi(2) - ip as f64).max(0.0)
        }
        // This is strictly cosine similarity, we don't anticipate every i8 scalar quantizer
        // correctly normalizing the vectors before quantization.
        VectorSimilarity::Dot => (ip as f64 / (norms.0 as f64 * norms.1 as f64))
            .mul_add(-0.5, 0.5)
            .clamp(0.0, 1.0),
    }
}

/// Symmetric distance comparison for "encoded" `i8` vectors.
pub(super) struct Distance {
    k: Kernel,
    s: VectorSimilarity,
}

impl Distance {
    pub fn new(s: VectorSimilarity) -> Self {
        Self {
            k: Kernel::default(),
            s,
        }
    }
}

impl VectorDistance for Distance {
    fn distance(&self, a: &[u8], b: &[u8]) -> f64 {
        let (anorm, av) = decode_vector(a);
        let (bnorm, bv) = decode_vector(b);
        compute_distance(self.s, inner_product(self.k, av, bv), (anorm, bnorm))
    }
}

pub(super) struct QueryDistance<'a> {
    k: Kernel,
    similarity: VectorSimilarity,

    query_norm: f32,
    query_vector: Cow<'a, [u8]>,
}

impl<'a> QueryDistance<'a> {
    pub fn new(similarity: VectorSimilarity, query: Cow<'a, [u8]>) -> Self {
        Self {
            k: Kernel::default(),
            similarity,
            query_norm: f32::from_le_bytes(query.as_chunks::<4>().0[0]),
            query_vector: query,
        }
    }

    fn query(&self) -> &[i8] {
        bytemuck::cast_slice(&self.query_vector[VectorCoder::HEADER_LEN..])
    }
}

impl QueryVectorDistance for QueryDistance<'_> {
    fn distance(&self, doc: &[u8]) -> f64 {
        let (dnorm, dv) = decode_vector(doc);
        compute_distance(
            self.similarity,
            inner_product(self.k, &self.query(), dv),
            (self.query_norm, dnorm),
        )
    }
}
