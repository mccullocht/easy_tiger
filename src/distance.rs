//! Dense vector scoring traits and implementations.

use std::{borrow::Cow, io, str::FromStr};

use serde::{Deserialize, Serialize};

/// Distance function for quantized vectors.
///
/// This trait is object-safe; it may be instantiated at runtime based on
/// data that appears in a file or other backing store.
pub trait VectorDistance: Send + Sync {
    /// Score the `query` vector against the `doc` vector. Returns a score
    /// where larger values are better matches.
    ///
    /// This function is not required to be commutative and may panic if
    /// one of the inputs is misshapen.
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64;
}

/// Distance function for `f32` vectors.
///
/// This trait is object-safe; it may be instantiated at runtime based on
/// data that appears in a file or other backing store.
pub trait F32VectorDistance: VectorDistance {
    /// Score vectors `a` and `b` against one another. Returns a score
    /// where larger values are better matches.
    ///
    /// Input vectors must be the same length or this function may panic.
    fn distance_f32(&self, a: &[f32], b: &[f32]) -> f64;

    /// Normalize a vector for use with this scoring function.
    /// By default, does nothing.
    fn normalize<'a>(&self, vector: Cow<'a, [f32]>) -> Cow<'a, [f32]> {
        vector
    }
}

/// Functions used for computing a similarity score for high fidelity vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorSimilarity {
    /// Euclidean (l2) distance.
    Euclidean,
    /// Dot product scoring, an approximation of cosine scoring.
    /// Vectors used for this distance function must be normalized.
    Dot,
}

impl VectorSimilarity {
    pub fn new_distance_function(self) -> Box<dyn F32VectorDistance> {
        match self {
            Self::Euclidean => Box::new(F32EuclideanDistance),
            Self::Dot => Box::new(F32DotProductDistance),
        }
    }
}

impl Default for VectorSimilarity {
    fn default() -> Self {
        Self::Dot
    }
}

impl FromStr for VectorSimilarity {
    type Err = io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "euclidean" => Ok(VectorSimilarity::Euclidean),
            "dot" => Ok(VectorSimilarity::Dot),
            x => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown similarity fuction {x}"),
            )),
        }
    }
}

/// Computes a score based on l2 distance.
#[derive(Debug, Copy, Clone)]
pub struct F32EuclideanDistance;

impl VectorDistance for F32EuclideanDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        l2sq_f32_bytes(query, doc)
    }
}

impl F32VectorDistance for F32EuclideanDistance {
    fn distance_f32(&self, a: &[f32], b: &[f32]) -> f64 {
        l2sq_f32(a, b)
    }
}

/// Computes a score based on the dot product.
#[derive(Debug, Copy, Clone)]
pub struct F32DotProductDistance;

impl VectorDistance for F32DotProductDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        dot_f32_bytes(query, doc)
    }
}

impl F32VectorDistance for F32DotProductDistance {
    fn distance_f32(&self, a: &[f32], b: &[f32]) -> f64 {
        // Assuming values are normalized, this will produce a distance in [0,1]
        (-dot_f32(a, b) + 1.0) / 2.0
    }

    fn normalize<'a>(&self, mut vector: Cow<'a, [f32]>) -> Cow<'a, [f32]> {
        let norm = dot_f32(&vector, &vector).sqrt() as f32;
        for d in vector.to_mut().iter_mut() {
            *d /= norm;
        }
        vector
    }
}

/// Computes a score from two bitmaps using hamming distance.
#[derive(Debug, Copy, Clone)]
pub struct HammingDistance;

impl VectorDistance for HammingDistance {
    fn distance(&self, a: &[u8], b: &[u8]) -> f64 {
        hamming(a, b)
    }
}

/// Computes a score between a query and doc vectors produced by [crate::quantization::AsymmetricBinaryQuantizer]
#[derive(Debug, Copy, Clone)]
pub struct AsymmetricHammingDistance;

impl VectorDistance for AsymmetricHammingDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        assert_eq!(query.len() % doc.len(), 0);
        query
            .chunks(doc.len())
            .enumerate()
            .map(|(i, v)| hamming(doc, v) as usize * (1usize << i))
            .sum::<usize>() as f64
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I8NaiveDistance(pub(crate) VectorSimilarity);

impl I8NaiveDistance {
    fn unpack(raw: &[u8]) -> (&[i8], f32) {
        assert!(raw.len() >= std::mem::size_of::<f32>());
        let (vector_bytes, norm_bytes) = raw.split_at(raw.len() - std::mem::size_of::<f32>());
        (
            unsafe {
                std::slice::from_raw_parts(vector_bytes.as_ptr() as *const i8, vector_bytes.len())
            },
            f32::from_le_bytes(norm_bytes.try_into().unwrap()),
        )
    }
}

impl VectorDistance for I8NaiveDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let (qv, qnorm) = Self::unpack(query);
        let (dv, dnorm) = Self::unpack(doc);
        let divisor = i8::MAX as f32 * i8::MAX as f32;
        // NB: we may be able to accelerate this further with manual SIMD implementations.
        let dot = qv
            .iter()
            .zip(dv.iter())
            .map(|(q, d)| *q as i32 * *d as i32)
            .sum::<i32>() as f64
            / divisor as f64;
        match self.0 {
            VectorSimilarity::Dot => (-dot + 1.0) / 2.0,
            VectorSimilarity::Euclidean => qnorm as f64 + dnorm as f64 - (2.0 * dot),
        }
    }
}

struct I8NonUniformNaiveVector<'a> {
    magnitude: f32,
    vector: &'a [i8],
}

impl I8NonUniformNaiveVector<'_> {
    fn dequantized(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        self.vector.iter().map(|d| *d as f32 * self.magnitude)
    }
}

impl<'a> From<&'a [u8]> for I8NonUniformNaiveVector<'a> {
    fn from(value: &'a [u8]) -> Self {
        let (magnitude, vector) = value.split_at(4);
        Self {
            magnitude: f32::from_le_bytes(magnitude.try_into().expect("4 bytes")),
            vector: bytemuck::cast_slice(vector),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I8NonUniformNaiveDotProduct;

impl VectorDistance for I8NonUniformNaiveDotProduct {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        // TODO: if we had a formal query distance abstraction we could avoid quantizing the query
        // vector and improve accuracy at the cost of speed (f32 instead of i32).
        let query = I8NonUniformNaiveVector::from(query);
        let doc = I8NonUniformNaiveVector::from(doc);
        query
            .vector
            .iter()
            .zip(doc.vector.iter())
            .map(|(q, d)| *q as i32 * *d as i32)
            .sum::<i32>() as f64
            * query.magnitude as f64
            * doc.magnitude as f64
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I8NonUniformNaiveEuclidean;

impl VectorDistance for I8NonUniformNaiveEuclidean {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        // TODO: if we had a formal query distance abstraction we could use the original query
        // vector or at least avoid dequantizing the query so many times.
        // XXX I want to normalize and encode the l2 norm so i can dot the hell out of this.
        let query = I8NonUniformNaiveVector::from(query);
        let doc = I8NonUniformNaiveVector::from(doc);
        // XXX ((qm * q0) - (dm * d0))^2 + ...
        query
            .dequantized()
            .zip(doc.dequantized())
            .map(|(q, d)| {
                let delta = q - d;
                delta * delta
            })
            .sum::<f32>() as f64
    }
}

#[inline(always)]
pub(crate) fn hamming(q: &[u8], d: &[u8]) -> f64 {
    use simsimd::BinarySimilarity;
    u8::hamming(q, d).expect("same dimensionality")
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
unsafe fn load_f32x4_le(p: *const u8) -> core::arch::aarch64::float32x4_t {
    core::arch::aarch64::vld1q_f32(p as *const f32)
}

#[cfg(not(target_arch = "aarch64"))]
fn f32_le_iter<'b>(b: &'b [u8]) -> impl ExactSizeIterator<Item = f32> + 'b {
    b.chunks_exact(4)
        .map(|f| f32::from_le_bytes(f.try_into().expect("4 bytes")))
}

// TODO: byte swapped load on big endian archs.

#[inline(always)]
pub(crate) fn l2sq_f32(q: &[f32], d: &[f32]) -> f64 {
    simsimd::SpatialSimilarity::l2sq(q, d).expect("same dimensions")
}

pub(crate) fn l2sq_f32_bytes(q: &[u8], d: &[u8]) -> f64 {
    assert_eq!(q.len(), d.len());
    assert_eq!(q.len() % 4, 0);
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vsubq_f32};
        let suffix_start = q.len() & !15;
        let mut l2sqv = vdupq_n_f32(0.0);
        for i in (0..suffix_start).step_by(16) {
            let dv = vsubq_f32(
                load_f32x4_le(q.as_ptr().add(i)),
                load_f32x4_le(d.as_ptr().add(i)),
            );
            l2sqv = vfmaq_f32(l2sqv, dv, dv);
        }
        let mut l2sq = vaddvq_f32(l2sqv);
        for i in (suffix_start..q.len()).step_by(4) {
            let delta = std::ptr::read_unaligned(q.as_ptr().add(i) as *const f32)
                - std::ptr::read_unaligned(d.as_ptr().add(i) as *const f32);
            l2sq += delta * delta;
        }
        l2sq as f64
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        f32_le_iter(q)
            .zip(f32_le_iter(d))
            .map(|(q, d)| {
                let delta = q - d;
                delta * delta
            })
            .sum::<f32>() as f64
    }
}

pub(crate) fn l2(q: &[f32], d: &[f32]) -> f64 {
    l2sq_f32(q, d).sqrt()
}

#[inline(always)]
pub(crate) fn dot_f32(q: &[f32], d: &[f32]) -> f64 {
    simsimd::SpatialSimilarity::dot(q, d).expect("same dimensions")
}

pub(crate) fn dot_f32_bytes(q: &[u8], d: &[u8]) -> f64 {
    assert_eq!(q.len(), d.len());
    assert_eq!(q.len() % 4, 0);
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32};
        let suffix_start = q.len() & !15;
        let mut dotv = vdupq_n_f32(0.0);
        for i in (0..suffix_start).step_by(16) {
            dotv = vfmaq_f32(
                dotv,
                load_f32x4_le(q.as_ptr().add(i)),
                load_f32x4_le(d.as_ptr().add(i)),
            );
        }
        let mut dot = vaddvq_f32(dotv);
        for i in (suffix_start..q.len()).step_by(4) {
            dot += std::ptr::read_unaligned(q.as_ptr().add(i) as *const f32)
                * std::ptr::read_unaligned(d.as_ptr().add(i) as *const f32);
        }
        dot as f64
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        f32_le_iter(q)
            .zip(f32_le_iter(d))
            .map(|(q, d)| q * d)
            .sum::<f32>() as f64
    }
}
