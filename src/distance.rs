//! Dense vector scoring traits and implementations.

use std::{borrow::Cow, io, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::query_distance::QueryVectorDistance;

/// Distance function for coded vectors.
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

#[derive(Debug, Copy, Clone)]
struct I8ScaledUniformVector<'a>(&'a [u8]);

impl I8ScaledUniformVector<'_> {
    fn dot_unnormalized(&self, other: &Self) -> f64 {
        self.vector()
            .iter()
            .zip(other.vector().iter())
            .map(|(s, o)| *s as i32 * *o as i32)
            .sum::<i32>() as f64
            * self.scale()
            * other.scale()
    }

    fn dequantized_unnormalized_iter(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        self.vector()
            .iter()
            .map(|d| *d as f32 * self.scale() as f32)
    }

    fn dequantized_normalized_iter(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        let scale = (self.scale() / self.l2_norm()) as f32;
        self.vector().iter().map(move |d| *d as f32 * scale)
    }

    fn scale(&self) -> f64 {
        f32::from_le_bytes(self.0[0..4].try_into().unwrap()).into()
    }

    fn l1_norm(&self) -> f64 {
        self.l2_norm() * self.l2_norm()
    }

    fn l2_norm(&self) -> f64 {
        f32::from_le_bytes(self.0[4..8].try_into().unwrap()).into()
    }

    fn vector(&self) -> &[i8] {
        bytemuck::cast_slice(&self.0[8..])
    }
}

impl<'a> From<&'a [u8]> for I8ScaledUniformVector<'a> {
    fn from(value: &'a [u8]) -> Self {
        assert!(value.len() >= 8);
        Self(value)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I8ScaledUniformDotProduct;

impl VectorDistance for I8ScaledUniformDotProduct {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = I8ScaledUniformVector::from(query);
        let doc = I8ScaledUniformVector::from(doc);
        let dot = query.dot_unnormalized(&doc) * query.l2_norm().recip() * doc.l2_norm().recip();
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Clone)]
pub struct I8ScaledUniformDotProductQueryDistance<'a>(Cow<'a, [f32]>);

impl<'a> I8ScaledUniformDotProductQueryDistance<'a> {
    pub fn new(query: &'a [f32]) -> Self {
        Self(F32DotProductDistance.normalize(query.into()))
    }
}

impl QueryVectorDistance for I8ScaledUniformDotProductQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        // TODO: benchmark performing dot product of query and doc without scaling, then scaling
        // afterward. This would avoid a multiplication per dimension.
        // XXX
        let vector = I8ScaledUniformVector::from(vector);
        let dot = self
            .0
            .iter()
            .zip(vector.dequantized_normalized_iter())
            .map(|(q, d)| *q * d)
            .sum::<f32>() as f64;
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I8ScaledUniformEuclidean;

impl VectorDistance for I8ScaledUniformEuclidean {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = I8ScaledUniformVector::from(query);
        let doc = I8ScaledUniformVector::from(doc);
        let dot = query.dot_unnormalized(&doc);
        query.l1_norm() + doc.l1_norm() - (2.0 * dot)
    }
}

#[derive(Debug, Clone)]
pub struct I8ScaledUniformEuclideanQueryDistance<'a>(&'a [f32], f64);

impl<'a> I8ScaledUniformEuclideanQueryDistance<'a> {
    pub fn new(query: &'a [f32]) -> Self {
        let l1_norm = dot_f32(query, query);
        Self(query, l1_norm)
    }
}

impl QueryVectorDistance for I8ScaledUniformEuclideanQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = I8ScaledUniformVector::from(vector);
        let dot = self
            .0
            .iter()
            .zip(vector.dequantized_unnormalized_iter())
            .map(|(q, d)| *q * d)
            .sum::<f32>() as f64;
        self.1 + vector.l1_norm() - (2.0 * dot)
    }
}

#[inline(always)]
pub(crate) fn hamming(q: &[u8], d: &[u8]) -> f64 {
    use simsimd::BinarySimilarity;
    u8::hamming(q, d).expect("same dimensionality")
}

#[cfg(all(target_arch = "aarch64"))]
unsafe fn load_f32x4_le(p: *const u8) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64;
    if cfg!(target_endian = "big") {
        aarch64::vreinterpretq_f32_u8(aarch64::vrev32q_u8(aarch64::vld1q_u8(p)))
    } else {
        aarch64::vld1q_f32(p as *const f32)
    }
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

#[cfg(test)]
mod test {
    use crate::{
        distance::{
            F32DotProductDistance, F32EuclideanDistance, F32VectorDistance,
            I8ScaledUniformDotProduct, I8ScaledUniformEuclidean, VectorDistance,
        },
        quantization::{I8ScaledUniformQuantizer, Quantizer},
    };

    struct TestVector {
        rvec: Vec<f32>,
        qvec: Vec<u8>,
    }

    impl TestVector {
        pub fn new(
            rvec: Vec<f32>,
            f32_dist_fn: impl F32VectorDistance,
            quantizer: impl Quantizer,
        ) -> Self {
            let qvec = quantizer.for_doc(&rvec);
            Self {
                rvec: f32_dist_fn.normalize(rvec.into()).to_vec(),
                qvec,
            }
        }
    }

    fn distance_compare_threshold(
        f32_dist_fn: impl F32VectorDistance + Copy,
        quantizer: impl Quantizer + Copy,
        dist_fn: impl VectorDistance + Copy,
        a: Vec<f32>,
        b: Vec<f32>,
        threshold: f64,
    ) {
        let a = TestVector::new(a, f32_dist_fn, quantizer);
        let b = TestVector::new(b, f32_dist_fn, quantizer);

        let rdist = f32_dist_fn.distance_f32(&a.rvec, &b.rvec);
        let qdist = dist_fn.distance(&a.qvec, &b.qvec);

        let range = (rdist * (1.0 - threshold))..=(rdist * (1.0 + threshold));
        assert!(
            range.contains(&qdist),
            "expected {} (range={:?}) actual {}",
            rdist,
            range,
            qdist,
        );
    }

    // XXX i get recall of 0 when using the asymmetric functions so clearly i fucked up _something_.

    #[test]
    fn i8_shaped_dot() {
        // TODO: randomly generate a bunch of vectors for this test.
        distance_compare_threshold(
            F32DotProductDistance,
            I8ScaledUniformQuantizer,
            I8ScaledUniformDotProduct,
            vec![-1.0f32, 2.5, 0.7, -1.7],
            vec![-0.6f32, -1.2, 0.4, 0.3],
            0.01,
        );
    }

    #[test]
    fn i8_shaped_l2() {
        distance_compare_threshold(
            F32EuclideanDistance,
            I8ScaledUniformQuantizer,
            I8ScaledUniformEuclidean,
            vec![-1.0f32, 2.5, 0.7, -1.7],
            vec![-0.6f32, -1.2, 0.4, 0.3],
            0.01,
        );
    }
}
