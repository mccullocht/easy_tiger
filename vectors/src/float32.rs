//! Raw float 32 vector coding and distance computation.
//!
//! Vectors are stored as a sequence of raw little-endian coded f32 values exactly as provided;
//! callers are responsible for any normalization (see [`crate::prepare_vector`]). Angular distance
//! functions assume the stored and query vectors are already l2 normalized.

#[cfg(target_arch = "aarch64")]
mod aarch64;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86_64;

use std::{borrow::Cow, sync::OnceLock};

use crate::{
    F32VectorCoder, F32VectorDistance, QueryVectorDistance as QueryVectorDistanceT, VectorDistance,
    VectorSimilarity,
};

#[derive(Debug, Clone, Copy)]
enum Kernel {
    #[allow(unused)]
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx512f,
}

impl Default for Kernel {
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    fn default() -> Self {
        Kernel::Scalar
    }

    #[cfg(target_arch = "aarch64")]
    fn default() -> Self {
        Kernel::Neon
    }

    #[cfg(target_arch = "x86_64")]
    fn default() -> Self {
        if std::arch::is_x86_feature_detected!("avx512f") {
            Kernel::Avx512f
        } else {
            Kernel::Scalar
        }
    }
}

#[inline]
fn dot(a: &[u8], b: &[u8], inst: Option<Kernel>) -> f64 {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len() % 4, 0);
    match inst.unwrap_or_default() {
        Kernel::Scalar => scalar::dot(a, b),
        #[cfg(target_arch = "aarch64")]
        Kernel::Neon => unsafe { aarch64::dot(a, b) },
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx512f => unsafe { x86_64::dot(a, b) },
    }
}

fn l2sq(a: &[u8], b: &[u8], inst: Option<Kernel>) -> f64 {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len() % 4, 0);
    match inst.unwrap_or_default() {
        Kernel::Scalar => scalar::l2sq(a, b),
        #[cfg(target_arch = "aarch64")]
        Kernel::Neon => unsafe { aarch64::l2sq(a, b) },
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx512f => unsafe { x86_64::l2sq(a, b) },
    }
}

/// Compute the l2 norm of `vector`.
pub fn l2_norm(vector: impl AsRef<[f32]>) -> f32 {
    dot(
        bytemuck::cast_slice(vector.as_ref()),
        bytemuck::cast_slice(vector.as_ref()),
        None,
    )
    .sqrt() as f32
}

/// Normalize the contents of vector in l2 space.
///
/// Returns the normalized vector and the l2 norm. The returned vector may be the input vector if
/// the input vector is already unit normalized.
pub fn l2_normalize<'a>(vector: impl Into<Cow<'a, [f32]>>) -> (Cow<'a, [f32]>, f32) {
    let mut vector: Cow<'a, [f32]> = vector.into();
    let norm = l2_norm(&vector);
    if norm != 1.0 {
        let norm_inv = norm.recip();
        for d in vector.to_mut().iter_mut() {
            *d *= norm_inv;
        }
    }
    (vector, norm)
}

#[derive(Debug, Copy, Clone, Default)]
pub struct VectorCoder;

impl VectorCoder {
    pub fn new() -> Self {
        Self
    }

    fn encode_it(vector: impl ExactSizeIterator<Item = f32>, out: &mut [u8]) {
        for (d, o) in vector.zip(out.as_chunks_mut::<{ std::mem::size_of::<f32>() }>().0) {
            *o = d.to_le_bytes();
        }
    }
}

impl F32VectorCoder for VectorCoder {
    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions * std::mem::size_of::<f32>()
    }

    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        assert!(out.len() >= std::mem::size_of_val(vector));
        Self::encode_it(vector.iter().copied(), out);
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        for (d, o) in scalar::f32_le_iter(encoded).zip(out.iter_mut()) {
            *o = d
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        byte_len / std::mem::size_of::<f32>()
    }
}

static L2_DIST: OnceLock<EuclideanDistance> = OnceLock::new();

/// Compute squared l2 (Euclidean) distance.
#[derive(Debug, Copy, Clone, Default)]
pub struct EuclideanDistance(Kernel);

impl EuclideanDistance {
    /// Returns a static instance of euclidean distance.
    pub fn get() -> &'static EuclideanDistance {
        L2_DIST.get_or_init(EuclideanDistance::default)
    }
}

impl VectorDistance for EuclideanDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        l2sq(query, doc, Some(self.0))
    }
}

impl F32VectorDistance for EuclideanDistance {
    fn distance_f32(&self, a: &[f32], b: &[f32]) -> f64 {
        l2sq(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(b),
            Some(self.0),
        )
    }
}

static DOT_DIST: OnceLock<DotProductDistance> = OnceLock::new();

/// Computes a score based on the dot product.
#[derive(Debug, Copy, Clone, Default)]
pub struct DotProductDistance(Kernel);

impl DotProductDistance {
    /// Returns a static instance of dot product distance.
    pub fn get() -> &'static DotProductDistance {
        DOT_DIST.get_or_init(DotProductDistance::default)
    }
}

impl VectorDistance for DotProductDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        // Assuming values are normalized, this will produce a distance in [0,1]
        (-dot(query, doc, Some(self.0)) + 1.0) / 2.0
    }
}

impl F32VectorDistance for DotProductDistance {
    fn distance_f32(&self, a: &[f32], b: &[f32]) -> f64 {
        // Assuming values are normalized, this will produce a distance in [0,1]
        (-dot(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(b),
            Some(self.0),
        ) + 1.0)
            / 2.0
    }
}

static COS_DIST: OnceLock<CosineDistance> = OnceLock::new();

#[derive(Debug, Default, Copy, Clone)]
pub struct CosineDistance(DotProductDistance);

impl CosineDistance {
    /// Returns a static instance of cosine distance.
    pub fn get() -> &'static CosineDistance {
        COS_DIST.get_or_init(CosineDistance::default)
    }
}

impl VectorDistance for CosineDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        // Vectors are normalized during encoding so we can make this fast.
        self.0.distance(query, doc)
    }
}

impl F32VectorDistance for CosineDistance {
    fn distance_f32(&self, a: &[f32], b: &[f32]) -> f64 {
        let ab = dot(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(b),
            Some(self.0.0),
        );
        let aa = dot(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(a),
            Some(self.0.0),
        );
        let bb = dot(
            bytemuck::cast_slice(b),
            bytemuck::cast_slice(b),
            Some(self.0.0),
        );
        let cos = ab / (aa * bb).sqrt();
        (-cos as f64 + 1.0) / 2.0
    }
}

#[derive(Debug, Clone)]
pub struct QueryVectorDistance<'a, D> {
    distance_fn: D,
    query: Cow<'a, [f32]>,
}

impl<'a, D: F32VectorDistance> QueryVectorDistance<'a, D> {
    pub fn new(distance_fn: D, query: Cow<'a, [f32]>) -> Self {
        Self { distance_fn, query }
    }
}

impl<'a, D: F32VectorDistance> QueryVectorDistanceT for QueryVectorDistance<'a, D> {
    fn distance(&self, vector: &[u8]) -> f64 {
        self.distance_fn
            .distance(bytemuck::cast_slice(self.query.as_ref()), vector)
    }
}

pub fn new_query_vector_distance<'a>(
    similarity: VectorSimilarity,
    query: Cow<'a, [f32]>,
) -> Box<dyn QueryVectorDistanceT + 'a> {
    match similarity {
        VectorSimilarity::Cosine => {
            Box::new(QueryVectorDistance::new(CosineDistance::default(), query))
        }
        VectorSimilarity::Dot => Box::new(QueryVectorDistance::new(
            DotProductDistance::default(),
            query,
        )),
        VectorSimilarity::Euclidean => Box::new(QueryVectorDistance::new(
            EuclideanDistance::default(),
            query,
        )),
    }
}
