//! Raw float 32 vector coding and distance computation.
//!
//! Vectors are stored as a sequence of raw little-endian coded f32 values.
//!
//! For Cosine similarity the vector will be normalized during encoding. When scoring float vectors
//! we will assume the vectors are unnormalized.

use std::{borrow::Cow, sync::OnceLock};

use crate::{
    F32VectorCoder, F32VectorDistance, QueryVectorDistance as QueryVectorDistanceT, VectorDistance,
    VectorSimilarity, float32::distance::InstructionSet,
};

mod distance {
    #![allow(unsafe_op_in_unsafe_fn)]

    #[derive(Debug, Clone, Copy)]
    pub enum InstructionSet {
        #[allow(unused)]
        Scalar,
        #[cfg(target_arch = "aarch64")]
        Neon,
        #[cfg(target_arch = "x86_64")]
        Avx512f,
    }

    impl Default for InstructionSet {
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        fn default() -> Self {
            InstructionSet::Scalar
        }

        #[cfg(target_arch = "aarch64")]
        fn default() -> Self {
            InstructionSet::Neon
        }

        #[cfg(target_arch = "x86_64")]
        fn default() -> Self {
            if std::arch::is_x86_feature_detected!("avx512f") {
                InstructionSet::Avx512f
            } else {
                InstructionSet::Scalar
            }
        }
    }

    #[inline]
    pub fn dot(a: &[u8], b: &[u8], inst: Option<InstructionSet>) -> f64 {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len() % 4, 0);
        match inst.unwrap_or_default() {
            InstructionSet::Scalar => dot_scalar(a, b),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => unsafe { dot_neon(a, b) },
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512f => unsafe { dot_avx512f(a, b) },
        }
    }

    #[inline]
    fn dot_scalar(a: &[u8], b: &[u8]) -> f64 {
        f32_le_iter(a)
            .zip(f32_le_iter(b))
            .map(|(a, b)| a * b)
            .sum::<f32>()
            .into()
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn dot_neon(a: &[u8], b: &[u8]) -> f64 {
        use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32};
        let len64 = a.len() & !63;
        let mut dot0 = vdupq_n_f32(0.0);
        let mut dot1 = vdupq_n_f32(0.0);
        let mut dot2 = vdupq_n_f32(0.0);
        let mut dot3 = vdupq_n_f32(0.0);
        for i in (0..len64).step_by(64) {
            dot0 = vfmaq_f32(
                dot0,
                load_f32x4_le(a.as_ptr().add(i)),
                load_f32x4_le(b.as_ptr().add(i)),
            );
            dot1 = vfmaq_f32(
                dot1,
                load_f32x4_le(a.as_ptr().add(i + 16)),
                load_f32x4_le(b.as_ptr().add(i + 16)),
            );
            dot2 = vfmaq_f32(
                dot2,
                load_f32x4_le(a.as_ptr().add(i + 32)),
                load_f32x4_le(b.as_ptr().add(i + 32)),
            );
            dot3 = vfmaq_f32(
                dot3,
                load_f32x4_le(a.as_ptr().add(i + 48)),
                load_f32x4_le(b.as_ptr().add(i + 48)),
            );
        }

        dot0 = vaddq_f32(vaddq_f32(dot0, dot1), vaddq_f32(dot2, dot3));
        let len16 = a.len() & !15;
        for i in (len64..len16).step_by(16) {
            dot0 = vfmaq_f32(
                dot0,
                load_f32x4_le(a.as_ptr().add(i)),
                load_f32x4_le(b.as_ptr().add(i)),
            );
        }

        let mut dot = vaddvq_f32(dot0);
        for i in (len16..a.len()).step_by(4) {
            dot += std::ptr::read_unaligned(a.as_ptr().add(i) as *const f32)
                * std::ptr::read_unaligned(b.as_ptr().add(i) as *const f32);
        }
        dot.into()
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn dot_avx512f(q: &[u8], d: &[u8]) -> f64 {
        use std::arch::x86_64::{
            _mm_cvtss_f32, _mm_hadd_ps, _mm512_add_ps, _mm512_castps512_ps128, _mm512_fmadd_ps,
            _mm512_maskz_loadu_ps, _mm512_set1_ps, _mm512_shuffle_f32x4,
        };
        let mut dot = _mm512_set1_ps(0.0);
        for i in (0..q.len()).step_by(64) {
            let rem = (q.len() - i).min(64) / 4;
            let mask = u16::MAX >> (16 - rem);
            let qv = _mm512_maskz_loadu_ps(mask, q.as_ptr().add(i) as *const f32);
            let dv = _mm512_maskz_loadu_ps(mask, d.as_ptr().add(i) as *const f32);
            dot = _mm512_fmadd_ps(qv, dv, dot);
        }

        let x = _mm512_add_ps(dot, _mm512_shuffle_f32x4(dot, dot, 0b00001110));
        let r = _mm512_castps512_ps128(_mm512_add_ps(x, _mm512_shuffle_f32x4(x, x, 0b00000001)));
        let r = _mm_hadd_ps(r, r);
        _mm_cvtss_f32(_mm_hadd_ps(r, r)).into()
    }

    pub fn l2sq(a: &[u8], b: &[u8], inst: Option<InstructionSet>) -> f64 {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len() % 4, 0);
        match inst.unwrap_or_default() {
            InstructionSet::Scalar => l2sq_scalar(a, b),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => unsafe { l2sq_neon(a, b) },
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512f => unsafe { l2sq_avx512(a, b) },
        }
    }

    #[inline]
    fn l2sq_scalar(a: &[u8], b: &[u8]) -> f64 {
        f32_le_iter(a)
            .zip(f32_le_iter(b))
            .map(|(a, b)| {
                let delta = a - b;
                delta * delta
            })
            .sum::<f32>()
            .into()
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn l2sq_neon(q: &[u8], d: &[u8]) -> f64 {
        use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vsubq_f32};

        let len64 = q.len() & !63;
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);
        for i in (0..len64).step_by(64) {
            let mut diff = vsubq_f32(
                load_f32x4_le(q.as_ptr().add(i)),
                load_f32x4_le(d.as_ptr().add(i)),
            );
            sum0 = vfmaq_f32(sum0, diff, diff);

            diff = vsubq_f32(
                load_f32x4_le(q.as_ptr().add(i + 16)),
                load_f32x4_le(d.as_ptr().add(i + 16)),
            );
            sum1 = vfmaq_f32(sum1, diff, diff);

            diff = vsubq_f32(
                load_f32x4_le(q.as_ptr().add(i + 32)),
                load_f32x4_le(d.as_ptr().add(i + 32)),
            );
            sum2 = vfmaq_f32(sum2, diff, diff);

            diff = vsubq_f32(
                load_f32x4_le(q.as_ptr().add(i + 48)),
                load_f32x4_le(d.as_ptr().add(i + 48)),
            );
            sum3 = vfmaq_f32(sum3, diff, diff);
        }

        sum0 = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
        let len16 = q.len() & !15;
        for i in (len64..len16).step_by(16) {
            let diff = vsubq_f32(
                load_f32x4_le(q.as_ptr().add(i)),
                load_f32x4_le(d.as_ptr().add(i)),
            );
            sum0 = vfmaq_f32(sum0, diff, diff);
        }

        let mut sum = vaddvq_f32(sum0);
        for i in (len16..q.len()).step_by(4) {
            let diff = std::ptr::read_unaligned(q.as_ptr().add(i) as *const f32)
                - std::ptr::read_unaligned(d.as_ptr().add(i) as *const f32);
            sum = diff.mul_add(diff, sum);
        }

        sum.into()
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn l2sq_avx512(q: &[u8], d: &[u8]) -> f64 {
        use std::arch::x86_64::{
            _mm_cvtss_f32, _mm_hadd_ps, _mm512_add_ps, _mm512_castps512_ps128, _mm512_fmadd_ps,
            _mm512_maskz_loadu_ps, _mm512_set1_ps, _mm512_shuffle_f32x4, _mm512_sub_ps,
        };
        let mut sum = _mm512_set1_ps(0.0);
        for i in (0..q.len()).step_by(64) {
            let rem = (q.len() - i).min(64) / 4;
            let mask = u16::MAX >> (16 - rem);
            let qv = _mm512_maskz_loadu_ps(mask, q.as_ptr().add(i) as *const f32);
            let dv = _mm512_maskz_loadu_ps(mask, d.as_ptr().add(i) as *const f32);
            let diff = _mm512_sub_ps(qv, dv);
            sum = _mm512_fmadd_ps(diff, diff, sum);
        }

        let x = _mm512_add_ps(sum, _mm512_shuffle_f32x4(sum, sum, 0b00_00_11_10));
        let r = _mm512_castps512_ps128(_mm512_add_ps(x, _mm512_shuffle_f32x4(x, x, 0b00_00_00_01)));
        let r = _mm_hadd_ps(r, r);
        _mm_cvtss_f32(_mm_hadd_ps(r, r)).into()
    }

    pub fn f32_le_iter<'b>(b: &'b [u8]) -> impl ExactSizeIterator<Item = f32> + 'b {
        let (chunks, rem) = b.as_chunks::<{ std::mem::size_of::<f32>() }>();
        debug_assert!(rem.is_empty());
        chunks.iter().map(|c| {
            f32::from_bits(u32::from_le(unsafe {
                // SAFETY: byte array chunk is guaranteed to be the same size as u32/f32.
                std::ptr::read_unaligned(c.as_ptr() as *const u32)
            }))
        })
    }

    #[inline(always)]
    #[cfg(target_arch = "aarch64")]
    unsafe fn load_f32x4_le(p: *const u8) -> core::arch::aarch64::float32x4_t {
        use core::arch::aarch64;
        if cfg!(target_endian = "big") {
            aarch64::vreinterpretq_f32_u8(aarch64::vrev32q_u8(aarch64::vld1q_u8(p)))
        } else {
            aarch64::vld1q_f32(p as *const f32)
        }
    }
} // mod distance

/// Compute the l2 norm of `vector`.
pub fn l2_norm(vector: impl AsRef<[f32]>) -> f32 {
    distance::dot(
        bytemuck::cast_slice(vector.as_ref()),
        bytemuck::cast_slice(vector.as_ref()),
        None,
    )
    .sqrt() as f32
}

/// Normalize the contents of vector in l2 space.
///
/// May return the input vector if it is already normalized.
pub fn l2_normalize<'a>(vector: impl Into<Cow<'a, [f32]>>) -> Cow<'a, [f32]> {
    let mut vector: Cow<'a, [f32]> = vector.into();
    let norm = l2_norm(&vector);
    if norm != 1.0 {
        let norm_inv = norm.recip();
        for d in vector.to_mut().iter_mut() {
            *d *= norm_inv;
        }
    }
    vector
}

#[derive(Debug, Copy, Clone)]
pub struct VectorCoder(VectorSimilarity);

impl VectorCoder {
    pub fn new(similarity: VectorSimilarity) -> Self {
        Self(similarity)
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
        let vector_it = vector.iter().copied();
        if self.0.l2_normalize() {
            let scale = 1.0 / l2_norm(vector);
            Self::encode_it(vector_it.map(|d| d * scale), out);
        } else {
            Self::encode_it(vector_it, out);
        }
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        for (d, o) in distance::f32_le_iter(encoded).zip(out.iter_mut()) {
            *o = d
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        byte_len / std::mem::size_of::<f32>()
    }
}

static L2_DIST: OnceLock<EuclideanDistance> = OnceLock::new();

/// Computes a score based on l2 distance.
#[derive(Debug, Copy, Clone, Default)]
pub struct EuclideanDistance(InstructionSet);

impl EuclideanDistance {
    /// Returns a static instance of euclidean distance.
    pub fn get() -> &'static EuclideanDistance {
        L2_DIST.get_or_init(EuclideanDistance::default)
    }
}

impl VectorDistance for EuclideanDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        distance::l2sq(query, doc, Some(self.0))
    }
}

impl F32VectorDistance for EuclideanDistance {
    fn distance_f32(&self, a: &[f32], b: &[f32]) -> f64 {
        distance::l2sq(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(b),
            Some(self.0),
        )
    }
}

static DOT_DIST: OnceLock<DotProductDistance> = OnceLock::new();

/// Computes a score based on the dot product.
#[derive(Debug, Copy, Clone, Default)]
pub struct DotProductDistance(InstructionSet);

impl DotProductDistance {
    /// Returns a static instance of dot product distance.
    pub fn get() -> &'static DotProductDistance {
        DOT_DIST.get_or_init(DotProductDistance::default)
    }
}

impl VectorDistance for DotProductDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        // Assuming values are normalized, this will produce a distance in [0,1]
        (-distance::dot(query, doc, Some(self.0)) + 1.0) / 2.0
    }
}

impl F32VectorDistance for DotProductDistance {
    fn distance_f32(&self, a: &[f32], b: &[f32]) -> f64 {
        // Assuming values are normalized, this will produce a distance in [0,1]
        (-distance::dot(
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
        // We can't assume the vectors have been processed/normalized here so we have to perform
        // full cosine similarity.
        let (ab, a2, b2) = a
            .iter()
            .zip(b.iter())
            .map(|(a, b)| (*a * b, *a * *a, *b * *b))
            .fold((0.0, 0.0, 0.0), |s, x| (s.0 + x.0, s.1 + x.1, s.2 + x.2));
        let cos = ab / (a2.sqrt() * b2.sqrt());
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
        VectorSimilarity::Cosine => Box::new(QueryVectorDistance::new(
            CosineDistance::default(),
            l2_normalize(query),
        )),
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
