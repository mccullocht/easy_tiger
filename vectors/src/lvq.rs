//! Locally adaptive Vector Quantization (LVQ): https://arxiv.org/pdf/2304.04759
//!
//! This has been modified in the same way as Optimized Scalar Quantization in Lucene where the
//! lower and upper bounds are selected by a grid search over the vector taking into account
//! anisotropic loss instead of simply taking min/max values. This grid search is more important
//! at lower bit rates.

#[cfg(target_arch = "aarch64")]
mod aarch64;
mod scalar;
#[cfg(test)]
mod test;
#[cfg(target_arch = "x86_64")]
mod x86_64;

use std::{
    borrow::Cow,
    cell::RefCell,
    ops::{Add, AddAssign},
};

use half::f16;
use thread_local::ThreadLocal;

use crate::{
    EstimatedDistance, F32VectorCoder, QueryVectorDistance, VectorDistance, VectorSimilarity,
    packing,
};

const SUPPORTED_PRIMARY_BITS: [usize; 4] = [1, 2, 4, 8];

const fn is_supported_bits(bits: usize, allowed: &[usize]) -> bool {
    let mut i = 0;
    while i < allowed.len() {
        if bits == allowed[i] {
            return true;
        }
        i += 1;
    }
    false
}

const fn check_primary_bits(bits: usize) {
    assert!(is_supported_bits(bits, &SUPPORTED_PRIMARY_BITS));
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Kernel {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl Kernel {
    const CANDIDATES: &'static [Self] = &[
        #[cfg(target_arch = "aarch64")]
        Self::Neon,
        #[cfg(target_arch = "x86_64")]
        Self::Avx512,
        Self::Scalar,
    ];

    /// Returns true if the specificed kernel is available for use on this cost.
    fn is_available(&self) -> bool {
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Neon => std::arch::is_aarch64_feature_detected!("dotprod"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => {
                use std::arch::is_x86_feature_detected as feature;
                feature!("avx2")
                    && feature!("avx512f")
                    && feature!("avx512bw")
                    && feature!("avx512vl")
                    && feature!("avx512vpopcntdq")
                    && feature!("avx512vnni")
            }
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

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct VectorStats {
    min: f32,
    max: f32,
    mean: f32,
    std_dev: f32,
    l2_norm_sq: f32,
}

impl VectorStats {
    fn new(k: Kernel, value: &[f32]) -> Self {
        if value.is_empty() {
            return VectorStats {
                l2_norm_sq: 1.0,
                ..Default::default()
            };
        }

        match k {
            Kernel::Scalar => scalar::compute_vector_stats(value),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => aarch64::compute_vector_stats(value),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe { x86_64::compute_vector_stats_avx512(value) },
        }
    }
}

fn optimize_interval(k: Kernel, vector: &[f32], stats: &VectorStats, bits: usize) -> (f32, f32) {
    // There are several spots in the optimization routine where we may divide by the input range
    // and if that range is zero then it produces NaNs.
    let (lower, upper) = if stats.min == stats.max {
        (stats.min, stats.min + f32::MIN_POSITIVE)
    } else {
        match k {
            Kernel::Scalar => scalar::optimize_interval_scalar(vector, stats, bits),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => aarch64::optimize_interval_neon(vector, stats, bits),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe { x86_64::optimize_interval_avx512(vector, stats, bits) },
        }
    };
    // The interval bounds are stored as f16, so round them here to keep the values used to
    // quantize the vector consistent with the values a caller sees when decoding it.
    (f16::from_f32(lower).to_f32(), f16::from_f32(upper).to_f32())
}

/// Prepare a vector for quantization by subtracting the center, if any. Returns the prepared
/// vector slice.
///
/// Callers are responsible for any l2 normalization (see [`crate::prepare_vector`]); for angular
/// similarity the input is expected to already be unit length. The center cancels in any distance
/// that reduces to a difference of vectors, so angular distance can be recovered from the
/// squared-Euclidean distance of the centered vectors without storing any cross terms.
///
/// `scratch` must be `Some` (and sized to `vector.len()`) when `center` is `Some`; it is unused
/// and may be `None` otherwise.
fn prepare_vector<'a>(
    vector: &'a [f32],
    scratch: Option<&'a mut [f32]>,
    center: Option<&[f32]>,
) -> &'a [f32] {
    let (Some(scratch), Some(center)) = (scratch, center) else {
        return vector;
    };

    for ((v, s), c) in vector.iter().zip(scratch.iter_mut()).zip(center.iter()) {
        *s = v - c;
    }

    scratch
}

/// Transform the unnormalized dot product of two vectors into an appropriate distance for the
/// similarity function.
fn distance_from_dot_unnormalized(
    similarity: VectorSimilarity,
    dot_unnormalized: f32,
    l2_norms_sq: (f32, f32),
) -> f32 {
    let l2_dist = l2_norms_sq.0 + l2_norms_sq.1 - (2.0 * dot_unnormalized);
    match similarity {
        VectorSimilarity::Euclidean => l2_dist,
        // Normalize angular distance into a value in [0,1] where lower is closer.
        VectorSimilarity::Cosine | VectorSimilarity::Dot => (0.25 * l2_dist).clamp(0.0, 1.0),
    }
}

#[derive(Debug, Copy, Clone)]
struct ErrorBoundTerms {
    l2_norm: f32,
    perpendicular_error_term: f32,
    mult: f32,
}

impl ErrorBoundTerms {
    fn from_header(header: &PrimaryVectorHeader, dim: usize, similarity: VectorSimilarity) -> Self {
        let mult = match similarity {
            VectorSimilarity::Cosine | VectorSimilarity::Dot => 0.5,
            VectorSimilarity::Euclidean => 2.0,
        } / ((dim.max(2) - 1) as f32).sqrt();
        Self {
            l2_norm: header.l2_norm,
            perpendicular_error_term: header.perpendicular_error_term,
            mult,
        }
    }

    fn error_bound<const B: usize>(&self, vector: &TurboPrimaryVector<B>) -> f32 {
        let query_error = self.perpendicular_error_term * vector.l2_norm;
        let doc_error = vector.perpendicular_error_term * self.l2_norm;
        (query_error.powi(2) + doc_error.powi(2)).sqrt() * self.mult
    }
}

/// Header for an LVQ primary vector.
///
/// Along with the bit configuration this carries enough metadata to transform a quantized vector
/// value stream back to an f32 representation or compute angular or l2 distance from another
/// vector.
#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(C)]
struct PrimaryVectorHeader {
    /// L2 norm (magnitude) of the (possibly centered) vector.
    /// This is used to compute euclidean and angular distance and the statistical bound on
    /// estimated distance. For angular similarity this is the norm of the centered unit vector.
    l2_norm: f32,
    /// The L2 norm of the residual vector (v - dequantize(quantize(v))).
    /// This term can be used to compute a statistical bound on the estimated distance.
    perpendicular_error_term: f32,
    /// Parallel error -- the projection of the vector onto the quantized residual divided by the
    /// squared l2 norm.
    ///
    /// Interval optimization minimizes this term so it is typically small.
    parallel_error_term: f32,
    /// Lower interval bound used for quantization, no smaller than the minimum component value.
    /// This is used to correct the uint dot product to an f32 dot product.
    lower: f32,
    /// Upper interval bound used for quantization, no larger than the maximum component value.
    /// This is used to correct the uint dot product to an f32 dot product.
    upper: f32,
    /// Sum of all the quantized components of the vector. This is used to correct the uint dot
    /// product to an f32 dot product.
    component_sum: u32,
}

impl PrimaryVectorHeader {
    /// Length of the encoded header in bytes.
    ///
    /// Stores 6 values -- 4 16-bit values and 2 32-bit values.
    /// * l2_norm (f32)
    /// * component_sum (u32)
    /// * perpendicular_error_term (f16)
    /// * parallel_error_term (f16)
    /// * lower (f16)
    /// * upper (f16)
    ///
    /// The first term is stored as 32 bits as it is combined directly into the final distance.
    const LEN: usize =
        std::mem::size_of::<f32>() + std::mem::size_of::<u32>() + std::mem::size_of::<f16>() * 4;

    fn new(stats: VectorStats) -> Self {
        Self {
            l2_norm: stats.l2_norm_sq.sqrt(),
            component_sum: 0,
            perpendicular_error_term: 0.0,
            parallel_error_term: 0.0,
            lower: stats.min,
            upper: stats.max,
        }
    }

    #[inline]
    fn split_output_buf(buf: &mut [u8]) -> Option<(&mut [u8], &mut [u8])> {
        buf.split_at_mut_checked(Self::LEN)
    }

    #[inline]
    fn serialize(&self, header_bytes: &mut [u8]) {
        let h32 = header_bytes[..8].as_chunks_mut::<4>().0;
        h32[0] = self.l2_norm.to_le_bytes();
        h32[1] = self.component_sum.to_le_bytes();

        let h16 = header_bytes[8..16].as_chunks_mut::<2>().0;
        h16[0] = f16::from_f32(self.perpendicular_error_term / self.l2_norm).to_le_bytes();
        h16[1] = f16::from_f32(self.parallel_error_term).to_le_bytes();
        h16[2] = f16::from_f32(self.lower).to_le_bytes();
        h16[3] = f16::from_f32(self.upper).to_le_bytes();
    }

    #[inline]
    fn deserialize(raw: &[u8]) -> Option<(Self, &[u8])> {
        let (header_bytes, vector_bytes) = raw.split_at_checked(Self::LEN)?;
        let h32 = header_bytes[..8].as_chunks::<4>().0;
        let h16 = header_bytes[8..16].as_chunks::<2>().0;
        let l2_norm = f32::from_le_bytes(h32[0]);
        Some((
            Self {
                l2_norm,
                component_sum: u32::from_le_bytes(h32[1]),
                perpendicular_error_term: f16::from_le_bytes(h16[0]).to_f32() * l2_norm,
                parallel_error_term: f16::from_le_bytes(h16[1]).to_f32(),
                lower: f16::from_le_bytes(h16[2]).to_f32(),
                upper: f16::from_le_bytes(h16[3]).to_f32(),
            },
            vector_bytes,
        ))
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
struct QuantizationStats {
    /// Sum of all quantized primary components.
    primary_component_sum: u32,
    /// Squared error of the quantization residual.
    residual_error_sq: f32,
    /// The inner product of each vector component and it's primary quantization residual.
    residual_ip: f32,
}

impl QuantizationStats {
    fn add_component(self, cp: u32, v: f32, r: f32) -> Self {
        Self {
            primary_component_sum: self.primary_component_sum + cp,
            residual_error_sq: r.mul_add(r, self.residual_error_sq),
            residual_ip: v.mul_add(r, self.residual_ip),
        }
    }
}

impl Add<QuantizationStats> for QuantizationStats {
    type Output = Self;

    fn add(self, rhs: QuantizationStats) -> Self::Output {
        Self {
            primary_component_sum: self.primary_component_sum + rhs.primary_component_sum,
            residual_error_sq: self.residual_error_sq + rhs.residual_error_sq,
            residual_ip: self.residual_ip + rhs.residual_ip,
        }
    }
}

impl AddAssign<QuantizationStats> for QuantizationStats {
    fn add_assign(&mut self, rhs: QuantizationStats) {
        *self = *self + rhs;
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct VectorDecodeTerms {
    lower: f32,
    delta: f32,
    component_sum: u32,
    parallel_error_term: f32,
}

impl VectorDecodeTerms {
    fn from_primary<const B: usize>(header: PrimaryVectorHeader) -> Self {
        Self {
            lower: header.lower,
            delta: (header.upper - header.lower) / ((1 << B) - 1) as f32,
            component_sum: header.component_sum,
            parallel_error_term: header.parallel_error_term,
        }
    }
}

const MINIMUM_MSE_GRID: [(f32, f32); 8] = [
    (-0.798, 0.798),
    (-1.493, 1.493),
    (-2.051, 2.051),
    (-2.514, 2.514),
    (-2.916, 2.916),
    (-3.278, 3.278),
    (-3.611, 3.611),
    (-3.922, 3.922),
];

const LAMBDA: f32 = 0.1;

#[derive(Debug, Copy, Clone, PartialEq)]
struct EncodedVector<'a> {
    terms: VectorDecodeTerms,
    data: &'a [u8],
}

impl<'a> EncodedVector<'a> {
    fn split_at(&self, i: usize) -> (Self, Self) {
        let (s, e) = self.data.split_at(i);
        (
            Self {
                terms: self.terms,
                data: s,
            },
            Self {
                terms: self.terms,
                data: e,
            },
        )
    }
}

/// Correct the dot product of two integer vectors using the stored vector terms.
fn correct_dot_uint(dot: u32, dim: usize, a: &VectorDecodeTerms, b: &VectorDecodeTerms) -> f32 {
    // Note that any dot value larger than (2 << 24) will be rounded when converted to f32 which can
    // cause vector comparisons a <-> b and b <-> a to return slightly different results. To prevent
    // this convert dot to f64 before including it in the correction.
    let fdot = (dot as f64 * (a.delta * b.delta) as f64
        + (a.component_sum as f32 * a.delta * b.lower
            + b.component_sum as f32 * b.delta * a.lower
            + a.lower * b.lower * dim as f32) as f64) as f32;
    fdot * (1.0 + a.parallel_error_term + b.parallel_error_term)
}

struct TurboPrimaryVector<'a, const B: usize> {
    rep: EncodedVector<'a>,
    l2_norm: f32,
    perpendicular_error_term: f32,
}

impl<'a, const B: usize> TurboPrimaryVector<'a, B> {
    fn new(data: &'a [u8]) -> Option<Self> {
        let (header, vector_bytes) = PrimaryVectorHeader::deserialize(data)?;
        Some(Self {
            rep: EncodedVector {
                terms: VectorDecodeTerms::from_primary::<B>(header),
                data: vector_bytes,
            },
            l2_norm: header.l2_norm,
            perpendicular_error_term: header.perpendicular_error_term,
        })
    }

    fn dim(&self) -> usize {
        (self.rep.data.len() * 8) / B
    }

    fn split_tail(&self, dim: usize) -> (usize, Self, Self) {
        let tail_dim = dim & !(packing::block_dim(B) - 1);
        let (headv, tailv) = self.rep.split_at(packing::byte_len(tail_dim, B));
        (
            tail_dim,
            Self {
                rep: headv,
                l2_norm: self.l2_norm,
                perpendicular_error_term: self.perpendicular_error_term,
            },
            Self {
                rep: tailv,
                l2_norm: self.l2_norm,
                perpendicular_error_term: self.perpendicular_error_term,
            },
        )
    }
}

#[derive(Debug)]
pub struct TurboPrimaryCoder<const B: usize> {
    center: Option<Vec<f32>>,
    scratch: ThreadLocal<RefCell<Vec<f32>>>,
    k: Kernel,
}

impl<const B: usize> TurboPrimaryCoder<B> {
    const B_CHECK: () = { check_primary_bits(B) };

    /// Quantization is similarity-agnostic; `similarity` is accepted for call-site symmetry with
    /// the other coders and is not retained.
    pub fn new(_similarity: VectorSimilarity, center: Option<Vec<f32>>) -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::B_CHECK;
        Self {
            center,
            scratch: ThreadLocal::new(),
            k: Kernel::default(),
        }
    }

    #[cfg(test)]
    fn with_kernel(k: Kernel, _similarity: VectorSimilarity, center: Option<Vec<f32>>) -> Self {
        assert!(k.is_available(), "{k:?}");
        #[allow(clippy::let_unit_value)]
        let _ = Self::B_CHECK;
        Self {
            center,
            scratch: ThreadLocal::new(),
            k,
        }
    }

    fn encode_parts(
        k: Kernel,
        vector: &[f32],
        center: Option<&[f32]>,
    ) -> (PrimaryVectorHeader, Vec<u8>) {
        let mut scratch_storage = if center.is_some() {
            vec![0.0f32; vector.len()]
        } else {
            vec![]
        };
        let scratch = center.is_some().then_some(scratch_storage.as_mut_slice());
        let prepared = prepare_vector(vector, scratch, center);
        let mut out = vec![0u8; packing::byte_len(vector.len(), B)];
        let header = Self::encode_parts_to(k, prepared, &mut out);
        (header, out)
    }

    fn encode_parts_to(k: Kernel, vector: &[f32], out: &mut [u8]) -> PrimaryVectorHeader {
        let stats = VectorStats::new(k, vector);
        let mut header = PrimaryVectorHeader::new(stats);
        (header.lower, header.upper) = optimize_interval(k, vector, &stats, B);

        let terms = VectorEncodeTerms::from_primary::<B>(&header);
        let quant_stats = match k {
            Kernel::Scalar => scalar::primary_quantize_and_pack::<B>(vector, terms, out),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => aarch64::primary_quantize_and_pack::<B>(vector, terms, out),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe {
                x86_64::primary_quantize_and_pack_avx512::<B>(vector, terms, out)
            },
        };
        header.component_sum = quant_stats.primary_component_sum;
        let perp_error_sq =
            quant_stats.residual_error_sq - (quant_stats.residual_ip.powi(2) / stats.l2_norm_sq);
        header.perpendicular_error_term = perp_error_sq.sqrt();
        header.parallel_error_term = quant_stats.residual_ip / stats.l2_norm_sq;

        header
    }
}

impl<const B: usize> F32VectorCoder for TurboPrimaryCoder<B> {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let (header_bytes, vector_bytes) = PrimaryVectorHeader::split_output_buf(out).unwrap();

        let mut scratch_guard = if self.center.is_some() {
            let mut g = self.scratch.get_or_default().borrow_mut();
            g.resize(vector.len(), 0.0);
            Some(g)
        } else {
            None
        };
        let prepared = prepare_vector(
            vector,
            scratch_guard.as_mut().map(|g| g.as_mut_slice()),
            self.center.as_deref(),
        );
        let header = Self::encode_parts_to(self.k, prepared, vector_bytes);
        header.serialize(header_bytes);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        PrimaryVectorHeader::LEN + packing::byte_len(dimensions, B)
    }

    fn decode_to(&self, vector: &[u8], out: &mut [f32]) {
        let vector = TurboPrimaryVector::<B>::new(vector).expect("valid primary vector");
        match self.k {
            Kernel::Scalar => scalar::primary_decode::<B>(vector, out),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => aarch64::primary_decode::<B>(vector, out),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe { x86_64::primary_decode_avx512::<B>(vector, out) },
        };
        if let Some(center) = &self.center {
            for (c, v) in center.iter().zip(out.iter_mut()) {
                *v += *c;
            }
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        let vector_bytes = byte_len - PrimaryVectorHeader::LEN;
        (vector_bytes * 8) / B
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TurboPrimaryDistance<const B: usize> {
    similarity: VectorSimilarity,
    inst: Kernel,
}

impl<const B: usize> TurboPrimaryDistance<B> {
    pub fn new(similarity: VectorSimilarity) -> Self {
        Self {
            similarity,
            inst: Kernel::default(),
        }
    }

    #[inline(always)]
    fn distance_internal(&self, query: &TurboPrimaryVector<B>, doc: &[u8]) -> f64 {
        let doc = TurboPrimaryVector::<B>::new(doc).unwrap();
        let uint_dot = match self.inst {
            Kernel::Scalar => scalar::dot_u8::<B>(query.rep.data, doc.rep.data),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => aarch64::dot_u8::<B>(query.rep.data, doc.rep.data),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe { x86_64::dot_u8_avx512::<B>(query.rep.data, doc.rep.data) },
        };
        let dot = correct_dot_uint(uint_dot, query.dim(), &query.rep.terms, &doc.rep.terms);
        distance_from_dot_unnormalized(
            self.similarity,
            dot,
            (query.l2_norm.powi(2), doc.l2_norm.powi(2)),
        )
        .into()
    }
}

impl<const B: usize> VectorDistance for TurboPrimaryDistance<B> {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = TurboPrimaryVector::<B>::new(query).unwrap();
        self.distance_internal(&query, doc)
    }

    fn bulk_distance(&self, query: &[u8], docs: &[&[u8]], out: &mut [f64]) {
        let query = TurboPrimaryVector::<B>::new(query).unwrap();
        for (doc, out) in docs.iter().zip(out.iter_mut()) {
            *out = self.distance_internal(&query, doc);
        }
    }
}

const PRIMARY_QUERY_BITS: usize = 8;

#[derive(Debug, Clone)]
pub struct TurboPrimaryQueryDistance<const B: usize> {
    k: Kernel,
    similarity: VectorSimilarity,

    query: Vec<u8>,
    l2_norm_sq: f32,
    terms: VectorDecodeTerms,
    error_terms: ErrorBoundTerms,
}

impl<const B: usize> TurboPrimaryQueryDistance<B> {
    pub fn new(
        similarity: VectorSimilarity,
        query: Cow<'_, [f32]>,
        center: Option<&[f32]>,
    ) -> Self {
        let k = Kernel::default();
        let (header, query) =
            TurboPrimaryCoder::<PRIMARY_QUERY_BITS>::encode_parts(k, query.as_ref(), center);
        let terms = VectorDecodeTerms::from_primary::<PRIMARY_QUERY_BITS>(header);
        let error_terms = ErrorBoundTerms::from_header(&header, query.len(), similarity);

        Self {
            k,
            similarity,
            query,
            l2_norm_sq: header.l2_norm.powi(2),
            terms,
            error_terms,
        }
    }

    #[inline(always)]
    fn distance_internal_raw(&self, vector: &[u8]) -> f64 {
        let vector = TurboPrimaryVector::<B>::new(vector).expect("valid primary vector");
        self.distance_internal(&vector)
    }

    #[inline(always)]
    fn distance_internal(&self, vector: &TurboPrimaryVector<B>) -> f64 {
        let uint8_dot = match self.k {
            Kernel::Scalar => scalar::primary_query8_dot_unnormalized::<B>(&self.query, vector),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => aarch64::primary_query8_dot_unnormalized::<B>(&self.query, vector),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe {
                x86_64::primary_query8_dot_unnormalized_avx512::<B>(&self.query, vector)
            },
        };
        let dot = correct_dot_uint(uint8_dot, self.query.len(), &self.terms, &vector.rep.terms);
        distance_from_dot_unnormalized(
            self.similarity,
            dot,
            (self.l2_norm_sq, vector.l2_norm.powi(2)),
        )
        .into()
    }
}

impl<const B: usize> QueryVectorDistance for TurboPrimaryQueryDistance<B> {
    fn distance(&self, vector: &[u8]) -> f64 {
        self.distance_internal_raw(vector)
    }

    fn bulk_distance(&self, vectors: &[&[u8]], out: &mut [f64]) {
        for (vector, out) in vectors.iter().zip(out.iter_mut()) {
            *out = self.distance_internal_raw(vector);
        }
    }

    // TODO: add tests for this. Right now it is not accurate enough to write good tests.
    fn estimated_distance(&self, vector: &[u8]) -> EstimatedDistance {
        let vector = TurboPrimaryVector::<B>::new(vector).expect("valid primary vector");
        let distance = self.distance_internal(&vector);
        let error = self.error_terms.error_bound(&vector).into();
        EstimatedDistance { distance, error }
    }
}

/// Asymmetric distance for 1-bit document with a 4-bit query.
///
/// This trades some of the accuracy gain from asymmetric distance for speed. The query bitplane
/// is split so that the dot product can be computed entirely with popcount.
#[derive(Debug, Clone)]
pub struct TurboPrimaryQueryDistance1 {
    k: Kernel,
    similarity: VectorSimilarity,

    query: Vec<u8>,
    l2_norm_sq: f32,
    terms: VectorDecodeTerms,
    error_terms: ErrorBoundTerms,
}

impl TurboPrimaryQueryDistance1 {
    pub fn new(
        similarity: VectorSimilarity,
        query: Cow<'_, [f32]>,
        center: Option<&[f32]>,
    ) -> Self {
        let k = Kernel::default();
        let (header, query) = TurboPrimaryCoder::<4>::encode_parts(k, query.as_ref(), center);
        let query = packing::bitplane_split4(&query);
        let terms = VectorDecodeTerms::from_primary::<4>(header);
        let error_terms = ErrorBoundTerms::from_header(&header, query.len() * 2, similarity);

        Self {
            k,
            similarity,
            query,
            l2_norm_sq: header.l2_norm.powi(2),
            terms,
            error_terms,
        }
    }

    #[inline(always)]
    fn distance_internal_raw(&self, vector: &[u8]) -> f64 {
        let vector = TurboPrimaryVector::<1>::new(vector).expect("valid primary vector");
        self.distance_internal(&vector)
    }

    #[inline(always)]
    fn distance_internal(&self, vector: &TurboPrimaryVector<1>) -> f64 {
        let uint8_dot = match self.k {
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => aarch64::query4_doc1_bitplane_dot(&self.query, vector.rep.data),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe {
                x86_64::query4_doc1_bitplane_dot_avx512(&self.query, vector.rep.data)
            },
            _ => scalar::query4_doc1_bitplane_dot(&self.query, vector.rep.data),
        };
        let dot = correct_dot_uint(
            uint8_dot,
            self.query.len() * 2,
            &self.terms,
            &vector.rep.terms,
        );
        distance_from_dot_unnormalized(
            self.similarity,
            dot,
            (self.l2_norm_sq, vector.l2_norm.powi(2)),
        )
        .into()
    }
}

impl QueryVectorDistance for TurboPrimaryQueryDistance1 {
    fn distance(&self, vector: &[u8]) -> f64 {
        self.distance_internal_raw(vector)
    }

    fn bulk_distance(&self, vectors: &[&[u8]], out: &mut [f64]) {
        for (vector, out) in vectors.iter().zip(out.iter_mut()) {
            *out = self.distance_internal_raw(vector);
        }
    }

    // TODO: add tests for this. Right now it is not accurate enough to write good tests.
    fn estimated_distance(&self, vector: &[u8]) -> EstimatedDistance {
        let vector = TurboPrimaryVector::<1>::new(vector).expect("valid primary vector");
        let distance = self.distance_internal(&vector);
        let error = self.error_terms.error_bound(&vector).into();
        EstimatedDistance { distance, error }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct VectorEncodeTerms {
    lower: f32,
    upper: f32,
    delta_inv: f32,
    delta: f32,
}

impl VectorEncodeTerms {
    fn from_primary<const B: usize>(primary: &PrimaryVectorHeader) -> Self {
        let delta_inv = ((1 << B) - 1) as f32 / (primary.upper - primary.lower);
        let delta = (primary.upper - primary.lower) / ((1 << B) - 1) as f32;
        Self {
            lower: primary.lower,
            upper: primary.upper,
            delta_inv,
            delta,
        }
    }
}
