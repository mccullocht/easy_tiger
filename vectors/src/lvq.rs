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
    float32::l2_norm,
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

/// Prepare a vector for quantization: optionally l2-normalize (for angular similarity) and/or
/// subtract the center. Returns the prepared vector slice and the center_dot correction term.
///
/// `scratch` must be `Some` (and sized to `vector.len()`) when `similarity.angular()` or `center`
/// is `Some`; it is unused and may be `None` otherwise.
fn prepare_vector<'a>(
    vector: &'a [f32],
    scratch: Option<&'a mut [f32]>,
    center: Option<&[f32]>,
    similarity: VectorSimilarity,
) -> (&'a [f32], f32) {
    let Some(scratch) = scratch else {
        return (vector, 0.0);
    };

    if similarity.angular() {
        let norm = l2_norm(vector);
        let scale = if norm > 0.0 { 1.0 / norm } else { 1.0 };
        for (v, s) in vector.iter().zip(scratch.iter_mut()) {
            *s = v * scale;
        }
    } else {
        scratch.copy_from_slice(vector);
    }

    let center_dot = if let Some(center) = center {
        let cd = if similarity.angular() {
            scratch
                .iter()
                .zip(center.iter())
                .map(|(&s, &c)| s * c)
                .sum()
        } else {
            0.0
        };
        for (s, c) in scratch.iter_mut().zip(center.iter()) {
            *s -= c;
        }
        cd
    } else {
        0.0
    };

    (scratch, center_dot)
}

fn uncenter_vector(center: &[f32], vector: &mut [f32]) {
    for (c, v) in center.iter().zip(vector.iter_mut()) {
        *v += *c;
    }
}

#[derive(Debug, Clone)]
enum DistanceCorrectionTerms {
    Euclidean {
        l2_norm_sq: f32,
    },
    Angular {
        center_dot: f32,
        center_center_dot: f32,
    },
}

impl DistanceCorrectionTerms {
    fn new(
        header: &PrimaryVectorHeader,
        center: Option<&[f32]>,
        similarity: VectorSimilarity,
    ) -> Self {
        match similarity {
            VectorSimilarity::Euclidean => Self::Euclidean {
                l2_norm_sq: header.l2_norm.powi(2),
            },
            VectorSimilarity::Dot | VectorSimilarity::Cosine => Self::Angular {
                center_dot: header.center_dot,
                center_center_dot: center
                    .map(|c| c.iter().map(|&v| v * v).sum())
                    .unwrap_or(0.0),
            },
        }
    }

    fn from_parts(
        l2_norm: f32,
        center_dot: f32,
        center_center_dot: f32,
        similarity: VectorSimilarity,
    ) -> Self {
        match similarity {
            VectorSimilarity::Euclidean => Self::Euclidean {
                l2_norm_sq: l2_norm.powi(2),
            },
            VectorSimilarity::Dot | VectorSimilarity::Cosine => Self::Angular {
                center_dot,
                center_center_dot,
            },
        }
    }

    fn distance_from_dot_unnormalized(
        &self,
        dot_unnormalized: f32,
        vector_l2_norm: f32,
        vector_center_dot: f32,
    ) -> f32 {
        match self {
            Self::Euclidean { l2_norm_sq } => {
                l2_norm_sq + vector_l2_norm.powi(2) - (2.0 * dot_unnormalized)
            }
            Self::Angular {
                center_dot,
                center_center_dot,
            } => (dot_unnormalized + center_dot + vector_center_dot - center_center_dot)
                .mul_add(-0.5, 0.5),
        }
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
    /// L2 norm (magnitude) of the vector.
    /// This is used to compute euclidean distance and the statistical bound on estimated distance.
    l2_norm: f32,
    /// The dot product of the vector and the centroid.
    /// This is used to compute angular distance when the vector is centered.
    center_dot: f32,
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
    /// * l2_norm or center_dot (f32)
    /// * component_sum (u32)
    /// * perpendicular_error_term (f16)
    /// * parallel_error_term (f16)
    /// * lower (f16)
    /// * upper (f16)
    ///
    /// The first term is stored as 32 bits as it is combined directly into the final distance.
    const LEN: usize =
        std::mem::size_of::<f32>() + std::mem::size_of::<u32>() + std::mem::size_of::<f16>() * 4;

    fn new(stats: VectorStats, center_dot: f32) -> Self {
        Self {
            l2_norm: stats.l2_norm_sq.sqrt(),
            center_dot,
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
    fn serialize(&self, header_bytes: &mut [u8], similarity: VectorSimilarity) {
        let h32 = header_bytes[..8].as_chunks_mut::<4>().0;
        let first = if similarity.angular() {
            self.center_dot
        } else {
            self.l2_norm
        };
        h32[0] = first.to_le_bytes();
        h32[1] = self.component_sum.to_le_bytes();

        let h16 = header_bytes[8..16].as_chunks_mut::<2>().0;
        h16[0] = f16::from_f32(self.perpendicular_error_term / self.l2_norm).to_le_bytes();
        h16[1] = f16::from_f32(self.parallel_error_term).to_le_bytes();
        h16[2] = f16::from_f32(self.lower).to_le_bytes();
        h16[3] = f16::from_f32(self.upper).to_le_bytes();
    }

    #[inline]
    fn deserialize(raw: &[u8], similarity: VectorSimilarity) -> Option<(Self, &[u8])> {
        let (header_bytes, vector_bytes) = raw.split_at_checked(Self::LEN)?;
        let h32 = header_bytes[..8].as_chunks::<4>().0;
        let h16 = header_bytes[8..16].as_chunks::<2>().0;
        let l2_norm = if similarity.angular() {
            1.0
        } else {
            f32::from_le_bytes(h32[0])
        };
        Some((
            Self {
                l2_norm,
                center_dot: if similarity.angular() {
                    f32::from_le_bytes(h32[0])
                } else {
                    0.0
                },
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

/// The turbo coder requires that all vector data be packed into 16-byte blocks.
const TURBO_BLOCK_SIZE: usize = 16;

struct TurboPrimaryVector<'a, const B: usize> {
    rep: EncodedVector<'a>,
    l2_norm: f32,
    perpendicular_error_term: f32,
    center_dot: f32,
}

impl<'a, const B: usize> TurboPrimaryVector<'a, B> {
    fn new(data: &'a [u8], similarity: VectorSimilarity) -> Option<Self> {
        let (header, vector_bytes) = PrimaryVectorHeader::deserialize(data, similarity)?;
        Some(Self {
            rep: EncodedVector {
                terms: VectorDecodeTerms::from_primary::<B>(header),
                data: vector_bytes,
            },
            l2_norm: header.l2_norm,
            perpendicular_error_term: header.perpendicular_error_term,
            center_dot: header.center_dot,
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
                center_dot: self.center_dot,
            },
            Self {
                rep: tailv,
                l2_norm: self.l2_norm,
                perpendicular_error_term: self.perpendicular_error_term,
                center_dot: self.center_dot,
            },
        )
    }
}

#[derive(Debug)]
pub struct TurboPrimaryCoder<const B: usize> {
    similarity: VectorSimilarity,
    center: Option<Vec<f32>>,
    scratch: ThreadLocal<RefCell<Vec<f32>>>,
    k: Kernel,
}

impl<const B: usize> TurboPrimaryCoder<B> {
    const B_CHECK: () = { check_primary_bits(B) };

    pub fn new(similarity: VectorSimilarity, center: Option<Vec<f32>>) -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::B_CHECK;
        Self {
            similarity,
            center,
            scratch: ThreadLocal::new(),
            k: Kernel::default(),
        }
    }

    #[cfg(test)]
    fn with_kernel(k: Kernel, similarity: VectorSimilarity, center: Option<Vec<f32>>) -> Self {
        assert!(k.is_available(), "{k:?}");
        #[allow(clippy::let_unit_value)]
        let _ = Self::B_CHECK;
        Self {
            similarity,
            center,
            scratch: ThreadLocal::new(),
            k,
        }
    }

    fn encode_parts(
        k: Kernel,
        similarity: VectorSimilarity,
        vector: &[f32],
        center: Option<&[f32]>,
    ) -> (PrimaryVectorHeader, Vec<u8>) {
        let needs_scratch = similarity.angular() || center.is_some();
        let mut scratch_storage = if needs_scratch {
            vec![0.0f32; vector.len()]
        } else {
            vec![]
        };
        let scratch = if needs_scratch {
            Some(scratch_storage.as_mut_slice())
        } else {
            None
        };
        let (prepared, center_dot) = prepare_vector(vector, scratch, center, similarity);
        let mut out = vec![0u8; packing::byte_len(vector.len(), B)];
        let header = Self::encode_parts_to(k, prepared, center_dot, &mut out);
        (header, out)
    }

    fn encode_parts_to(
        k: Kernel,
        vector: &[f32],
        center_dot: f32,
        out: &mut [u8],
    ) -> PrimaryVectorHeader {
        let stats = VectorStats::new(k, vector);
        let mut header = PrimaryVectorHeader::new(stats, center_dot);
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

        let needs_scratch = self.similarity.angular() || self.center.is_some();
        let mut scratch_guard = if needs_scratch {
            let mut g = self.scratch.get_or_default().borrow_mut();
            g.resize(vector.len(), 0.0);
            Some(g)
        } else {
            None
        };
        let (prepared, center_dot) = prepare_vector(
            vector,
            scratch_guard.as_mut().map(|g| g.as_mut_slice()),
            self.center.as_deref(),
            self.similarity,
        );
        let header = Self::encode_parts_to(self.k, prepared, center_dot, vector_bytes);
        header.serialize(header_bytes, self.similarity);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        PrimaryVectorHeader::LEN + packing::byte_len(dimensions, B)
    }

    fn decode_to(&self, vector: &[u8], out: &mut [f32]) {
        let vector =
            TurboPrimaryVector::<B>::new(vector, self.similarity).expect("valid primary vector");
        match self.k {
            Kernel::Scalar => scalar::primary_decode::<B>(vector, out),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => aarch64::primary_decode::<B>(vector, out),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe { x86_64::primary_decode_avx512::<B>(vector, out) },
        };
        if let Some(c) = &self.center {
            uncenter_vector(c, out);
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
    center_center_dot: f32,
    inst: Kernel,
}

impl<const B: usize> TurboPrimaryDistance<B> {
    pub fn new(similarity: VectorSimilarity, center: Option<&[f32]>) -> Self {
        let center_center_dot = if let Some(c) = center
            && similarity.angular()
        {
            c.iter().map(|v| v * v).sum()
        } else {
            0.0
        };
        Self {
            similarity,
            center_center_dot,
            inst: Kernel::default(),
        }
    }

    #[inline(always)]
    fn distance_internal(
        &self,
        correction_terms: &DistanceCorrectionTerms,
        query: &TurboPrimaryVector<B>,
        doc: &[u8],
    ) -> f64 {
        let doc = TurboPrimaryVector::<B>::new(doc, self.similarity).unwrap();
        let uint_dot = match self.inst {
            Kernel::Scalar => scalar::dot_u8::<B>(query.rep.data, doc.rep.data),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => aarch64::dot_u8::<B>(query.rep.data, doc.rep.data),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe { x86_64::dot_u8_avx512::<B>(query.rep.data, doc.rep.data) },
        };
        let dot = correct_dot_uint(uint_dot, query.dim(), &query.rep.terms, &doc.rep.terms);
        correction_terms
            .distance_from_dot_unnormalized(dot, doc.l2_norm, doc.center_dot)
            .into()
    }
}

impl<const B: usize> VectorDistance for TurboPrimaryDistance<B> {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = TurboPrimaryVector::<B>::new(query, self.similarity).unwrap();
        let correction_terms = DistanceCorrectionTerms::from_parts(
            query.l2_norm,
            query.center_dot,
            self.center_center_dot,
            self.similarity,
        );
        self.distance_internal(&correction_terms, &query, doc)
    }

    fn bulk_distance(&self, query: &[u8], docs: &[&[u8]], out: &mut [f64]) {
        let query = TurboPrimaryVector::<B>::new(query, self.similarity).unwrap();
        let correction_terms = DistanceCorrectionTerms::from_parts(
            query.l2_norm,
            query.center_dot,
            self.center_center_dot,
            self.similarity,
        );
        for (doc, out) in docs.iter().zip(out.iter_mut()) {
            *out = self.distance_internal(&correction_terms, &query, doc);
        }
    }
}

const PRIMARY_QUERY_BITS: usize = 8;

#[derive(Debug, Clone)]
pub struct TurboPrimaryQueryDistance<const B: usize> {
    k: Kernel,
    similarity: VectorSimilarity,

    query: Vec<u8>,
    terms: VectorDecodeTerms,
    correction_terms: DistanceCorrectionTerms,
    error_terms: ErrorBoundTerms,
}

impl<const B: usize> TurboPrimaryQueryDistance<B> {
    pub fn new(
        similarity: VectorSimilarity,
        query: Cow<'_, [f32]>,
        center: Option<&[f32]>,
    ) -> Self {
        let k = Kernel::default();
        let (header, query) = TurboPrimaryCoder::<PRIMARY_QUERY_BITS>::encode_parts(
            k,
            similarity,
            query.as_ref(),
            center,
        );
        let terms = VectorDecodeTerms::from_primary::<PRIMARY_QUERY_BITS>(header);
        let correction_terms = DistanceCorrectionTerms::new(&header, center, similarity);
        let error_terms = ErrorBoundTerms::from_header(&header, query.len(), similarity);

        Self {
            k,
            similarity,
            query,
            terms,
            correction_terms,
            error_terms,
        }
    }

    #[inline(always)]
    fn distance_internal_raw(&self, vector: &[u8]) -> f64 {
        let vector =
            TurboPrimaryVector::<B>::new(vector, self.similarity).expect("valid primary vector");
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
        self.correction_terms
            .distance_from_dot_unnormalized(dot, vector.l2_norm, vector.center_dot)
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
        let vector =
            TurboPrimaryVector::<B>::new(vector, self.similarity).expect("valid primary vector");
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
    #[allow(dead_code)]
    k: Kernel,
    similarity: VectorSimilarity,

    query: Vec<u8>,
    terms: VectorDecodeTerms,
    correction_terms: DistanceCorrectionTerms,
    error_terms: ErrorBoundTerms,
}

impl TurboPrimaryQueryDistance1 {
    pub fn new(
        similarity: VectorSimilarity,
        query: Cow<'_, [f32]>,
        center: Option<&[f32]>,
    ) -> Self {
        let k = Kernel::default();
        let (header, query) =
            TurboPrimaryCoder::<4>::encode_parts(k, similarity, query.as_ref(), center);
        let query = packing::bitplane_split4(&query);
        let terms = VectorDecodeTerms::from_primary::<4>(header);
        let correction_terms = DistanceCorrectionTerms::new(&header, center, similarity);
        let error_terms = ErrorBoundTerms::from_header(&header, query.len() * 2, similarity);

        Self {
            k,
            similarity,
            query,
            terms,
            correction_terms,
            error_terms,
        }
    }

    #[inline(always)]
    fn distance_internal_raw(&self, vector: &[u8]) -> f64 {
        let vector =
            TurboPrimaryVector::<1>::new(vector, self.similarity).expect("valid primary vector");
        self.distance_internal(&vector)
    }

    #[inline(always)]
    fn distance_internal(&self, vector: &TurboPrimaryVector<1>) -> f64 {
        let uint8_dot = {
            let (qhead, qtail) = self.query.as_chunks::<64>();
            let (dhead, dtail) = vector.rep.data.split_at(qhead.len() * 16);
            let dhead = dhead.as_chunks::<16>().0;
            let mut pdot = [0u32; 4];
            for (q, d) in qhead.iter().zip(dhead.iter()) {
                let qc = q.as_chunks::<16>().0;
                let q0 = u128::from_le_bytes(qc[0]);
                let q1 = u128::from_le_bytes(qc[1]);
                let q2 = u128::from_le_bytes(qc[2]);
                let q3 = u128::from_le_bytes(qc[3]);
                let d = u128::from_le_bytes(*d);
                pdot[0] += (q0 & d).count_ones();
                pdot[1] += (q1 & d).count_ones();
                pdot[2] += (q2 & d).count_ones();
                pdot[3] += (q3 & d).count_ones();
            }

            if !qtail.is_empty() {
                let mut qit = qtail.chunks(qtail.len() / 4);
                let q = [
                    qit.next().unwrap(),
                    qit.next().unwrap(),
                    qit.next().unwrap(),
                    qit.next().unwrap(),
                ];

                for (i, &d) in dtail.iter().enumerate() {
                    pdot[0] += (q[0][i] & d).count_ones();
                    pdot[1] += (q[1][i] & d).count_ones();
                    pdot[2] += (q[2][i] & d).count_ones();
                    pdot[3] += (q[3][i] & d).count_ones();
                }
            }

            pdot[0] + pdot[1] * 2 + pdot[2] * 4 + pdot[3] * 8
        };
        let dot = correct_dot_uint(
            uint8_dot,
            self.query.len() * 2,
            &self.terms,
            &vector.rep.terms,
        );
        self.correction_terms
            .distance_from_dot_unnormalized(dot, vector.l2_norm, vector.center_dot)
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
        let vector =
            TurboPrimaryVector::<1>::new(vector, self.similarity).expect("valid primary vector");
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

pub(super) mod packing {
    use std::iter::FusedIterator;

    use crate::lvq::TURBO_BLOCK_SIZE;

    /// The number of bytes required to pack `dimensions` with `bits` per entry.
    pub const fn byte_len(dimensions: usize, bits: usize) -> usize {
        (dimensions * bits).div_ceil(8)
    }

    pub struct TurboPacker<'a, const B: usize> {
        blocks: &'a mut [[u8; TURBO_BLOCK_SIZE]],
        tail: &'a mut [u8],
        block: usize,
        nbuf: usize,
    }

    impl<'a, const B: usize> TurboPacker<'a, B> {
        pub fn new(vector_bytes: &'a mut [u8]) -> Self {
            let (blocks, tail) = vector_bytes.as_chunks_mut::<TURBO_BLOCK_SIZE>();
            Self {
                blocks,
                tail,
                block: 0,
                nbuf: 0,
            }
        }

        pub fn push(&mut self, q: u8) {
            if self.block < self.blocks.len() {
                let block = &mut self.blocks[self.block];
                let byte = self.nbuf % TURBO_BLOCK_SIZE;
                let shift = self.nbuf / TURBO_BLOCK_SIZE * B;
                block[byte] |= q << shift;
                self.nbuf += 1;
                if self.nbuf == (TURBO_BLOCK_SIZE * 8) / B {
                    self.block += 1;
                    self.nbuf = 0;
                }
            } else {
                let byte = self.nbuf % self.tail.len();
                let shift = self.nbuf / self.tail.len() * B;
                self.tail[byte] |= q << shift;
                self.nbuf += 1;
                if self.nbuf == self.tail.len() * 8 / B {
                    self.block += 1;
                    self.nbuf = 0;
                }
            }
        }
    }

    pub struct TurboUnpacker<'a, const B: usize> {
        blocks: &'a [[u8; TURBO_BLOCK_SIZE]],
        tail: &'a [u8],
        block: usize,
        pos: usize,
    }

    impl<'a, const B: usize> TurboUnpacker<'a, B> {
        pub fn new(vector_bytes: &'a [u8]) -> Self {
            let (blocks, tail) = vector_bytes.as_chunks::<TURBO_BLOCK_SIZE>();
            Self {
                blocks,
                tail,
                block: 0,
                pos: 0,
            }
        }
    }

    impl<'a, const B: usize> Iterator for TurboUnpacker<'a, B> {
        type Item = u8;

        fn next(&mut self) -> Option<Self::Item> {
            if self.block < self.blocks.len() {
                let block = &self.blocks[self.block];
                let byte = self.pos % TURBO_BLOCK_SIZE;
                let shift = self.pos / TURBO_BLOCK_SIZE * B;
                let v = (block[byte] >> shift) & u8::MAX >> (8 - B);
                self.pos += 1;
                if self.pos == (TURBO_BLOCK_SIZE * 8) / B {
                    self.block += 1;
                    self.pos = 0;
                }
                Some(v)
            } else if !self.tail.is_empty() && self.block == self.blocks.len() {
                let byte = self.pos % self.tail.len();
                let shift = self.pos / self.tail.len() * B;
                let v = (self.tail[byte] >> shift) & u8::MAX >> (8 - B);
                self.pos += 1;
                if self.pos == self.tail.len() * 8 / B {
                    self.block += 1;
                    self.pos = 0;
                }
                Some(v)
            } else {
                None
            }
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let total = (self.blocks.len() * TURBO_BLOCK_SIZE * 8) / B + self.tail.len() * 8 / B;
            let next = self.block * TURBO_BLOCK_SIZE + self.pos;
            (total - next, Some(total - next))
        }
    }

    impl<'a, const B: usize> FusedIterator for TurboUnpacker<'a, B> {}

    impl<'a, const B: usize> ExactSizeIterator for TurboUnpacker<'a, B> {}

    /// Take a 4 bit encoded input and split it into 4 bitplanes.
    ///
    /// The resulting bitplanes are interleaved at 16 bytes chunks until the tail when they are
    /// interleaved in the turbo packing format.
    pub fn bitplane_split4(vector: &[u8]) -> Vec<u8> {
        // 64 bytes contains 128 dims, which is enough to populate 4 128 bit bitplanes.
        let head_len = vector.len() & !63;
        let tail_dim = (vector.len() & 63) * 2;
        let tail_len = tail_dim.div_ceil(8) * 4;
        let len = head_len + tail_len;
        let mut out = vec![0u8; len];
        let (head, tail) = vector.as_chunks::<64>();
        let (ohead, otail) = out.split_at_mut(head.len() * 64);
        let ohead = ohead.as_chunks_mut::<64>().0;
        let nibble_mask = u128::from_ne_bytes([0xf; 16]);
        let bit_mask = u128::from_ne_bytes([1; 16]);
        for (c, o) in head.iter().zip(ohead.iter_mut()) {
            let mut b0 = 0u128;
            let mut b1 = 0u128;
            let mut b2 = 0u128;
            let mut b3 = 0u128;
            for (i, b) in c.as_chunks::<16>().0.iter().enumerate() {
                let b = u128::from_le_bytes(*b);
                let lo = b & nibble_mask;
                let hi = (b >> 4) & nibble_mask;

                b0 |= (lo & bit_mask) << (i * 2);
                b0 |= (hi & bit_mask) << (i * 2 + 1);
                b1 |= ((lo >> 1) & bit_mask) << (i * 2);
                b1 |= ((hi >> 1) & bit_mask) << (i * 2 + 1);
                b2 |= ((lo >> 2) & bit_mask) << (i * 2);
                b2 |= ((hi >> 2) & bit_mask) << (i * 2 + 1);
                b3 |= ((lo >> 3) & bit_mask) << (i * 2);
                b3 |= ((hi >> 3) & bit_mask) << (i * 2 + 1);
            }

            let planes = o.as_chunks_mut::<16>().0;
            planes[0] = b0.to_le_bytes();
            planes[1] = b1.to_le_bytes();
            planes[2] = b2.to_le_bytes();
            planes[3] = b3.to_le_bytes();
        }

        if !tail.is_empty() {
            assert!(otail.len().is_multiple_of(4));
            let mut oiter = otail.chunks_mut(otail.len() / 4);
            let mut b0 = TurboPacker::<1>::new(oiter.next().unwrap());
            let mut b1 = TurboPacker::<1>::new(oiter.next().unwrap());
            let mut b2 = TurboPacker::<1>::new(oiter.next().unwrap());
            let mut b3 = TurboPacker::<1>::new(oiter.next().unwrap());
            for d in TurboUnpacker::<4>::new(tail) {
                b0.push(d & 1);
                b1.push((d >> 1) & 1);
                b2.push((d >> 2) & 1);
                b3.push((d >> 3) & 1);
            }
        }

        out
    }

    /// Return the number of dimensions that can be packed into a single block.
    ///
    /// So long as `bits` is a power of 2 the returned value will _also_ be a power of 2.
    /// This is useful for splitting between the head and tail during vector coding tasks.
    pub const fn block_dim(bits: usize) -> usize {
        (TURBO_BLOCK_SIZE * 8) / bits
    }
}
