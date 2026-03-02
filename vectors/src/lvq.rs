//! Locally adaptive Vector Quantization (LVQ): https://arxiv.org/pdf/2304.04759
//!
//! This supports both primary quantization and two level quantization, which allows splitting the
//! representation for initial scoring and re-ranking.
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

use thread_local::ThreadLocal;

use crate::{
    F32VectorCoder, QueryVectorDistance, VectorDistance, VectorSimilarity,
    dot_unnormalized_to_distance,
};

#[derive(Debug, Copy, Clone)]
enum InstructionSet {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl Default for InstructionSet {
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    fn default() -> Self {
        InstructionSet::Scalar
    }

    #[cfg(target_arch = "aarch64")]
    fn default() -> Self {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            InstructionSet::Neon
        } else {
            InstructionSet::Scalar
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn default() -> Self {
        use std::arch::is_x86_feature_detected as feature;
        if feature!("avx2")
            && feature!("avx512f")
            && feature!("avx512bw")
            && feature!("avx512vl")
            && feature!("avx512vpopcntdq")
            && feature!("avx512vnni")
        {
            InstructionSet::Avx512
        } else {
            InstructionSet::Scalar
        }
    }
}

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

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct VectorStats {
    min: f32,
    max: f32,
    mean: f32,
    std_dev: f32,
    l2_norm_sq: f32,
}

impl VectorStats {
    #[allow(dead_code)]
    fn from_scalar(value: &[f32]) -> Self {
        if value.is_empty() {
            return VectorStats {
                l2_norm_sq: 1.0,
                ..Default::default()
            };
        }

        scalar::compute_vector_stats(value)
    }
}

impl From<&[f32]> for VectorStats {
    fn from(value: &[f32]) -> Self {
        if value.is_empty() {
            return VectorStats {
                l2_norm_sq: 1.0,
                ..Default::default()
            };
        }

        match InstructionSet::default() {
            InstructionSet::Scalar => scalar::compute_vector_stats(value),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::compute_vector_stats(value),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe { x86_64::compute_vector_stats_avx512(value) },
        }
    }
}

fn optimize_interval(vector: &[f32], stats: &VectorStats, bits: usize) -> (f32, f32) {
    // There are several spots in the optimization routine where we may divide by the input range
    // and if that range is zero then it produces NaNs.
    if stats.min == stats.max {
        return (stats.min, stats.min + f32::MIN_POSITIVE);
    }
    match InstructionSet::default() {
        InstructionSet::Scalar => scalar::optimize_interval_scalar(vector, stats, bits),
        #[cfg(target_arch = "aarch64")]
        InstructionSet::Neon => aarch64::optimize_interval_neon(vector, stats, bits),
        #[cfg(target_arch = "x86_64")]
        InstructionSet::Avx512 => unsafe { x86_64::optimize_interval_avx512(vector, stats, bits) },
    }
}

fn maybe_center_vector_to<'a>(
    vector: &'a [f32],
    center: Option<(&[f32], &'a mut [f32])>,
    similarity: VectorSimilarity,
) -> (&'a [f32], f32) {
    if let Some((center, out)) = center {
        center_vector(vector, center, out);
        let center_dot = match similarity {
            VectorSimilarity::Dot | VectorSimilarity::Cosine => {
                vector.iter().zip(center.iter()).map(|(&v, &c)| v * c).sum()
            }
            _ => 0.0,
        };
        (out, center_dot)
    } else {
        (vector, 0.0)
    }
}

fn center_vector(vector: &[f32], center: &[f32], out: &mut [f32]) {
    for ((v, c), o) in vector.iter().zip(center.iter()).zip(out.iter_mut()) {
        *o = *v - *c;
    }
}

fn uncenter_vector(center: &[f32], vector: &mut [f32]) {
    for (c, v) in center.iter().zip(vector.iter_mut()) {
        *v += *c;
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
    l2_norm: f32,
    /// Lower interval bound used for quantization, no smaller than the minimum component value.
    /// This is used to correct the uint dot product to an f32 dot product.
    lower: f32,
    /// Upper interval bound used for quantization, no larger than the maximum component value.
    /// This is used to correct the uint dot product to an f32 dot product.
    upper: f32,
    /// The L2 norm of the residual vector (v - dequantize(quantize(v))).
    /// This term can be used to compute a statistical bound on the estimated distance.
    residual_error_term: f32,
    /// Sum of all the quantized components of the vector. This is used to correct the uint dot
    /// product to an f32 dot product.
    component_sum: u32,
}

impl PrimaryVectorHeader {
    /// Encoded buffer size.
    const LEN: usize = std::mem::size_of::<Self>();

    #[inline]
    fn split_output_buf(buf: &mut [u8]) -> Option<(&mut [u8], &mut [u8])> {
        buf.split_at_mut_checked(Self::LEN)
    }

    #[inline]
    fn serialize(&self, header_bytes: &mut [u8]) {
        let header = header_bytes.as_chunks_mut::<4>().0;
        header[0] = self.l2_norm.to_le_bytes();
        header[1] = self.lower.to_le_bytes();
        header[2] = self.upper.to_le_bytes();
        header[3] = self.residual_error_term.to_le_bytes();
        header[4] = self.component_sum.to_le_bytes();
    }

    #[inline]
    fn deserialize(raw: &[u8]) -> Option<(Self, &[u8])> {
        let (header_bytes, vector_bytes) = raw.split_at_checked(Self::LEN)?;
        let header_entries = header_bytes.as_chunks::<4>().0;
        Some((
            Self {
                l2_norm: f32::from_le_bytes(header_entries[0]),
                lower: f32::from_le_bytes(header_entries[1]),
                upper: f32::from_le_bytes(header_entries[2]),
                residual_error_term: f32::from_le_bytes(header_entries[3]),
                component_sum: u32::from_le_bytes(header_entries[4]),
            },
            vector_bytes,
        ))
    }
}

impl From<VectorStats> for PrimaryVectorHeader {
    fn from(value: VectorStats) -> Self {
        Self {
            l2_norm: value.l2_norm_sq.sqrt(),
            lower: value.min,
            upper: value.max,
            residual_error_term: 0.0,
            component_sum: 0,
        }
    }
}

/// Header for an LVQ residual vector.
///
/// This contains additional information to decode the residual vector, but the primary vector and
/// header are required to interpret the data.
/// vector.
#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(C)]
struct ResidualVectorHeader {
    magnitude: f32,
    component_sum: u32,
}

impl ResidualVectorHeader {
    /// Encoded buffer size.
    const LEN: usize = std::mem::size_of::<Self>();

    #[inline]
    fn split_output_buf(buf: &mut [u8]) -> Option<(&mut [u8], &mut [u8])> {
        buf.split_at_mut_checked(Self::LEN)
    }

    #[inline]
    fn serialize(&self, header_bytes: &mut [u8]) {
        let header = header_bytes.as_chunks_mut::<4>().0;
        header[0] = self.magnitude.to_le_bytes();
        header[1] = self.component_sum.to_le_bytes();
    }

    #[inline]
    fn deserialize(raw: &[u8]) -> Option<(Self, &[u8])> {
        let (header_bytes, vector_bytes) = raw.split_at_checked(Self::LEN)?;
        let header_entries = header_bytes.as_chunks::<4>().0;
        Some((
            Self {
                magnitude: f32::from_le_bytes(header_entries[0]),
                component_sum: u32::from_le_bytes(header_entries[1]),
            },
            vector_bytes,
        ))
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct VectorDecodeTerms {
    lower: f32,
    delta: f32,
    component_sum: u32,
}

impl VectorDecodeTerms {
    fn from_primary<const B: usize>(header: PrimaryVectorHeader) -> Self {
        Self {
            lower: header.lower,
            delta: (header.upper - header.lower) / ((1 << B) - 1) as f32,
            component_sum: header.component_sum,
        }
    }

    fn from_residual(header: ResidualVectorHeader) -> Self {
        Self {
            lower: -header.magnitude / 2.0,
            delta: header.magnitude / RESIDUAL_MAX,
            component_sum: header.component_sum,
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

/// Compute an unnormalized dot product beween two vectors encoded with the same bits/dimension..
fn dot_unnormalized_uint_symmetric<const B: usize>(
    inst: InstructionSet,
    dim: usize,
    a: &EncodedVector<'_>,
    b: &EncodedVector<'_>,
) -> f32 {
    let dot = match inst {
        InstructionSet::Scalar => scalar::dot_u8::<B>(a.data, b.data),
        #[cfg(target_arch = "aarch64")]
        InstructionSet::Neon => aarch64::dot_u8::<B>(a.data, b.data),
        #[cfg(target_arch = "x86_64")]
        InstructionSet::Avx512 => unsafe { x86_64::dot_u8_avx512::<B>(a.data, b.data) },
    };
    correct_dot_uint(dot, dim, &a.terms, &b.terms)
}

/// Correct the dot product of two integer vectors using the stored vector terms.
fn correct_dot_uint(dot: u32, dim: usize, a: &VectorDecodeTerms, b: &VectorDecodeTerms) -> f32 {
    // Note that any dot value larger than (2 << 24) will be rounded when converted to f32 which can
    // cause vector comparisons a <-> b and b <-> a to return slightly different results. To prevent
    // this convert dot to f64 before including it in the correction.
    (dot as f64 * (a.delta * b.delta) as f64
        + (a.component_sum as f32 * a.delta * b.lower
            + b.component_sum as f32 * b.delta * a.lower
            + a.lower * b.lower * dim as f32) as f64) as f32
}

/// The four components of a residual dot product.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
#[repr(C)]
struct ResidualDotComponents {
    ap_dot_bp: u32,
    ap_dot_br: u32,
    ar_dot_bp: u32,
    ar_dot_br: u32,
}

impl ResidualDotComponents {
    fn compute_dot(
        &self,
        dim: usize,
        a: (&VectorDecodeTerms, &VectorDecodeTerms),
        b: (&VectorDecodeTerms, &VectorDecodeTerms),
    ) -> f32 {
        correct_dot_uint(self.ap_dot_bp, dim, a.0, b.0)
            + correct_dot_uint(self.ap_dot_br, dim, a.0, b.1)
            + correct_dot_uint(self.ar_dot_bp, dim, a.1, b.0)
            + correct_dot_uint(self.ar_dot_br, dim, a.1, b.1)
    }
}

impl Add for ResidualDotComponents {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl AddAssign for ResidualDotComponents {
    fn add_assign(&mut self, rhs: Self) {
        self.ap_dot_bp += rhs.ap_dot_bp;
        self.ap_dot_br += rhs.ap_dot_br;
        self.ar_dot_bp += rhs.ar_dot_bp;
        self.ar_dot_br += rhs.ar_dot_br;
    }
}

/// The turbo coder requires that all vector data be packed into 16-byte blocks.
const TURBO_BLOCK_SIZE: usize = 16;

struct TurboPrimaryVector<'a, const B: usize> {
    rep: EncodedVector<'a>,
    l2_norm: f32,
    residual_error_term: f32,
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
            residual_error_term: header.residual_error_term,
        })
    }

    fn dim(&self) -> usize {
        (self.rep.data.len() * 8) / B
    }

    fn l2_norm(&self) -> f64 {
        self.l2_norm.into()
    }

    fn residual_error_term(&self) -> f64 {
        self.residual_error_term.into()
    }

    fn split_tail(&self, dim: usize) -> (usize, Self, Self) {
        let tail_dim = dim & !(packing::block_dim(B) - 1);
        let (headv, tailv) = self.rep.split_at(packing::byte_len(tail_dim, B));
        (
            tail_dim,
            Self {
                rep: headv,
                l2_norm: self.l2_norm,
                residual_error_term: self.residual_error_term,
            },
            Self {
                rep: tailv,
                l2_norm: self.l2_norm,
                residual_error_term: self.residual_error_term,
            },
        )
    }
}

#[derive(Debug)]
struct CenteringState {
    center: Vec<f32>,
    scratch: ThreadLocal<RefCell<Vec<f32>>>,
}

impl CenteringState {
    fn new(center: Vec<f32>) -> Self {
        Self {
            center,
            scratch: ThreadLocal::new(),
        }
    }
}

#[derive(Debug)]
pub struct TurboPrimaryCoder<const B: usize> {
    similarity: VectorSimilarity,
    centering_state: Option<CenteringState>,
    inst: InstructionSet,
}

impl<const B: usize> TurboPrimaryCoder<B> {
    const B_CHECK: () = { check_primary_bits(B) };

    pub fn new(similarity: VectorSimilarity, center: Option<Vec<f32>>) -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::B_CHECK;
        Self {
            similarity,
            centering_state: center.map(CenteringState::new),
            inst: InstructionSet::default(),
        }
    }

    #[allow(unused)]
    pub fn scalar(similarity: VectorSimilarity, center: Option<Vec<f32>>) -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::B_CHECK;
        Self {
            similarity,
            centering_state: center.map(CenteringState::new),
            inst: InstructionSet::Scalar,
        }
    }

    fn encode_parts(
        inst: InstructionSet,
        similarity: VectorSimilarity,
        vector: &[f32],
        center: Option<&[f32]>,
    ) -> (PrimaryVectorHeader, Vec<u8>) {
        let mut scratch = center.map(|c| vec![0.0f32; c.len()]);
        let mut out = vec![0u8; packing::byte_len(vector.len(), B)];
        let header = Self::encode_parts_to(
            inst,
            similarity,
            vector,
            center.zip(scratch.as_deref_mut()),
            &mut out,
        );
        (header, out)
    }

    fn encode_parts_to(
        inst: InstructionSet,
        similarity: VectorSimilarity,
        vector: &[f32],
        center: Option<(&[f32], &mut [f32])>,
        out: &mut [u8],
    ) -> PrimaryVectorHeader {
        let (vector, _center_dot) = maybe_center_vector_to(vector, center, similarity);

        let stats = VectorStats::from(vector);
        let mut header = PrimaryVectorHeader::from(stats);
        (header.lower, header.upper) = optimize_interval(vector, &stats, B);

        let terms = VectorEncodeTerms::from_primary::<B>(&header);
        let residual_error_sq;
        (header.component_sum, residual_error_sq) = match inst {
            InstructionSet::Scalar => scalar::primary_quantize_and_pack::<B>(vector, terms, out),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::primary_quantize_and_pack::<B>(vector, terms, out),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe {
                x86_64::primary_quantize_and_pack_avx512::<B>(vector, terms, out)
            },
        };
        header.residual_error_term = residual_error_sq.sqrt();

        header
    }
}

impl<const B: usize> F32VectorCoder for TurboPrimaryCoder<B> {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let (header_bytes, vector_bytes) = PrimaryVectorHeader::split_output_buf(out).unwrap();

        let mut center_scratch = self.centering_state.as_ref().map(|c| {
            let mut scratch = c.scratch.get_or_default().borrow_mut();
            scratch.resize(vector.len(), 0.0);
            scratch
        });
        let center_state = self
            .centering_state
            .as_ref()
            .map(|c| c.center.as_slice())
            .zip(center_scratch.as_mut().map(|s| s.as_mut_slice()));
        let header = Self::encode_parts_to(
            self.inst,
            self.similarity,
            vector,
            center_state,
            vector_bytes,
        );
        header.serialize(header_bytes);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        PrimaryVectorHeader::LEN + packing::byte_len(dimensions, B)
    }

    fn decode_to(&self, vector: &[u8], out: &mut [f32]) {
        let vector = TurboPrimaryVector::<B>::new(vector).expect("valid primary vector");
        match self.inst {
            InstructionSet::Scalar => scalar::primary_decode::<B>(vector, out),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::primary_decode::<B>(vector, out),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe { x86_64::primary_decode_avx512::<B>(vector, out) },
        };
        if let Some(c) = self.centering_state.as_ref() {
            uncenter_vector(&c.center, out);
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        let vector_bytes = byte_len - PrimaryVectorHeader::LEN;
        (vector_bytes * 8) / B
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TurboPrimaryDistance<const B: usize>(VectorSimilarity, InstructionSet);

impl<const B: usize> TurboPrimaryDistance<B> {
    pub fn new(similarity: VectorSimilarity) -> Self {
        Self(similarity, InstructionSet::default())
    }

    #[inline(always)]
    fn distance_internal(&self, query: &TurboPrimaryVector<B>, doc: &[u8]) -> f64 {
        let doc = TurboPrimaryVector::<B>::new(doc).unwrap();
        dot_unnormalized_to_distance(
            self.0,
            dot_unnormalized_uint_symmetric::<B>(self.1, query.dim(), &query.rep, &doc.rep).into(),
            (query.l2_norm(), doc.l2_norm()),
        )
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
    similarity: VectorSimilarity,

    query: Vec<u8>,
    terms: VectorDecodeTerms,
    l2_norm: f64,

    inst: InstructionSet,
}

impl<const B: usize> TurboPrimaryQueryDistance<B> {
    pub fn new(similarity: VectorSimilarity, query: Cow<'_, [f32]>) -> Self {
        let inst = InstructionSet::default();
        let (header, query) = TurboPrimaryCoder::<PRIMARY_QUERY_BITS>::encode_parts(
            inst,
            similarity,
            query.as_ref(),
            None, /* XXX center */
        );
        let terms = VectorDecodeTerms::from_primary::<PRIMARY_QUERY_BITS>(header);
        let l2_norm = header.l2_norm.into();

        Self {
            similarity,
            query,
            terms,
            l2_norm,
            inst,
        }
    }

    #[inline(always)]
    fn distance_internal(&self, vector: &[u8]) -> f64 {
        let vector = TurboPrimaryVector::<B>::new(vector).expect("valid primary vector");
        let uint8_dot = match self.inst {
            InstructionSet::Scalar => {
                scalar::primary_query8_dot_unnormalized::<B>(&self.query, &vector)
            }
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => {
                aarch64::primary_query8_dot_unnormalized::<B>(&self.query, &vector)
            }
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe {
                x86_64::primary_query8_dot_unnormalized_avx512::<B>(&self.query, &vector)
            },
        };
        let dot = correct_dot_uint(uint8_dot, self.query.len(), &self.terms, &vector.rep.terms);
        dot_unnormalized_to_distance(
            self.similarity,
            dot.into(),
            (self.l2_norm, vector.l2_norm()),
        )
    }
}

impl<const B: usize> QueryVectorDistance for TurboPrimaryQueryDistance<B> {
    fn distance(&self, vector: &[u8]) -> f64 {
        self.distance_internal(vector)
    }

    fn bulk_distance(&self, vectors: &[&[u8]], out: &mut [f64]) {
        for (vector, out) in vectors.iter().zip(out.iter_mut()) {
            *out = self.distance_internal(vector);
        }
    }
}

/// An implementation of distance against one-bit quantized vectors.
///
/// This implementation quantizes the query 1x8 bits and provides a bounded distance function that
/// short circuits residual evaluation if the result would not fall below the maximum distance.
/// This can provide significant savings, particularly when using memory bound indexes like a flat
/// index or partitioned index.
pub struct TurboPrimaryQueryDistance1 {
    similarity: VectorSimilarity,

    primary_query: Vec<u8>,
    primary_terms: VectorDecodeTerms,
    residual_query: Vec<u8>,
    residual_terms: VectorDecodeTerms,
    l2_norm: f64,
    residual_error_term: f64,
    sqrt_dim_inv: f64,

    inst: InstructionSet,
}

impl TurboPrimaryQueryDistance1 {
    pub fn new(similarity: VectorSimilarity, query: Cow<'_, [f32]>) -> Self {
        let inst = InstructionSet::default();
        let (primary_header, primary_query, residual_header, residual_query) =
            TurboResidualCoder::<1>::encode_parts(inst, similarity, query.as_ref(), None);
        Self {
            similarity,
            primary_query,
            primary_terms: VectorDecodeTerms::from_primary::<1>(primary_header),
            residual_query,
            residual_terms: VectorDecodeTerms::from_residual(residual_header),
            l2_norm: primary_header.l2_norm.into(),
            residual_error_term: primary_header.residual_error_term.into(),
            sqrt_dim_inv: 1.0 / (query.len() as f64).sqrt(),
            inst,
        }
    }
}

impl QueryVectorDistance for TurboPrimaryQueryDistance1 {
    fn distance(&self, vector: &[u8]) -> f64 {
        self.distance_with_bound(vector, f64::INFINITY).unwrap()
    }

    fn distance_with_bound(&self, vector: &[u8], max_distance: f64) -> Option<f64> {
        let vector = TurboPrimaryVector::<1>::new(vector).expect("valid primary vector");
        let uint8_dot_primary = match self.inst {
            InstructionSet::Scalar => scalar::dot_u8::<1>(&self.primary_query, vector.rep.data),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::dot_u8::<1>(&self.primary_query, vector.rep.data),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe {
                x86_64::dot_u8_avx512::<1>(&self.primary_query, vector.rep.data)
            },
        };
        let dot_primary = correct_dot_uint(
            uint8_dot_primary,
            self.residual_query.len(),
            &self.primary_terms,
            &vector.rep.terms,
        );
        let distance_primary = dot_unnormalized_to_distance(
            self.similarity,
            dot_primary.into(),
            (self.l2_norm, vector.l2_norm()),
        );
        let error = (self.residual_error_term + vector.residual_error_term()) * self.sqrt_dim_inv;
        let mult = match self.similarity {
            VectorSimilarity::Dot | VectorSimilarity::Cosine => 0.5,
            VectorSimilarity::Euclidean => 2.0,
        };
        let estimated_error = 1.96 * mult * error;
        if distance_primary - estimated_error > max_distance {
            return None;
        }

        let uint8_dot_residual = match self.inst {
            InstructionSet::Scalar => {
                scalar::primary_query8_dot_unnormalized::<1>(&self.residual_query, &vector)
            }
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => {
                aarch64::primary_query8_dot_unnormalized::<1>(&self.residual_query, &vector)
            }
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe {
                x86_64::primary_query8_dot_unnormalized_avx512::<1>(&self.residual_query, &vector)
            },
        };
        let dot_residual = correct_dot_uint(
            uint8_dot_residual,
            self.residual_query.len(),
            &self.residual_terms,
            &vector.rep.terms,
        );
        Some(dot_unnormalized_to_distance(
            self.similarity,
            (dot_primary + dot_residual).into(),
            (self.l2_norm, vector.l2_norm()),
        ))
    }
}

const RESIDUAL_BITS: usize = 8;
const RESIDUAL_MAX: f32 = ((1 << RESIDUAL_BITS) - 1) as f32;

struct TurboResidualVector<'a, const B: usize> {
    primary: EncodedVector<'a>,
    residual: EncodedVector<'a>,
    l2_norm: f32,
}

impl<'a, const B: usize> TurboResidualVector<'a, B> {
    const B_CHECK: () = { check_primary_bits(B) };

    fn new(data: &'a [u8]) -> Option<Self> {
        #[allow(clippy::let_unit_value)]
        let _ = Self::B_CHECK;

        let (primary_header, vector_bytes) = PrimaryVectorHeader::deserialize(data)?;
        let (residual_header, vector_bytes) = ResidualVectorHeader::deserialize(vector_bytes)?;
        let (primary_vector, residual_vector) = vector_bytes.split_at(packing::two_vector_split(
            vector_bytes.len(),
            B,
            RESIDUAL_BITS,
        ));
        Some(Self {
            primary: EncodedVector {
                terms: VectorDecodeTerms::from_primary::<B>(primary_header),
                data: primary_vector,
            },
            residual: EncodedVector {
                terms: VectorDecodeTerms::from_residual(residual_header),
                data: residual_vector,
            },
            l2_norm: primary_header.l2_norm,
        })
    }

    fn dim(&self) -> usize {
        self.residual.data.len()
    }

    fn l2_norm(&self) -> f64 {
        self.l2_norm.into()
    }

    fn split_tail(&self, dim: usize) -> (usize, Self, Self) {
        let tail_dim = dim & !(packing::block_dim(B) - 1);
        let (primary_headv, primary_tailv) = self.primary.split_at(packing::byte_len(tail_dim, B));
        let (residual_headv, residual_tailv) = self.residual.split_at(tail_dim);
        (
            tail_dim,
            Self {
                primary: primary_headv,
                residual: residual_headv,
                l2_norm: self.l2_norm,
            },
            Self {
                primary: primary_tailv,
                residual: residual_tailv,
                l2_norm: self.l2_norm,
            },
        )
    }

    /// Splits (primary, residual) vector bytes into head and tail pairs.
    /// The head contains full 16 byte blocks, while the tail contains the remaining bytes.
    /// Returns the dimension chosen for the split in addition to the head and tail pairs.
    fn split_vector_tail<'b>(
        vector: ResidualVectorBytes<'b>,
    ) -> (usize, ResidualVectorBytes<'b>, ResidualVectorBytes<'b>) {
        let tail_split = vector.1.len() & !(packing::block_dim(B) - 1);
        let primary = vector.0.split_at(packing::byte_len(tail_split, B));
        let residual = vector.1.split_at(tail_split);
        (tail_split, (primary.0, residual.0), (primary.1, residual.1))
    }
}

type ResidualVectorBytes<'b> = (&'b [u8], &'b [u8]);

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

    fn from_residual(magnitude: f32) -> Self {
        Self {
            lower: -magnitude / 2.0,
            upper: magnitude / 2.0,
            delta_inv: RESIDUAL_MAX / magnitude,
            delta: 0.0,
        }
    }
}

#[derive(Debug)]
pub struct TurboResidualCoder<const B: usize> {
    similarity: VectorSimilarity,
    centering_state: Option<CenteringState>,
    inst: InstructionSet,
}

impl<const B: usize> TurboResidualCoder<B> {
    const B_CHECK: () = { check_primary_bits(B) };

    pub fn new(similarity: VectorSimilarity, center: Option<Vec<f32>>) -> Self {
        Self {
            similarity,
            centering_state: center.map(CenteringState::new),
            inst: InstructionSet::default(),
        }
    }

    #[allow(unused)]
    pub fn scalar(similarity: VectorSimilarity, center: Option<Vec<f32>>) -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::B_CHECK;
        Self {
            similarity,
            centering_state: center.map(CenteringState::new),
            inst: InstructionSet::Scalar,
        }
    }

    fn encode_parts(
        inst: InstructionSet,
        similarity: VectorSimilarity,
        vector: &[f32],
        center: Option<&[f32]>,
    ) -> (PrimaryVectorHeader, Vec<u8>, ResidualVectorHeader, Vec<u8>) {
        let mut scratch = center.map(|c| vec![0.0f32; c.len()]);
        let mut primary = vec![0u8; packing::byte_len(vector.len(), B)];
        let mut residual = vec![0u8; vector.len()];
        let (primary_header, residual_header) = Self::encode_parts_to(
            inst,
            similarity,
            vector,
            center.zip(scratch.as_deref_mut()),
            &mut primary,
            &mut residual,
        );
        (primary_header, primary, residual_header, residual)
    }

    fn encode_parts_to(
        inst: InstructionSet,
        similarity: VectorSimilarity,
        vector: &[f32],
        center: Option<(&[f32], &mut [f32])>,
        primary: &mut [u8],
        residual: &mut [u8],
    ) -> (PrimaryVectorHeader, ResidualVectorHeader) {
        // XXX produce correction term for angular distance.
        let (vector, _center_dot) = maybe_center_vector_to(vector, center, similarity);

        let stats = VectorStats::from(vector);
        let mut primary_header = PrimaryVectorHeader::from(stats);
        // NB: this interval optimization reduces loss for the primary vector, but this loss
        // reduction can make the residual vector more lossy if we derive the delta from the primary
        // interval as described in the LVQ paper. Compute and store a residual delta that is large
        // enough to encode both the min and max value in the vector.
        // TODO: investigate interval optimization on the derived residual vector.
        let interval = optimize_interval(vector, &stats, B);
        // For the residual interval choose the maximum based on primary delta, or the min/max
        // values we may need to encode based on the gap between the initial and optimized interval.
        let residual_magnitude = [
            (interval.1 - interval.0) / ((1 << B) - 1) as f32,
            (primary_header.lower.abs() - interval.0.abs()) * 2.0,
            (primary_header.upper.abs() - interval.1.abs()) * 2.0,
        ]
        .into_iter()
        .max_by(f32::total_cmp)
        .expect("3 values input");
        (primary_header.lower, primary_header.upper) = interval;
        let mut residual_header = ResidualVectorHeader {
            magnitude: residual_magnitude,
            component_sum: 0,
        };

        let primary_terms = VectorEncodeTerms::from_primary::<B>(&primary_header);
        let residual_terms = VectorEncodeTerms::from_residual(residual_magnitude);
        let residual_error_sq;
        (
            primary_header.component_sum,
            residual_header.component_sum,
            residual_error_sq,
        ) = match inst {
            InstructionSet::Scalar => scalar::residual_quantize_and_pack::<B>(
                vector,
                primary_terms,
                residual_terms,
                primary,
                residual,
            ),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::residual_quantize_and_pack::<B>(
                vector,
                primary_terms,
                residual_terms,
                primary,
                residual,
            ),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe {
                x86_64::residual_quantize_and_pack_avx512::<B>(
                    vector,
                    primary_terms,
                    residual_terms,
                    primary,
                    residual,
                )
            },
        };
        primary_header.residual_error_term = residual_error_sq.sqrt();

        (primary_header, residual_header)
    }
}

impl<const B: usize> F32VectorCoder for TurboResidualCoder<B> {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let (primary_header_bytes, vector_bytes) =
            PrimaryVectorHeader::split_output_buf(out).unwrap();
        let (residual_header_bytes, vector_bytes) =
            ResidualVectorHeader::split_output_buf(vector_bytes).unwrap();
        let split = packing::two_vector_split(vector_bytes.len(), B, RESIDUAL_BITS);
        let (primary, residual) = vector_bytes.split_at_mut(split);

        let mut center_scratch = self.centering_state.as_ref().map(|c| {
            let mut scratch = c.scratch.get_or_default().borrow_mut();
            scratch.resize(vector.len(), 0.0);
            scratch
        });
        let center_state = self
            .centering_state
            .as_ref()
            .map(|c| c.center.as_slice())
            .zip(center_scratch.as_mut().map(|s| s.as_mut_slice()));
        let (primary_header, residual_header) = Self::encode_parts_to(
            self.inst,
            self.similarity,
            vector,
            center_state,
            primary,
            residual,
        );
        primary_header.serialize(primary_header_bytes);
        residual_header.serialize(residual_header_bytes);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        PrimaryVectorHeader::LEN
            + ResidualVectorHeader::LEN
            + packing::byte_len(dimensions, B)
            + dimensions
    }

    fn decode_to(&self, vector: &[u8], out: &mut [f32]) {
        let vector = TurboResidualVector::<B>::new(vector).expect("valid vector");
        match self.inst {
            InstructionSet::Scalar => scalar::residual_decode::<B>(&vector, out),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::residual_decode::<B>(&vector, out),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe { x86_64::residual_decode_avx512::<B>(&vector, out) },
        }
        if let Some(c) = self.centering_state.as_ref() {
            uncenter_vector(&c.center, out);
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        let len_no_corrective_terms =
            byte_len - PrimaryVectorHeader::LEN - ResidualVectorHeader::LEN;
        let split = packing::two_vector_split(len_no_corrective_terms, B, RESIDUAL_BITS);
        // Residual vector always uses a byte per dimension.
        len_no_corrective_terms - split
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TurboResidualDistance<const B: usize>(VectorSimilarity, InstructionSet);

impl<const B: usize> TurboResidualDistance<B> {
    pub fn new(similarity: VectorSimilarity) -> Self {
        Self(similarity, InstructionSet::default())
    }
}

impl<const B: usize> VectorDistance for TurboResidualDistance<B> {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = TurboResidualVector::<B>::new(query).unwrap();
        let doc = TurboResidualVector::<B>::new(doc).unwrap();

        let component_dot = match self.1 {
            InstructionSet::Scalar => scalar::residual_dot_unnormalized::<B>(
                (query.primary.data, query.residual.data),
                (doc.primary.data, doc.residual.data),
            ),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::residual_dot_unnormalized::<B>(
                (query.primary.data, query.residual.data),
                (doc.primary.data, doc.residual.data),
            ),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe {
                x86_64::residual_dot_unnormalized_avx512::<B>(
                    (query.primary.data, query.residual.data),
                    (doc.primary.data, doc.residual.data),
                )
            },
        };
        let dot = component_dot.compute_dot(
            query.dim(),
            (&query.primary.terms, &query.residual.terms),
            (&doc.primary.terms, &doc.residual.terms),
        );
        dot_unnormalized_to_distance(self.0, dot.into(), (query.l2_norm(), doc.l2_norm()))
    }
}

pub struct TurboResidualQueryDistance<const B: usize> {
    similarity: VectorSimilarity,

    primary_vector: Vec<u8>,
    primary_terms: VectorDecodeTerms,
    residual_vector: Vec<u8>,
    residual_terms: VectorDecodeTerms,
    l2_norm: f64,

    inst: InstructionSet,
}

impl<const B: usize> TurboResidualQueryDistance<B> {
    pub fn new(similarity: VectorSimilarity, query: Cow<'_, [f32]>) -> Self {
        let inst = InstructionSet::default();
        let (primary_header, primary_vector, residual_header, residual_vector) =
            TurboResidualCoder::<B>::encode_parts(
                inst,
                similarity,
                query.as_ref(),
                None, /* XXX center */
            );
        let primary_terms = VectorDecodeTerms::from_primary::<B>(primary_header);
        let residual_terms = VectorDecodeTerms::from_residual(residual_header);
        let l2_norm = primary_header.l2_norm.into();
        Self {
            similarity,
            primary_vector,
            primary_terms,
            residual_vector,
            residual_terms,
            l2_norm,
            inst: InstructionSet::default(),
        }
    }
}

impl<const B: usize> QueryVectorDistance for TurboResidualQueryDistance<B> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = TurboResidualVector::<B>::new(vector).expect("valid vector");
        let component_dot = match self.inst {
            InstructionSet::Scalar => scalar::residual_dot_unnormalized::<B>(
                (&self.primary_vector, &self.residual_vector),
                (vector.primary.data, vector.residual.data),
            ),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::residual_dot_unnormalized::<B>(
                (&self.primary_vector, &self.residual_vector),
                (vector.primary.data, vector.residual.data),
            ),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe {
                x86_64::residual_dot_unnormalized_avx512::<B>(
                    (&self.primary_vector, &self.residual_vector),
                    (vector.primary.data, vector.residual.data),
                )
            },
        };
        let dot = component_dot.compute_dot(
            vector.dim(),
            (&self.primary_terms, &self.residual_terms),
            (&vector.primary.terms, &vector.residual.terms),
        );
        dot_unnormalized_to_distance(
            self.similarity,
            dot.into(),
            (self.l2_norm, vector.l2_norm.into()),
        )
    }
}

mod packing {
    use std::iter::FusedIterator;

    use crate::lvq::TURBO_BLOCK_SIZE;

    /// The number of bytes required to pack `dimensions` with `bits` per entry.
    pub const fn byte_len(dimensions: usize, bits: usize) -> usize {
        (dimensions * bits).div_ceil(8)
    }

    /// Pick where to split between primary and residual vector representation based on the number
    /// of payload bits, and bits per section.
    pub const fn two_vector_split(vector_bytes: usize, bits1: usize, bits2: usize) -> usize {
        if bits1 < bits2 {
            (vector_bytes * bits1).div_ceil(bits1 + bits2)
        } else if bits1 == bits2 {
            vector_bytes / 2
        } else {
            (vector_bytes * bits2).div_ceil(bits1 + bits2)
        }
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

    /// Return the number of dimensions that can be packed into a single block.
    ///
    /// So long as `bits` is a power of 2 the returned value will _also_ be a power of 2.
    /// This is useful for splitting between the head and tail during vector coding tasks.
    pub const fn block_dim(bits: usize) -> usize {
        (TURBO_BLOCK_SIZE * 8) / bits
    }
}
