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
#[cfg(target_arch = "x86_64")]
mod x86_64;

use half::f16;
use std::{
    borrow::Cow,
    ops::{Add, AddAssign},
};

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
    component_sum: f32,
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

/// Header for an LVQ primary vector.
///
/// Along with the bit configuration this carries enough metadata to transform a quantized vector
/// value stream back to an f32 representation or compute angular or l2 distance from another
/// vector.
#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(C)]
struct PrimaryVectorHeader {
    l2_norm: f16,
    lower: f16,
    delta: f16,
    component_sum: f16,
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
        let header = header_bytes.as_chunks_mut::<2>().0;
        header[0] = self.l2_norm.to_le_bytes();
        header[1] = self.lower.to_le_bytes();
        header[2] = self.delta.to_le_bytes();
        header[3] = self.component_sum.to_le_bytes();
    }

    #[inline]
    fn deserialize(raw: &[u8]) -> Option<(Self, &[u8])> {
        let (header_bytes, vector_bytes) = raw.split_at_checked(Self::LEN)?;
        let header_entries = header_bytes.as_chunks::<2>().0;
        Some((
            Self {
                l2_norm: f16::from_le_bytes(header_entries[0]),
                lower: f16::from_le_bytes(header_entries[1]),
                delta: f16::from_le_bytes(header_entries[2]),
                component_sum: f16::from_le_bytes(header_entries[3]),
            },
            vector_bytes,
        ))
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
    magnitude: f16,
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
        let header = header_bytes.as_chunks_mut::<2>().0;
        header[0] = self.magnitude.to_le_bytes();
    }

    #[inline]
    fn deserialize(raw: &[u8]) -> Option<(Self, &[u8])> {
        let (header_bytes, vector_bytes) = raw.split_at_checked(Self::LEN)?;
        let header_entries = header_bytes.as_chunks::<2>().0;
        Some((
            Self {
                magnitude: f16::from_le_bytes(header_entries[0]),
            },
            vector_bytes,
        ))
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

/// The four components of a residual dot product.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
#[repr(C)]
struct ResidualDotComponents {
    ap_dot_bp: u32,
    ap_dot_br: u32,
    ar_dot_bp: u32,
    ar_dot_br: u32,
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

#[derive(Debug, Clone, Copy)]
struct PrimaryVectorTerms {
    l2_norm: f32,
    component_sum: f32,
    lower: f32,
    delta: f32,
}

impl PrimaryVectorTerms {
    fn correct_dot(&self, dot: u32, dim: usize, other: &PrimaryVectorTerms) -> f32 {
        // Note that any dot value larger than (2 << 24) will be rounded when converted to f32 which can
        // cause vector comparisons a <-> b and b <-> a to return slightly different results. To prevent
        // this convert dot to f64 before including it in the correction.
        ((dot as f64 * (self.delta * other.delta) as f64) as f32)
            + self.component_sum * other.lower
            + other.component_sum * self.lower
            - self.lower * other.lower * dim as f32
    }
}

impl From<PrimaryVectorHeader> for PrimaryVectorTerms {
    fn from(header: PrimaryVectorHeader) -> Self {
        Self {
            l2_norm: header.l2_norm.to_f32(),
            component_sum: header.component_sum.to_f32(),
            lower: header.lower.to_f32(),
            delta: header.delta.to_f32(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ResidualVectorTerms {
    primary: PrimaryVectorTerms,
    lower: f32,
    delta: f32,
}

impl ResidualVectorTerms {
    fn correct_dot(&self, dot: &ResidualDotComponents, dim: usize, other: &Self) -> f32 {
        let dot_raw = dot.ap_dot_bp as f64 * (self.primary.delta * other.primary.delta) as f64
            + dot.ap_dot_br as f64 * (self.primary.delta * other.delta) as f64
            + dot.ar_dot_bp as f64 * (self.delta * other.primary.delta) as f64
            + dot.ar_dot_br as f64 * (self.delta * other.delta) as f64;
        let a_lower = self.primary.lower + self.lower;
        let b_lower = other.primary.lower + other.lower;
        dot_raw as f32
            + self.primary.component_sum * b_lower
            + other.primary.component_sum * a_lower
            - dim as f32 * a_lower * b_lower
    }
}

impl From<(PrimaryVectorHeader, ResidualVectorHeader)> for ResidualVectorTerms {
    fn from(header: (PrimaryVectorHeader, ResidualVectorHeader)) -> Self {
        Self {
            primary: PrimaryVectorTerms::from(header.0),
            lower: -header.1.magnitude.to_f32() / 2.0,
            delta: header.1.magnitude.to_f32() / RESIDUAL_MAX,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct TurboPrimaryVector<'a, const B: usize> {
    data: &'a [u8],
    terms: PrimaryVectorTerms,
}

impl<'a, const B: usize> TurboPrimaryVector<'a, B> {
    fn new(data: &'a [u8]) -> Option<Self> {
        let (header, vector_bytes) = PrimaryVectorHeader::deserialize(data)?;
        Some(Self {
            data: vector_bytes,
            terms: PrimaryVectorTerms::from(header),
        })
    }

    fn dim(&self) -> usize {
        (self.data.len() * 8) / B
    }

    fn l2_norm(&self) -> f64 {
        self.terms.l2_norm.into()
    }

    fn split_tail(&self, dim: usize) -> (usize, Self, Self) {
        let tail_dim = dim & !(packing::block_dim(B) - 1);
        let tail_byte_len = packing::byte_len(tail_dim, B);
        let (head_bytes, tail_bytes) = self.data.split_at(tail_byte_len);
        let mut head = *self;
        head.data = head_bytes;
        let mut tail = *self;
        tail.data = tail_bytes;
        (tail_dim, head, tail)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct TurboPrimaryCoder<const B: usize>(InstructionSet);

impl<const B: usize> TurboPrimaryCoder<B> {
    const B_CHECK: () = { check_primary_bits(B) };

    #[allow(unused)]
    pub fn scalar() -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::B_CHECK;
        Self(InstructionSet::Scalar)
    }

    fn encode_parts(inst: InstructionSet, vector: &[f32]) -> (PrimaryVectorHeader, Vec<u8>) {
        let mut out = vec![0u8; packing::byte_len(vector.len(), B)];
        let header = Self::encode_parts_to(inst, vector, &mut out);
        (header, out)
    }

    fn encode_parts_to(
        inst: InstructionSet,
        vector: &[f32],
        out: &mut [u8],
    ) -> PrimaryVectorHeader {
        let stats = VectorStats::from(vector);
        let interval = optimize_interval(vector, &stats, B);
        let header = PrimaryVectorHeader {
            l2_norm: f16::from_f32(stats.l2_norm_sq.sqrt()),
            component_sum: f16::from_f32(stats.component_sum),
            lower: f16::from_f32(interval.0),
            delta: f16::from_f32((interval.1 - interval.0) / ((1 << B) - 1) as f32),
        };

        // XXX interval.1 need to do a f16 round trip.
        let terms = VectorEncodeTerms::from_primary::<B>(&header, interval.1);
        match inst {
            InstructionSet::Scalar => scalar::primary_quantize_and_pack::<B>(vector, terms, out),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::primary_quantize_and_pack::<B>(vector, terms, out),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe {
                x86_64::primary_quantize_and_pack_avx512::<B>(vector, terms, out)
            },
        };

        header
    }
}

impl<const B: usize> Default for TurboPrimaryCoder<B> {
    fn default() -> Self {
        Self(InstructionSet::default())
    }
}

impl<const B: usize> F32VectorCoder for TurboPrimaryCoder<B> {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let (header_bytes, vector_bytes) = PrimaryVectorHeader::split_output_buf(out).unwrap();
        let header = Self::encode_parts_to(self.0, vector, vector_bytes);
        header.serialize(header_bytes);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        PrimaryVectorHeader::LEN + packing::byte_len(dimensions, B)
    }

    fn decode_to(&self, vector: &[u8], out: &mut [f32]) {
        let vector = TurboPrimaryVector::<B>::new(vector).expect("valid primary vector");
        match self.0 {
            InstructionSet::Scalar => scalar::primary_decode::<B>(vector, out),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::primary_decode::<B>(vector, out),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe { x86_64::primary_decode_avx512::<B>(vector, out) },
        };
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
        let uint_dot = match self.1 {
            InstructionSet::Scalar => scalar::dot_u8::<B>(query.data, doc.data),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::dot_u8::<B>(query.data, doc.data),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe { x86_64::dot_u8_avx512::<B>(query.data, doc.data) },
        };
        let dot = query.terms.correct_dot(uint_dot, query.dim(), &doc.terms);
        dot_unnormalized_to_distance(self.0, dot.into(), (query.l2_norm(), doc.l2_norm()))
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
    terms: PrimaryVectorTerms,

    inst: InstructionSet,
}

impl<const B: usize> TurboPrimaryQueryDistance<B> {
    pub fn new(similarity: VectorSimilarity, query: Cow<'_, [f32]>) -> Self {
        let inst = InstructionSet::default();
        let (header, query) =
            TurboPrimaryCoder::<PRIMARY_QUERY_BITS>::encode_parts(inst, query.as_ref());

        Self {
            similarity,
            query,
            terms: PrimaryVectorTerms::from(header),
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
        let dot = self
            .terms
            .correct_dot(uint8_dot, self.query.len(), &vector.terms);
        dot_unnormalized_to_distance(
            self.similarity,
            dot.into(),
            (self.terms.l2_norm.into(), vector.l2_norm()),
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

const RESIDUAL_BITS: usize = 8;
const RESIDUAL_MAX: f32 = ((1 << RESIDUAL_BITS) - 1) as f32;

#[derive(Debug, Copy, Clone)]
struct TurboResidualVector<'a, const B: usize> {
    primary_data: &'a [u8],
    residual_data: &'a [u8],
    terms: ResidualVectorTerms,
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
            primary_data: primary_vector,
            residual_data: residual_vector,
            terms: ResidualVectorTerms::from((primary_header, residual_header)),
        })
    }

    fn dim(&self) -> usize {
        self.residual_data.len()
    }

    fn l2_norm(&self) -> f64 {
        self.terms.primary.l2_norm.into()
    }

    fn split_tail(&self, dim: usize) -> (usize, Self, Self) {
        let tail_dim = dim & !(packing::block_dim(B) - 1);
        let tail_primary_byte_len = packing::byte_len(tail_dim, B);
        let (primary_head, primary_tail) = self.primary_data.split_at(tail_primary_byte_len);
        let (residual_head, residual_tail) = self.residual_data.split_at(tail_dim);

        let mut head = *self;
        head.primary_data = primary_head;
        head.residual_data = residual_head;

        let mut tail = *self;
        tail.primary_data = primary_tail;
        tail.residual_data = residual_tail;

        (tail_dim, head, tail)
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
}

impl VectorEncodeTerms {
    // XXX should consume the upper bound, it did the wrong thing here.
    fn from_primary<const B: usize>(primary: &PrimaryVectorHeader, _upper: f32) -> Self {
        let lower = primary.lower.to_f32();
        let delta = primary.delta.to_f32();
        let upper = lower + delta * ((1 << B) - 1) as f32;
        let delta_inv = 1.0 / delta;
        Self {
            lower,
            upper,
            delta_inv,
        }
    }

    fn from_residual(magnitude: f32) -> Self {
        Self {
            lower: -magnitude / 2.0,
            upper: magnitude / 2.0,
            delta_inv: RESIDUAL_MAX / magnitude,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct TurboResidualCoder<const B: usize>(InstructionSet);

impl<const B: usize> TurboResidualCoder<B> {
    const B_CHECK: () = { check_primary_bits(B) };

    #[allow(unused)]
    pub fn scalar() -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::B_CHECK;
        Self(InstructionSet::Scalar)
    }

    fn encode_parts(
        inst: InstructionSet,
        vector: &[f32],
    ) -> (PrimaryVectorHeader, Vec<u8>, ResidualVectorHeader, Vec<u8>) {
        let mut primary = vec![0u8; packing::byte_len(vector.len(), B)];
        let mut residual = vec![0u8; vector.len()];
        let (primary_header, residual_header) =
            Self::encode_parts_to(inst, vector, &mut primary, &mut residual);
        (primary_header, primary, residual_header, residual)
    }

    fn encode_parts_to(
        inst: InstructionSet,
        vector: &[f32],
        primary: &mut [u8],
        residual: &mut [u8],
    ) -> (PrimaryVectorHeader, ResidualVectorHeader) {
        let stats = VectorStats::from(vector);
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
            (stats.min.abs() - interval.0.abs()) * 2.0,
            (stats.max.abs() - interval.1.abs()) * 2.0,
        ]
        .into_iter()
        .max_by(f32::total_cmp)
        .expect("3 values input");
        let primary_header = PrimaryVectorHeader {
            l2_norm: f16::from_f32(stats.l2_norm_sq.sqrt()),
            component_sum: f16::from_f32(stats.component_sum),
            lower: f16::from_f32(interval.0),
            delta: f16::from_f32((interval.1 - interval.0) / ((1 << B) - 1) as f32),
        };
        let residual_header = ResidualVectorHeader {
            magnitude: f16::from_f32(residual_magnitude),
        };

        // XXX interval.1 need to do a f16 round trip.
        let primary_terms = VectorEncodeTerms::from_primary::<B>(&primary_header, interval.1);
        let residual_terms = VectorEncodeTerms::from_residual(residual_header.magnitude.to_f32());
        match inst {
            InstructionSet::Scalar => scalar::residual_quantize_and_pack::<B>(
                vector,
                primary_terms,
                residual_terms,
                primary_header.delta.to_f32(),
                primary,
                residual,
            ),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::residual_quantize_and_pack::<B>(
                vector,
                primary_terms,
                residual_terms,
                primary_header.delta.to_f32(),
                primary,
                residual,
            ),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe {
                x86_64::residual_quantize_and_pack_avx512::<B>(
                    vector,
                    primary_terms,
                    residual_terms,
                    primary_header.delta.to_f32(),
                    primary,
                    residual,
                )
            },
        };

        (primary_header, residual_header)
    }
}

impl<const B: usize> Default for TurboResidualCoder<B> {
    fn default() -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::B_CHECK;
        Self(InstructionSet::default())
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

        let (primary_header, residual_header) =
            Self::encode_parts_to(self.0, vector, primary, residual);
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
        match self.0 {
            InstructionSet::Scalar => scalar::residual_decode::<B>(&vector, out),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::residual_decode::<B>(&vector, out),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe { x86_64::residual_decode_avx512::<B>(&vector, out) },
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
                (query.primary_data, query.residual_data),
                (doc.primary_data, doc.residual_data),
            ),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::residual_dot_unnormalized::<B>(
                (query.primary_data, query.residual_data),
                (doc.primary_data, doc.residual_data),
            ),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe {
                x86_64::residual_dot_unnormalized_avx512::<B>(
                    (query.primary_data, query.residual_data),
                    (doc.primary_data, doc.residual_data),
                )
            },
        };
        let dot = query
            .terms
            .correct_dot(&component_dot, query.dim(), &doc.terms);
        dot_unnormalized_to_distance(self.0, dot.into(), (query.l2_norm(), doc.l2_norm()))
    }
}

pub struct TurboResidualQueryDistance<const B: usize> {
    similarity: VectorSimilarity,

    primary_vector: Vec<u8>,
    residual_vector: Vec<u8>,
    terms: ResidualVectorTerms,

    inst: InstructionSet,
}

impl<const B: usize> TurboResidualQueryDistance<B> {
    pub fn new(similarity: VectorSimilarity, query: Cow<'_, [f32]>) -> Self {
        let inst = InstructionSet::default();
        let (primary_header, primary_vector, residual_header, residual_vector) =
            TurboResidualCoder::<B>::encode_parts(inst, query.as_ref());
        Self {
            similarity,
            primary_vector,
            residual_vector,
            terms: ResidualVectorTerms::from((primary_header, residual_header)),
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
                (vector.primary_data, vector.residual_data),
            ),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::residual_dot_unnormalized::<B>(
                (&self.primary_vector, &self.residual_vector),
                (vector.primary_data, vector.residual_data),
            ),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe {
                x86_64::residual_dot_unnormalized_avx512::<B>(
                    (&self.primary_vector, &self.residual_vector),
                    (vector.primary_data, vector.residual_data),
                )
            },
        };
        let dot = self
            .terms
            .correct_dot(&component_dot, self.residual_vector.len(), &vector.terms);
        dot_unnormalized_to_distance(
            self.similarity,
            dot.into(),
            (self.terms.primary.l2_norm.into(), vector.l2_norm()),
        )
    }
}

/// Compute a "fast" distance by comparing primary vectors and ignoring residual vectors.
///
/// This quantizes the query vector the same as the primary vector rather than using an 8-bit
/// representation. This makes the comparison symmetrical and faster, but at the cost of accuracy.
/// For 1-bit primary quantization this is 2-3x faster than comparing and 8-bit quantization.
pub struct TurboResidualFastQueryDistance<const B: usize> {
    similarity: VectorSimilarity,

    query: Vec<u8>,
    terms: PrimaryVectorTerms,

    inst: InstructionSet,
}

impl<const B: usize> TurboResidualFastQueryDistance<B> {
    pub fn new(similarity: VectorSimilarity, query: &[f32]) -> Self {
        let inst = InstructionSet::default();
        let (header, query) = TurboPrimaryCoder::<B>::encode_parts(inst, query);
        let terms = PrimaryVectorTerms::from(header);

        Self {
            similarity,
            query,
            terms,
            inst,
        }
    }
}

impl<const B: usize> QueryVectorDistance for TurboResidualFastQueryDistance<B> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = TurboResidualVector::<B>::new(vector).expect("valid vector");
        let dot_uint = match self.inst {
            InstructionSet::Scalar => scalar::dot_u8::<B>(&self.query, vector.primary_data),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::dot_u8::<B>(&self.query, vector.primary_data),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe {
                x86_64::dot_u8_avx512::<B>(&self.query, vector.primary_data)
            },
        };
        dot_unnormalized_to_distance(
            self.similarity,
            self.terms
                .correct_dot(dot_uint, self.query.len(), &vector.terms.primary)
                .into(),
            (self.terms.l2_norm.into(), vector.l2_norm()),
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

#[cfg(test)]
mod test {
    use approx::{AbsDiffEq, abs_diff_eq, assert_abs_diff_eq};
    use half::f16;

    use crate::lvq::{
        PrimaryVectorHeader, ResidualVectorHeader, TurboPrimaryCoder, TurboResidualCoder,
        VectorStats,
    };
    use crate::{F32VectorCoder, F32VectorCoding, VectorSimilarity};

    impl AbsDiffEq for PrimaryVectorHeader {
        type Epsilon = f32;

        fn default_epsilon() -> Self::Epsilon {
            0.00001
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            abs_diff_eq!(
                self.l2_norm.to_f32(),
                other.l2_norm.to_f32(),
                epsilon = epsilon
            ) && abs_diff_eq!(self.lower.to_f32(), other.lower.to_f32(), epsilon = epsilon)
                && abs_diff_eq!(self.delta.to_f32(), other.delta.to_f32(), epsilon = epsilon)
                && abs_diff_eq!(
                    self.component_sum.to_f32(),
                    other.component_sum.to_f32(),
                    epsilon = epsilon
                )
        }
    }

    impl AbsDiffEq for ResidualVectorHeader {
        type Epsilon = f32;

        fn default_epsilon() -> Self::Epsilon {
            0.00001
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            // TODO: tlvq8x8 fails on aarch64 when epsilon = 0; figure this out
            abs_diff_eq!(
                self.magnitude.to_f32(),
                other.magnitude.to_f32(),
                epsilon = epsilon
            )
        }
    }

    impl AbsDiffEq for VectorStats {
        type Epsilon = f32;

        fn default_epsilon() -> Self::Epsilon {
            0.00001
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            abs_diff_eq!(self.min, other.min, epsilon = epsilon)
                && abs_diff_eq!(self.max, other.max, epsilon = epsilon)
                && abs_diff_eq!(self.mean, other.mean, epsilon = epsilon)
                && abs_diff_eq!(self.std_dev, other.std_dev, epsilon = epsilon)
                && abs_diff_eq!(self.l2_norm_sq, other.l2_norm_sq, epsilon = epsilon)
                && abs_diff_eq!(self.component_sum, other.component_sum, epsilon = epsilon)
        }
    }

    // This test vector contains randomly generated numbers in [-1,1] but is not l2 normalized.
    // It has 19 elements -- long enough to trigger SIMD optimizations but with some remainder to
    // test scalar tail paths.
    const TEST_VECTOR: [f32; 19] = [
        -0.921, -0.061, 0.659, 0.67, 0.573, 0.431, 0.646, 0.001, -0.2, -0.428, 0.73, -0.704,
        -0.273, 0.539, -0.731, 0.436, 0.913, 0.694, 0.202,
    ];

    #[test]
    fn vector_stats_simd() {
        let simd_stats = VectorStats::from(TEST_VECTOR.as_ref());
        let scalar_stats = VectorStats::from_scalar(TEST_VECTOR.as_ref());
        assert_abs_diff_eq!(simd_stats, scalar_stats);
    }

    #[test]
    fn tlvq1() {
        let coder = TurboPrimaryCoder::<1>::default();
        let encoded = coder.encode(&TEST_VECTOR);
        assert_abs_diff_eq!(
            PrimaryVectorHeader::deserialize(&encoded).unwrap().0,
            PrimaryVectorHeader {
                l2_norm: f16::from_f32(2.5226507),
                lower: f16::from_f32(-0.49564388),
                delta: f16::from_f32(1.2012576),
                component_sum: f16::from_f32(3.176),
            }
        );
        let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
        coder.decode_to(&encoded, &mut decoded);
        assert_abs_diff_eq!(
            decoded.as_ref(),
            [
                -0.49560547,
                -0.49560547,
                0.7055664,
                0.7055664,
                0.7055664,
                0.7055664,
                0.7055664,
                -0.49560547,
                -0.49560547,
                -0.49560547,
                0.7055664,
                -0.49560547,
                -0.49560547,
                0.7055664,
                -0.49560547,
                0.7055664,
                0.7055664,
                0.7055664,
                0.7055664
            ]
            .as_ref(),
            epsilon = 0.00001
        );
    }

    #[test]
    fn tlvq2() {
        let coder = TurboPrimaryCoder::<2>::default();
        let encoded = coder.encode(&TEST_VECTOR);
        assert_abs_diff_eq!(
            PrimaryVectorHeader::deserialize(&encoded).unwrap().0,
            PrimaryVectorHeader {
                l2_norm: f16::from_f32(2.5226507),
                lower: f16::from_f32(-0.6709247),
                delta: f16::from_f32(0.5039812),
                component_sum: f16::from_f32(3.176),
            }
        );
        let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
        coder.decode_to(&encoded, &mut decoded);
        assert_abs_diff_eq!(
            decoded.as_ref(),
            [
                -0.67089844,
                -0.16699219,
                0.8408203,
                0.8408203,
                0.33691406,
                0.33691406,
                0.8408203,
                -0.16699219,
                -0.16699219,
                -0.67089844,
                0.8408203,
                -0.67089844,
                -0.16699219,
                0.33691406,
                -0.67089844,
                0.33691406,
                0.8408203,
                0.8408203,
                0.33691406
            ]
            .as_ref(),
            epsilon = 0.0001,
        );
    }

    #[test]
    fn tlvq4() {
        let coder = TurboPrimaryCoder::<4>::default();
        let encoded = coder.encode(&TEST_VECTOR);
        assert_abs_diff_eq!(
            PrimaryVectorHeader::deserialize(&encoded).unwrap().0,
            PrimaryVectorHeader {
                l2_norm: f16::from_f32(2.5226507),
                lower: f16::from_f32(-0.93474734),
                delta: f16::from_f32(0.12319123),
                component_sum: f16::from_f32(3.176),
            }
        );
        let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
        coder.decode_to(&encoded, &mut decoded);
        assert_abs_diff_eq!(
            decoded.as_ref(),
            [
                -0.9345703,
                -0.072387695,
                0.666626,
                0.666626,
                0.54345703,
                0.4202881,
                0.666626,
                0.05078125,
                -0.19555664,
                -0.44189453,
                0.7897949,
                -0.6882324,
                -0.3187256,
                0.54345703,
                -0.6882324,
                0.4202881,
                0.91296387,
                0.666626,
                0.1739502
            ]
            .as_ref(),
            epsilon = 0.0001,
        );
    }

    #[test]
    fn tlvq8() {
        let coder = TurboPrimaryCoder::<8>::default();
        let encoded = coder.encode(&TEST_VECTOR);
        assert_abs_diff_eq!(
            PrimaryVectorHeader::deserialize(&encoded).unwrap().0,
            PrimaryVectorHeader {
                l2_norm: f16::from_f32(2.5226507),
                lower: f16::from_f32(-0.92000645),
                delta: f16::from_f32(0.0071822493),
                component_sum: f16::from_f32(3.176),
            }
        );
        let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
        coder.decode_to(&encoded, &mut decoded);
        assert_abs_diff_eq!(
            decoded.as_ref(),
            [
                -0.9199219,
                -0.05795288,
                0.6603546,
                0.6675377,
                0.5741577,
                0.43049622,
                0.64598846,
                -0.00048828125,
                -0.20161438,
                -0.43147278,
                0.73218536,
                -0.7044296,
                -0.27344513,
                0.53824234,
                -0.7331619,
                0.4376793,
                0.91176224,
                0.69627,
                0.20063782
            ]
            .as_ref(),
            epsilon = 0.0001
        );
    }

    #[test]
    fn tlvq1x8() {
        let coder = TurboResidualCoder::<1>::default();
        let encoded = coder.encode(&TEST_VECTOR);
        let (primary_header, vector_bytes) = PrimaryVectorHeader::deserialize(&encoded).unwrap();
        assert_abs_diff_eq!(
            primary_header,
            PrimaryVectorHeader {
                l2_norm: f16::from_f32(2.5226507),
                lower: f16::from_f32(-0.49564388),
                delta: f16::from_f32(1.2012576),
                component_sum: f16::from_f32(3.176),
            }
        );
        let (residual_header, _) = ResidualVectorHeader::deserialize(&vector_bytes).unwrap();
        assert_abs_diff_eq!(
            residual_header,
            ResidualVectorHeader {
                magnitude: f16::from_f32(1.2012576),
            }
        );
        let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
        coder.decode_to(&encoded, &mut decoded);
        assert_abs_diff_eq!(
            decoded.as_ref(),
            [
                -0.9219037,
                -0.059886307,
                0.66081685,
                0.6702378,
                0.5713178,
                0.43000343,
                0.6466854,
                0.001349926,
                -0.20120063,
                -0.42730355,
                0.73147404,
                -0.7052218,
                -0.2718578,
                0.53834444,
                -0.72877413,
                0.4347139,
                0.91518265,
                0.6937902,
                0.20390052
            ]
            .as_ref(),
            epsilon = 0.00001
        );
    }

    #[test]
    fn tlvq2x8() {
        let coder = TurboResidualCoder::<2>::default();
        let encoded = coder.encode(&TEST_VECTOR);
        let (primary_header, vector_bytes) = PrimaryVectorHeader::deserialize(&encoded).unwrap();
        assert_abs_diff_eq!(
            primary_header,
            PrimaryVectorHeader {
                l2_norm: f16::from_f32(2.5226507),
                lower: f16::from_f32(-0.6709247),
                delta: f16::from_f32(0.5039812),
                component_sum: f16::from_f32(3.176),
            }
        );
        let (residual_header, _) = ResidualVectorHeader::deserialize(&vector_bytes).unwrap();
        assert_abs_diff_eq!(
            residual_header,
            ResidualVectorHeader {
                magnitude: f16::from_f32(0.5039812),
            }
        );
        let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
        coder.decode_to(&encoded, &mut decoded);
        assert_abs_diff_eq!(
            decoded.as_ref(),
            [
                -0.92087543,
                -0.06127066,
                0.6580308,
                0.6698874,
                0.57305837,
                0.43077898,
                0.6461742,
                0.0019646436,
                -0.19959787,
                -0.4288258,
                0.7291705,
                -0.70350415,
                -0.2727137,
                0.53946465,
                -0.7311696,
                0.4367073,
                0.9129481,
                0.69360065,
                0.20155102
            ]
            .as_ref(),
            epsilon = 0.0001,
        );
    }

    #[test]
    fn tlvq4x8() {
        let coder = TurboResidualCoder::<4>::default();
        let encoded = coder.encode(&TEST_VECTOR);
        let (primary_header, vector_bytes) = PrimaryVectorHeader::deserialize(&encoded).unwrap();
        assert_abs_diff_eq!(
            primary_header,
            PrimaryVectorHeader {
                l2_norm: f16::from_f32(2.5226507),
                lower: f16::from_f32(-0.93474734),
                delta: f16::from_f32(0.12319123),
                component_sum: f16::from_f32(3.176),
            }
        );
        let (residual_header, _) = ResidualVectorHeader::deserialize(&vector_bytes).unwrap();
        assert_abs_diff_eq!(
            residual_header,
            ResidualVectorHeader {
                magnitude: f16::from_f32(0.12319123),
            }
        );
        let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
        coder.decode_to(&encoded, &mut decoded);
        assert_abs_diff_eq!(
            decoded.as_ref(),
            [
                -0.9208044,
                -0.061036833,
                0.6591392,
                0.6697656,
                0.5731625,
                0.43115592,
                0.64609784,
                0.00078914687,
                -0.20014529,
                -0.4281286,
                0.73014253,
                -0.70393044,
                -0.27308062,
                0.53886837,
                -0.73097926,
                0.4359861,
                0.9132054,
                0.6939163,
                0.2022066
            ]
            .as_ref(),
            epsilon = 0.0001,
        );
    }

    #[test]
    fn tlvq8x8() {
        let coder = TurboResidualCoder::<8>::default();
        let encoded = coder.encode(&TEST_VECTOR);
        let (primary_header, vector_bytes) = PrimaryVectorHeader::deserialize(&encoded).unwrap();
        assert_abs_diff_eq!(
            primary_header,
            PrimaryVectorHeader {
                l2_norm: f16::from_f32(2.5226507),
                lower: f16::from_f32(-0.92000645),
                delta: f16::from_f32(0.0071822493),
                component_sum: f16::from_f32(3.176),
            }
        );
        let (residual_header, _) = ResidualVectorHeader::deserialize(&vector_bytes).unwrap();
        assert_abs_diff_eq!(
            residual_header,
            ResidualVectorHeader {
                magnitude: f16::from_f32(0.0071822493),
            }
        );
        let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
        coder.decode_to(&encoded, &mut decoded);
        assert_abs_diff_eq!(
            decoded.as_ref(),
            [
                -0.9210063,
                -0.06099535,
                0.65900403,
                0.66998863,
                0.572986,
                0.43100283,
                0.64599144,
                0.0009973188,
                -0.199993,
                -0.42799422,
                0.7300097,
                -0.70398974,
                -0.27299845,
                0.53899,
                -0.7310006,
                0.43598813,
                0.9130022,
                0.69401395,
                0.20198764
            ]
            .as_ref(),
            epsilon = 0.0001
        );
    }

    #[test]
    fn null_vector_decode() {
        let vector = vec![0.0f32; 256];
        for coding in [
            F32VectorCoding::TLVQ1,
            F32VectorCoding::TLVQ2,
            F32VectorCoding::TLVQ4,
            F32VectorCoding::TLVQ8,
            F32VectorCoding::TLVQ1x8,
            F32VectorCoding::TLVQ2x8,
            F32VectorCoding::TLVQ4x8,
            F32VectorCoding::TLVQ8x8,
        ] {
            let coder = coding.new_coder(VectorSimilarity::Dot);
            let encoded = coder.encode(&vector);
            let decoded = coder.decode(&encoded);
            assert_abs_diff_eq!(decoded.as_slice(), vector.as_ref());
        }
    }

    #[test]
    fn fill_vector_decode() {
        let vector = vec![1.0f32; 256];
        for coding in [
            F32VectorCoding::TLVQ1,
            F32VectorCoding::TLVQ2,
            F32VectorCoding::TLVQ4,
            F32VectorCoding::TLVQ8,
            F32VectorCoding::TLVQ1x8,
            F32VectorCoding::TLVQ2x8,
            F32VectorCoding::TLVQ4x8,
            F32VectorCoding::TLVQ8x8,
        ] {
            let coder = coding.new_coder(VectorSimilarity::Dot);
            let encoded = coder.encode(&vector);
            let decoded = coder.decode(&encoded);
            assert_abs_diff_eq!(decoded.as_slice(), vector.as_ref());
        }
    }
}
