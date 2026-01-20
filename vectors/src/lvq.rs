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

use std::borrow::Cow;

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
const SUPPORTED_RESIDUAL_BITS: [usize; 2] = [4, 8];

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

const fn check_residual_bits(primary: usize, residual: usize) {
    assert!(is_supported_bits(primary, &SUPPORTED_PRIMARY_BITS));
    assert!(is_supported_bits(residual, &SUPPORTED_RESIDUAL_BITS));
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
    l2_norm: f32,
    lower: f32,
    upper: f32,
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
        header[3] = self.component_sum.to_le_bytes();
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
                component_sum: u32::from_le_bytes(header_entries[3]),
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
struct VectorTerms {
    lower: f32,
    delta: f32,
    component_sum: u32,
}

impl VectorTerms {
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
    terms: VectorTerms,
    data: &'a [u8],
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
        InstructionSet::Avx512 => unsafe { x86_64::dot_u8::<B>(a.data, b.data) },
    };
    correct_dot_uint(dot, dim, &a.terms, &b.terms)
}

fn dot_unnormalized_uint8_asymmetric<const B: usize>(
    inst: InstructionSet,
    query: &[u8],
    query_terms: VectorTerms,
    doc: &TurboPrimaryVector<'_, B>,
) -> f32 {
    let dot = match inst {
        InstructionSet::Scalar => scalar::primary_query8_dot_unnormalized::<B>(query, doc),
        #[cfg(target_arch = "aarch64")]
        InstructionSet::Neon => aarch64::primary_query8_dot_unnormalized::<B>(query, doc),
        #[cfg(target_arch = "x86_64")]
        InstructionSet::Avx512 => unsafe {
            x86_64::primary_query8_dot_unnormalized_avx512::<B>(query, doc)
        },
    };
    correct_dot_uint(dot, query.len(), &query_terms, &doc.rep.terms)
}

/// Correct the dot product of two integer vectors using the stored vector terms.
fn correct_dot_uint(dot: u32, dim: usize, a: &VectorTerms, b: &VectorTerms) -> f32 {
    dot as f32 * a.delta * b.delta
        + a.component_sum as f32 * a.delta * b.lower
        + b.component_sum as f32 * b.delta * a.lower
        + a.lower * b.lower * dim as f32
}

/// An LVQ1 coded primary vector.
///
/// There may be a parallel residual vector that can be composed with this one to increase accuracy.
struct PrimaryVector<'a, const B: usize> {
    l2_norm: f32,
    v: EncodedVector<'a>,
}

impl<'a, const B: usize> PrimaryVector<'a, B> {
    const B_CHECK: () = { check_primary_bits(B) };

    fn new(encoded: &'a [u8]) -> Option<Self> {
        #[allow(clippy::let_unit_value)]
        let _ = Self::B_CHECK;

        let (header, vector) = PrimaryVectorHeader::deserialize(encoded)?;
        Some(Self::with_header(header, vector))
    }

    fn with_header(header: PrimaryVectorHeader, vector: &'a [u8]) -> Self {
        Self {
            l2_norm: header.l2_norm,
            v: EncodedVector {
                terms: VectorTerms {
                    lower: header.lower,
                    delta: (header.upper - header.lower) / ((1 << B) - 1) as f32,
                    component_sum: header.component_sum,
                },
                data: vector,
            },
        }
    }

    fn l2_norm(&self) -> f64 {
        self.l2_norm.into()
    }

    fn dim(&self) -> usize {
        (self.v.data.len() * 8).div_ceil(B)
    }

    fn f32_iter(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        packing::unpack_iter::<B>(self.v.data)
            .map(|q| (q as f32).mul_add(self.v.terms.delta, self.v.terms.lower))
    }
}

struct TwoLevelVector<'a, const B1: usize, const B2: usize> {
    primary: PrimaryVector<'a, B1>,
    residual: EncodedVector<'a>,
}

impl<'a, const B1: usize, const B2: usize> TwoLevelVector<'a, B1, B2> {
    const B_CHECK: () = { check_residual_bits(B1, B2) };

    fn new(encoded: &'a [u8]) -> Option<Self> {
        #[allow(clippy::let_unit_value)]
        let _ = Self::B_CHECK;

        let (primary_header, vector_bytes) = PrimaryVectorHeader::deserialize(encoded)?;
        let (residual_header, vector_bytes) = ResidualVectorHeader::deserialize(vector_bytes)?;
        let split = packing::two_vector_split(vector_bytes.len(), B1, B2);
        let (primary_vector, residual_vector) = vector_bytes.split_at(split);
        let primary = PrimaryVector::<'_, B1>::with_header(primary_header, primary_vector);
        let residual = EncodedVector {
            terms: VectorTerms {
                lower: -residual_header.magnitude / 2.0,
                delta: residual_header.magnitude / ((1 << B2) - 1) as f32,
                component_sum: residual_header.component_sum,
            },
            data: residual_vector,
        };
        Some(Self { primary, residual })
    }

    pub fn l2_norm(&self) -> f64 {
        self.primary.l2_norm()
    }

    pub fn dim(&self) -> usize {
        self.primary
            .dim()
            .min((self.residual.data.len() * 8).div_ceil(B2))
    }

    fn f32_iter(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        self.primary
            .f32_iter()
            .zip(
                packing::unpack_iter::<B2>(self.residual.data).map(|r| {
                    (r as f32).mul_add(self.residual.terms.delta, self.residual.terms.lower)
                }),
            )
            .map(|(q, r)| q + r)
    }

    fn f32_dot_correction(&self, query_sum: f32, primary_dot: f32, residual_dot: f32) -> f32 {
        primary_dot * self.primary.v.terms.delta
            + query_sum * self.primary.v.terms.lower
            + residual_dot * self.residual.terms.delta
            + query_sum * self.residual.terms.lower
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PrimaryVectorCoder<const B: usize>(InstructionSet);

impl<const B: usize> PrimaryVectorCoder<B> {
    const B_CHECK: () = { check_primary_bits(B) };

    #[allow(unused)]
    pub fn scalar() -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::B_CHECK;
        PrimaryVectorCoder::<B>(InstructionSet::Scalar)
    }
}

impl<const B: usize> Default for PrimaryVectorCoder<B> {
    fn default() -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::B_CHECK;
        PrimaryVectorCoder::<B>(InstructionSet::default())
    }
}

impl<const B: usize> F32VectorCoder for PrimaryVectorCoder<B> {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let stats = VectorStats::from(vector);
        let mut header = PrimaryVectorHeader::from(stats);
        (header.lower, header.upper) = optimize_interval(vector, &stats, B);
        let (header_bytes, vector_bytes) = PrimaryVectorHeader::split_output_buf(out).unwrap();
        header.component_sum = match self.0 {
            InstructionSet::Scalar => scalar::lvq1_quantize_and_pack::<B>(
                vector,
                header.lower,
                header.upper,
                vector_bytes,
            ),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::lvq1_quantize_and_pack::<B>(
                vector,
                header.lower,
                header.upper,
                vector_bytes,
            ),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe {
                x86_64::lvq1_quantize_and_pack_avx512::<B>(
                    vector,
                    header.lower,
                    header.upper,
                    vector_bytes,
                )
            },
        };
        header.serialize(header_bytes);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        PrimaryVectorHeader::LEN + packing::byte_len(dimensions, B)
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        let v = PrimaryVector::<B>::new(encoded).expect("valid vector");
        match self.0 {
            InstructionSet::Scalar => scalar::lvq1_decode::<B>(&v, out),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::lvq1_decode::<B>(&v, out),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe { x86_64::lvq1_decode_avx512::<B>(&v, out) },
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        (byte_len - PrimaryVectorHeader::LEN) * 8 / B
    }
}

/// Store a two level vector.
///
/// Format is as follows:
/// * Vector header (16 bytes)
/// * Primary vector -- length depends on dimensionality and B1
/// * Residual interval value (4 bytes)
/// * Residual vector -- length depends on dimensionality and B2
///
/// This format allows the representation to be split -- the primary vector can be used with just
/// the header and primary vector contents; the residual delta and vector can be used along with
/// the primary vector parts to provide a higher fidelity representation.
#[derive(Debug, Copy, Clone)]
pub struct TwoLevelVectorCoder<const B1: usize, const B2: usize>(InstructionSet);

impl<const B1: usize, const B2: usize> TwoLevelVectorCoder<B1, B2> {
    const B_CHECK: () = { check_residual_bits(B1, B2) };

    #[allow(unused)]
    pub fn scalar() -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::B_CHECK;
        TwoLevelVectorCoder::<B1, B2>(InstructionSet::Scalar)
    }
}

impl<const B1: usize, const B2: usize> Default for TwoLevelVectorCoder<B1, B2> {
    fn default() -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::B_CHECK;
        TwoLevelVectorCoder::<B1, B2>(InstructionSet::default())
    }
}

impl<const B1: usize, const B2: usize> F32VectorCoder for TwoLevelVectorCoder<B1, B2> {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let stats = VectorStats::from(vector);
        let mut primary_header = PrimaryVectorHeader::from(stats);
        // NB: this interval optimization reduces loss for the primary vector, but this loss
        // reduction can make the residual vector more lossy if we derive the delta from the primary
        // interval as described in the LVQ paper. Compute and store a residual delta that is large
        // enough to encode both the min and max value in the vector.
        // TODO: investigate interval optimization on the derived residual vector.
        let interval = optimize_interval(vector, &stats, B1);
        // For the residual interval choose the maximum based on primary delta, or the min/max
        // values we may need to encode based on the gap between the initial and optimized interval.
        let residual_interval = [
            (interval.1 - interval.0) / ((1 << B1) - 1) as f32,
            (primary_header.lower.abs() - interval.0.abs()) * 2.0,
            (primary_header.upper.abs() - interval.1.abs()) * 2.0,
        ]
        .into_iter()
        .max_by(f32::total_cmp)
        .expect("3 values input");
        (primary_header.lower, primary_header.upper) = interval;

        let (primary_header_bytes, vector_bytes) =
            PrimaryVectorHeader::split_output_buf(out).unwrap();
        let (residual_header_bytes, vector_bytes) =
            ResidualVectorHeader::split_output_buf(vector_bytes).unwrap();
        let split = packing::two_vector_split(vector_bytes.len(), B1, B2);
        let (primary, residual) = vector_bytes.split_at_mut(split);
        let mut residual_header = ResidualVectorHeader {
            magnitude: residual_interval,
            component_sum: 0,
        };
        (primary_header.component_sum, residual_header.component_sum) = match self.0 {
            InstructionSet::Scalar => scalar::lvq2_quantize_and_pack::<B1, B2>(
                vector,
                primary_header.lower,
                primary_header.upper,
                primary,
                residual_interval,
                residual,
            ),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::lvq2_quantize_and_pack::<B1, B2>(
                vector,
                primary_header.lower,
                primary_header.upper,
                primary,
                residual_interval,
                residual,
            ),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe {
                x86_64::lvq2_quantize_and_pack::<B1, B2>(
                    vector,
                    primary_header.lower,
                    primary_header.upper,
                    primary,
                    residual_interval,
                    residual,
                )
            },
        };
        primary_header.serialize(primary_header_bytes);
        residual_header.serialize(residual_header_bytes);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        PrimaryVectorHeader::LEN
            + ResidualVectorHeader::LEN
            + packing::byte_len(dimensions, B1)
            + packing::byte_len(dimensions, B2)
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        let v = TwoLevelVector::<B1, B2>::new(encoded).expect("valid vector");
        match self.0 {
            InstructionSet::Scalar => scalar::lvq2_decode::<B1, B2>(&v, out),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::lvq2_decode::<B1, B2>(&v, out),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe { x86_64::lvq2_decode_avx512::<B1, B2>(&v, out) },
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        let len_no_corrective_terms =
            byte_len - PrimaryVectorHeader::LEN - ResidualVectorHeader::LEN;
        let split = packing::two_vector_split(len_no_corrective_terms, B1, B2);
        let dim1 = split * 8 / B1;
        let dim2 = (len_no_corrective_terms - split) * 8 / B2;
        dim1.min(dim2)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PrimaryDistance<const B: usize>(VectorSimilarity, InstructionSet);

impl<const B: usize> PrimaryDistance<B> {
    pub fn new(similarity: VectorSimilarity) -> Self {
        Self(similarity, InstructionSet::default())
    }
}

impl<const B: usize> VectorDistance for PrimaryDistance<B> {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = PrimaryVector::<B>::new(query).unwrap();
        let doc = PrimaryVector::<B>::new(doc).unwrap();
        dot_unnormalized_to_distance(
            self.0,
            dot_unnormalized_uint_symmetric::<B>(self.1, query.dim(), &query.v, &doc.v).into(),
            (query.l2_norm(), doc.l2_norm()),
        )
    }
}

// XXX rename to ResidualComponentDot
/// The four components of a residual dot product.
#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
struct LVQ2Dot {
    ap_dot_bp: u32,
    ap_dot_br: u32,
    ar_dot_bp: u32,
    ar_dot_br: u32,
}

impl LVQ2Dot {
    fn compute_dot(
        &self,
        dim: usize,
        a: (&VectorTerms, &VectorTerms),
        b: (&VectorTerms, &VectorTerms),
    ) -> f32 {
        correct_dot_uint(self.ap_dot_bp, dim, a.0, b.0)
            + correct_dot_uint(self.ap_dot_br, dim, a.0, b.1)
            + correct_dot_uint(self.ar_dot_bp, dim, a.1, b.0)
            + correct_dot_uint(self.ar_dot_br, dim, a.1, b.1)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TwoLevelDistance<const B1: usize, const B2: usize>(VectorSimilarity, InstructionSet);

impl<const B1: usize, const B2: usize> TwoLevelDistance<B1, B2> {
    pub fn new(similarity: VectorSimilarity) -> Self {
        Self(similarity, InstructionSet::default())
    }
}

impl<const B1: usize, const B2: usize> VectorDistance for TwoLevelDistance<B1, B2> {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = TwoLevelVector::<B1, B2>::new(query).unwrap();
        let doc = TwoLevelVector::<B1, B2>::new(doc).unwrap();

        let qv = (query.primary.v, query.residual);
        let dv = (doc.primary.v, doc.residual);
        let lvq2_dot = match self.1 {
            InstructionSet::Scalar => {
                scalar::dot_residual_u8::<B1, B2>(qv.0.data, qv.1.data, dv.0.data, dv.1.data)
            }
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => {
                aarch64::dot_residual_u8::<B1, B2>(qv.0.data, qv.1.data, dv.0.data, dv.1.data)
            }
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe {
                x86_64::dot_residual_u8::<B1, B2>(qv.0.data, qv.1.data, dv.0.data, dv.1.data)
            },
        };
        let dim = query.dim();
        let dot = correct_dot_uint(lvq2_dot.ap_dot_bp, dim, &qv.0.terms, &dv.0.terms)
            + correct_dot_uint(lvq2_dot.ap_dot_br, dim, &qv.0.terms, &dv.1.terms)
            + correct_dot_uint(lvq2_dot.ar_dot_bp, dim, &qv.1.terms, &dv.0.terms)
            + correct_dot_uint(lvq2_dot.ar_dot_br, dim, &qv.1.terms, &dv.1.terms);
        dot_unnormalized_to_distance(self.0, dot.into(), (query.l2_norm(), doc.l2_norm()))
    }
}

#[derive(Debug, Clone)]
pub struct TwoLevelQueryDistance<'a, const B1: usize, const B2: usize> {
    similarity: VectorSimilarity,
    query: Cow<'a, [f32]>,
    query_l2_norm: f64,
    query_sum: f32,
    inst: InstructionSet,
}

impl<'a, const B1: usize, const B2: usize> TwoLevelQueryDistance<'a, B1, B2> {
    pub fn new(similarity: VectorSimilarity, query: Cow<'a, [f32]>) -> Self {
        let query_l2_norm = match similarity {
            VectorSimilarity::Dot => 1.0,
            _ => super::l2_norm(&query).into(),
        };
        let query_sum = query.iter().copied().sum::<f32>();
        Self {
            similarity,
            query,
            query_l2_norm,
            query_sum,
            inst: InstructionSet::default(),
        }
    }
}

impl<const B1: usize, const B2: usize> QueryVectorDistance for TwoLevelQueryDistance<'_, B1, B2> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = TwoLevelVector::<B1, B2>::new(vector).unwrap();
        let dot = match self.inst {
            InstructionSet::Scalar => scalar::lvq2_f32_dot_unnormalized::<B1, B2>(
                self.query.as_ref(),
                self.query_sum,
                &vector,
            ),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::lvq2_f32_dot_unnormalized::<B1, B2>(
                self.query.as_ref(),
                self.query_sum,
                &vector,
            ),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe {
                x86_64::lvq2_f32_dot_unnormalized::<B1, B2>(
                    self.query.as_ref(),
                    self.query_sum,
                    &vector,
                )
            },
        };
        dot_unnormalized_to_distance(self.similarity, dot, (self.query_l2_norm, vector.l2_norm()))
    }
}

#[derive(Debug, Clone)]
pub struct FastTwoLevelQueryDistance<const B1: usize, const B2: usize> {
    similarity: VectorSimilarity,
    query: Vec<u8>,
    dim: usize,
    l2_norm: f64,
    terms: VectorTerms,
    inst: InstructionSet,
}

impl<const B1: usize, const B2: usize> FastTwoLevelQueryDistance<B1, B2> {
    pub fn new(similarity: VectorSimilarity, query: &[f32]) -> Self {
        let query = PrimaryVectorCoder::<B1>::default().encode(query);
        let vec = PrimaryVector::<B1>::new(&query).unwrap();
        let dim = query.len();
        let l2_norm = vec.l2_norm();
        let terms = vec.v.terms;
        Self {
            similarity,
            query,
            dim,
            l2_norm,
            terms,
            inst: InstructionSet::default(),
        }
    }
}

impl<const B1: usize, const B2: usize> QueryVectorDistance for FastTwoLevelQueryDistance<B1, B2> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let query = EncodedVector {
            terms: self.terms,
            data: &self.query[PrimaryVectorHeader::LEN..],
        };
        let doc = TwoLevelVector::<B1, B2>::new(vector).unwrap();
        dot_unnormalized_to_distance(
            self.similarity,
            dot_unnormalized_uint_symmetric::<B1>(self.inst, self.dim, &query, &doc.primary.v)
                .into(),
            (self.l2_norm, doc.l2_norm()),
        )
    }
}

/// The turbo coder requires that all vector data be packed into 16-byte blocks.
const TURBO_BLOCK_SIZE: usize = 16;

struct TurboPrimaryVector<'a, const B: usize> {
    rep: EncodedVector<'a>,
    l2_norm: f32,
}

impl<'a, const B: usize> TurboPrimaryVector<'a, B> {
    fn new(data: &'a [u8]) -> Option<Self> {
        let (header, vector_bytes) = PrimaryVectorHeader::deserialize(data)?;
        Some(Self {
            rep: EncodedVector {
                terms: VectorTerms::from_primary::<B>(header),
                data: vector_bytes,
            },
            l2_norm: header.l2_norm,
        })
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = u8> {
        packing::TurboUnpacker::<B>::new(self.rep.data)
    }

    fn l2_norm(&self) -> f64 {
        self.l2_norm.into()
    }

    fn split_tail(&self, dim: usize) -> (usize, Self, Self) {
        let tail_dim = dim & !(packing::block_dim(B) - 1);
        let (head_data, tail_data) = self.rep.data.split_at(packing::byte_len(tail_dim, B));
        (
            tail_dim,
            Self {
                rep: EncodedVector {
                    terms: self.rep.terms,
                    data: head_data,
                },
                l2_norm: self.l2_norm,
            },
            Self {
                rep: EncodedVector {
                    terms: self.rep.terms,
                    data: tail_data,
                },
                l2_norm: self.l2_norm,
            },
        )
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
}

impl<const B: usize> Default for TurboPrimaryCoder<B> {
    fn default() -> Self {
        Self(InstructionSet::default())
    }
}

impl<const B: usize> F32VectorCoder for TurboPrimaryCoder<B> {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let stats = VectorStats::from(vector);
        let mut header = PrimaryVectorHeader::from(stats);
        (header.lower, header.upper) = optimize_interval(vector, &stats, B);
        let (header_bytes, vector_bytes) = PrimaryVectorHeader::split_output_buf(out).unwrap();

        let delta_inv = ((1 << B) - 1) as f32 / (header.upper - header.lower);
        header.component_sum = match self.0 {
            InstructionSet::Scalar => scalar::primary_quantize_and_pack::<B>(
                vector,
                header.lower,
                header.upper,
                delta_inv,
                vector_bytes,
            ),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::primary_quantize_and_pack::<B>(
                vector,
                header.lower,
                header.upper,
                delta_inv,
                vector_bytes,
            ),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe {
                x86_64::primary_quantize_and_pack_avx512::<B>(
                    vector,
                    header.lower,
                    header.upper,
                    delta_inv,
                    vector_bytes,
                )
            },
        };

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

const PRIMARY_QUERY_BITS: usize = 8;

#[derive(Debug, Clone)]
pub struct TurboPrimaryQueryDistance<const B: usize> {
    similarity: VectorSimilarity,

    query: Vec<u8>,
    terms: VectorTerms,
    l2_norm: f64,

    inst: InstructionSet,
}

impl<const B: usize> TurboPrimaryQueryDistance<B> {
    pub fn new(similarity: VectorSimilarity, query: Cow<'_, [f32]>) -> Self {
        // XXX consider optimizing interval selection in this case.
        let mut header = PrimaryVectorHeader::from(VectorStats::from(query.as_ref()));
        let delta_inv = ((1 << PRIMARY_QUERY_BITS) - 1) as f32 / (header.upper - header.lower);
        let mut qvec = vec![0u8; packing::byte_len(query.len(), PRIMARY_QUERY_BITS)];
        let inst = InstructionSet::default();
        header.component_sum = match inst {
            InstructionSet::Scalar => scalar::primary_quantize_and_pack::<PRIMARY_QUERY_BITS>(
                query.as_ref(),
                header.lower,
                header.upper,
                delta_inv,
                &mut qvec,
            ),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => aarch64::primary_quantize_and_pack::<PRIMARY_QUERY_BITS>(
                query.as_ref(),
                header.lower,
                header.upper,
                delta_inv,
                &mut qvec,
            ),
            #[cfg(target_arch = "x86_64")]
            InstructionSet::Avx512 => unsafe {
                x86_64::primary_quantize_and_pack_avx512::<PRIMARY_QUERY_BITS>(
                    query.as_ref(),
                    header.lower,
                    header.upper,
                    delta_inv,
                    &mut qvec,
                )
            },
        };
        let terms = VectorTerms::from_primary::<PRIMARY_QUERY_BITS>(header);
        let l2_norm = header.l2_norm.into();

        Self {
            similarity,
            query: qvec,
            terms,
            l2_norm,
            inst,
        }
    }
}

impl<const B: usize> QueryVectorDistance for TurboPrimaryQueryDistance<B> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = TurboPrimaryVector::<B>::new(vector).expect("valid primary vector");
        // XXX inline this method.
        let dot = dot_unnormalized_uint8_asymmetric(self.inst, &self.query, self.terms, &vector);
        dot_unnormalized_to_distance(
            self.similarity,
            dot.into(),
            (self.l2_norm, vector.l2_norm()),
        )
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
                terms: VectorTerms::from_primary::<B>(primary_header),
                data: primary_vector,
            },
            residual: EncodedVector {
                terms: VectorTerms::from_residual(residual_header),
                data: residual_vector,
            },
            l2_norm: primary_header.l2_norm,
        })
    }

    fn dim(&self) -> usize {
        self.residual.data.len()
    }

    #[allow(dead_code)] // XXX
    fn iter(&self) -> impl ExactSizeIterator<Item = (u8, u8)> {
        packing::TurboUnpacker::<B>::new(self.primary.data).zip(packing::TurboUnpacker::<
            RESIDUAL_BITS,
        >::new(self.residual.data))
    }

    #[allow(dead_code)] // XXX
    fn l2_norm(&self) -> f64 {
        self.l2_norm.into()
    }

    #[allow(dead_code)] // XXX
    fn split_tail(&self, _dim: usize) -> (usize, Self, Self) {
        /*
        let tail_dim = dim & !(packing::block_dim(B) - 1);
        let (head_data, tail_data) = self.rep.data.split_at(packing::byte_len(tail_dim, B));
        (
            tail_dim,
            Self {
                rep: EncodedVector {
                    terms: self.rep.terms,
                    data: head_data,
                },
                l2_norm: self.l2_norm,
            },
            Self {
                rep: EncodedVector {
                    terms: self.rep.terms,
                    data: tail_data,
                },
                l2_norm: self.l2_norm,
            },
        )
        */
        todo!("XXX")
    }
}

// XXX use struct for primary as well
struct VectorEncodeTerms {
    lower: f32,
    upper: f32,
    delta_inv: f32,
}

impl VectorEncodeTerms {
    fn from_primary<const B: usize>(primary: &PrimaryVectorHeader) -> Self {
        let delta_inv = ((1 << B) - 1) as f32 / (primary.upper - primary.lower);
        Self {
            lower: primary.lower,
            upper: primary.upper,
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

#[allow(dead_code)] // XXX
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

        let (primary_header_bytes, vector_bytes) =
            PrimaryVectorHeader::split_output_buf(out).unwrap();
        let (residual_header_bytes, vector_bytes) =
            ResidualVectorHeader::split_output_buf(vector_bytes).unwrap();
        let split = packing::two_vector_split(vector_bytes.len(), B, RESIDUAL_BITS);
        let (primary, residual) = vector_bytes.split_at_mut(split);
        let mut residual_header = ResidualVectorHeader {
            magnitude: residual_magnitude,
            component_sum: 0,
        };

        // XXX arch impls
        (primary_header.component_sum, residual_header.component_sum) =
            scalar::residual_quantize_and_pack::<B>(
                vector,
                VectorEncodeTerms::from_primary::<B>(&primary_header),
                VectorEncodeTerms::from_residual(residual_magnitude),
                (primary_header.upper - primary_header.lower) / ((1 << B) - 1) as f32,
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
        // XXX arch impls
        scalar::residual_decode::<B>(&vector, out);
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        let len_no_corrective_terms =
            byte_len - PrimaryVectorHeader::LEN - ResidualVectorHeader::LEN;
        let split = packing::two_vector_split(len_no_corrective_terms, B, RESIDUAL_BITS);
        // Residual vector always uses a byte per dimension.
        len_no_corrective_terms - split
    }
}

#[allow(dead_code)] // XXX
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

        /*
        // XXX arch impls
        let component_dot = scalar::residual_dot_unnormalized::<B>(
            (&query.primary.data, &query.residual.data),
            (&doc.primary.data, &doc.residual.data),
        );
        let dot = component_dot.compute_dot(
            query.dim(),
            (&query.primary.terms, &query.residual.terms),
            (&doc.primary.terms, &doc.residual.terms),
        );
        */
        let mut qd = vec![0.0f32; query.dim()];
        let mut dd = vec![0.0f32; query.dim()];
        scalar::residual_decode::<B>(&query, &mut qd);
        scalar::residual_decode::<B>(&doc, &mut dd);
        let dot: f32 = qd.iter().zip(dd.iter()).map(|(q, d)| q * d).sum();
        dot_unnormalized_to_distance(self.0, dot.into(), (query.l2_norm(), doc.l2_norm()))
    }
}

pub struct TurboResidualQueryDistance<const B: usize> {
    similarity: VectorSimilarity,

    primary_vector: Vec<u8>,
    primary_terms: VectorTerms,
    residual_vector: Vec<u8>,
    residual_terms: VectorTerms,
    l2_norm: f64,

    #[allow(dead_code)] // XXX
    inst: InstructionSet,
}

impl<const B: usize> TurboResidualQueryDistance<B> {
    pub fn new(similarity: VectorSimilarity, query: Cow<'_, [f32]>) -> Self {
        // XXX refactor to share with residual coder.
        let stats = VectorStats::from(query.as_ref());
        let mut primary_header = PrimaryVectorHeader::from(stats);
        // NB: this interval optimization reduces loss for the primary vector, but this loss
        // reduction can make the residual vector more lossy if we derive the delta from the primary
        // interval as described in the LVQ paper. Compute and store a residual delta that is large
        // enough to encode both the min and max value in the vector.
        // TODO: investigate interval optimization on the derived residual vector.
        let interval = optimize_interval(query.as_ref(), &stats, B);
        // For the residual interval choose the maximum based on primary delta, or the min/max
        // values we may need to encode based on the gap between the initial and optimized interval.
        let residual_magnitude = [
            (interval.1 - interval.0) / RESIDUAL_MAX,
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

        // XXX arch impls
        // XXX consider encoding an 8-bit primary vector in all cases.
        let mut primary_vector = vec![0u8; packing::byte_len(query.len(), B)];
        let mut residual_vector = vec![0u8; query.len()];
        (primary_header.component_sum, residual_header.component_sum) =
            scalar::residual_quantize_and_pack::<B>(
                query.as_ref(),
                VectorEncodeTerms::from_primary::<B>(&primary_header),
                VectorEncodeTerms::from_residual(residual_magnitude),
                (primary_header.upper - primary_header.lower) / ((1 << B) - 1) as f32,
                &mut primary_vector,
                &mut residual_vector,
            );

        let primary_terms = VectorTerms::from_primary::<B>(primary_header);
        let residual_terms = VectorTerms::from_residual(residual_header);
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
        let component_dot = scalar::residual_dot_unnormalized::<B>(
            (&self.primary_vector, &self.residual_vector),
            (&vector.primary.data, &vector.residual.data),
        );
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

    /// Pack the contents of `it` into `out`.
    /// REQUIRES: B must be in 1..=8 and B % 8 == 0
    pub fn pack_iter<const B: usize>(it: impl ExactSizeIterator<Item = u8>, out: &mut [u8]) {
        let dims_per_byte = 8 / B;
        for (i, q) in it.enumerate() {
            out[i / dims_per_byte] |= q << ((i % dims_per_byte) * B);
        }
    }

    /// Pack the contents of `it` into `primary` and `residual`.
    /// REQUIRES: B1 must be in 1..=8 and B1 % 8 == 0
    /// REQUIRES: B2 must be in 1..=8 and B2 % 8 == 0
    pub fn pack_iter2<const B1: usize, const B2: usize>(
        it: impl ExactSizeIterator<Item = (u8, u8)>,
        primary: &mut [u8],
        residual: &mut [u8],
    ) {
        let dims_per_byte1 = 8 / B1;
        let dims_per_byte2 = 8 / B2;
        for (i, (q, r)) in it.enumerate() {
            primary[i / dims_per_byte1] |= q << ((i % dims_per_byte1) * B1);
            residual[i / dims_per_byte2] |= r << ((i % dims_per_byte2) * B2);
        }
    }
    pub struct UnpackIter<'a, const B: usize> {
        inner: std::slice::Iter<'a, u8>,
        buf: u8,
        nbuf: usize,
    }

    impl<'a, const B: usize> UnpackIter<'a, B> {
        const MASK: u8 = u8::MAX >> (8 - B);
        const BIT_CHECK: () = { assert!(B == 1 || B == 4 || B == 8) };

        fn new(packed: &'a [u8]) -> Self {
            #[allow(clippy::let_unit_value)]
            let _ = Self::BIT_CHECK;
            Self {
                inner: packed.iter(),
                buf: 0,
                nbuf: 0,
            }
        }
    }

    impl<'a, const B: usize> Iterator for UnpackIter<'a, B> {
        type Item = u8;

        fn next(&mut self) -> Option<Self::Item> {
            if B == 8 {
                return self.inner.next().copied();
            }

            if self.nbuf == 0 {
                self.buf = *self.inner.next()?;
                self.nbuf = 8;
            }

            let v = self.buf & Self::MASK;
            self.buf >>= B;
            self.nbuf -= B;
            Some(v)
        }

        fn nth(&mut self, n: usize) -> Option<Self::Item> {
            let mut skip_bits = n * B;
            if skip_bits > self.nbuf {
                skip_bits -= self.nbuf;
                self.nbuf = 0;
                let skip_bytes = skip_bits / 8;
                skip_bits -= skip_bytes * 8;
                self.buf = *self.inner.nth(skip_bytes)?;
            }

            self.buf >>= skip_bits;
            self.next()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = (self.nbuf + 8 * self.inner.len()) / B;
            (len, Some(len))
        }
    }

    impl<'a, const B: usize> ExactSizeIterator for UnpackIter<'a, B> {}
    impl<'a, const B: usize> FusedIterator for UnpackIter<'a, B> {}

    /// Iterate over the value in each dimension where each dimension is `B` bits in `packed`
    /// REQUIRES: B must be in 1..=8 and B % 8 == 0
    pub fn unpack_iter<const B: usize>(packed: &[u8]) -> impl ExactSizeIterator<Item = u8> + '_ {
        UnpackIter::<B>::new(packed)
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

    use crate::lvq::{ResidualVectorHeader, TurboPrimaryCoder, TurboResidualCoder, VectorStats};

    use super::{
        F32VectorCoder, PrimaryVector, PrimaryVectorCoder, PrimaryVectorHeader, TwoLevelVector,
        TwoLevelVectorCoder,
    };

    impl AbsDiffEq for PrimaryVectorHeader {
        type Epsilon = f32;

        fn default_epsilon() -> Self::Epsilon {
            0.00001
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            abs_diff_eq!(self.l2_norm, other.l2_norm, epsilon = epsilon)
                && abs_diff_eq!(self.lower, other.lower, epsilon = epsilon)
                && abs_diff_eq!(self.upper, other.upper, epsilon = epsilon)
        }
    }

    impl AbsDiffEq for ResidualVectorHeader {
        type Epsilon = f32;

        fn default_epsilon() -> Self::Epsilon {
            0.00001
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            abs_diff_eq!(self.magnitude, other.magnitude, epsilon = epsilon)
                && abs_diff_eq!(self.component_sum, other.component_sum)
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
        }
    }

    // This test vector contains randomly generated numbers in [-1,1] but is not l2 normalized.
    // It has 19 elements -- long enough to trigger SIMD optimizations but with some remainder to
    // test scalar tail paths.
    const TEST_VECTOR: [f32; 19] = [
        -0.921, -0.061, 0.659, 0.67, 0.573, 0.431, 0.646, 0.001, -0.2, -0.428, 0.73, -0.704,
        -0.273, 0.539, -0.731, 0.436, 0.913, 0.694, 0.202,
    ];

    fn unpack_primary<const B: usize>(vector: &PrimaryVector<'_, B>) -> Vec<u8> {
        super::packing::unpack_iter::<B>(vector.v.data).collect()
    }

    fn unpack_residual<const B1: usize, const B2: usize>(
        vector: &TwoLevelVector<'_, B1, B2>,
    ) -> Vec<u8> {
        super::packing::unpack_iter::<B2>(vector.residual.data).collect()
    }

    #[test]
    fn lvq1_1() {
        let encoded = PrimaryVectorCoder::<1>::default().encode(&TEST_VECTOR);
        let lvq = PrimaryVector::<1>::new(&encoded).expect("readable");
        assert_eq!(
            &unpack_primary(&lvq)[..TEST_VECTOR.len()],
            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1]
        );
        assert_abs_diff_eq!(
            PrimaryVectorHeader::deserialize(&encoded).unwrap().0,
            PrimaryVectorHeader {
                l2_norm: 2.5226507,
                lower: -0.49564388,
                upper: 0.70561373,
                component_sum: 11,
            }
        );
        // NB: vector dimensionality is not a multiple of 8 so we're producting extra dimensions.
        assert_abs_diff_eq!(
            lvq.f32_iter()
                .take(TEST_VECTOR.len())
                .collect::<Vec<_>>()
                .as_ref(),
            [
                -0.49564388,
                -0.49564388,
                0.70561373,
                0.70561373,
                0.70561373,
                0.70561373,
                0.70561373,
                -0.49564388,
                -0.49564388,
                -0.49564388,
                0.70561373,
                -0.49564388,
                -0.49564388,
                0.70561373,
                -0.49564388,
                0.70561373,
                0.70561373,
                0.70561373,
                0.70561373
            ]
            .as_ref(),
            epsilon = 0.00001
        );
    }

    #[test]
    fn lvq1_4() {
        let encoded = PrimaryVectorCoder::<4>::default().encode(&TEST_VECTOR);
        let lvq = PrimaryVector::<4>::new(&encoded).expect("readable");
        assert_eq!(
            &unpack_primary(&lvq)[..TEST_VECTOR.len()],
            [
                0, 7, 13, 13, 12, 11, 13, 8, 6, 4, 14, 2, 5, 12, 2, 11, 15, 13, 9
            ]
        );
        assert_abs_diff_eq!(
            PrimaryVectorHeader::deserialize(&encoded).unwrap().0,
            PrimaryVectorHeader {
                l2_norm: 2.5226507,
                lower: -0.93474734,
                upper: 0.9131211,
                component_sum: 170,
            }
        );
        assert_abs_diff_eq!(
            lvq.f32_iter()
                .take(TEST_VECTOR.len())
                .collect::<Vec<_>>()
                .as_ref(),
            [
                -0.93474734,
                -0.072408736,
                0.6667386,
                0.6667386,
                0.5435474,
                0.42035615,
                0.6667386,
                0.0507825,
                -0.19559997,
                -0.44198242,
                0.78992987,
                -0.68836486,
                -0.3187912,
                0.5435474,
                -0.68836486,
                0.42035615,
                0.9131211,
                0.6667386,
                0.17397368
            ]
            .as_ref(),
            epsilon = 0.0001,
        );
    }

    #[test]
    fn lvq1_8() {
        let encoded = PrimaryVectorCoder::<8>::default().encode(&TEST_VECTOR);
        let lvq = PrimaryVector::<8>::new(&encoded).expect("readable");
        assert_eq!(
            unpack_primary(&lvq),
            [
                0, 120, 220, 221, 208, 188, 218, 128, 100, 69, 230, 30, 90, 203, 26, 189, 255, 225,
                156
            ]
        );
        assert_abs_diff_eq!(
            PrimaryVectorHeader::deserialize(&encoded).unwrap().0,
            PrimaryVectorHeader {
                l2_norm: 2.5226507,
                lower: -0.92000645,
                upper: 0.91146713,
                component_sum: 2876,
            }
        );
        assert_abs_diff_eq!(
            lvq.f32_iter().collect::<Vec<_>>().as_ref(),
            [
                -0.92000645,
                -0.058136523,
                0.66008836,
                0.6672706,
                0.57390136,
                0.43025643,
                0.6457239,
                -0.0006785393,
                -0.20178151,
                -0.42443126,
                0.7319109,
                -0.70453894,
                -0.27360404,
                0.53799015,
                -0.73326796,
                0.43743867,
                0.91146713,
                0.6959997,
                0.20042449
            ]
            .as_ref(),
            epsilon = 0.0001
        );
    }

    #[test]
    fn lvq2_1_8() {
        let encoded = TwoLevelVectorCoder::<1, 8>::default().encode(&TEST_VECTOR);
        let lvq = TwoLevelVector::<1, 8>::new(&encoded).expect("readable");
        assert_eq!(
            &unpack_primary(&lvq.primary)[..TEST_VECTOR.len()],
            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1]
        );
        assert_abs_diff_eq!(
            unpack_residual(&lvq).as_ref(),
            [
                37, 220, 118, 120, 99, 69, 115, 233, 190, 142, 133, 83, 175, 92, 78, 70, 172, 125,
                21
            ]
            .as_ref(),
            epsilon = 1
        );
        let (primary_header, vector_bytes) = PrimaryVectorHeader::deserialize(&encoded).unwrap();
        assert_abs_diff_eq!(
            primary_header,
            PrimaryVectorHeader {
                l2_norm: 2.5226507,
                lower: -0.49564388,
                upper: 0.70561373,
                component_sum: 11,
            }
        );
        let (residual_header, _) = ResidualVectorHeader::deserialize(&vector_bytes).unwrap();
        assert_abs_diff_eq!(
            residual_header,
            ResidualVectorHeader {
                magnitude: 1.2012575,
                component_sum: 2292,
            }
        );
        assert_abs_diff_eq!(
            lvq.f32_iter().collect::<Vec<_>>().as_ref(),
            [
                -0.9219725,
                -0.05989358,
                0.660861,
                0.6702826,
                0.5713555,
                0.4300311,
                0.6467286,
                0.0013469756,
                -0.20121804,
                -0.42733708,
                0.7315232,
                -0.7052751,
                -0.27188024,
                0.5383798,
                -0.72882915,
                0.4347419,
                0.91524494,
                0.6938367,
                0.20391202
            ]
            .as_ref(),
            epsilon = 0.01
        );
    }

    #[test]
    fn lvq2_4_4() {
        let encoded = TwoLevelVectorCoder::<4, 4>::default().encode(&TEST_VECTOR);
        let lvq = TwoLevelVector::<4, 4>::new(&encoded).expect("readable");
        assert_abs_diff_eq!(
            &unpack_primary(&lvq.primary)[..TEST_VECTOR.len()],
            [
                0, 7, 13, 13, 12, 11, 13, 8, 6, 4, 14, 2, 5, 12, 2, 11, 15, 13, 9
            ]
            .as_ref(),
            epsilon = 1,
        );
        assert_abs_diff_eq!(
            &unpack_residual(&lvq)[..TEST_VECTOR.len()],
            [9, 9, 7, 8, 11, 9, 5, 1, 7, 9, 0, 6, 13, 7, 2, 9, 7, 11, 11].as_ref(),
            epsilon = 1
        );
        assert_abs_diff_eq!(
            PrimaryVectorHeader::deserialize(&encoded).unwrap().0,
            PrimaryVectorHeader {
                l2_norm: 2.5226507,
                lower: -0.93474734,
                upper: 0.9131211,
                component_sum: 170,
            }
        );
        assert_abs_diff_eq!(
            lvq.f32_iter().collect::<Vec<_>>().as_ref(),
            [
                -0.9224282,
                -0.060089614,
                0.6626322,
                0.67084503,
                0.57229203,
                0.43267527,
                0.64620674,
                -0.0026003644,
                -0.19970635,
                -0.4296633,
                0.72833425,
                -0.700684,
                -0.27362108,
                0.539441,
                -0.733535,
                0.43267527,
                0.9090147,
                0.69548327,
                0.2027183,
                -0.99634296
            ]
            .as_ref(),
            epsilon = 0.0001,
        );
    }

    #[test]
    fn lvq2_4_8() {
        let encoded = TwoLevelVectorCoder::<4, 8>::default().encode(&TEST_VECTOR);
        let lvq = TwoLevelVector::<4, 8>::new(&encoded).expect("readable");
        assert_abs_diff_eq!(
            &unpack_primary(&lvq.primary)[..TEST_VECTOR.len()],
            [
                0, 7, 13, 13, 12, 11, 13, 8, 6, 4, 14, 2, 5, 12, 2, 11, 15, 13, 9
            ]
            .as_ref(),
            epsilon = 1,
        );
        assert_abs_diff_eq!(
            unpack_residual(&lvq).as_ref(),
            [
                156, 151, 111, 134, 188, 150, 85, 24, 118, 156, 3, 95, 222, 118, 39, 160, 127, 184,
                186
            ]
            .as_ref(),
            epsilon = 1
        );
        assert_abs_diff_eq!(
            PrimaryVectorHeader::deserialize(&encoded).unwrap().0,
            PrimaryVectorHeader {
                l2_norm: 2.5226507,
                lower: -0.93474734,
                upper: 0.9131211,
                component_sum: 170,
            }
        );
        assert_abs_diff_eq!(
            lvq.f32_iter().collect::<Vec<_>>().as_ref(),
            [
                -0.9209789,
                -0.06105582,
                0.65876746,
                0.6698788,
                0.5727751,
                0.43122596,
                0.64620674,
                0.0007813573,
                -0.20018944,
                -0.42821398,
                0.72978354,
                -0.7040657,
                -0.273138,
                0.5389579,
                -0.73111945,
                0.436057,
                0.9128795,
                0.6940339,
                0.20223519
            ]
            .as_ref(),
            epsilon = 0.0001,
        );
    }

    #[test]
    fn lvq2_8_8() {
        let encoded = TwoLevelVectorCoder::<8, 8>::default().encode(&TEST_VECTOR);
        let lvq = TwoLevelVector::<8, 8>::new(&encoded).expect("readable");
        assert_abs_diff_eq!(
            &unpack_primary(&lvq.primary)[..TEST_VECTOR.len()],
            [
                0, 120, 220, 221, 208, 188, 218, 128, 100, 69, 230, 30, 90, 203, 26, 189, 255, 225,
                156
            ]
            .as_ref(),
            epsilon = 1,
        );
        assert_abs_diff_eq!(
            unpack_residual(&lvq).as_ref(),
            [
                92, 26, 89, 224, 95, 154, 137, 187, 191, 1, 60, 147, 149, 163, 208, 76, 182, 57,
                183
            ]
            .as_ref(),
            epsilon = 1
        );
        assert_abs_diff_eq!(
            PrimaryVectorHeader::deserialize(&encoded).unwrap().0,
            PrimaryVectorHeader {
                l2_norm: 2.5226507,
                lower: -0.92000645,
                upper: 0.91146713,
                component_sum: 2876,
            }
        );
        assert_abs_diff_eq!(
            lvq.f32_iter().collect::<Vec<_>>().as_ref(),
            [
                -0.9210063,
                -0.06099534,
                0.659004,
                0.6699886,
                0.57298595,
                0.43100283,
                0.64599144,
                0.0009973187,
                -0.19999298,
                -0.42799422,
                0.7300097,
                -0.7039897,
                -0.27299848,
                0.53899,
                -0.7310006,
                0.43598813,
                0.9130022,
                0.694014,
                0.20198768
            ]
            .as_ref(),
            epsilon = 0.0001
        );
    }

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
                l2_norm: 2.5226507,
                lower: -0.49564388,
                upper: 0.70561373,
                component_sum: 11,
            }
        );
        let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
        coder.decode_to(&encoded, &mut decoded);
        assert_abs_diff_eq!(
            decoded.as_ref(),
            [
                -0.49564388,
                -0.49564388,
                0.70561373,
                0.70561373,
                0.70561373,
                0.70561373,
                0.70561373,
                -0.49564388,
                -0.49564388,
                -0.49564388,
                0.70561373,
                -0.49564388,
                -0.49564388,
                0.70561373,
                -0.49564388,
                0.70561373,
                0.70561373,
                0.70561373,
                0.70561373
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
                l2_norm: 2.5226507,
                lower: -0.6709247,
                upper: 0.8410188,
                component_sum: 32,
            }
        );
        let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
        coder.decode_to(&encoded, &mut decoded);
        assert_abs_diff_eq!(
            decoded.as_ref(),
            [
                -0.6709247,
                -0.16694355,
                0.8410188,
                0.8410188,
                0.33703762,
                0.33703762,
                0.8410188,
                -0.16694355,
                -0.16694355,
                -0.6709247,
                0.8410188,
                -0.6709247,
                -0.16694355,
                0.33703762,
                -0.6709247,
                0.33703762,
                0.8410188,
                0.8410188,
                0.33703762
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
                l2_norm: 2.5226507,
                lower: -0.93474734,
                upper: 0.9131211,
                component_sum: 170,
            }
        );
        let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
        coder.decode_to(&encoded, &mut decoded);
        assert_abs_diff_eq!(
            decoded.as_ref(),
            [
                -0.93474734,
                -0.072408736,
                0.6667386,
                0.6667386,
                0.5435474,
                0.42035615,
                0.6667386,
                0.0507825,
                -0.19559997,
                -0.44198242,
                0.78992987,
                -0.68836486,
                -0.3187912,
                0.5435474,
                -0.68836486,
                0.42035615,
                0.9131211,
                0.6667386,
                0.17397368
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
                l2_norm: 2.5226507,
                lower: -0.92000645,
                upper: 0.91146713,
                component_sum: 2876,
            }
        );
        let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
        coder.decode_to(&encoded, &mut decoded);
        assert_abs_diff_eq!(
            decoded.as_ref(),
            [
                -0.92000645,
                -0.058136523,
                0.66008836,
                0.6672706,
                0.57390136,
                0.43025643,
                0.6457239,
                -0.0006785393,
                -0.20178151,
                -0.42443126,
                0.7319109,
                -0.70453894,
                -0.27360404,
                0.53799015,
                -0.73326796,
                0.43743867,
                0.91146713,
                0.6959997,
                0.20042449
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
                l2_norm: 2.5226507,
                lower: -0.49564388,
                upper: 0.70561373,
                component_sum: 11,
            }
        );
        let (residual_header, _) = ResidualVectorHeader::deserialize(&vector_bytes).unwrap();
        assert_abs_diff_eq!(
            residual_header,
            ResidualVectorHeader {
                magnitude: 1.2012575,
                component_sum: 2292,
            }
        );
        let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
        coder.decode_to(&encoded, &mut decoded);
        assert_abs_diff_eq!(
            decoded.as_ref(),
            [
                -0.9219725,
                -0.05989358,
                0.660861,
                0.6702826,
                0.5713555,
                0.4300311,
                0.6467286,
                0.0013469756,
                -0.20121804,
                -0.42733708,
                0.7315232,
                -0.7052751,
                -0.27188024,
                0.5383798,
                -0.72882915,
                0.4347419,
                0.91524494,
                0.6938367,
                0.20391202
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
                l2_norm: 2.5226507,
                lower: -0.6709247,
                upper: 0.8410188,
                component_sum: 32,
            }
        );
        let (residual_header, _) = ResidualVectorHeader::deserialize(&vector_bytes).unwrap();
        assert_abs_diff_eq!(
            residual_header,
            ResidualVectorHeader {
                magnitude: 0.5039812,
                component_sum: 2319,
            }
        );
        let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
        coder.decode_to(&encoded, &mut decoded);
        assert_abs_diff_eq!(
            decoded.as_ref(),
            [
                -0.9209389,
                -0.06120634,
                0.6582021,
                0.67006046,
                0.57321703,
                0.43091646,
                0.6463437,
                6.195903e-5,
                -0.19955412,
                -0.42881614,
                0.72935236,
                -0.70353526,
                -0.2726808,
                0.53961825,
                -0.7312048,
                0.43684563,
                0.9131573,
                0.6937772,
                0.20165443
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
                l2_norm: 2.5226507,
                lower: -0.93474734,
                upper: 0.9131211,
                component_sum: 170,
            }
        );
        let (residual_header, _) = ResidualVectorHeader::deserialize(&vector_bytes).unwrap();
        assert_abs_diff_eq!(
            residual_header,
            ResidualVectorHeader {
                magnitude: 0.123191215,
                component_sum: 2407,
            }
        );
        let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
        coder.decode_to(&encoded, &mut decoded);
        assert_abs_diff_eq!(
            decoded.as_ref(),
            [
                -0.9209789,
                -0.06105582,
                0.65876746,
                0.6698788,
                0.5727751,
                0.43122596,
                0.64620674,
                0.0007813573,
                -0.20018944,
                -0.42821398,
                0.72978354,
                -0.7040657,
                -0.273138,
                0.5389579,
                -0.73111945,
                0.436057,
                0.9128795,
                0.6940339,
                0.20223519
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
                l2_norm: 2.5226507,
                lower: -0.92000645,
                upper: 0.91146713,
                component_sum: 2876,
            }
        );
        let (residual_header, _) = ResidualVectorHeader::deserialize(&vector_bytes).unwrap();
        assert_abs_diff_eq!(
            residual_header,
            ResidualVectorHeader {
                magnitude: 0.0071822493,
                component_sum: 2422,
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
}
