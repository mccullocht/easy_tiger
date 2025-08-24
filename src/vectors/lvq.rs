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

use std::{borrow::Cow, iter::FusedIterator};

use crate::vectors::{F32VectorCoder, QueryVectorDistance, VectorDistance};

#[derive(Debug, Clone, Copy, Default)]
struct VectorStats {
    min: f32,
    max: f32,
    mean: f64,
    std_dev: f64,
    l2_norm_sq: f64,
}

impl From<&[f32]> for VectorStats {
    fn from(value: &[f32]) -> Self {
        if value.is_empty() {
            return VectorStats {
                l2_norm_sq: 1.0,
                ..Default::default()
            };
        }

        let (min, max, mean, variance, dot) = value.iter().copied().enumerate().fold(
            (f32::MAX, f32::MIN, 0.0, 0.0, 0.0),
            |mut stats, (i, x)| {
                stats.0 = x.min(stats.0);
                stats.1 = x.max(stats.1);
                let x: f64 = x.into();
                let delta = x - stats.2;
                stats.2 += delta / (i + 1) as f64;
                stats.3 += delta * (x - stats.2);
                stats.4 += x * x;
                stats
            },
        );
        Self {
            min,
            max,
            mean,
            std_dev: (variance / value.len() as f64).sqrt(),
            l2_norm_sq: dot,
        }
    }
}

/// Header for an LVQ vector.
///
/// Along with the bit configuration this carries enough metadata to transform a quantized vector
/// value stream back to an f32 representation or compute angular or l2 distance from another
/// vector.
#[derive(Debug, Copy, Clone, PartialEq)]
struct VectorHeader {
    l2_norm: f32,
    lower: f32,
    upper: f32,
    component_sum: u32,
}

impl VectorHeader {
    /// Encoded buffer size.
    const LEN: usize = 16;

    fn split_output_buf(buf: &mut [u8]) -> Option<(&mut [u8], &mut [u8])> {
        buf.split_at_mut_checked(Self::LEN)
    }

    fn serialize(&self, header_bytes: &mut [u8]) {
        let header = header_bytes.as_chunks_mut::<4>().0;
        header[0] = self.l2_norm.to_le_bytes();
        header[1] = self.lower.to_le_bytes();
        header[2] = self.upper.to_le_bytes();
        header[3] = self.component_sum.to_le_bytes();
    }

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

impl From<VectorStats> for VectorHeader {
    fn from(value: VectorStats) -> Self {
        VectorHeader {
            l2_norm: value.l2_norm_sq.sqrt() as f32,
            lower: value.min,
            upper: value.max,
            component_sum: 0,
        }
    }
}

const MINIMUM_MSE_GRID: [(f64, f64); 8] = [
    (-0.798, 0.798),
    (-1.493, 1.493),
    (-2.051, 2.051),
    (-2.514, 2.514),
    (-2.916, 2.916),
    (-3.278, 3.278),
    (-3.611, 3.611),
    (-3.922, 3.922),
];

const LAMBDA: f64 = 0.1;

fn optimize_interval(vector: &[f32], stats: &VectorStats, bits: usize) -> (f32, f32) {
    let norm_sq: f64 = stats.l2_norm_sq;
    let mut loss = compute_loss(vector, (stats.min, stats.max), norm_sq, bits);

    let scale = (1.0 - LAMBDA) / norm_sq;
    let mut lower: f64 = (MINIMUM_MSE_GRID[bits - 1].0 * stats.std_dev + stats.mean)
        .clamp(stats.min.into(), stats.max.into());
    let mut upper: f64 = (MINIMUM_MSE_GRID[bits - 1].1 * stats.std_dev + stats.mean)
        .clamp(stats.min.into(), stats.max.into());

    let points_incl = ((1 << bits) - 1) as f64;
    for _ in 0..5 {
        let step_inv = points_incl / (upper - lower);
        // calculate the grid points for coordinate descent.
        let mut daa = 0.0;
        let mut dab = 0.0;
        let mut dbb = 0.0;
        let mut dax = 0.0;
        let mut dbx = 0.0;
        for xi in vector.iter().copied().map(f64::from) {
            let k = ((xi.clamp(lower, upper) - lower) * step_inv).round();
            let s = k / points_incl;
            daa += (1.0 - s) * (1.0 - s);
            dab += (1.0 - s) * s;
            dbb += s * s;
            dax += xi * (1.0 - s);
            dbx += xi * s;
        }
        let m0 = scale * dax * dax + LAMBDA * daa;
        let m1 = scale * dax * dbx + LAMBDA * dab;
        let m2 = scale * dbx * dbx + LAMBDA * dbb;
        let det = m0 * m2 - m1 * m1;
        // if the determinant is zero we can't update the interval
        if det == 0.0 {
            break;
        }

        let lower_candidate = (m2 * dax - m1 * dbx) / det;
        let upper_candidate = (m0 * dbx - m1 * dax) / det;
        if (lower - lower_candidate).abs() < 1e-8 && (upper - upper_candidate).abs() < 1e-8 {
            break;
        }
        let loss_candidate = compute_loss(
            vector,
            (lower_candidate as f32, upper_candidate as f32),
            norm_sq,
            bits,
        );
        if loss_candidate > loss {
            break;
        }
        lower = lower_candidate;
        upper = upper_candidate;
        loss = loss_candidate;
    }
    (lower as f32, upper as f32)
}

#[cfg(target_arch = "aarch64")]
use aarch64::compute_loss;
#[cfg(not(target_arch = "aarch64"))]
use scalar::compute_loss;

struct PrimaryQuantizer<'a> {
    it: std::slice::Iter<'a, f32>,
    header: VectorHeader,
    delta: f32,
}

impl<'a> PrimaryQuantizer<'a> {
    fn new(v: &'a [f32], bits: usize) -> Self {
        let stats = VectorStats::from(v);
        let mut header = VectorHeader::from(stats);
        (header.lower, header.upper) = optimize_interval(v, &stats, bits);
        let delta = (header.upper - header.lower) / ((1 << bits) - 1) as f32;
        Self {
            it: v.iter(),
            header,
            delta,
        }
    }

    fn into_header(self) -> VectorHeader {
        assert_eq!(self.it.len(), 0);
        self.header
    }
}

impl Iterator for PrimaryQuantizer<'_> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|x| {
            let q = ((x.clamp(self.header.lower, self.header.upper) - self.header.lower)
                / self.delta)
                .round() as u8;
            self.header.component_sum += u32::from(q);
            q
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl ExactSizeIterator for PrimaryQuantizer<'_> {}

impl FusedIterator for PrimaryQuantizer<'_> {}

struct TwoLevelQuantizer<'a> {
    primary_it: PrimaryQuantizer<'a>,
    f32_it: std::slice::Iter<'a, f32>,
    lower: f32,
    delta: f32,
}

impl<'a> TwoLevelQuantizer<'a> {
    pub fn new(v: &'a [f32], bits1: usize, bits2: usize) -> Self {
        let primary_it = PrimaryQuantizer::new(v, bits1);
        let f32_it = v.iter();
        let lower = -primary_it.delta / 2.0;
        let delta = primary_it.delta / ((1 << bits2) - 1) as f32;
        Self {
            primary_it,
            f32_it,
            lower,
            delta,
        }
    }

    fn into_header(self) -> VectorHeader {
        self.primary_it.into_header()
    }
}

impl Iterator for TwoLevelQuantizer<'_> {
    type Item = (u8, u8);

    fn next(&mut self) -> Option<Self::Item> {
        let x = self.f32_it.next()?;
        let q = self.primary_it.next()?;
        let res = x.clamp(self.primary_it.header.lower, self.primary_it.header.upper)
            - ((q as f32 * self.primary_it.delta) + self.primary_it.header.lower);
        let r = ((res - self.lower) / self.delta).round() as u8;
        Some((q, r))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.primary_it.size_hint()
    }
}

impl ExactSizeIterator for TwoLevelQuantizer<'_> {}

impl FusedIterator for TwoLevelQuantizer<'_> {}

/// An LVQ1 coded primary vector.
///
/// There may be a parallel residual vector that can be composed with this one to increase accuracy.
struct PrimaryVector<'a, const B: usize> {
    header: VectorHeader,
    delta: f32,
    vector: &'a [u8],
}

impl<'a, const B: usize> PrimaryVector<'a, B> {
    fn new(encoded: &'a [u8]) -> Option<Self> {
        let (header, vector) = VectorHeader::deserialize(encoded)?;
        Some(Self::with_header(header, vector))
    }

    fn with_header(header: VectorHeader, vector: &'a [u8]) -> Self {
        let delta = (header.upper - header.lower) / ((1 << B) - 1) as f32;
        Self {
            header,
            delta,
            vector,
        }
    }

    fn l2_norm(&self) -> f64 {
        self.header.l2_norm.into()
    }

    fn f32_iter(&self) -> impl Iterator<Item = f32> + '_ {
        packing::unpack_iter::<B>(self.vector).map(|q| q as f32 * self.delta + self.header.lower)
    }

    fn dot_unnormalized(&self, other: &Self) -> f64 {
        let dot_quantized = if B == 1 {
            self.vector
                .iter()
                .zip(other.vector.iter())
                .map(|(s, o)| (*s & *o).count_ones())
                .sum::<u32>()
        } else {
            packing::unpack_iter::<B>(self.vector)
                .zip(packing::unpack_iter::<B>(other.vector))
                .map(|(s, o)| s as u32 * o as u32)
                .sum::<u32>()
        };
        let sdelta = f64::from(self.delta);
        let slower = f64::from(self.header.lower);
        let odelta = f64::from(other.delta);
        let olower = f64::from(other.header.lower);
        dot_quantized as f64 * sdelta * odelta
            + self.header.component_sum as f64 * sdelta * olower
            + other.header.component_sum as f64 * odelta * slower
            + slower * olower * (self.vector.len() * 8).div_ceil(B) as f64
    }
}

struct TwoLevelVector<'a, const B1: usize, const B2: usize> {
    primary: PrimaryVector<'a, B1>,
    vector: &'a [u8],
    delta: f32,
    lower: f32,
}

impl<'a, const B1: usize, const B2: usize> TwoLevelVector<'a, B1, B2> {
    fn new(encoded: &'a [u8]) -> Option<Self> {
        let (header, vector) = VectorHeader::deserialize(encoded)?;
        let split = packing::two_vector_split(vector.len(), B1, B2);
        let (primary_vector, residual_vector) = vector.split_at(split);
        let primary = PrimaryVector::<'_, B1>::with_header(header, primary_vector);
        let delta = primary.delta / ((1 << B2) - 1) as f32;
        let lower = -primary.delta / 2.0;
        Some(Self {
            primary,
            vector: residual_vector,
            delta,
            lower,
        })
    }

    pub fn l2_norm(&self) -> f64 {
        self.primary.l2_norm()
    }

    fn f32_iter(&self) -> impl Iterator<Item = f32> + '_ {
        self.primary
            .f32_iter()
            .zip(
                packing::unpack_iter::<B2>(self.vector).map(|r| r as f32 * self.delta + self.lower),
            )
            .map(|(q, r)| q + r)
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct PrimaryVectorCoder<const B: usize>;

impl<const B: usize> F32VectorCoder for PrimaryVectorCoder<B> {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let mut it = PrimaryQuantizer::new(vector, B);
        let (header_bytes, vector_bytes) = VectorHeader::split_output_buf(out).unwrap();
        packing::pack_iter::<B>(it.by_ref(), vector_bytes);
        it.into_header().serialize(header_bytes);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        VectorHeader::LEN + packing::byte_len(dimensions, B)
    }

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        Some(PrimaryVector::<B>::new(encoded)?.f32_iter().collect())
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct TwoLevelVectorCoder<const B1: usize, const B2: usize>;

impl<const B1: usize, const B2: usize> F32VectorCoder for TwoLevelVectorCoder<B1, B2> {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let mut it = TwoLevelQuantizer::new(vector, B1, B2);
        let (header_bytes, vector_bytes) = VectorHeader::split_output_buf(out).unwrap();
        let split = packing::two_vector_split(vector_bytes.len(), B1, B2);
        let (primary, residual) = vector_bytes.split_at_mut(split);
        packing::pack_iter2::<B1, B2>(it.by_ref(), primary, residual);
        it.into_header().serialize(header_bytes);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        VectorHeader::LEN + packing::byte_len(dimensions, B1) + packing::byte_len(dimensions, B2)
    }

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        Some(TwoLevelVector::<B1, B2>::new(encoded)?.f32_iter().collect())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PrimaryDotProductDistance<const B: usize>;

impl<const B: usize> VectorDistance for PrimaryDotProductDistance<B> {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = PrimaryVector::<B>::new(query).unwrap();
        let doc = PrimaryVector::<B>::new(doc).unwrap();
        let dot = query.dot_unnormalized(&doc) / (query.l2_norm() * doc.l2_norm());
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Clone)]
pub struct PrimaryQueryDotProductDistance<'a, const B: usize>(Cow<'a, [f32]>);

impl<'a, const B: usize> PrimaryQueryDotProductDistance<'a, B> {
    pub fn new(query: Cow<'a, [f32]>) -> Self {
        Self(query)
    }
}

impl<const B: usize> QueryVectorDistance for PrimaryQueryDotProductDistance<'_, B> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = PrimaryVector::<B>::new(vector).unwrap();
        let dot = self
            .0
            .iter()
            .zip(vector.f32_iter())
            .map(|(q, d)| *q * d)
            .sum::<f32>() as f64;
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TwoLevelDotProductDistance<const B1: usize, const B2: usize>;

impl<const B1: usize, const B2: usize> VectorDistance for TwoLevelDotProductDistance<B1, B2> {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = TwoLevelVector::<B1, B2>::new(query).unwrap();
        let doc = TwoLevelVector::<B1, B2>::new(doc).unwrap();
        let dot = query
            .f32_iter()
            .zip(doc.f32_iter())
            .map(|(q, d)| q * d)
            .sum::<f32>() as f64
            / (query.l2_norm() * doc.l2_norm());
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Clone)]
pub struct TwoLevelQueryDotProductDistance<'a, const B1: usize, const B2: usize>(Cow<'a, [f32]>);

impl<'a, const B1: usize, const B2: usize> TwoLevelQueryDotProductDistance<'a, B1, B2> {
    pub fn new(query: Cow<'a, [f32]>) -> Self {
        Self(query)
    }
}

impl<const B1: usize, const B2: usize> QueryVectorDistance
    for TwoLevelQueryDotProductDistance<'_, B1, B2>
{
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = TwoLevelVector::<B1, B2>::new(vector).unwrap();
        let dot = self
            .0
            .iter()
            .zip(vector.f32_iter())
            .map(|(s, o)| *s * o)
            .sum::<f32>() as f64
            / vector.l2_norm();
        (-dot + 1.0) / 2.0
    }
}

mod packing {
    use std::iter::FusedIterator;

    /// The number of bytes required to pack `dimensions` with `bits` per entry.
    pub const fn byte_len(dimensions: usize, bits: usize) -> usize {
        dimensions.div_ceil(8 / bits)
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

    /// Iterate over the value in each dimension where each dimension is `B` bits in `packed`
    /// REQUIRES: B must be in 1..=8 and B % 8 == 0
    pub fn unpack_iter<const B: usize>(packed: &[u8]) -> impl FusedIterator<Item = u8> + '_ {
        let mask = ((1 << B) - 1) as u8;
        packed
            .iter()
            .flat_map(move |b| (0..8).step_by(B).map(move |i| (*b >> i) & mask))
    }
}

#[cfg(test)]
mod test {
    use crate::vectors::lvq::{
        F32VectorCoder, PrimaryVector, PrimaryVectorCoder, TwoLevelVector, TwoLevelVectorCoder,
        VectorHeader,
    };

    #[test]
    fn lvq1_1() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let encoded = PrimaryVectorCoder::<1>::default().encode(&vec);
        let lvq = PrimaryVector::<1>::new(&encoded).expect("readable");
        assert_eq!(lvq.vector, &[0b11100000, 0b11]);
        assert_eq!(
            lvq.header,
            VectorHeader {
                l2_norm: 0.92195445,
                lower: -0.38059703,
                upper: 0.25373134,
                component_sum: 5,
            }
        );
        // NB: vector dimensionality is not a multiple of 8 so we're producting extra dimensions.
        assert_eq!(
            lvq.f32_iter().take(10).collect::<Vec<_>>(),
            &[
                -0.38059703,
                -0.38059703,
                -0.38059703,
                -0.38059703,
                -0.38059703,
                0.25373134,
                0.25373134,
                0.25373134,
                0.25373134,
                0.25373134
            ]
        );
    }

    #[test]
    fn lvq1_4() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let encoded = PrimaryVectorCoder::<4>::default().encode(&vec);
        let lvq = PrimaryVector::<4>::new(&encoded).expect("readable");
        assert_eq!(lvq.vector, &[0x20, 0x53, 0x87, 0xca, 0xfd]);
        assert_eq!(
            lvq.header,
            VectorHeader {
                l2_norm: 0.92195445,
                lower: -0.50325716,
                upper: 0.40300408,
                component_sum: 75,
            }
        );
        assert_eq!(
            lvq.f32_iter().collect::<Vec<_>>(),
            &[
                -0.50325716,
                -0.38242233,
                -0.3220049,
                -0.20117009,
                -0.08033526,
                -0.019917846,
                0.10091698,
                0.22175181,
                0.28216922,
                0.40300405
            ]
        );
    }

    #[test]
    fn lvq1_8() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let encoded = PrimaryVectorCoder::<8>::default().encode(&vec);
        let lvq = PrimaryVector::<8>::new(&encoded).expect("readable");
        assert_eq!(lvq.vector, &[0, 28, 57, 85, 113, 142, 170, 198, 227, 255]);
        let component_sum = lvq.vector.iter().copied().map(u32::from).sum::<u32>();
        assert_eq!(
            lvq.header,
            VectorHeader {
                l2_norm: 0.92195445,
                lower: -0.49980745,
                upper: 0.3998066,
                component_sum,
            }
        );
        assert_eq!(
            lvq.f32_iter().collect::<Vec<_>>(),
            &[
                -0.49980745,
                -0.4010263,
                -0.29871726,
                -0.19993612,
                -0.10115495,
                0.0011540949,
                0.099935204,
                0.19871637,
                0.30102542,
                0.3998066
            ]
        );
    }

    #[test]
    fn lvq2_1_8() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let encoded = TwoLevelVectorCoder::<1, 8>::default().encode(&vec);
        let lvq = TwoLevelVector::<1, 8>::new(&encoded).expect("readable");
        assert_eq!(lvq.primary.vector, &[0b11100000, 0b11]);
        assert_eq!(
            lvq.vector,
            &[128, 128, 160, 200, 240, 26, 66, 106, 128, 128]
        );
        assert_eq!(
            lvq.primary.header,
            VectorHeader {
                l2_norm: 0.92195445,
                lower: -0.38059703,
                upper: 0.25373134,
                component_sum: 5,
            }
        );
        assert_eq!(
            lvq.f32_iter().collect::<Vec<_>>(),
            &[
                -0.37935326,
                -0.37935326,
                -0.29975125,
                -0.20024878,
                -0.100746304,
                0.0012437701,
                0.10074626,
                0.20024875,
                0.2549751,
                0.2549751
            ]
        );
    }

    #[test]
    fn lvq2_4_4() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let encoded = TwoLevelVectorCoder::<4, 4>::default().encode(&vec);
        let lvq = TwoLevelVector::<4, 4>::new(&encoded).expect("readable");
        assert_eq!(lvq.primary.vector, &[0x20, 0x53, 0x87, 0xca, 0xfd]);
        assert_eq!(lvq.vector, &[0x38, 0x8d, 0xc3, 0x27, 0x7c]);
        assert_eq!(
            lvq.primary.header,
            VectorHeader {
                l2_norm: 0.92195445,
                lower: -0.50325716,
                upper: 0.40300408,
                component_sum: 75,
            }
        );
        assert_eq!(
            lvq.f32_iter().collect::<Vec<_>>(),
            &[
                -0.50124323,
                -0.40054756,
                -0.29985186,
                -0.19915617,
                -0.09846048,
                -0.0017926209,
                0.09890307,
                0.19959876,
                0.30029446,
                0.40099013
            ]
        );
    }

    #[test]
    fn lvq2_4_8() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let encoded = TwoLevelVectorCoder::<4, 8>::default().encode(&vec);
        let lvq = TwoLevelVector::<4, 8>::new(&encoded).expect("readable");
        assert_eq!(lvq.primary.vector, &[0x20, 0x53, 0x87, 0xca, 0xfd]);
        assert_eq!(lvq.vector, &[141, 53, 220, 132, 45, 212, 124, 36, 203, 115]);
        assert_eq!(
            lvq.primary.header,
            VectorHeader {
                l2_norm: 0.92195445,
                lower: -0.50325716,
                upper: 0.40300408,
                component_sum: 75,
            }
        );
        assert_eq!(
            lvq.f32_iter().collect::<Vec<_>>(),
            &[
                -0.5000586,
                -0.40007368,
                -0.3000888,
                -0.2001039,
                -0.099882066,
                0.00010282919,
                0.100087725,
                0.20007262,
                0.30005753,
                0.4000424,
            ]
        );
    }

    #[test]
    fn lvq2_8_8() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let encoded = TwoLevelVectorCoder::<8, 8>::default().encode(&vec);
        let lvq = TwoLevelVector::<8, 8>::new(&encoded).expect("readable");
        assert_eq!(
            lvq.primary.vector,
            &[0, 28, 57, 85, 113, 142, 170, 198, 227, 255]
        );
        assert_eq!(lvq.vector, &[128, 202, 35, 123, 211, 44, 132, 220, 53, 128]);
        let component_sum = lvq
            .primary
            .vector
            .iter()
            .copied()
            .map(u32::from)
            .sum::<u32>();
        assert_eq!(
            lvq.primary.header,
            VectorHeader {
                l2_norm: 0.92195445,
                lower: -0.49980745,
                upper: 0.3998066,
                component_sum,
            }
        );
        assert_eq!(
            lvq.f32_iter().collect::<Vec<_>>(),
            &[
                -0.49980053,
                -0.3999956,
                -0.299997,
                -0.19999838,
                -0.09999974,
                -1.1187512e-6,
                0.09999746,
                0.1999961,
                0.2999947,
                0.3998135
            ]
        );
    }
}
