//! Locally adaptive Vector Quantization (LVQ): https://arxiv.org/pdf/2304.04759
//!
//! This is Optimized Scalar Quantization without anisotropic loss compensation. Without loss
//! compensation it still does fine at 4+ bits per level but is quite poor at 1 or 2 bits.

use std::{borrow::Cow, iter::FusedIterator};

use crate::vectors::{F32VectorCoder, QueryVectorDistance, VectorDistance};

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

    fn partial_init_from_vector(v: &[f32]) -> Self {
        if v.is_empty() {
            return VectorHeader {
                l2_norm: 1.0,
                lower: 0.0,
                upper: 0.0,
                component_sum: 0,
            };
        }
        v.iter().fold(
            VectorHeader {
                l2_norm: 0.0,
                lower: f32::MAX,
                upper: f32::MIN,
                component_sum: 0,
            },
            |mut h, x| {
                h.l2_norm += *x * *x;
                h.lower = x.min(h.lower);
                h.upper = x.max(h.upper);
                h
            },
        )
    }

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

struct PrimaryQuantizer<'a> {
    it: std::slice::Iter<'a, f32>,
    header: VectorHeader,
    delta: f32,
}

impl<'a> PrimaryQuantizer<'a> {
    fn new(v: &'a [f32], bits: usize) -> Self {
        let header = VectorHeader::partial_init_from_vector(v);
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
            let q = ((*x - self.header.lower) / self.delta).round() as u8;
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
        let res = *x - ((q as f32 * self.primary_it.delta) + self.primary_it.header.lower);
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

// XXX quantized rep dot product for lvq1 (lvq2 would be more complicated).
// XXX a0 * b0 + a1 * b1 + ...
// XXX let X = (1 << B) - 1
// XXX (a0q * (au - al) / X + al) * (b0q * (bu - bl) / X + bl) + ...
// XXX (a0q * (au - al) / X * b0q * (bu - bl) / X) + (a0q * (au - al) / X * bl) + (b0q * (bu - bl) / X * al) + (al * bl)
// XXX (a0q * b0q * adelta * bdelta) + (a0q * adelta * bl) + (b0q * bdelta * al) + (al * bl)
// XXX (a dot b * adelta * bdelta) + (sum(a) * adelta * bl) + (sum(b) * bdelta * al) + (al * bl * dim)
// XXX this derivation works for lvq1 but not lvq2
// XXX this derivation can probably be simplified for f32xlvq1

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
        let dot_quantized = packing::unpack_iter::<B>(self.vector)
            .zip(packing::unpack_iter::<B>(other.vector))
            .map(|(s, o)| s as u32 * o as u32)
            .sum::<u32>();
        let sdelta = f64::from(self.delta);
        let slower = f64::from(self.header.lower);
        let odelta = f64::from(other.delta);
        let olower = f64::from(other.header.lower);
        dot_quantized as f64 * sdelta * odelta
            + self.header.component_sum as f64 * sdelta * olower
            + other.header.component_sum as f64 * odelta * slower
            + slower * olower * self.vector.len().div_ceil(B) as f64
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
        // XXX this can likely be made to work more efficiently by adopting a simpler version of
        // the factorization from lvq1xlvq1
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

    pub const fn two_vector_split(vector_bytes: usize, bits1: usize, bits2: usize) -> usize {
        match (bits1, bits2) {
            (1, 1) | (2, 2) | (4, 4) | (8, 8) => vector_bytes / 2,
            (1, 8) => vector_bytes.div_ceil(9),
            (4, 8) => vector_bytes / 3,
            (8, 4) => vector_bytes * 2 / 3,
            _ => panic!("XXX fill this out completely"),
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
                l2_norm: 0.8500001,
                lower: -0.5,
                upper: 0.4,
                component_sum: 5,
            }
        );
        // NB: vector dimensionality is not a multiple of 8 so we're producting extra dimensions.
        assert_eq!(
            lvq.f32_iter().take(10).collect::<Vec<_>>(),
            // yikes bikes, interpreting everything as min/max feels bad. i guess that's why you
            // should use anisotropic loss.
            &[
                -0.5, -0.5, -0.5, -0.5, -0.5, 0.39999998, 0.39999998, 0.39999998, 0.39999998,
                0.39999998
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
                l2_norm: 0.8500001,
                lower: -0.5,
                upper: 0.4,
                component_sum: 75,
            }
        );
        assert_eq!(
            lvq.f32_iter().collect::<Vec<_>>(),
            &[
                -0.5f32,
                -0.38,
                -0.32,
                -0.20000002,
                -0.08000001,
                -0.02000001,
                0.099999964,
                0.21999997,
                0.27999997,
                0.39999998
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
                l2_norm: 0.8500001,
                lower: -0.5,
                upper: 0.4,
                component_sum,
            }
        );
        assert_eq!(
            lvq.f32_iter().collect::<Vec<_>>(),
            &[
                -0.5f32,
                -0.40117645,
                -0.29882354,
                -0.19999999,
                -0.10117647,
                0.0011764765,
                0.100000024,
                0.19882351,
                0.3011765,
                0.39999998
            ]
        );
    }

    #[test]
    fn lvq2_1_8() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let encoded = TwoLevelVectorCoder::<1, 8>::default().encode(&vec);
        let lvq = TwoLevelVector::<1, 8>::new(&encoded).expect("readable");
        assert_eq!(lvq.primary.vector, &[0b11100000, 0b11]);
        // notably, this is only using 7 bits of residual.
        assert_eq!(lvq.vector, &[128, 156, 184, 213, 241, 14, 43, 71, 99, 128]);
        assert_eq!(
            lvq.primary.header,
            VectorHeader {
                l2_norm: 0.8500001,
                lower: -0.5,
                upper: 0.4,
                component_sum: 5,
            }
        );
        assert_eq!(
            lvq.f32_iter().collect::<Vec<_>>(),
            &[
                -0.4982353,
                -0.39941174,
                -0.30058825,
                -0.19823527,
                -0.099411786,
                -0.00058823824,
                0.10176468,
                0.20058823,
                0.29941174,
                0.4017647
            ]
        );
    }

    #[test]
    fn lvq2_4_4() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let encoded = TwoLevelVectorCoder::<4, 4>::default().encode(&vec);
        let lvq = TwoLevelVector::<4, 4>::new(&encoded).expect("readable");
        assert_eq!(lvq.primary.vector, &[0x20, 0x53, 0x87, 0xca, 0xfd]);
        assert_eq!(lvq.vector, &[0x28, 0x8c, 0xd3, 0x38, 0x8d]);
        assert_eq!(
            lvq.primary.header,
            VectorHeader {
                l2_norm: 0.8500001,
                lower: -0.5,
                upper: 0.4,
                component_sum: 75,
            }
        );
        assert_eq!(
            lvq.f32_iter().collect::<Vec<_>>(),
            &[
                -0.498,
                -0.402,
                -0.302,
                -0.19800001,
                -0.09800001,
                0.0019999873,
                0.10199996,
                0.20199996,
                0.30199996,
                0.40199998
            ]
        );
    }

    #[test]
    fn lvq2_4_8() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let encoded = TwoLevelVectorCoder::<4, 8>::default().encode(&vec);
        let lvq = TwoLevelVector::<4, 8>::new(&encoded).expect("readable");
        assert_eq!(lvq.primary.vector, &[0x20, 0x53, 0x87, 0xca, 0xfd]);
        assert_eq!(lvq.vector, &[128, 42, 212, 128, 43, 213, 128, 43, 213, 128]);
        assert_eq!(
            lvq.primary.header,
            VectorHeader {
                l2_norm: 0.8500001,
                lower: -0.5,
                upper: 0.4,
                component_sum: 75,
            }
        );
        assert_eq!(
            lvq.f32_iter().collect::<Vec<_>>(),
            &[
                -0.49988234,
                -0.40011764,
                -0.30011764,
                -0.19988237,
                -0.099882364,
                0.000117635354,
                0.10011761,
                0.20011762,
                0.3001176,
                0.40011764
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
        assert_eq!(lvq.vector, &[128, 212, 42, 127, 212, 42, 127, 213, 42, 128]);
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
                l2_norm: 0.8500001,
                lower: -0.5,
                upper: 0.4,
                component_sum,
            }
        );
        assert_eq!(
            lvq.f32_iter().collect::<Vec<_>>(),
            &[
                -0.4999931,
                -0.4000069,
                -0.30000693,
                -0.2000069,
                -0.10000692,
                -6.914488e-6,
                0.0999931,
                0.2000069,
                0.2999931,
                0.4000069
            ]
        );
    }
}
