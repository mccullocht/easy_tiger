//! Locally adaptive Vector Quantization (LVQ): https://arxiv.org/pdf/2304.04759
//!
//! This is Optimized Scalar Quantization without anisotropic loss compensation.

// XXX quantized rep dot product for lvq1 (lvq2 would be more complicated).
// XXX a0 * b0 + a1 * b1 + ...
// XXX let X = (1 << B) - 1
// XXX (a0q * (au - al) / X + al) * (b0q * (bu - bl) / X + bl) + ...
// XXX (a0q * (au - al) / X * b0q * (bu - bl) / X) + (a0q * (au - al) / X * bl) + (b0q * (bu - bl) / X * al) + (al * bl)
// XXX (a0q * b0q * adelta * bdelta) + (a0q * adelta * bl) + (b0q * bdelta * al) + (al * bl)
// XXX (a dot b * adelta * bdelta) + (sum(a) * adelta * bl) + (sum(b) * bdelta * al) + (al * bl * dim)
// XXX this derivation works for lvq1 but not lvq2

use std::iter::FusedIterator;

use crate::vectors::F32VectorCoder;

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

    fn deserialize<'a>(raw: &'a [u8]) -> Option<(Self, &'a [u8])> {
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
struct PrimaryVector<'a> {
    header: VectorHeader,
    delta: f32,
    vector: &'a [u8],
}

impl<'a> PrimaryVector<'a> {
    fn new(encoded: &'a [u8], bits: usize) -> Option<Self> {
        let (header, vector) = VectorHeader::deserialize(encoded)?;
        Some(Self::with_header(header, bits, vector))
    }

    fn with_header(header: VectorHeader, bits: usize, vector: &'a [u8]) -> Self {
        let delta = (header.upper - header.lower) / ((1 << bits) - 1) as f32;
        Self {
            header,
            delta,
            vector,
        }
    }

    fn l2_norm(&self) -> f32 {
        self.header.l2_norm
    }

    fn f32_iter(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        self.vector
            .iter()
            .map(|q| *q as f32 * self.delta + self.header.lower)
    }
}

struct TwoLevelVector<'a> {
    primary: PrimaryVector<'a>,
    vector: &'a [u8],
    delta: f32,
    lower: f32,
}

impl<'a> TwoLevelVector<'a> {
    fn new(encoded: &'a [u8], bits1: usize, bits2: usize) -> Option<Self> {
        let (header, vector) = VectorHeader::deserialize(encoded)?;
        let (primary_vector, residual_vector) = vector.split_at(vector.len() / 2);
        let primary = PrimaryVector::with_header(header, bits1, primary_vector);
        let delta = primary.delta / ((1 << bits2) - 1) as f32;
        let lower = -primary.delta / 2.0;
        Some(Self {
            primary,
            vector: residual_vector,
            delta,
            lower,
        })
    }

    fn f32_iter(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        self.primary
            .f32_iter()
            .zip(
                self.vector
                    .iter()
                    .map(|r| *r as f32 * self.delta + self.lower),
            )
            .map(|(q, r)| q + r)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PrimaryVectorCoder(usize);

impl PrimaryVectorCoder {
    pub fn new(bits: usize) -> Self {
        assert!((1..=8).contains(&bits));
        Self(bits)
    }
}

impl F32VectorCoder for PrimaryVectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let mut it = PrimaryQuantizer::new(vector, self.0);
        let (header_bytes, vector_bytes) = VectorHeader::split_output_buf(out).unwrap();
        for (i, o) in it.by_ref().zip(vector_bytes.iter_mut()) {
            *o = i;
        }
        it.into_header().serialize(header_bytes);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        VectorHeader::LEN + dimensions
    }

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        Some(PrimaryVector::new(encoded, self.0)?.f32_iter().collect())
    }
}

#[derive(Debug, Copy, Clone)]
pub struct TwoLevelVectorCoder(usize, usize);

impl TwoLevelVectorCoder {
    pub fn new(bits1: usize, bits2: usize) -> Self {
        assert!((1..=8).contains(&bits1));
        assert!((1..=8).contains(&bits2));
        Self(bits1, bits2)
    }
}

impl F32VectorCoder for TwoLevelVectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let mut it = TwoLevelQuantizer::new(vector, self.0, self.1);
        let (header_bytes, vector_bytes) = VectorHeader::split_output_buf(out).unwrap();
        let (primary, residual) = vector_bytes.split_at_mut(vector.len());
        for (i, o) in it.by_ref().zip(primary.iter_mut().zip(residual.iter_mut())) {
            *o.0 = i.0;
            *o.1 = i.1;
        }
        it.into_header().serialize(header_bytes);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        VectorHeader::LEN + (dimensions * 2)
    }

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        Some(
            TwoLevelVector::new(encoded, self.0, self.1)?
                .f32_iter()
                .collect(),
        )
    }
}

#[cfg(test)]
mod test {
    use crate::vectors::lvq::{
        F32VectorCoder, PrimaryVector, PrimaryVectorCoder, TwoLevelVector, TwoLevelVectorCoder,
        VectorHeader,
    };

    #[test]
    fn lvq1_4() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let encoded = PrimaryVectorCoder::new(4).encode(&vec);
        let lvq = PrimaryVector::new(&encoded, 4).expect("readable");
        assert_eq!(lvq.vector, &[0, 2, 3, 5, 7, 8, 10, 12, 13, 15]);
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
        let encoded = PrimaryVectorCoder::new(8).encode(&vec);
        let lvq = PrimaryVector::new(&encoded, 8).expect("readable");
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
    fn lvq2_4_4() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let encoded = TwoLevelVectorCoder::new(4, 4).encode(&vec);
        let lvq = TwoLevelVector::new(&encoded, 4, 4).expect("readable");
        assert_eq!(lvq.primary.vector, &[0, 2, 3, 5, 7, 8, 10, 12, 13, 15]);
        assert_eq!(lvq.vector, &[8, 2, 12, 8, 3, 13, 8, 3, 13, 8]);
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
        let encoded = TwoLevelVectorCoder::new(4, 8).encode(&vec);
        let lvq = TwoLevelVector::new(&encoded, 4, 8).expect("readable");
        assert_eq!(lvq.primary.vector, &[0, 2, 3, 5, 7, 8, 10, 12, 13, 15]);
        assert_eq!(lvq.vector, &[128, 42, 212, 128, 43, 213, 128, 43, 213, 128]);
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
        let encoded = TwoLevelVectorCoder::new(8, 8).encode(&vec);
        let lvq = TwoLevelVector::new(&encoded, 8, 8).expect("readable");
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
