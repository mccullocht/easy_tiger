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
struct LVQHeader {
    l2_norm: f32,
    lower: f32,
    upper: f32,
    component_sum: u32,
}

impl LVQHeader {
    /// Encoded buffer size.
    const LEN: usize = 16;

    fn partial_init_from_vector(v: &[f32]) -> Self {
        if v.is_empty() {
            return LVQHeader {
                l2_norm: 1.0,
                lower: 0.0,
                upper: 0.0,
                component_sum: 0,
            };
        }
        v.iter().fold(
            LVQHeader {
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

/// Iterator over a float slice that produces a stream of quantized (primary, residual) vectors.
/// The header may be extracted once all of the quantized values have been consumed.
///
/// `B1` is the dimension bit width in the primary vectors and must be in 1..=8.
/// `B2` is the dimension bit width in the residual vectors and must bin in 0..=8. If set to zero
/// the residual values will always be 0.
// XXX replace with composed approach used for the vector implementation.
struct LVQuantizingIter<'a, const B1: usize, const B2: usize> {
    it: std::slice::Iter<'a, f32>,
    header: LVQHeader,
    delta: f32,
    res_delta: f32,
    res_lower: f32,
}

impl<'a, const B1: usize, const B2: usize> LVQuantizingIter<'a, B1, B2> {
    fn new(v: &'a [f32]) -> Self {
        let header = LVQHeader::partial_init_from_vector(v);
        let delta = (header.upper - header.lower) / ((1 << B1) - 1) as f32;
        let (res_delta, res_lower) = if B2 > 0 {
            (delta / ((1 << B2) - 1) as f32, -delta / 2.0)
        } else {
            (0.0, 0.0)
        };
        Self {
            it: v.iter(),
            header,
            delta,
            res_delta,
            res_lower,
        }
    }

    fn into_header(self) -> LVQHeader {
        assert_eq!(self.it.len(), 0);
        self.header
    }
}

impl<const B1: usize, const B2: usize> Iterator for LVQuantizingIter<'_, B1, B2> {
    type Item = (u8, u8);

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|x| {
            let q = ((*x - self.header.lower) / self.delta).round() as u8;
            let r = if B2 > 0 {
                let res = *x - ((q as f32 * self.delta) + self.header.lower);
                ((res - self.res_lower) / self.res_delta).round() as u8
            } else {
                0
            };
            // Component sum is only ever computed against the primary vector.
            // The algebraic expansion for scoring with residuals is probably more effort than
            // converting back to floats, and
            self.header.component_sum += u32::from(q);
            (q, r)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<const B1: usize, const B2: usize> ExactSizeIterator for LVQuantizingIter<'_, B1, B2> {}

impl<const B1: usize, const B2: usize> FusedIterator for LVQuantizingIter<'_, B1, B2> {}

/// An LVQ1 coded primary vector.
///
/// There may be a parallel residual vector that can be composed with this one to increase accuracy.
struct LVQPrimaryVector<'a> {
    header: LVQHeader,
    delta: f32,
    vector: &'a [u8],
}

impl<'a> LVQPrimaryVector<'a> {
    fn new(encoded: &'a [u8], bits: usize) -> Option<Self> {
        let (header, vector) = LVQHeader::deserialize(encoded)?;
        Some(Self::with_header(header, bits, vector))
    }

    fn with_header(header: LVQHeader, bits: usize, vector: &'a [u8]) -> Self {
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

struct LVQResidualVector<'a> {
    primary: LVQPrimaryVector<'a>,
    vector: &'a [u8],
    delta: f32,
    lower: f32,
}

impl<'a> LVQResidualVector<'a> {
    fn new(encoded: &'a [u8], bits1: usize, bits2: usize) -> Option<Self> {
        let (header, vector) = LVQHeader::deserialize(encoded)?;
        let (primary_vector, residual_vector) = vector.split_at(vector.len() / 2);
        let primary = LVQPrimaryVector::with_header(header, bits1, primary_vector);
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

// XXX we still want both of these coders but I don't need to template on the bit width.
#[derive(Debug, Copy, Clone, Default)]
pub struct LVQ1VectorCoder<const B: usize>;

impl<const B: usize> F32VectorCoder for LVQ1VectorCoder<B> {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let mut it = LVQuantizingIter::<B, 0>::new(vector);
        let (header_bytes, vector_bytes) = LVQHeader::split_output_buf(out).unwrap();
        for (i, o) in it.by_ref().map(|i| i.0).zip(vector_bytes.iter_mut()) {
            *o = i;
        }
        it.into_header().serialize(header_bytes);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        LVQHeader::LEN + dimensions
    }

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        Some(LVQPrimaryVector::new(encoded, B)?.f32_iter().collect())
    }
}

pub type LVQ1x4VectorCoder = LVQ1VectorCoder<4>;
pub type LVQ1x8VectorCoder = LVQ1VectorCoder<8>;

#[derive(Debug, Copy, Clone, Default)]
pub struct LVQ2VectorCoder<const B1: usize, const B2: usize>;

impl<const B1: usize, const B2: usize> F32VectorCoder for LVQ2VectorCoder<B1, B2> {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let mut it = LVQuantizingIter::<B1, B2>::new(vector);
        let (header_bytes, vector_bytes) = LVQHeader::split_output_buf(out).unwrap();
        let (primary, residual) = vector_bytes.split_at_mut(vector.len());
        for (i, o) in it.by_ref().zip(primary.iter_mut().zip(residual.iter_mut())) {
            *o.0 = i.0;
            *o.1 = i.1;
        }
        it.into_header().serialize(header_bytes);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        LVQHeader::LEN + (dimensions * 2)
    }

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        Some(
            LVQResidualVector::new(encoded, B1, B2)?
                .f32_iter()
                .collect(),
        )
    }
}

pub type LVQ2x4x4VectorCoder = LVQ2VectorCoder<4, 4>;
pub type LVQ2x4x8VectorCoder = LVQ2VectorCoder<4, 8>;
pub type LVQ2x8x8VectorCoder = LVQ2VectorCoder<8, 8>;

#[cfg(test)]
mod test {
    use crate::vectors::lvq::{
        F32VectorCoder, LVQ1x4VectorCoder, LVQ1x8VectorCoder, LVQ2x4x4VectorCoder,
        LVQ2x4x8VectorCoder, LVQ2x8x8VectorCoder, LVQHeader, LVQPrimaryVector, LVQResidualVector,
    };

    #[test]
    fn lvq1_4() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let encoded = LVQ1x4VectorCoder::default().encode(&vec);
        let lvq = LVQPrimaryVector::new(&encoded, 4).expect("readable");
        assert_eq!(lvq.vector, &[0, 2, 3, 5, 7, 8, 10, 12, 13, 15]);
        let component_sum = lvq.vector.iter().copied().map(u32::from).sum::<u32>();
        assert_eq!(
            lvq.header,
            LVQHeader {
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
        let encoded = LVQ1x8VectorCoder::default().encode(&vec);
        let lvq = LVQPrimaryVector::new(&encoded, 8).expect("readable");
        assert_eq!(lvq.vector, &[0, 28, 57, 85, 113, 142, 170, 198, 227, 255]);
        let component_sum = lvq.vector.iter().copied().map(u32::from).sum::<u32>();
        assert_eq!(
            lvq.header,
            LVQHeader {
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
        let encoded = LVQ2x4x4VectorCoder::default().encode(&vec);
        let lvq = LVQResidualVector::new(&encoded, 4, 4).expect("readable");
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
            LVQHeader {
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
        let encoded = LVQ2x4x8VectorCoder::default().encode(&vec);
        let lvq = LVQResidualVector::new(&encoded, 4, 8).expect("readable");
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
            LVQHeader {
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
        let encoded = LVQ2x8x8VectorCoder::default().encode(&vec);
        let lvq = LVQResidualVector::new(&encoded, 8, 8).expect("readable");
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
            LVQHeader {
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
