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

struct QuantizerMeta {
    lower: f32,
    upper: f32,
    l2_norm: f32,
}

impl From<&[f32]> for QuantizerMeta {
    fn from(value: &[f32]) -> Self {
        if value.is_empty() {
            return QuantizerMeta {
                l2_norm: 1.0,
                lower: 0.0,
                upper: 0.0,
            };
        }
        value.iter().fold(
            QuantizerMeta {
                l2_norm: 0.0,
                lower: f32::MAX,
                upper: f32::MIN,
            },
            |mut m, x| {
                m.l2_norm += *x * *x;
                m.lower = x.min(m.lower);
                m.upper = x.max(m.upper);
                m
            },
        )
    }
}

struct LVQHeader {
    l2_norm: f32,
    lower: f32,
    upper: f32,
    component_sum: u32,
}

impl LVQHeader {
    fn split_output_buf(buf: &mut [u8]) -> Option<(&mut [u8], &mut [u8])> {
        buf.split_at_mut_checked(16)
    }

    fn serialize(&self, header_bytes: &mut [u8]) {
        let header = header_bytes.as_chunks_mut::<4>().0;
        header[0] = self.l2_norm.to_le_bytes();
        header[1] = self.lower.to_le_bytes();
        header[2] = self.upper.to_le_bytes();
        header[3] = self.component_sum.to_le_bytes();
    }

    fn deserialize<'a>(raw: &'a [u8]) -> Option<(Self, &'a [u8])> {
        let (header_bytes, vector_bytes) = raw.split_at_checked(16)?;
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

struct LVQ1Iter<'a> {
    it: std::slice::Iter<'a, f32>,
    header: LVQHeader,
    delta: f32,
}

impl<'a> LVQ1Iter<'a> {
    fn new<const B: usize>(v: &'a [f32]) -> Self {
        let meta = QuantizerMeta::from(v);
        let header = LVQHeader {
            l2_norm: meta.l2_norm,
            lower: meta.lower,
            upper: meta.upper,
            component_sum: 0,
        };
        let delta = (meta.upper - meta.lower) / ((1 << B) - 1) as f32;
        Self {
            it: v.iter(),
            header,
            delta,
        }
    }
}

impl Iterator for LVQ1Iter<'_> {
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

impl ExactSizeIterator for LVQ1Iter<'_> {}

impl FusedIterator for LVQ1Iter<'_> {}

/// Compute single-level LVQ, with `B` as the number of quantized bits.
///
/// Returns min, max, and an iterator over quantizatized values with a single entry per byte.
fn lvq1<const B: usize>(v: &[f32]) -> (QuantizerMeta, impl ExactSizeIterator<Item = u8> + '_) {
    let meta = QuantizerMeta::from(v);
    let delta = (meta.upper - meta.lower) / ((1 << B) - 1) as f32;
    let it = v
        .iter()
        .map(move |x| ((x - meta.lower) / delta).round() as u8);
    (meta, it)
}

/// Invert one-level LVQ to produce an f32 vector.
///
/// The `B` value must be the same as was used in the original [lvq1] call.
fn unlvq1<'q, const B: usize>(
    l: f32,
    u: f32,
    quantized: impl ExactSizeIterator<Item = u8> + 'q,
) -> impl ExactSizeIterator<Item = f32> + 'q {
    let delta = (u - l) / ((1 << B) - 1) as f32;
    quantized.map(move |q| (q as f32 * delta) + l)
}

/// Compute two-level LVQ, with `B1` primary vector bits and `B2` residual bits.
///
/// Returns min, max, and an iterator over (primary, residual) quantized values
fn lvq2<const B1: usize, const B2: usize>(
    v: &[f32],
) -> (QuantizerMeta, impl ExactSizeIterator<Item = (u8, u8)> + '_) {
    let meta = QuantizerMeta::from(v);
    let delta = (meta.upper - meta.lower) / ((1 << B1) - 1) as f32;

    let res_l = -delta / 2.0;
    let res_delta = delta / ((1 << B2) - 1) as f32;

    let it = v.iter().map(move |x| {
        let q = ((x - meta.lower) / delta).round();
        let res = *x - ((q * delta) + meta.lower);
        let res_q = ((res - res_l) / res_delta).round();
        (q as u8, res_q as u8)
    });
    (meta, it)
}

/// Invert two-level LVQ to produce an f32 vector.
///
/// The `B1` and `B2` values must be the same as was used in the original [lvq2] call.
fn unlvq2<'q, const B1: usize, const B2: usize>(
    l: f32,
    u: f32,
    quantized: impl ExactSizeIterator<Item = (u8, u8)> + 'q,
) -> impl ExactSizeIterator<Item = f32> + 'q {
    let delta = (u - l) / ((1 << B1) - 1) as f32;
    let res_l = -delta / 2.0;
    let res_delta = delta / ((1 << B2) - 1) as f32;
    quantized.map(move |(q, r)| {
        let uq = q as f32 * delta + l;
        let ur = r as f32 * res_delta + res_l;
        uq + ur
    })
}

#[derive(Debug, Copy, Clone, Default)]
pub struct LVQ1VectorCoder<const B: usize>;

impl<const B: usize> F32VectorCoder for LVQ1VectorCoder<B> {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let mut it = LVQ1Iter::new::<B>(vector);
        let (header_bytes, vector_bytes) = LVQHeader::split_output_buf(out).unwrap();
        for (i, o) in it.by_ref().zip(vector_bytes.iter_mut()) {
            *o = i;
        }
        it.header.serialize(header_bytes);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        std::mem::size_of::<f32>() * 2 + dimensions
    }

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        let (header, vector) = LVQHeader::deserialize(encoded)?;
        Some(unlvq1::<B>(header.lower, header.upper, vector.iter().copied()).collect())
    }
}

pub type LVQ1x4VectorCoder = LVQ1VectorCoder<4>;
pub type LVQ1x8VectorCoder = LVQ1VectorCoder<8>;

#[derive(Debug, Copy, Clone, Default)]
pub struct LVQ2VectorCoder<const B1: usize, const B2: usize>;

impl<const B1: usize, const B2: usize> F32VectorCoder for LVQ2VectorCoder<B1, B2> {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let (meta, it) = lvq2::<B1, B2>(vector);
        let (header_bytes, vec_bytes) = out.split_at_mut(std::mem::size_of::<f32>() * 2);
        let header = header_bytes
            .as_chunks_mut::<{ std::mem::size_of::<f32>() }>()
            .0;
        header[0] = meta.lower.to_le_bytes();
        header[1] = meta.upper.to_le_bytes();
        let (o1_vec, o2_vec) = vec_bytes.split_at_mut(vector.len());
        for ((i1, i2), (o1, o2)) in it.zip(o1_vec.iter_mut().zip(o2_vec.iter_mut())) {
            *o1 = i1;
            *o2 = i2;
        }
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        (std::mem::size_of::<f32>() * 2) + (dimensions * 2)
    }

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        let (meta_bytes, vec_bytes) = encoded.split_at(std::mem::size_of::<f32>() * 2);
        let meta = meta_bytes.as_chunks::<{ std::mem::size_of::<f32>() }>().0;
        let l = f32::from_le_bytes(meta[0]);
        let u = f32::from_le_bytes(meta[1]);
        let (b1_vec, b2_vec) = vec_bytes.split_at(vec_bytes.len() / 2);
        Some(unlvq2::<B1, B2>(l, u, b1_vec.iter().copied().zip(b2_vec.iter().copied())).collect())
    }
}

pub type LVQ2x4x4VectorCoder = LVQ2VectorCoder<4, 4>;
pub type LVQ2x4x8VectorCoder = LVQ2VectorCoder<4, 8>;
pub type LVQ2x8x8VectorCoder = LVQ2VectorCoder<8, 8>;

#[cfg(test)]
mod test {
    use crate::vectors::lvq::{lvq1, lvq2, unlvq1, unlvq2};

    #[test]
    fn lvq1_4() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let (meta, it) = lvq1::<4>(&vec);
        assert_eq!(meta.lower, -0.5f32);
        assert_eq!(meta.upper, 0.4f32);
        let q = it.collect::<Vec<_>>();
        assert_eq!(&q, &[0, 2, 3, 5, 7, 8, 10, 12, 13, 15]);
        assert_eq!(
            unlvq1::<4>(meta.lower, meta.upper, q.into_iter()).collect::<Vec<_>>(),
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
        let (meta, it) = lvq1::<8>(&vec);
        assert_eq!(meta.lower, -0.5f32);
        assert_eq!(meta.upper, 0.4f32);
        let q = it.collect::<Vec<_>>();
        assert_eq!(q, &[0, 28, 57, 85, 113, 142, 170, 198, 227, 255]);
        assert_eq!(
            unlvq1::<8>(meta.lower, meta.upper, q.into_iter()).collect::<Vec<_>>(),
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
    fn lvq2_4_8() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let (meta, it) = lvq2::<4, 8>(&vec);
        assert_eq!(meta.lower, -0.5f32);
        assert_eq!(meta.upper, 0.4f32);
        let q = it.collect::<Vec<_>>();
        assert_eq!(
            q,
            &[
                (0, 128),
                (2, 42),
                (3, 212),
                (5, 128),
                (7, 43),
                (8, 213),
                (10, 128),
                (12, 43),
                (13, 213),
                (15, 128)
            ]
        );
        // TODO: very specific rounding error in dequantization; investigate.
        assert_eq!(
            unlvq2::<4, 8>(meta.lower, meta.upper, q.into_iter()).collect::<Vec<_>>(),
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
        let (meta, it) = lvq2::<8, 8>(&vec);
        assert_eq!(meta.lower, -0.5f32);
        assert_eq!(meta.upper, 0.4f32);
        let q = it.collect::<Vec<_>>();
        assert_eq!(
            q,
            &[
                (0, 128),
                (28, 212),
                (57, 42),
                (85, 127),
                (113, 212),
                (142, 42),
                (170, 127),
                (198, 213),
                (227, 42),
                (255, 128),
            ]
        );
        assert_eq!(
            unlvq2::<8, 8>(meta.lower, meta.upper, q.into_iter()).collect::<Vec<_>>(),
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
