//! Binary vector coding and distance computation.

use std::borrow::Cow;

use simsimd::BinarySimilarity;

use crate::{
    distance::l2_normalize,
    vectors::{F32VectorCoder, QueryVectorDistance, VectorDistance},
};

// XXX for euclidean we could store the l2 norm, which would help with adjustments.
// XXX we could also store a mean absolute value to use in the euclidean case.
#[derive(Debug, Copy, Clone)]
pub struct BinaryQuantizedVectorCoder;

impl F32VectorCoder for BinaryQuantizedVectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        for (c, o) in vector.chunks(8).zip(out.iter_mut()) {
            *o = c
                .iter()
                .enumerate()
                .filter_map(|(i, d)| if *d > 0.0 { Some(1u8 << i) } else { None })
                .fold(0, |a, b| a | b)
        }
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions.div_ceil(8)
    }

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        let value = (1.0 / (encoded.len() * 8) as f64).sqrt() as f32;
        Some(
            encoded
                .iter()
                .flat_map(|x| {
                    let mut values = [0.0f32; 8];
                    for i in 0..8 {
                        values[i] = if (*x & (1 << i)) != 0 { value } else { -value };
                    }
                    values
                })
                .collect(),
        )
    }
}

#[derive(Debug, Copy, Clone)]
pub struct AsymmetricBinaryQuantizedVectorCoder(usize);

impl AsymmetricBinaryQuantizedVectorCoder {
    /// Create a new quantizer.
    ///
    /// *Panics* if `!(1..=8).contains(&n)`
    pub fn new(n: usize) -> Self {
        assert!((1..=8).contains(&n));
        Self(n)
    }

    /// Produce a packed u8 representation containing only `bit` from each byte of `dword`.
    fn summarize_chunk(mut dword: u64, bit: usize) -> u8 {
        // First, shift and mask so that only `bit` is present as the lowest bit in each byte.
        // Shift bit to the lowest position in each byte of word.
        dword = (dword >> bit) & 0x0101010101010101;

        // Add a magic constant and mask to get a different bit set for each value if the
        // bit for that byte is present in the input word.
        dword = (dword + 0x7f3f1f0f_07030100) & 0x80402010_08040201;

        // Reduce and mask down to a single byte;
        let word = dword | (dword >> 32);
        let hword = word | (word >> 16);
        (hword | (hword >> 8)) as u8
    }
}

impl F32VectorCoder for AsymmetricBinaryQuantizedVectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        if vector.is_empty() || self.0 == 1 {
            BinaryQuantizedVectorCoder.encode_to(vector, out);
            return;
        }

        // Scale each dimension to be in [0, 2^n) and produce a trivially quantized vector.
        let (min, max) = vector.iter().fold((f32::MAX, f32::MIN), |(min, max), d| {
            assert!(!d.is_nan());
            (min.min(*d), max.max(*d))
        });
        assert!(min <= max);
        let scale = (max - min) / ((1 << self.0) - 1) as f32;
        let trivial_quantized = vector
            .iter()
            .map(|d| ((*d - min) / scale).round() as u8)
            .collect::<Vec<_>>();

        let doc_bytes = vector.len().div_ceil(8);

        for (i, chunk) in trivial_quantized.chunks(8).enumerate() {
            let word = if chunk.len() == 8 {
                u64::from_le_bytes(chunk.try_into().expect("exactly 8 bytes"))
            } else {
                assert!(chunk.len() <= 8);
                let mut bytes = [0u8; 8];
                bytes[0..chunk.len()].copy_from_slice(chunk);
                u64::from_le_bytes(bytes)
            };
            for bit in 0..self.0 {
                out[doc_bytes * bit + i] = Self::summarize_chunk(word, bit);
            }
        }
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions.div_ceil(8) * self.0
    }
}

/// Computes a score from two bitmaps using hamming distance.
#[derive(Debug, Copy, Clone)]
pub struct HammingDistance;

impl VectorDistance for HammingDistance {
    fn distance(&self, a: &[u8], b: &[u8]) -> f64 {
        // XXX this is actually a dot distance, I need the l2 norm (at least!) to do euclidean.
        // XXX the avg absolute error is 0.445 or basically 45deg lol.
        let dist = BinarySimilarity::hamming(a, b).expect("same dimensionality");
        // map distance to a value in [0,2] using dimensions (a.len() * 8)
        dist / (a.len() * 4) as f64
    }
}

// XXX this is 175x slower and 12x more accurate. still 353x less accurate than i8-su
// absolute mean error is 0.037 vs 0.445 for bin x bin.
#[derive(Debug, Clone)]
pub struct DotProductQueryDistance<'a>(Cow<'a, [f32]>, f32);

impl<'a> DotProductQueryDistance<'a> {
    pub fn new(query: &'a [f32]) -> Self {
        let value = (1.0 / query.len() as f64).sqrt() as f32;
        Self(l2_normalize(query), value)
    }
}

impl DotProductQueryDistance<'_> {
    fn dot_scalar(&self, vector: &[u8], start_dim: usize) -> f64 {
        let values = [-self.1, self.1];
        self.0[start_dim..]
            .iter()
            .zip(vector[(start_dim / 8)..].iter().flat_map(|b| {
                // XXX this is a lot of scalar instructions.
                // can i generate a value, add + mask + or to get the right outcome???
                // XXX currently 1.2usecs, absolutely putrid.
                // XXX if i pack in 4s this can be _way_ better by using a fixed bit across multiple
                // dims i can at least kind of dupq my way out of it.
                // XXX VTBL MADNESS
                // - the low 3 bytes are always the same, the high 2 may vary.
                // actually might be best to vtbl instead of vtblq and then i only need 4 different
                // vtbl masks to do the work instead of 16. i can vcombine these to do the thing.
                [
                    values[*b as usize & 0x1],
                    values[(*b as usize >> 1) & 0x1],
                    values[(*b as usize >> 2) & 0x1],
                    values[(*b as usize >> 3) & 0x1],
                    values[(*b as usize >> 4) & 0x1],
                    values[(*b as usize >> 5) & 0x1],
                    values[(*b as usize >> 6) & 0x1],
                    values[(*b as usize >> 7) & 0x1],
                ]
            }))
            .map(|(q, d)| *q * d)
            .sum::<f32>() as f64
    }

    // XXX this is at least near correct and is also near-ish f32 dot performance.
    // XXX this brings me to: I really want to be able to confidence bound the approximate score
    // to decide whether or not to pivot into something more expensive. the worst case scenario is
    // essentially re-ranking.
    #[cfg(target_arch = "aarch64")]
    fn dot(&self, vector: &[u8]) -> f64 {
        let split = self.0.len() & !7;
        let dot = unsafe {
            use std::arch::aarch64::{
                vaddvq_f32, vcombine_f32, vdupq_n_f32, vfmaq_f32, vld1_f32, vld1_u8, vld1q_f32,
                vreinterpret_f32_u8, vreinterpret_u8_f32, vtbl1_u8,
            };

            let mut dot = vdupq_n_f32(0.0);
            // 0 is positive, one is negative.
            let values = vreinterpret_u8_f32(vld1_f32([-self.1, self.1].as_ptr()));
            // XXX this is little-endian specific.
            let vtbl_masks = [
                vld1_u8([0, 1, 2, 3, 0, 1, 2, 3].as_ptr()),
                vld1_u8([4, 5, 6, 7, 0, 1, 2, 3].as_ptr()),
                vld1_u8([0, 1, 2, 3, 4, 5, 6, 7].as_ptr()),
                vld1_u8([4, 5, 6, 7, 4, 5, 6, 7].as_ptr()),
            ];
            for i in (0..split).step_by(8) {
                let b = vector[i / 8] as usize;
                let d = vcombine_f32(
                    vreinterpret_f32_u8(vtbl1_u8(values, vtbl_masks[b & 3])),
                    vreinterpret_f32_u8(vtbl1_u8(values, vtbl_masks[(b >> 2) & 3])),
                );
                let q = vld1q_f32(self.0.as_ptr().add(i));
                dot = vfmaq_f32(dot, q, d);
                let d = vcombine_f32(
                    vreinterpret_f32_u8(vtbl1_u8(values, vtbl_masks[(b >> 4) & 3])),
                    vreinterpret_f32_u8(vtbl1_u8(values, vtbl_masks[(b >> 6) & 3])),
                );
                let q = vld1q_f32(self.0.as_ptr().add(i + 4));
                dot = vfmaq_f32(dot, q, d);
            }

            vaddvq_f32(dot)
        };

        dot as f64 + self.dot_scalar(vector, split)
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn dot(&self, vector: &[u8]) -> f64 {
        self.dot_scalar(vector, 0)
    }
}

impl QueryVectorDistance for DotProductQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let dot = self.dot(vector);
        (-dot + 1.0) / 2.0
    }
}

/// Computes a score between a query and doc vectors produced by [AsymmetricBinaryQuantizedVectorCoder]
#[derive(Debug, Copy, Clone)]
pub struct AsymmetricHammingDistance;

impl VectorDistance for AsymmetricHammingDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        // XXX it's not clear to me how I would go about turning this into a proper angular distance.
        assert_eq!(query.len() % doc.len(), 0);
        query
            .chunks(doc.len())
            .enumerate()
            .map(|(i, v)| {
                BinarySimilarity::hamming(doc, v).expect("same dimensionality") as usize
                    * (1usize << i)
            })
            .sum::<usize>() as f64
    }
}
