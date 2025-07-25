//! Utilities for vector quantization.
//!
//! Graph navigation during search uses these quantized vectors.

use std::{io, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::distance::{
    AsymmetricHammingDistance, HammingDistance, I8NaiveDistance, I8ScaledUniformDotProduct,
    I8ScaledUniformEuclidean, VectorDistance, VectorSimilarity,
};

/// `Quantizer` is used to perform lossy quantization of input vectors.
///
/// Methods are exposed to quantize for indexing vectors in the database or for
/// querying. In simpler schemes these methods are identical and quantization is
/// symmetric; some implementations are asymmetric and use larger vectors for
/// querying to produce higher fidelity scores.
pub trait Quantizer: Send + Sync {
    /// Quantize this vector for querying an index.
    fn for_query(&self, vector: &[f32]) -> Vec<u8>;

    /// Return the size of a quantized query vector for the provided dimensionality.
    fn query_bytes(&self, dimensions: usize) -> usize;

    /// Quantize this vector for indexing.
    fn for_doc(&self, vector: &[f32]) -> Vec<u8>;

    /// Return the size of a quantized index vector for the provided dimensionality.
    fn doc_bytes(&self, dimensions: usize) -> usize;
}

/// Methods for quantizing vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorQuantizer {
    /// Reduces each dimension to a single bit around the origin point.
    Binary,
    /// Binary quantizes indexed vectors; produces an n-bit representation at
    /// query time to increase precision. This is also used during indexing.
    AsymmetricBinary { n: usize },
    /// Reduces each dimension to an 8-bit integer.
    /// This implementation does not train on any input to decide how to quantize.
    I8Naive,
    /// Reduces each dimension to an 8-bit integer but shaped to the input vector with uniform
    /// scaling across all dimensions.
    ///
    /// The maximum magnitude across all dimensions is used in scaling rather than l2 normalizing
    /// the vector and bounding by [-1,1], which greatly reduces loss. This quantization scheme
    /// works pretty well on transformer models where all values are in roughly the same range.
    /// Scoring can be performed directly on the quantized representation.
    ///
    /// This implementation does _not_ train on a sample of the dataset to decide parameters.
    I8ScaledUniform,
}

impl VectorQuantizer {
    /// Create a new quantizer for this quantization method.
    pub fn new_quantizer(&self) -> Box<dyn Quantizer> {
        match self {
            Self::Binary => Box::new(BinaryQuantizer),
            Self::AsymmetricBinary { n } => Box::new(AsymmetricBinaryQuantizer::new(*n)),
            Self::I8Naive => Box::new(I8NaiveQuantizer),
            Self::I8ScaledUniform => Box::new(I8ScaledUniformQuantizer),
        }
    }

    /// Create a new distance function for this quantization method.
    pub fn new_distance_function(&self, similarity: &VectorSimilarity) -> Box<dyn VectorDistance> {
        match (self, similarity) {
            (Self::Binary, _) => Box::new(HammingDistance),
            (Self::AsymmetricBinary { n: _ }, _) => Box::new(AsymmetricHammingDistance),
            (Self::I8Naive, _) => Box::new(I8NaiveDistance(*similarity)),
            (Self::I8ScaledUniform, VectorSimilarity::Dot) => Box::new(I8ScaledUniformDotProduct),
            (Self::I8ScaledUniform, VectorSimilarity::Euclidean) => {
                Box::new(I8ScaledUniformEuclidean)
            }
        }
    }
}

impl Default for VectorQuantizer {
    fn default() -> Self {
        Self::Binary
    }
}

impl FromStr for VectorQuantizer {
    type Err = io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let input_err = |s| io::Error::new(io::ErrorKind::InvalidInput, s);
        match s {
            "binary" => Ok(Self::Binary),
            ab if ab.starts_with("asymmetric_binary:") => {
                let bits_str = ab
                    .strip_prefix("asymmetric_binary:")
                    .expect("prefix matched");
                bits_str
                    .parse::<usize>()
                    .ok()
                    .and_then(|b| if (1..=8).contains(&b) { Some(b) } else { None })
                    .map(|n| Self::AsymmetricBinary { n })
                    .ok_or_else(|| input_err(format!("invalid asymmetric_binary bits {bits_str}")))
            }
            "i8naive" => Ok(Self::I8Naive),
            "i8scaled-uniform" => Ok(Self::I8ScaledUniform),
            _ => Err(input_err(format!("unknown quantizer function {s}"))),
        }
    }
}

/// Reduce each dimension to a single bit.
///
/// This quantizer is stateless. It quantizes all values > 0 to a 1 bit and all other values to 0.
/// This works best if the data set is centered around the origin.
#[derive(Debug, Copy, Clone)]
pub struct BinaryQuantizer;

impl Quantizer for BinaryQuantizer {
    fn for_doc(&self, vector: &[f32]) -> Vec<u8> {
        vector
            .chunks(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .filter_map(|(i, d)| if *d > 0.0 { Some(1u8 << i) } else { None })
                    .fold(0, |a, b| a | b)
            })
            .collect()
    }

    fn doc_bytes(&self, dimensions: usize) -> usize {
        dimensions.div_ceil(8)
    }

    fn for_query(&self, vector: &[f32]) -> Vec<u8> {
        self.for_doc(vector)
    }

    fn query_bytes(&self, dimensions: usize) -> usize {
        self.doc_bytes(dimensions)
    }
}

/// An asymmetric binary quantizer that uses a larger query vector.
///
/// Indexed vectors are reduced to a single bit per dimension like in [BinaryQuantizer].
///
/// Query vectors are quantized into `N` bits per dimension in a stateless fashion, then
/// transformed to group by bit index (all the lowest bits in a sequence, then the next
/// lowest, etc). This allows us to perform dot product-like scoring using hamming distance
/// iteratively.
#[derive(Debug, Copy, Clone)]
pub struct AsymmetricBinaryQuantizer {
    n: usize,
}

impl AsymmetricBinaryQuantizer {
    /// Create a new quantizer.
    ///
    /// *Panics* if `!(1..=8).contains(&n)`
    pub fn new(n: usize) -> Self {
        assert!((1..=8).contains(&n));
        Self { n }
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

impl Quantizer for AsymmetricBinaryQuantizer {
    fn for_doc(&self, vector: &[f32]) -> Vec<u8> {
        BinaryQuantizer.for_doc(vector)
    }

    fn doc_bytes(&self, dimensions: usize) -> usize {
        BinaryQuantizer.doc_bytes(dimensions)
    }

    fn for_query(&self, vector: &[f32]) -> Vec<u8> {
        if vector.is_empty() {
            return vec![];
        }

        if self.n == 1 {
            return BinaryQuantizer.for_query(vector);
        }

        // Scale each dimension to be in [0, 2^n) and produce a trivially quantized vector.
        let (min, max) = vector.iter().fold((f32::MAX, f32::MIN), |(min, max), d| {
            assert!(!d.is_nan());
            (min.min(*d), max.max(*d))
        });
        assert!(min <= max);
        let scale = (max - min) / ((1 << self.n) - 1) as f32;
        let trivial_quantized = vector
            .iter()
            .map(|d| ((*d - min) / scale).round() as u8)
            .collect::<Vec<_>>();

        // Transform the trivially quantized vector to produce a vector containing `n` bitvectors,
        // each containing only the i'th bit of each dimension.
        let doc_bytes = self.doc_bytes(vector.len());
        let mut quantized = vec![0u8; self.query_bytes(vector.len())];
        for (i, chunk) in trivial_quantized.chunks(8).enumerate() {
            let word = if chunk.len() == 8 {
                u64::from_le_bytes(chunk.try_into().expect("exactly 8 bytes"))
            } else {
                assert!(chunk.len() <= 8);
                let mut bytes = [0u8; 8];
                bytes[0..chunk.len()].copy_from_slice(chunk);
                u64::from_le_bytes(bytes)
            };
            for bit in 0..self.n {
                quantized[doc_bytes * bit + i] = Self::summarize_chunk(word, bit);
            }
        }

        quantized
    }

    fn query_bytes(&self, dimensions: usize) -> usize {
        self.doc_bytes(dimensions) * self.n
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I8NaiveQuantizer;

impl Quantizer for I8NaiveQuantizer {
    fn for_doc(&self, vector: &[f32]) -> Vec<u8> {
        let norm = crate::distance::dot_f32(vector, vector).sqrt() as f32;
        let mut normalized_vector = vector.to_vec();
        for d in normalized_vector.iter_mut() {
            *d /= norm;
        }

        // In the normalized vector all dimensions are in [-1.0, 1.0], so we can multiply by i8::MAX and round.
        // TODO: this is very conservative, we could probably be less conservative and choose a larger value
        // (with a clamp) based on dimensionality. We could also adjust values for matryoshka models.
        //
        // Save the squared l2 norm for use computing l2 distance.
        // TODO: only store this for l2 similarity, it's unnecessary for dot.
        let mut quantized = Vec::with_capacity(self.doc_bytes(vector.len()));
        for f in normalized_vector {
            quantized.push(((f * i8::MAX as f32).round() as i8).to_le_bytes()[0]);
        }
        quantized.extend_from_slice(&(norm * norm).to_le_bytes());
        quantized
    }

    fn doc_bytes(&self, dimensions: usize) -> usize {
        dimensions + std::mem::size_of::<f32>()
    }

    // TODO: "quantize" the query as [f32] and score asymmetrically to preserve precision.
    fn for_query(&self, vector: &[f32]) -> Vec<u8> {
        self.for_doc(vector)
    }

    fn query_bytes(&self, dimensions: usize) -> usize {
        self.doc_bytes(dimensions)
    }
}

/// Quantize to an i8 value per dimension but shaped to the input vector.
///
/// This stores two additional float values:
/// * A scale value that can be used to de-quantized into a normalized float vector.
/// * The l2 norm.
#[derive(Debug, Copy, Clone)]
pub struct I8ScaledUniformQuantizer;

impl Quantizer for I8ScaledUniformQuantizer {
    fn doc_bytes(&self, dimensions: usize) -> usize {
        dimensions + std::mem::size_of::<f32>() * 2
    }

    fn for_doc(&self, vector: &[f32]) -> Vec<u8> {
        let l2_norm = crate::distance::dot_f32(vector, vector).sqrt() as f32;
        let (scale, inv_scale) =
            if let Some(max) = vector.iter().map(|d| d.abs()).max_by(|a, b| a.total_cmp(b)) {
                (
                    (f64::from(i8::MAX) / max as f64) as f32,
                    (max as f64 / f64::from(i8::MAX)) as f32,
                )
            } else {
                (0.0, 0.0)
            };

        let mut quantized = Vec::with_capacity(self.doc_bytes(vector.len()));
        quantized.extend_from_slice(&inv_scale.to_le_bytes());
        quantized.extend_from_slice(&l2_norm.to_le_bytes());
        quantized.extend(
            vector
                .iter()
                .map(|d| ((*d * scale).round() as i8).to_le_bytes()[0]),
        );
        quantized
    }

    fn query_bytes(&self, dimensions: usize) -> usize {
        self.doc_bytes(dimensions)
    }

    fn for_query(&self, vector: &[f32]) -> Vec<u8> {
        self.for_doc(vector)
    }
}

// TODO: quantizer that is non-uniform for MRL vectors.
// Bonus points if it can still be scored on the quantized rep instead of de-quantizing.
