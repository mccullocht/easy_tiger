//! Utilities for vector quantization.
//!
//! Graph navigation during search uses these quantized vectors.

use std::{io, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::distance::{
    AsymmetricHammingDistance, HammingDistance, I8NaiveDistance, OptimizedScalarDistance1,
    OptimizedScalarDistance7, QuantizedVectorDistance, TrivialQuantizedDistance, VectorSimilarity,
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
    /// Encodes a float vector as a byte array in little-endian order.
    Trivial,
    /// Optimized scalar; 7 bits.
    OSQ7,
    /// Optimized scalar; 1 bit.
    OSQ1,
}

impl VectorQuantizer {
    /// Create a new quantizer for this quantization method.
    pub fn new_quantizer(&self) -> Box<dyn Quantizer> {
        match self {
            Self::Binary => Box::new(BinaryQuantizer),
            Self::AsymmetricBinary { n } => Box::new(AsymmetricBinaryQuantizer::new(*n)),
            Self::I8Naive => Box::new(I8NaiveQuantizer),
            Self::Trivial => Box::new(TrivialQuantizer),
            Self::OSQ7 => Box::new(OptimizedScalarQuantizer7),
            Self::OSQ1 => Box::new(OptimizedScalarQuantizer1),
        }
    }

    /// Create a new distance function for this quantization method.
    pub fn new_distance_function(
        &self,
        similarity: &VectorSimilarity,
    ) -> Box<dyn QuantizedVectorDistance> {
        match self {
            Self::Binary => Box::new(HammingDistance),
            Self::AsymmetricBinary { n: _ } => Box::new(AsymmetricHammingDistance),
            Self::I8Naive => Box::new(I8NaiveDistance(*similarity)),
            Self::Trivial => Box::new(TrivialQuantizedDistance(*similarity)),
            Self::OSQ7 => Box::new(OptimizedScalarDistance7(*similarity)),
            Self::OSQ1 => Box::new(OptimizedScalarDistance1(*similarity)),
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
                    .ok_or_else(|| {
                        input_err(format!("invalid asymmetric_binary bits {}", bits_str))
                    })
            }
            "i8naive" => Ok(Self::I8Naive),
            "trivial" => Ok(Self::Trivial),
            "osq7" => Ok(Self::OSQ7),
            "osq1" => Ok(Self::OSQ1),
            _ => Err(input_err(format!("unknown quantizer function {}", s))),
        }
    }
}

/// Trivial quantization just encodes a float vector as a little-endian byte vector.
pub struct TrivialQuantizer;

impl Quantizer for TrivialQuantizer {
    fn for_doc(&self, vector: &[f32]) -> Vec<u8> {
        vector.iter().flat_map(|d| d.to_le_bytes()).collect()
    }

    fn doc_bytes(&self, dimensions: usize) -> usize {
        dimensions * std::mem::size_of::<f32>()
    }

    fn for_query(&self, vector: &[f32]) -> Vec<u8> {
        self.for_doc(vector)
    }

    fn query_bytes(&self, dimensions: usize) -> usize {
        self.doc_bytes(dimensions)
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
        let norm = crate::distance::dot(vector, vector).sqrt() as f32;
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

/// Based on https://github.com/apache/lucene/blob/c8147c9e6fa19e12390f1b2f66e18c0af3654d44/lucene/core/src/java/org/apache/lucene/util/quantization/OptimizedScalarQuantizer.java
///
/// Unlike lucene OSQ this does not center, which means we don't need to re-quantize if the dataset
/// change. It also doesn't adjust for anisotropic loss, mostly because it's complicated but would
/// probably make the results better.
///
/// Compared to I8Naive this has a lower range (7 bits instead of 8) and stores 12 more bytes of
/// metadata in each vector but also reduces quantization error in the bits it does use.
pub struct OptimizedScalarQuantizer7;

impl Quantizer for OptimizedScalarQuantizer7 {
    fn for_doc(&self, vector: &[f32]) -> Vec<u8> {
        let (quantized_it, stats) = optimized_scalar_quantize::<7>(vector);
        let (mut quantized, component_sum) = quantized_it.fold(
            (Vec::with_capacity(self.doc_bytes(vector.len())), 0u32),
            |(mut vec, sum), q| {
                vec.push(q);
                (vec, sum + u32::from(q))
            },
        );
        OptimizedScalarQuantizedVector::pack_meta(&mut quantized, stats, component_sum);
        quantized
    }

    fn doc_bytes(&self, dimensions: usize) -> usize {
        dimensions + OptimizedScalarQuantizedVector::META_BYTES
    }

    fn for_query(&self, vector: &[f32]) -> Vec<u8> {
        self.for_doc(vector)
    }

    fn query_bytes(&self, dimensions: usize) -> usize {
        self.doc_bytes(dimensions)
    }
}

/// Like [`OptimizedScalarQuantizer7``] but produces a 1-bit quantization.
pub struct OptimizedScalarQuantizer1;

impl Quantizer for OptimizedScalarQuantizer1 {
    fn for_doc(&self, vector: &[f32]) -> Vec<u8> {
        let (quantized_it, stats) = optimized_scalar_quantize::<1>(vector);
        // Pack 8 dimensions into a single byte.
        let mut quantized = Vec::with_capacity(self.doc_bytes(vector.len()));
        quantized.resize(vector.len().div_ceil(8), 0);
        let component_sum = quantized_it.enumerate().fold(0u32, |sum, (i, q)| {
            quantized[i / 8] |= q << (i % 8);
            sum + u32::from(q)
        });
        OptimizedScalarQuantizedVector::pack_meta(&mut quantized, stats, component_sum);
        quantized
    }

    fn doc_bytes(&self, dimensions: usize) -> usize {
        dimensions.div_ceil(8) + OptimizedScalarQuantizedVector::META_BYTES
    }

    fn for_query(&self, vector: &[f32]) -> Vec<u8> {
        self.for_doc(vector)
    }

    fn query_bytes(&self, dimensions: usize) -> usize {
        self.doc_bytes(dimensions)
    }
}

/// Representation of an optimized scalar quantized vector.
#[derive(Debug, Copy, Clone)]
pub(crate) struct OptimizedScalarQuantizedVector<'a> {
    vector: &'a [u8],
    lower: f32,
    upper: f32,
    norm_sq: f32,
    component_sum: u32,
}

impl<'a> OptimizedScalarQuantizedVector<'a> {
    const META_BYTES: usize = 16;

    /// Pack metadata on the end of the vector.
    fn pack_meta(packed_vector: &mut Vec<u8>, stats: OSQStats, component_sum: u32) {
        packed_vector.extend_from_slice(&stats.lower.to_le_bytes());
        packed_vector.extend_from_slice(&stats.upper.to_le_bytes());
        packed_vector.extend_from_slice(&stats.norm_sq.to_le_bytes());
        packed_vector.extend_from_slice(&component_sum.to_le_bytes());
    }

    /// Initialize from raw bytes.
    ///
    /// Provides access to the packed, quantized vector bytes and unpacked metadata.
    ///
    /// Returns `None` if the input bytes are too short to contain a vector + metadata.
    pub fn from_bytes(bytes: &'a [u8]) -> Option<Self> {
        if bytes.len() <= Self::META_BYTES {
            return None;
        }

        let (vector, meta) = bytes.split_at(bytes.len() - Self::META_BYTES);
        let lower = f32::from_le_bytes(meta[0..4].try_into().expect("4 bytes"));
        let upper = f32::from_le_bytes(meta[4..8].try_into().expect("4 bytes"));
        let norm_sq = f32::from_le_bytes(meta[8..12].try_into().expect("4 bytes"));
        let component_sum = u32::from_le_bytes(meta[12..16].try_into().expect("4 bytes"));
        Some(Self {
            vector,
            lower,
            upper,
            norm_sq,
            component_sum,
        })
    }

    /// Return the packed, quantized vector.
    ///
    /// This may pack multiple dimensions into a single byte, depending on configuration.
    pub fn vector(&self) -> &[u8] {
        self.vector
    }

    /// Compute the distance between `query` and `doc`.
    ///
    /// `QBITS` is the number of bits each dimension of query vector is quantized into.
    /// `DBITS` is the number of bits each dimension of doc vector is quantized into.
    /// `dimensions` is the number of dimensions in both vectors.
    /// `dot` is the dot product distance between the two vectors.
    /// `similarity` is the expected similarity mode.
    pub fn distance<const QBITS: usize, const DBITS: usize>(
        query: &Self,
        doc: &Self,
        dimensions: usize,
        dot: f64,
        similarity: VectorSimilarity,
    ) -> f64 {
        let qrange = (query.upper as f64 - query.lower as f64) / ((1 << QBITS) - 1) as f64;
        let drange = (doc.upper as f64 - doc.lower as f64) / ((1 << DBITS) - 1) as f64;
        let dist = doc.lower as f64 * query.lower as f64 * dimensions as f64
            + query.lower as f64 * drange * doc.component_sum as f64
            + doc.lower as f64 * qrange * query.component_sum as f64
            + drange * qrange * dot;
        match similarity {
            VectorSimilarity::Dot => (-dist + 1.0) / 2.0,
            VectorSimilarity::Euclidean => query.norm_sq as f64 + doc.norm_sq as f64 - (2.0 * dist),
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct OSQStats {
    lower: f32,
    upper: f32,
    norm_sq: f32,
    mean: f64,
    variance: f64,
}

impl Default for OSQStats {
    fn default() -> Self {
        Self {
            lower: f32::MAX,
            upper: f32::MIN,
            norm_sq: 0.0,
            mean: 0.0,
            variance: 0.0,
        }
    }
}

impl From<&[f32]> for OSQStats {
    fn from(vector: &[f32]) -> Self {
        let mut stats = Self::default();
        for (i, v) in vector.iter().copied().enumerate() {
            stats.lower = stats.lower.min(v);
            stats.upper = stats.upper.max(v);
            stats.norm_sq += v * v;
            let delta = v as f64 - stats.mean;
            stats.mean += delta / (i as f64 + 1.0);
            stats.variance += delta * (v as f64 - stats.mean);
        }
        stats
    }
}

fn optimized_scalar_quantize<'a, const NBITS: usize>(
    vector: &'a [f32],
) -> (impl ExactSizeIterator<Item = u8> + 'a, OSQStats) {
    let stats = OSQStats::from(vector);
    let lower = stats.lower;
    let upper = stats.upper;
    let step = (stats.upper - stats.lower) / ((1 << NBITS) - 1) as f32;
    (
        vector
            .iter()
            .map(move |d| ((d.clamp(lower, upper) - lower) / step).round() as u8),
        stats,
    )
}

// compute min, max, norm^2 (sum(dx*dx))
// compute quantized value for each dimension
// compute sum of quantized values
// store (packed) vector, min, max, norm^2, sum of quantized values.

// for 1 we need only store magnitude and norm^2. can do signed values, 8 bytes

// for 2 add sum of assigned values to output and try to do correction

// for 3 allow n bit reps, we really only do 1, 4, 8.

// for 4 try doing the loss function.
