//! Utilities for vector quantization.
//!
//! Graph navigation during search uses these quantized vectors.

use std::{io, iter::FusedIterator, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::distance::{
    AsymmetricHammingDistance, HammingDistance, I8NaiveDistance, I8ScaledUniformDotProduct,
    I8ScaledUniformEuclidean, OptimizedScalarDistance1, OptimizedScalarDistance7, VectorDistance,
    VectorSimilarity,
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
            Self::I8ScaledUniform => Box::new(I8ScaledUniformQuantizer),
            Self::OSQ7 => Box::new(OptimizedScalarQuantizer7),
            Self::OSQ1 => Box::new(OptimizedScalarQuantizer1),
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
            (Self::OSQ7, _) => Box::new(OptimizedScalarDistance7(*similarity)),
            (Self::OSQ1, _) => Box::new(OptimizedScalarDistance1(*similarity)),
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
            "osq7" => Ok(Self::OSQ7),
            "osq1" => Ok(Self::OSQ1),
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
        let mut qiter = OSQIter::new(vector, 7);
        let mut quantized = Vec::with_capacity(self.doc_bytes(vector.len()));
        for q in qiter.by_ref() {
            quantized.push(q);
        }
        OptimizedScalarQuantizedVector::pack_meta(&mut quantized, *qiter.meta());
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
        let mut qiter = OSQIter::new(vector, 1);
        let mut quantized = Vec::with_capacity(self.doc_bytes(vector.len()));
        // Pack 8 dimensions into a single byte.
        quantized.resize(vector.len().div_ceil(8), 0);
        for (i, q) in qiter.by_ref().enumerate() {
            quantized[i / 8] |= q << (i % 8);
        }
        OptimizedScalarQuantizedVector::pack_meta(&mut quantized, *qiter.meta());
        quantized
    }

    fn doc_bytes(&self, dimensions: usize) -> usize {
        dimensions.div_ceil(8) + OptimizedScalarQuantizedVector::META_BYTES
    }

    fn for_query(&self, vector: &[f32]) -> Vec<u8> {
        let mut qiter = OSQIter::new(vector, 4);
        let mut quantized = Vec::with_capacity(self.query_bytes(vector.len()));
        let doc_bytes = vector.len().div_ceil(8);
        quantized.resize(doc_bytes * 4, 0);
        for (i, q) in qiter.by_ref().enumerate() {
            // Like in asymmetric binary each bit is packed into a like group.
            for bit in 0..4 {
                quantized[doc_bytes * bit + i / 8] |= ((q >> bit) & 0x1) << (i % 8);
            }
        }
        OptimizedScalarQuantizedVector::pack_meta(&mut quantized, *qiter.meta());
        quantized
    }

    fn query_bytes(&self, dimensions: usize) -> usize {
        dimensions.div_ceil(8) * 4 + OptimizedScalarQuantizedVector::META_BYTES
    }
}

/// Representation of an optimized scalar quantized vector.
#[derive(Debug, Copy, Clone)]
pub(crate) struct OptimizedScalarQuantizedVector<'a> {
    vector: &'a [u8],
    // XXX the rep of this should just be OSQMeta.
    lower: f32,
    upper: f32,
    norm_sq: f32,
    component_sum: u32,
}

impl<'a> OptimizedScalarQuantizedVector<'a> {
    const META_BYTES: usize = 16;

    /// Pack metadata on the end of the vector.
    fn pack_meta(packed_vector: &mut Vec<u8>, meta: OSQMeta) {
        packed_vector.extend_from_slice(&meta.lower.to_le_bytes());
        packed_vector.extend_from_slice(&meta.upper.to_le_bytes());
        packed_vector.extend_from_slice(&meta.norm_sq.to_le_bytes());
        packed_vector.extend_from_slice(&meta.component_sum.to_le_bytes());
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

    fn normalized_range<const BITS: usize>(&self) -> f64 {
        let r = self.upper as f64 - self.lower as f64;
        if BITS == 1 {
            r
        } else {
            r / ((1 << BITS) - 1) as f64
        }
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
        let adjusted_dot = doc.lower as f64 * query.lower as f64 * dimensions as f64
            + query.lower as f64 * doc.normalized_range::<DBITS>() * doc.component_sum as f64
            + doc.lower as f64 * query.normalized_range::<QBITS>() * query.component_sum as f64
            + doc.normalized_range::<DBITS>() * query.normalized_range::<QBITS>() * dot;
        match similarity {
            VectorSimilarity::Dot => (-adjusted_dot + 1.0) / 2.0,
            VectorSimilarity::Euclidean => {
                query.norm_sq as f64 + doc.norm_sq as f64 - (2.0 * adjusted_dot)
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct OSQMeta {
    lower: f32,
    upper: f32,
    norm_sq: f32,
    component_sum: u32,
}

struct OSQIter<'a> {
    raw_iter: std::slice::Iter<'a, f32>,
    meta: OSQMeta,
    step: f32,
}

impl<'a> OSQIter<'a> {
    fn new(vector: &'a [f32], bits: usize) -> Self {
        assert!(bits <= 8);
        let meta = Self::compute_meta(vector, bits);
        let step = (meta.upper - meta.lower) as f64 / ((1 << bits) - 1) as f64;
        Self {
            raw_iter: vector.iter(),
            meta,
            step: step as f32,
        }
    }

    fn meta(&self) -> &OSQMeta {
        // This should only be read when the iterator has been consumed.
        assert_eq!(self.raw_iter.len(), 0);
        &self.meta
    }

    // The intial interval is set to the minimum MSE grid for each number of bits.
    // These starting points are derived from the optimal MSE grid for a uniform distribution.
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

    fn compute_meta(vector: &[f32], bits: usize) -> OSQMeta {
        let (min, max, mean, var, norm_sq) = vector.iter().enumerate().fold(
            (f32::MAX, f32::MIN, 0.0, 0.0, 0.0),
            |(min, max, mean, var, norm_sq), (i, v)| {
                let delta = *v as f64 - mean;
                let mean = mean + delta / (i + 1) as f64;
                (
                    min.min(*v),
                    max.max(*v),
                    mean,
                    var + delta * (*v as f64 - mean),
                    norm_sq + (*v as f64 * *v as f64),
                )
            },
        );
        let std_dev = (var / vector.len() as f64).sqrt();
        let (lower, upper) = Self::optimize_intervals(
            vector,
            bits,
            (
                ((Self::MINIMUM_MSE_GRID[bits - 1].0 as f64 * std_dev + mean) as f32)
                    .clamp(min, max),
                ((Self::MINIMUM_MSE_GRID[bits - 1].1 as f64 * std_dev + mean) as f32)
                    .clamp(min, max),
            ),
            norm_sq,
            0.1,
            5,
        );
        OSQMeta {
            lower,
            upper,
            norm_sq: norm_sq as f32,
            component_sum: 0,
        }
    }

    fn optimize_intervals(
        vector: &[f32],
        bits: usize,
        mut interval: (f32, f32),
        norm_sq: f64,
        lambda: f64,
        iters: usize,
    ) -> (f32, f32) {
        // XXX osq1 recall is mid -- poorer at the outset, about the same when re-scored.
        // might be related to this.
        let qmax = ((1 << bits) - 1) as f64;
        let mut loss = Self::loss(vector, interval, qmax, norm_sq, lambda);
        let scale = (1.0 - lambda) / norm_sq;
        if !scale.is_finite() {
            return interval;
        }
        for _ in 0..iters {
            let step_inv = qmax as f32 / (interval.1 - interval.0);
            let (daa, dab, dbb, dax, dbx) =
                vector
                    .iter()
                    .fold((0.0, 0.0, 0.0, 0.0, 0.0), |(daa, dab, dbb, dax, dbx), v| {
                        let s = (v.clamp(interval.0, interval.1) * step_inv).round() as f64 / qmax;
                        (
                            daa + (1.0 - s) * (1.0 - s),
                            dab + (1.0 - s) * s,
                            dbb + s * s,
                            dax + *v as f64 * (1.0 - s),
                            dbx + *v as f64 * s,
                        )
                    });
            let m0 = scale * dax * dax + lambda * daa;
            let m1 = scale * dax * dbx + lambda * dab;
            let m2 = scale * dbx * dbx + lambda * dbb;
            let det = m0 * m2 - m1 * m1;
            // If the determinant is 0 then we can't update the interval.
            if det == 0.0 {
                break;
            }
            let new_interval = (
                ((m2 * dax - m1 * dbx) / det) as f32,
                (((m0 * dbx - m1 * dax) / det) as f32),
            );
            // If the interval doesn't change appreciably, then stop.
            if (new_interval.0 - interval.0).abs() < 1e-8
                && (new_interval.1 - interval.1).abs() < 1e-8
            {
                break;
            }
            let new_loss = Self::loss(vector, new_interval, qmax, norm_sq, lambda);
            // If the new loss is worse, skip updating the interval and exit.
            if new_loss > loss {
                break;
            }
            interval = new_interval;
            loss = new_loss;
        }
        interval
    }

    fn loss(vector: &[f32], interval: (f32, f32), qmax: f64, norm_sq: f64, lambda: f64) -> f64 {
        let step = (interval.1 - interval.0) as f64 / qmax;
        let step_inv = 1.0 / step;
        let (ve, e) = vector.iter().fold((0.0, 0.0), |(ve, e), v| {
            // quantize and dequantize the vector to measure the difference.
            let vq = interval.0 as f64
                + step * ((v.clamp(interval.0, interval.1) - interval.0) as f64 * step_inv).round();
            let delta = *v as f64 - vq;
            (ve + *v as f64 * delta, e + delta * delta)
        });
        (1.0 - lambda) * ve * ve / norm_sq + lambda * e
    }
}

impl Iterator for OSQIter<'_> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        let d = *self.raw_iter.next()?;
        let q = ((d.clamp(self.meta.lower, self.meta.upper) - self.meta.lower) / self.step).round()
            as u8;
        self.meta.component_sum += q as u32;
        Some(q)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.raw_iter.size_hint()
    }
}

impl FusedIterator for OSQIter<'_> {}

impl ExactSizeIterator for OSQIter<'_> {}

// compute min, max, norm^2 (sum(dx*dx))
// compute quantized value for each dimension
// compute sum of quantized values
// store (packed) vector, min, max, norm^2, sum of quantized values.

// for 1 we need only store magnitude and norm^2. can do signed values, 8 bytes

// for 2 add sum of assigned values to output and try to do correction

// for 3 allow n bit reps, we really only do 1, 4, 8.

// for 4 try doing the loss function.
