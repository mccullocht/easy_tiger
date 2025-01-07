//! Utilities for vector quantization.
//!
//! Graph navigation during search uses these quantized vectors.

/// Return the number of output bytes required to binary quantize a vector of `dimensions` length.
pub fn binary_quantized_bytes(dimensions: usize) -> usize {
    BinaryQuantizer.index_bytes(dimensions)
}

/// Return binary quantized form of `vector`.
pub fn binary_quantize(vector: &[f32]) -> Vec<u8> {
    BinaryQuantizer.for_index(vector)
}

/// `Quantizer` is used to perform lossy quantization of input vectors.
///
/// Methods are exposed to quantize for indexing vectors in the database or for
/// querying. In simpler schemes these methods are identical and quantization is
/// symmetric; some implementations are asymmetric and use larger vectors for
/// querying to produce higher fidelity scores.
pub trait Quantizer {
    /// Quantize this vector for querying an index.
    fn for_query(&self, vector: &[f32]) -> Vec<u8>;

    /// Return the size of a quantized query vector for the provided dimensionality.
    fn query_bytes(&self, dimensions: usize) -> usize;

    /// Quantize this vector for indexing.
    fn for_index(&self, vector: &[f32]) -> Vec<u8>;

    /// Return the size of a quantized index vector for the provided dimensionality.
    fn index_bytes(&self, dimensions: usize) -> usize;
}

/// Reduce each dimension to a single bit.
///
/// This quantizer is stateless. It quantizes all values > 0 to a 1 bit and all other values to 0.
/// This works best if the data set is centered around the origin.
#[derive(Debug, Copy, Clone)]
pub struct BinaryQuantizer;

impl Quantizer for BinaryQuantizer {
    fn for_index(&self, vector: &[f32]) -> Vec<u8> {
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

    fn index_bytes(&self, dimensions: usize) -> usize {
        (dimensions + 7) / 8
    }

    fn for_query(&self, vector: &[f32]) -> Vec<u8> {
        self.for_index(vector)
    }

    fn query_bytes(&self, dimensions: usize) -> usize {
        self.index_bytes(dimensions)
    }
}
