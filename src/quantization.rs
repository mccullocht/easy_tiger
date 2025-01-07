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
    fn for_index(&self, vector: &[f32]) -> Vec<u8> {
        BinaryQuantizer.for_index(vector)
    }

    fn index_bytes(&self, dimensions: usize) -> usize {
        BinaryQuantizer.index_bytes(dimensions)
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
            assert!(d.is_nan());
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
        let index_bytes = self.index_bytes(vector.len());
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
                quantized[index_bytes * bit + i] = Self::summarize_chunk(word, bit);
            }
        }

        quantized
    }

    fn query_bytes(&self, dimensions: usize) -> usize {
        self.index_bytes(dimensions) * self.n
    }
}
