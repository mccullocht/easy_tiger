//! Vector handling: formatting/quantization and distance computation.

use std::fmt::Debug;

use crate::{
    quantization::{
        AsymmetricBinaryQuantizer, BinaryQuantizer, I8NaiveQuantizer, I8ScaledUniformQuantizer,
        Quantizer,
    },
    vectors::raw::{RawF32VectorCoder, RawL2NormalizedF32VectorCoder},
};

mod raw;

// XXX immediate TODOs
// * invert relationship between Quantizer and F32VectorCoding.

// XXX some changes that need to happen:
// * in bulk loading we should always use the WT table because it represent a transform.
// * mod quantization goes away entirely, including factory functions.
// * mod query_distance and mod distance also go away. factory functions move here.

// XXX docos
pub enum F32VectorCoding {
    /// Little-endian f32 values encoded as bytes.
    Raw,
    /// Little-endian f32 values encoded as bytes, but l2 normalized first.
    /// The resulting unit vectors can be used to cheaply compute angular distance.
    RawL2Normalized,
    /// Single bit (sign bit) per dimension.
    BinaryQuantized,
    /// Quantize to n-bits per dimension, but format to facilitate hamming distance calculations.
    NBitBinaryQuantized(usize),
    /// Normalize and quantize into an i8 value.
    /// This normalizes the input vector but otherwise does not shape quantization to the input.
    I8NaiveQuantized,
    /// Normalize and quantize into an i8 value, shaped to the input vector.
    /// This uses the contents of the vector to try to reduce quantization error.
    I8ScaledUniform,
}

impl F32VectorCoding {
    /// Create a new coder for this format.
    pub fn new_coder(&self) -> Box<dyn F32VectorCoder> {
        match self {
            Self::Raw => Box::new(RawF32VectorCoder),
            Self::RawL2Normalized => Box::new(RawL2NormalizedF32VectorCoder),
            Self::BinaryQuantized => Box::new(QuantizedVectorCoding(BinaryQuantizer)),
            Self::NBitBinaryQuantized(n) => {
                Box::new(NBitBinaryQuantized(AsymmetricBinaryQuantizer::new(*n)))
            }
            Self::I8NaiveQuantized => Box::new(QuantizedVectorCoding(I8NaiveQuantizer)),
            Self::I8ScaledUniform => Box::new(QuantizedVectorCoding(I8ScaledUniformQuantizer)),
        }
    }

    // XXX we can do symmetrical distance by using the same F32VectorCoding on both sides.
    // XXX we can also do asymmetrical for some combinations
}

// XXX docos
pub trait F32VectorCoder: Send + Sync {
    /// Encode the input vector and return the encoded byte buffer.
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut out = vec![0; self.byte_len(vector.len())];
        self.encode_to(vector, &mut out);
        out
    }

    /// Encode `vector` and write to `out`.
    ///
    /// *Panics* if `out.len() < self.byte_len(vector.len())`.
    fn encode_to(&self, vector: &[f32], out: &mut [u8]);

    /// Return the number of bytes required to encode a vector of length `dimensions`.
    fn byte_len(&self, dimensions: usize) -> usize;
}

// XXX remove this entirely, write native coders for each format.
#[derive(Debug, Copy, Clone)]
struct QuantizedVectorCoding<Q: Debug + Copy + Clone>(Q);

impl<Q: Quantizer + Debug + Copy + Clone> F32VectorCoder for QuantizedVectorCoding<Q> {
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        self.0.for_doc(vector)
    }

    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        out.copy_from_slice(&self.encode(vector));
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        self.0.doc_bytes(dimensions)
    }
}

// XXX remove this entirely, write a native coder for this format.
#[derive(Debug, Copy, Clone)]
struct NBitBinaryQuantized(AsymmetricBinaryQuantizer);

impl F32VectorCoder for NBitBinaryQuantized {
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        self.0.for_query(vector)
    }

    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        out.copy_from_slice(&self.encode(vector));
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        self.0.query_bytes(dimensions)
    }
}
