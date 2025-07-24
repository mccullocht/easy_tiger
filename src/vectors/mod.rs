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

// XXX so we are able to create a formatter that is run on ingress and can be used for both the
// nav vector and raw vector tables, which also makes the raw vector table _optional_.

// XXX i don't want the raw format to dictate the scorer we use because that would force scoring
// to be symmetric and that is not great for us. for a given table we know both the similarity
// and the format so we can create a symmetric implementation but we might need more information
// to create an asymmetric implementation.

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

// XXX should there be a raw normalized format??? it's not really an alternate format but rather
// a processing step.
pub enum F32VectorCoding {
    /// Little-endian f32 values encoded as bytes.
    Raw,
    /// Little-endian f32 values encoded as bytes, but l2 normalized first.
    /// This loses vector magnitude but the result will always be suitable for angular distance.
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
    // XXX should this accept the VectorSimilarity? it would allow me to normalize.
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

// TODO: remove this entirely.
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

// TODO: remove this entirely.
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
