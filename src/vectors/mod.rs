//! Vector handling: formatting/quantization and distance computation.

use std::fmt::Debug;

use crate::{
    distance::{
        AsymmetricHammingDistance, F32DotProductDistance, F32EuclideanDistance, HammingDistance,
        I8NaiveDistance, I8ScaledUniformDotProduct, I8ScaledUniformEuclidean, VectorDistance,
        VectorSimilarity,
    },
    quantization::VectorQuantizer,
    vectors::{
        raw::{RawF32VectorCoder, RawL2NormalizedF32VectorCoder},
        scaled_uniform::I8ScaledUniformVectorCoder,
    },
};

mod binary;
mod i8naive;
mod raw;
mod scaled_uniform;

pub(crate) use binary::{AsymmetricBinaryQuantizedVectorCoder, BinaryQuantizedVectorCoder};
pub(crate) use i8naive::I8NaiveVectorCoder;

// XXX immediate TODOs
// * invert relationship between Quantizer and F32VectorCoding.

// XXX some changes that need to happen:
// * in bulk loading we should always use the WT table because it represent a transform.
// * mod quantization goes away entirely, including factory functions.
// * mod query_distance and mod distance also go away. factory functions move here.

/// Supported coding schemes for input f32 vectors.
///
/// Raw vectors are stored little endian but the remaining formats are all lossy in some way with
/// varying degrees of compression and fidelity in distance computation.
#[derive(Debug, Copy, Clone, Default)]
pub enum F32VectorCoding {
    /// Little-endian f32 values encoded as bytes.
    #[default]
    Raw,
    /// Little-endian f32 values encoded as bytes, but l2 normalized first.
    /// The resulting unit vectors can be used to cheaply compute angular distance.
    RawL2Normalized,
    /// Single bit (sign bit) per dimension.
    ///
    /// This encoding is very compact and efficient for distance computation but also does not have
    /// high fidelity with distances computed between raw vectors.
    BinaryQuantized,
    /// Quantize to N bits per dimension but transpose to create N bit vectors.
    ///
    /// This allows us to emulate dot product when comparing to a binary quantized vector, and
    /// produces higher fidelity distance than binary quantization on its own.
    NBitBinaryQuantized(usize),
    /// Normalize and quantize into an i8 value.
    ///
    /// This normalizes the input vector but otherwise does not shape quantization to the input.
    ///
    /// This uses 1 byte per dimension + 4 bytes for the l2 norm for euclidean distances.
    I8NaiveQuantized,
    /// Normalize and quantize into an i8 value, shaped to the input vector.
    ///
    /// This uses the contents of the vector to try to reduce quantization error but no data from
    /// othr vectors in the data set.
    ///
    /// This uses 1 byte per dimension and 8 additional bytes for a scaling factor and l2 norm.
    I8ScaledUniform,
}

impl F32VectorCoding {
    /// Create a new coder for this format.
    pub fn new_coder(&self) -> Box<dyn F32VectorCoder> {
        match self {
            Self::Raw => Box::new(RawF32VectorCoder),
            Self::RawL2Normalized => Box::new(RawL2NormalizedF32VectorCoder),
            Self::BinaryQuantized => Box::new(BinaryQuantizedVectorCoder),
            Self::NBitBinaryQuantized(n) => Box::new(AsymmetricBinaryQuantizedVectorCoder::new(*n)),
            Self::I8NaiveQuantized => Box::new(I8NaiveVectorCoder),
            Self::I8ScaledUniform => Box::new(I8ScaledUniformVectorCoder),
        }
    }

    // XXX I don't like this very much because it assumes the same format on both sides but abinary
    // is obviously not symmetric. i'm just going it to make this easier to retrofit existing code
    // using VectorQuantizer. This is pretty obviously wrong.
    // XXX maybe just return Option<...> and add 'symmetric' to the name? or maybe i hide abinary
    // in QueryVectorDistance once and for all, if that is even possible???
    pub fn new_distance_fn(&self, similarity: VectorSimilarity) -> Box<dyn VectorDistance> {
        match (self, similarity) {
            (Self::Raw, VectorSimilarity::Dot) => Box::new(F32DotProductDistance),
            (Self::RawL2Normalized, VectorSimilarity::Dot) => Box::new(F32DotProductDistance),
            (Self::Raw, VectorSimilarity::Euclidean) => Box::new(F32EuclideanDistance),
            (Self::RawL2Normalized, VectorSimilarity::Euclidean) => Box::new(F32EuclideanDistance),
            (Self::BinaryQuantized, _) => Box::new(HammingDistance),
            (Self::NBitBinaryQuantized(_), _) => Box::new(AsymmetricHammingDistance),
            (Self::I8NaiveQuantized, _) => Box::new(I8NaiveDistance(similarity)),
            (Self::I8ScaledUniform, VectorSimilarity::Dot) => Box::new(I8ScaledUniformDotProduct),
            (Self::I8ScaledUniform, VectorSimilarity::Euclidean) => {
                Box::new(I8ScaledUniformEuclidean)
            }
        }
    }
}

impl From<VectorQuantizer> for F32VectorCoding {
    fn from(value: VectorQuantizer) -> Self {
        match value {
            VectorQuantizer::Binary => Self::BinaryQuantized,
            VectorQuantizer::AsymmetricBinary { n } => Self::NBitBinaryQuantized(n),
            VectorQuantizer::I8Naive => Self::I8NaiveQuantized,
            VectorQuantizer::I8ScaledUniform => Self::I8ScaledUniform,
        }
    }
}

/// Encode an f32 vector into byte stream, possibly quantizing the vector in the process.
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
