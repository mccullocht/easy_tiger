//! Vector handling: formatting/quantization and distance computation.

use std::{borrow::Cow, fmt::Debug, io, str::FromStr};

use crate::vectors::{
    binary::{AsymmetricHammingDistance, HammingDistance},
    i8naive::I8NaiveDistance,
    raw::{
        F32DotProductDistance, F32EuclideanDistance, F32QueryVectorDistance, RawF32VectorCoder,
        RawL2NormalizedF32VectorCoder,
    },
    scaled_uniform::{
        I8ScaledUniformDotProduct, I8ScaledUniformDotProductQueryDistance,
        I8ScaledUniformEuclidean, I8ScaledUniformEuclideanQueryDistance,
    },
};

mod binary;
mod i8naive;
mod raw;
mod scaled_uniform;

pub(crate) use binary::{AsymmetricBinaryQuantizedVectorCoder, BinaryQuantizedVectorCoder};
pub(crate) use i8naive::I8NaiveVectorCoder;
pub(crate) use scaled_uniform::I8ScaledUniformVectorCoder;
use serde::{Deserialize, Serialize};

/// Functions used for to compute the distance between two vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum VectorSimilarity {
    /// Euclidean (l2) distance.
    Euclidean,
    /// Dot product scoring, an approximation of cosine scoring.
    /// Vectors used for this distance function must be normalized.
    #[default]
    Dot,
}

impl VectorSimilarity {
    /// Return an [`F32VectorDistance`] for this similarity function.
    pub fn new_distance_function(self) -> Box<dyn F32VectorDistance> {
        match self {
            Self::Euclidean => Box::new(F32EuclideanDistance),
            Self::Dot => Box::new(F32DotProductDistance),
        }
    }

    /// Returns the default [F32VectorCoding] to use for this similarity function.
    pub fn vector_coding(&self) -> F32VectorCoding {
        match self {
            Self::Euclidean => F32VectorCoding::Raw,
            Self::Dot => F32VectorCoding::RawL2Normalized,
        }
    }
}

impl FromStr for VectorSimilarity {
    type Err = io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "euclidean" => Ok(VectorSimilarity::Euclidean),
            "dot" => Ok(VectorSimilarity::Dot),
            x => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown similarity function {x}"),
            )),
        }
    }
}

/// Distance function for coded vectors.
///
/// This trait is object-safe; it may be instantiated at runtime based on
/// data that appears in a file or other backing store.
pub trait VectorDistance: Send + Sync {
    /// Score the `query` vector against the `doc` vector. Returns a score
    /// where larger values are better matches.
    ///
    /// This function is not required to be commutative and may panic if
    /// one of the inputs is misshapen.
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64;
}

/// Distance function for `f32` vectors.
///
/// This trait is object-safe; it may be instantiated at runtime based on
/// data that appears in a file or other backing store.
pub trait F32VectorDistance: VectorDistance {
    /// Score vectors `a` and `b` against one another. Returns a score
    /// where larger values are better matches.
    ///
    /// Input vectors must be the same length or this function may panic.
    fn distance_f32(&self, a: &[f32], b: &[f32]) -> f64;

    /// Normalize a vector for use with this scoring function.
    /// By default, does nothing.
    // TODO: remove this in favor of F32VectorCoding.
    fn normalize<'a>(&self, vector: Cow<'a, [f32]>) -> Cow<'a, [f32]> {
        vector
    }
}

/// Supported coding schemes for input f32 vectors.
///
/// Raw vectors are stored little endian but the remaining formats are all lossy in some way with
/// varying degrees of compression and fidelity in distance computation.
#[derive(Debug, Copy, Clone, Default, Serialize, Deserialize)]
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
    I8ScaledUniformQuantized,
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
            Self::I8ScaledUniformQuantized => Box::new(I8ScaledUniformVectorCoder),
        }
    }

    /// Returns a [VectorDistance] for symmetrical vector codings, or [None] if the encoding is not
    /// symmetrical.
    pub fn new_symmetric_vector_distance(
        &self,
        similarity: VectorSimilarity,
    ) -> Option<Box<dyn VectorDistance>> {
        match (self, similarity) {
            (Self::Raw, VectorSimilarity::Dot) => Some(Box::new(F32DotProductDistance)),
            (Self::RawL2Normalized, VectorSimilarity::Dot) => Some(Box::new(F32DotProductDistance)),
            (Self::Raw, VectorSimilarity::Euclidean) => Some(Box::new(F32EuclideanDistance)),
            (Self::RawL2Normalized, VectorSimilarity::Euclidean) => {
                Some(Box::new(F32EuclideanDistance))
            }
            (Self::BinaryQuantized, _) => Some(Box::new(HammingDistance)),
            (Self::NBitBinaryQuantized(_), _) => None,
            (Self::I8NaiveQuantized, _) => Some(Box::new(I8NaiveDistance(similarity))),
            (Self::I8ScaledUniformQuantized, VectorSimilarity::Dot) => {
                Some(Box::new(I8ScaledUniformDotProduct))
            }
            (Self::I8ScaledUniformQuantized, VectorSimilarity::Euclidean) => {
                Some(Box::new(I8ScaledUniformEuclidean))
            }
        }
    }

    /// Returns true if this coding can be scored symmetrically, where both vectors are using the
    /// same coding. Only encodings that are symmetrical can be used for vectors stored on disk.
    pub fn is_symmetric(&self) -> bool {
        !matches!(self, Self::NBitBinaryQuantized(_))
    }
}

impl FromStr for F32VectorCoding {
    type Err = io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let input_err = |s| io::Error::new(io::ErrorKind::InvalidInput, s);
        match s {
            "raw" => Ok(Self::Raw),
            "raw-l2-norm" => Ok(Self::RawL2Normalized),
            "binary" => Ok(Self::BinaryQuantized),
            ab if ab.starts_with("asymmetric_binary:") => {
                let bits_str = ab
                    .strip_prefix("asymmetric_binary:")
                    .expect("prefix matched");
                bits_str
                    .parse::<usize>()
                    .ok()
                    .and_then(|b| if (1..=8).contains(&b) { Some(b) } else { None })
                    .map(Self::NBitBinaryQuantized)
                    .ok_or_else(|| input_err(format!("invalid asymmetric_binary bits {bits_str}")))
            }
            "i8-naive" => Ok(Self::I8NaiveQuantized),
            "i8-scaled-uniform" => Ok(Self::I8ScaledUniformQuantized),
            _ => Err(input_err(format!("unknown quantizer function {s}"))),
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

/// Compute the distance between a fixed vector provided at creation time and other vectors.
/// This is often useful in query flows where everything references a specific point.
pub trait QueryVectorDistance: Send + Sync {
    fn distance(&self, vector: &[u8]) -> f64;
}

#[derive(Debug, Clone)]
struct QuantizedQueryVectorDistance<'a, D> {
    distance_fn: D,
    query: Cow<'a, [u8]>,
}

impl<'a, D: VectorDistance> QuantizedQueryVectorDistance<'a, D> {
    fn from_f32(distance_fn: D, query: &'a [f32], coder: impl F32VectorCoder) -> Self {
        let query = coder.encode(query).into();
        Self { distance_fn, query }
    }

    fn from_quantized(distance_fn: D, query: &'a [u8]) -> Self {
        Self {
            distance_fn,
            query: query.into(),
        }
    }
}

impl<'a, D: VectorDistance> QueryVectorDistance for QuantizedQueryVectorDistance<'a, D> {
    fn distance(&self, vector: &[u8]) -> f64 {
        self.distance_fn.distance(self.query.as_ref(), vector)
    }
}

/// Create a new [QueryVectorDistance] given a query, similarity function, and vector coding.
pub fn new_query_vector_distance_f32<'a>(
    query: &'a [f32],
    similarity: VectorSimilarity,
    coding: F32VectorCoding,
) -> Box<dyn QueryVectorDistance + 'a> {
    match (similarity, coding) {
        (VectorSimilarity::Dot, F32VectorCoding::Raw)
        | (VectorSimilarity::Dot, F32VectorCoding::RawL2Normalized) => {
            Box::new(F32QueryVectorDistance::new(
                F32DotProductDistance,
                query,
                matches!(coding, F32VectorCoding::RawL2Normalized),
            ))
        }
        (VectorSimilarity::Euclidean, F32VectorCoding::Raw)
        | (VectorSimilarity::Euclidean, F32VectorCoding::RawL2Normalized) => {
            Box::new(F32QueryVectorDistance::new(
                F32EuclideanDistance,
                query,
                matches!(coding, F32VectorCoding::RawL2Normalized),
            ))
        }
        (_, F32VectorCoding::BinaryQuantized) => Box::new(QuantizedQueryVectorDistance::from_f32(
            HammingDistance,
            query,
            BinaryQuantizedVectorCoder,
        )),
        (_, F32VectorCoding::NBitBinaryQuantized(n)) => {
            Box::new(QuantizedQueryVectorDistance::from_f32(
                AsymmetricHammingDistance,
                query,
                AsymmetricBinaryQuantizedVectorCoder::new(n),
            ))
        }
        (_, F32VectorCoding::I8NaiveQuantized) => Box::new(QuantizedQueryVectorDistance::from_f32(
            I8NaiveDistance(similarity),
            query,
            I8NaiveVectorCoder,
        )),
        (VectorSimilarity::Dot, F32VectorCoding::I8ScaledUniformQuantized) => {
            Box::new(I8ScaledUniformDotProductQueryDistance::new(query))
        }
        (VectorSimilarity::Euclidean, F32VectorCoding::I8ScaledUniformQuantized) => {
            Box::new(I8ScaledUniformEuclideanQueryDistance::new(query))
        }
    }
}

/// Create a new [QueryVectorDistance] for indexing that _requires_ symmetrical distance computation.
pub fn new_query_vector_distance_indexing<'a>(
    query: &'a [u8],
    similarity: VectorSimilarity,
    coding: F32VectorCoding,
) -> Box<dyn QueryVectorDistance + 'a> {
    match (similarity, coding) {
        (VectorSimilarity::Dot, F32VectorCoding::Raw)
        | (VectorSimilarity::Dot, F32VectorCoding::RawL2Normalized) => Box::new(
            QuantizedQueryVectorDistance::from_quantized(F32DotProductDistance, query),
        ),
        (VectorSimilarity::Euclidean, F32VectorCoding::Raw)
        | (VectorSimilarity::Euclidean, F32VectorCoding::RawL2Normalized) => Box::new(
            QuantizedQueryVectorDistance::from_quantized(F32EuclideanDistance, query),
        ),
        (_, F32VectorCoding::BinaryQuantized) => Box::new(
            QuantizedQueryVectorDistance::from_quantized(HammingDistance, query),
        ),
        (_, F32VectorCoding::NBitBinaryQuantized(_)) => Box::new(
            QuantizedQueryVectorDistance::from_quantized(HammingDistance, query),
        ),
        (_, F32VectorCoding::I8NaiveQuantized) => Box::new(
            QuantizedQueryVectorDistance::from_quantized(I8NaiveDistance(similarity), query),
        ),
        (VectorSimilarity::Dot, F32VectorCoding::I8ScaledUniformQuantized) => Box::new(
            QuantizedQueryVectorDistance::from_quantized(I8ScaledUniformDotProduct, query),
        ),
        (VectorSimilarity::Euclidean, F32VectorCoding::I8ScaledUniformQuantized) => Box::new(
            QuantizedQueryVectorDistance::from_quantized(I8ScaledUniformEuclidean, query),
        ),
    }
}

#[cfg(test)]
mod test {
    use super::raw::{F32DotProductDistance, F32EuclideanDistance};
    use super::scaled_uniform::{I8ScaledUniformDotProduct, I8ScaledUniformEuclidean};
    use crate::vectors::i8naive::I8NaiveDistance;
    use crate::vectors::{
        F32VectorCoder, F32VectorDistance, I8NaiveVectorCoder, I8ScaledUniformVectorCoder,
        VectorDistance, VectorSimilarity,
    };

    struct TestVector {
        rvec: Vec<f32>,
        qvec: Vec<u8>,
    }

    impl TestVector {
        pub fn new(
            rvec: Vec<f32>,
            f32_dist_fn: impl F32VectorDistance,
            coder: impl F32VectorCoder,
        ) -> Self {
            let qvec = coder.encode(&rvec);
            Self {
                rvec: f32_dist_fn.normalize(rvec.into()).to_vec(),
                qvec,
            }
        }
    }

    macro_rules! assert_float_near {
        ($expected:expr, $actual:expr, $epsilon:expr) => {{
            let range = ($expected * (1.0 - $epsilon))..=($expected * (1.0 + $epsilon));
            assert!(
                range.contains(&$actual),
                "expected {} (range={:?}) actual {}",
                $expected,
                range,
                $actual
            );
        }};
    }

    fn distance_compare_threshold(
        f32_dist_fn: impl F32VectorDistance + Copy,
        coder: impl F32VectorCoder + Copy,
        dist_fn: impl VectorDistance + Copy,
        a: Vec<f32>,
        b: Vec<f32>,
        threshold: f64,
    ) {
        let a = TestVector::new(a, f32_dist_fn, coder);
        let b = TestVector::new(b, f32_dist_fn, coder);

        let rf32_dist = f32_dist_fn.distance_f32(&a.rvec, &b.rvec);
        let ru8_dist =
            f32_dist_fn.distance(bytemuck::cast_slice(&a.rvec), bytemuck::cast_slice(&b.rvec));
        assert_float_near!(rf32_dist, ru8_dist, 0.00001);
        let qdist = dist_fn.distance(&a.qvec, &b.qvec);
        assert_float_near!(rf32_dist, qdist, threshold);
    }

    #[test]
    fn i8_naive_dot() {
        // TODO: randomly generate a bunch of vectors for this test.
        distance_compare_threshold(
            F32DotProductDistance,
            I8NaiveVectorCoder,
            I8NaiveDistance(VectorSimilarity::Dot),
            vec![-1.0f32, 2.5, 0.7, -1.7],
            vec![-0.6f32, -1.2, 0.4, 0.3],
            0.01,
        );
    }

    #[test]
    fn i8_naive_l2() {
        distance_compare_threshold(
            F32EuclideanDistance,
            I8NaiveVectorCoder,
            I8NaiveDistance(VectorSimilarity::Euclidean),
            vec![-1.0f32, 2.5, 0.7, -1.7],
            vec![-0.6f32, -1.2, 0.4, 0.3],
            0.01,
        );
    }

    #[test]
    fn i8_scaled_dot() {
        // TODO: randomly generate a bunch of vectors for this test.
        distance_compare_threshold(
            F32DotProductDistance,
            I8ScaledUniformVectorCoder,
            I8ScaledUniformDotProduct,
            vec![-1.0f32, 2.5, 0.7, -1.7],
            vec![-0.6f32, -1.2, 0.4, 0.3],
            0.01,
        );
    }

    #[test]
    fn i8_scaled_l2() {
        distance_compare_threshold(
            F32EuclideanDistance,
            I8ScaledUniformVectorCoder,
            I8ScaledUniformEuclidean,
            vec![-1.0f32, 2.5, 0.7, -1.7],
            vec![-0.6f32, -1.2, 0.4, 0.3],
            0.01,
        );
    }
}
