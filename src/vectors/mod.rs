//! Vector handling: formatting/quantization and distance computation.

use std::{borrow::Cow, fmt::Debug, io, str::FromStr};

use crate::vectors::{
    binary::{AsymmetricHammingDistance, HammingDistance},
    i8naive::I8NaiveDistance,
    raw::{
        F32DotProductDistance, F32EuclideanDistance, F32QueryVectorDistance, RawF32VectorCoder,
        RawL2NormalizedF32VectorCoder,
    },
};

mod binary;
mod i8naive;
mod raw;
mod scaled_uniform;

pub(crate) use binary::{AsymmetricBinaryQuantizedVectorCoder, BinaryQuantizedVectorCoder};
pub(crate) use i8naive::I8NaiveVectorCoder;
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
}

/// Supported coding schemes for input f32 vectors.
///
/// Raw vectors are stored little endian but the remaining formats are all lossy in some way with
/// varying degrees of compression and fidelity in distance computation.
#[derive(Debug, Copy, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
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
    /// Quantize into an i8 value shaped to the input vector.
    ///
    /// This uses the contents of the vector to try to reduce quantization error but no data from
    /// other vectors in the data set.
    ///
    /// This uses 1 byte per dimension and 8 additional bytes for a scaling factor and l2 norm.
    I8ScaledUniformQuantized,
    /// Quantize into an i4 value shaped to the input vector and pack 2 dimensions per byte.
    ///
    /// This uses the contents of the vector to try to reduce quantization error but no data from
    /// other vectors in the data set.
    ///
    /// This uses 1 byte per 2 dimensions and 8 additional bytes for a scaling factor and l2 norm.
    I4ScaledUniformQuantized,
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
            Self::I8ScaledUniformQuantized => Box::new(scaled_uniform::I8VectorCoder),
            Self::I4ScaledUniformQuantized => Box::new(scaled_uniform::I4PackedVectorCoder),
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
                Some(Box::new(scaled_uniform::I8DotProductDistance))
            }
            (Self::I8ScaledUniformQuantized, VectorSimilarity::Euclidean) => {
                Some(Box::new(scaled_uniform::I8EuclideanDistance))
            }
            (Self::I4ScaledUniformQuantized, VectorSimilarity::Dot) => {
                Some(Box::new(scaled_uniform::I4PackedDotProductDistance))
            }
            (Self::I4ScaledUniformQuantized, VectorSimilarity::Euclidean) => {
                Some(Box::new(scaled_uniform::I4PackedEuclideanDistance))
            }
        }
    }

    /// Returns true if this coding can be scored symmetrically, where both vectors are using the
    /// same coding. Only encodings that are symmetrical can be used for vectors stored on disk.
    pub fn is_symmetric(&self) -> bool {
        !matches!(self, Self::NBitBinaryQuantized(_))
    }

    /// Adjust raw format to normalize for angular similarity.
    pub fn adjust_raw_format(&self, similarity: VectorSimilarity) -> Self {
        match (self, similarity) {
            (Self::Raw, VectorSimilarity::Dot) => Self::RawL2Normalized,
            (_, _) => *self,
        }
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
            "i4-scaled-uniform" => Ok(Self::I4ScaledUniformQuantized),
            _ => Err(input_err(format!("unknown vector coding {s}"))),
        }
    }
}

impl std::fmt::Display for F32VectorCoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Raw => write!(f, "raw"),
            Self::RawL2Normalized => write!(f, "raw-l2-norm"),
            Self::BinaryQuantized => write!(f, "binary"),
            Self::NBitBinaryQuantized(n) => write!(f, "asymmetric_binary:{}", *n),
            Self::I8NaiveQuantized => write!(f, "i8-naive"),
            Self::I8ScaledUniformQuantized => write!(f, "i8-scaled-uniform"),
            Self::I4ScaledUniformQuantized => write!(f, "i4-scaled-uniform"),
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

    /// Decode `encoded` to a float vector.
    ///
    /// This is not supported for all codecs, and in cases where the format is packed may
    /// return more dimensions than originally specified.
    fn decode(&self, _encoded: &[u8]) -> Option<Vec<f32>> {
        None
    }
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
            Box::new(scaled_uniform::I8DotProductQueryDistance::new(query))
        }
        (VectorSimilarity::Euclidean, F32VectorCoding::I8ScaledUniformQuantized) => {
            Box::new(scaled_uniform::I8EuclideanQueryDistance::new(query))
        }
        (VectorSimilarity::Dot, F32VectorCoding::I4ScaledUniformQuantized) => {
            Box::new(scaled_uniform::I4PackedDotProductQueryDistance::new(query))
        }
        (VectorSimilarity::Euclidean, F32VectorCoding::I4ScaledUniformQuantized) => {
            Box::new(scaled_uniform::I4PackedEuclideanQueryDistance::new(query))
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
        (VectorSimilarity::Dot, F32VectorCoding::I8ScaledUniformQuantized) => {
            Box::new(QuantizedQueryVectorDistance::from_quantized(
                scaled_uniform::I8DotProductDistance,
                query,
            ))
        }
        (VectorSimilarity::Euclidean, F32VectorCoding::I8ScaledUniformQuantized) => {
            Box::new(QuantizedQueryVectorDistance::from_quantized(
                scaled_uniform::I8EuclideanDistance,
                query,
            ))
        }
        (VectorSimilarity::Dot, F32VectorCoding::I4ScaledUniformQuantized) => {
            Box::new(QuantizedQueryVectorDistance::from_quantized(
                scaled_uniform::I4PackedDotProductDistance,
                query,
            ))
        }
        (VectorSimilarity::Euclidean, F32VectorCoding::I4ScaledUniformQuantized) => {
            Box::new(QuantizedQueryVectorDistance::from_quantized(
                scaled_uniform::I4PackedEuclideanDistance,
                query,
            ))
        }
    }
}

#[cfg(test)]
mod test {
    use crate::vectors::{
        new_query_vector_distance_f32, F32VectorCoder, F32VectorCoding, VectorSimilarity,
    };

    struct TestVector {
        rvec: Vec<f32>,
        qvec: Vec<u8>,
    }

    impl TestVector {
        pub fn new(
            vec: &[f32],
            similarity: VectorSimilarity,
            coder: &(impl F32VectorCoder + ?Sized),
        ) -> Self {
            let f32_coder = match similarity {
                VectorSimilarity::Dot => F32VectorCoding::RawL2Normalized,
                VectorSimilarity::Euclidean => F32VectorCoding::Raw,
            }
            .new_coder();
            let rvec = f32_coder
                .encode(vec)
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>();
            let qvec = coder.encode(&vec);
            Self { rvec, qvec }
        }
    }

    macro_rules! assert_float_near {
        ($expected:expr, $actual:expr, $epsilon:expr, $index:expr) => {{
            let range = ($expected * (1.0 - $epsilon))..=($expected * (1.0 + $epsilon));
            assert!(
                range.contains(&$actual),
                "expected {} (range={:?}) actual {} index={}",
                $expected,
                range,
                $actual,
                $index
            );
        }};
    }

    fn distance_compare(
        similarity: VectorSimilarity,
        format: F32VectorCoding,
        index: usize,
        a: &[f32],
        b: &[f32],
        threshold: f64,
    ) {
        let coder = format.new_coder();
        let a = TestVector::new(a, similarity, coder.as_ref());
        let b = TestVector::new(b, similarity, coder.as_ref());

        let f32_dist_fn = similarity.new_distance_function();
        let rf32_dist = f32_dist_fn.distance_f32(&a.rvec, &b.rvec);
        let ru8_dist =
            f32_dist_fn.distance(bytemuck::cast_slice(&a.rvec), bytemuck::cast_slice(&b.rvec));
        assert_float_near!(rf32_dist, ru8_dist, 0.00001, index);

        let dist_fn = format.new_symmetric_vector_distance(similarity).unwrap();
        let qdist = dist_fn.distance(&a.qvec, &b.qvec);
        assert_float_near!(rf32_dist, qdist, threshold, index);
    }

    fn query_distance_compare(
        similarity: VectorSimilarity,
        format: F32VectorCoding,
        index: usize,
        a: &[f32],
        b: &[f32],
        threshold: f64,
    ) {
        let coder = format.new_coder();
        let a = TestVector::new(a, similarity, coder.as_ref());
        let b = TestVector::new(b, similarity, coder.as_ref());

        let f32_dist_fn = similarity.new_distance_function();
        let f32_dist = f32_dist_fn.distance_f32(&a.rvec, &b.rvec);

        let query_dist_fn = new_query_vector_distance_f32(&a.rvec, similarity, format);
        let query_dist = query_dist_fn.distance(&b.qvec);

        assert_float_near!(f32_dist, query_dist, threshold, index);
    }

    fn test_float_vectors() -> Vec<(Vec<f32>, Vec<f32>)> {
        // TODO: randomly generate a bunch of vectors for this test.
        vec![
            (vec![-1.0f32, 2.5, 0.7, -1.7], vec![-0.6f32, -1.2, 0.4, 0.3]),
            (
                vec![
                    1.22f32, 1.25, 2.37, -2.21, 2.28, -2.8, -0.61, 2.29, -2.56, -0.57, -2.62,
                    -1.56, 1.92, -0.63, 0.77, -2.86,
                ],
                vec![
                    3.19, 2.91, 0.23, -2.51, -0.76, 1.82, 1.97, 2.19, -0.15, -3.85, -3.14, -0.43,
                    1.06, -0.05, 2.05, -2.51,
                ],
            ),
        ]
    }

    use F32VectorCoding::{I4ScaledUniformQuantized, I8NaiveQuantized, I8ScaledUniformQuantized};
    use VectorSimilarity::{Dot, Euclidean};

    #[test]
    fn i8_naive_dot() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Dot, I8NaiveQuantized, i, &a, &b, 0.01);
        }
    }

    #[test]
    fn i8_naive_l2() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Euclidean, I8NaiveQuantized, i, &a, &b, 0.01);
        }
    }

    #[test]
    fn i8_scaled_dot() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Dot, I8ScaledUniformQuantized, i, &a, &b, 0.01);
            query_distance_compare(Dot, I8ScaledUniformQuantized, i, &a, &b, 0.01);
        }
    }

    #[test]
    fn i8_scaled_l2() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Euclidean, I8ScaledUniformQuantized, i, &a, &b, 0.01);
            query_distance_compare(Euclidean, I8ScaledUniformQuantized, i, &a, &b, 0.01);
        }
    }

    #[test]
    fn i4_scaled_dot() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Dot, I4ScaledUniformQuantized, i, &a, &b, 0.10);
            query_distance_compare(Dot, I4ScaledUniformQuantized, i, &a, &b, 0.10);
        }
    }

    #[test]
    fn i4_scaled_l2() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Euclidean, I4ScaledUniformQuantized, i, &a, &b, 0.10);
            query_distance_compare(Euclidean, I4ScaledUniformQuantized, i, &a, &b, 0.10);
        }
    }
}
