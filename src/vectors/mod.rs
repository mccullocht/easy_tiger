//! Vector handling: formatting/quantization and distance computation.

use std::{borrow::Cow, fmt::Debug, io, str::FromStr};

use crate::distance::l2_normalize;

mod binary;
mod float16;
mod float32;
mod lvq;
mod scaled_uniform;
mod truncated;

use serde::{Deserialize, Serialize};

/// Functions used for to compute the distance between two vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorSimilarity {
    /// Euclidean (l2) distance.
    Euclidean,
    /// Dot product distance.
    ///
    /// Assuming all input vectors are normalized this produces the same distance as `Cosine`.
    /// If your vectors are already l2 normalized this will be _much_ faster than `Cosine`.
    Dot,
    /// Cosine (angular) distance.
    ///
    /// Vectors stored in an index will be l2 normalized to speed up distance computation so
    /// egress vectors may not be identical to ingress vectors.
    ///
    /// If your vectors are already l2 normalized `Dot` will be _much_ faster.
    ///
    /// This function produces a distance in [0.0, 1.0]
    Cosine,
}

impl VectorSimilarity {
    /// Return an [`F32VectorDistance`] for this similarity function.
    pub fn new_distance_function(self) -> Box<dyn F32VectorDistance> {
        match self {
            Self::Euclidean => Box::new(float32::EuclideanDistance),
            Self::Dot => Box::new(float32::DotProductDistance),
            Self::Cosine => Box::new(float32::CosineDistance),
        }
    }

    /// Return true if vectors must be l2 normalized during encoding.
    pub fn l2_normalize(&self) -> bool {
        *self == Self::Cosine
    }

    pub fn all() -> impl ExactSizeIterator<Item = VectorSimilarity> {
        [
            VectorSimilarity::Euclidean,
            VectorSimilarity::Dot,
            VectorSimilarity::Cosine,
        ]
        .into_iter()
    }
}

impl FromStr for VectorSimilarity {
    type Err = io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "euclidean" | "l2" => Ok(VectorSimilarity::Euclidean),
            "cosine" | "cos" => Ok(VectorSimilarity::Cosine),
            "dot" => Ok(VectorSimilarity::Dot),
            x => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown similarity function {x}"),
            )),
        }
    }
}

impl std::fmt::Display for VectorSimilarity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Euclidean => write!(f, "l2"),
            Self::Cosine => write!(f, "cos"),
            Self::Dot => write!(f, "dot"),
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
    /// Little-endian f32 values.
    ///
    /// Depending on the similarity function this may be normalized or transformed in some other way
    /// so users should not rely on the value being identical.
    #[default]
    F32,
    /// Little-endian f32 values truncated at some maximum dimension.
    ///
    /// This is useful for Matryoshka Representation Learned (MRL) models.
    ///
    /// Note that for dot product similarity the truncated value will be l2 normalized before it is
    /// encoded.
    TruncatedF32(usize),
    /// Little-endian IEEE f16 encoding.
    F16,
    /// Single bit (sign bit) per dimension.
    ///
    /// This encoding is very compact and efficient for distance computation but also does not have
    /// high fidelity with distances computed between raw vectors.
    BinaryQuantized,
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
    /// Quantize into an i16 value shaped to the input vector.
    ///
    /// This uses the contents of the vector to try to reduce quantization error but no data from
    /// other vectors in the data set.
    ///
    /// This uses 2 bytes per dimension and 8 additional bytes for a scaling factor and l2 norm.
    I16ScaledUniformQuantized,
    /// LVQ one-level; 1 bit
    LVQ1x1,
    /// LVQ one-level; 4 bits
    LVQ1x4,
    /// LVQ one-level; 8 bits
    LVQ1x8,
    /// LVQ two-level; 1 bit primary 8 bits residual
    LVQ2x1x8,
    /// LVQ two-level; 1 bit primary 16 bits residual
    LVQ2x1x16,
    /// LVQ two-level; 4 bits primary 4 bits residual
    LVQ2x4x4,
    /// LVQ two-level; 4 bits primary 8 bits residual
    LVQ2x4x8,
    /// LVQ one-level; 8 bits primary 8 bits residual
    LVQ2x8x8,
}

impl F32VectorCoding {
    /// Create a new coder for this format.
    pub fn new_coder(&self, similarity: VectorSimilarity) -> Box<dyn F32VectorCoder> {
        match self {
            Self::F32 => Box::new(float32::VectorCoder::new(similarity)),
            Self::TruncatedF32(d) => Box::new(truncated::VectorCoder::new(similarity, *d)),
            Self::F16 => Box::new(float16::VectorCoder::new(similarity)),
            Self::BinaryQuantized => Box::new(binary::BinaryQuantizedVectorCoder),
            Self::I8ScaledUniformQuantized => {
                Box::new(scaled_uniform::I8VectorCoder::new(similarity))
            }
            Self::I4ScaledUniformQuantized => {
                Box::new(scaled_uniform::I4PackedVectorCoder::new(similarity))
            }
            Self::I16ScaledUniformQuantized => {
                Box::new(scaled_uniform::I16VectorCoder::new(similarity))
            }
            Self::LVQ1x1 => Box::new(lvq::PrimaryVectorCoder::<1>),
            Self::LVQ1x4 => Box::new(lvq::PrimaryVectorCoder::<4>),
            Self::LVQ1x8 => Box::new(lvq::PrimaryVectorCoder::<8>),
            Self::LVQ2x1x8 => Box::new(lvq::TwoLevelVectorCoder::<1, 8>),
            Self::LVQ2x1x16 => Box::new(lvq::TwoLevelVectorCoder::<1, 16>),
            Self::LVQ2x4x4 => Box::new(lvq::TwoLevelVectorCoder::<4, 4>),
            Self::LVQ2x4x8 => Box::new(lvq::TwoLevelVectorCoder::<4, 8>),
            Self::LVQ2x8x8 => Box::new(lvq::TwoLevelVectorCoder::<8, 8>),
        }
    }

    /// Returns a [VectorDistance] between vectors encoded using this coder.
    pub fn new_vector_distance(&self, similarity: VectorSimilarity) -> Box<dyn VectorDistance> {
        use VectorSimilarity::{Cosine, Dot, Euclidean};

        match (self, similarity) {
            (Self::F32, Cosine) => Box::new(float32::CosineDistance),
            (Self::F32, Dot) => Box::new(float32::DotProductDistance),
            (Self::F32, Euclidean) => Box::new(float32::EuclideanDistance),
            (Self::TruncatedF32(_), _) => F32VectorCoding::F32.new_vector_distance(similarity),
            (Self::F16, Dot) | (Self::F16, Cosine) => Box::new(float16::DotProductDistance),
            (Self::F16, Euclidean) => Box::new(float16::EuclideanDistance),
            (Self::BinaryQuantized, _) => Box::new(binary::HammingDistance),
            (Self::I8ScaledUniformQuantized, Dot) | (Self::I8ScaledUniformQuantized, Cosine) => {
                Box::new(scaled_uniform::I8DotProductDistance)
            }
            (Self::I8ScaledUniformQuantized, Euclidean) => {
                Box::new(scaled_uniform::I8EuclideanDistance)
            }
            (Self::I4ScaledUniformQuantized, Dot) | (Self::I4ScaledUniformQuantized, Cosine) => {
                Box::new(scaled_uniform::I4PackedDotProductDistance)
            }
            (Self::I4ScaledUniformQuantized, Euclidean) => {
                Box::new(scaled_uniform::I4PackedEuclideanDistance)
            }
            (Self::I16ScaledUniformQuantized, Dot) | (Self::I16ScaledUniformQuantized, Cosine) => {
                Box::new(scaled_uniform::I16DotProductDistance)
            }
            (Self::I16ScaledUniformQuantized, Euclidean) => {
                Box::new(scaled_uniform::I16EuclideanDistance)
            }
            (Self::LVQ1x1, Dot) | (Self::LVQ1x1, Cosine) => {
                Box::new(lvq::PrimaryDotProductDistance::<1>)
            }
            (Self::LVQ1x1, Euclidean) => Box::new(lvq::PrimaryEuclideanDistance::<1>),
            (Self::LVQ1x4, Dot) | (Self::LVQ1x4, Cosine) => {
                Box::new(lvq::PrimaryDotProductDistance::<4>)
            }
            (Self::LVQ1x4, Euclidean) => Box::new(lvq::PrimaryEuclideanDistance::<4>),
            (Self::LVQ1x8, Dot) | (Self::LVQ1x8, Cosine) => {
                Box::new(lvq::PrimaryDotProductDistance::<8>)
            }
            (Self::LVQ1x8, Euclidean) => Box::new(lvq::PrimaryEuclideanDistance::<8>),
            (Self::LVQ2x1x8, Dot) | (Self::LVQ2x1x8, Cosine) => {
                Box::new(lvq::TwoLevelDotProductDistance::<1, 8>)
            }
            (Self::LVQ2x1x8, Euclidean) => Box::new(lvq::TwoLevelEuclideanDistance::<1, 8>),
            (Self::LVQ2x1x16, Dot) | (Self::LVQ2x1x16, Cosine) => {
                Box::new(lvq::TwoLevelDotProductDistance::<1, 16>)
            }
            (Self::LVQ2x1x16, Euclidean) => Box::new(lvq::TwoLevelEuclideanDistance::<1, 16>),
            (Self::LVQ2x4x4, Dot) | (Self::LVQ2x4x4, Cosine) => {
                Box::new(lvq::TwoLevelDotProductDistance::<4, 4>)
            }
            (Self::LVQ2x4x4, Euclidean) => Box::new(lvq::TwoLevelEuclideanDistance::<4, 4>),
            (Self::LVQ2x4x8, Dot) | (Self::LVQ2x4x8, Cosine) => {
                Box::new(lvq::TwoLevelDotProductDistance::<4, 8>)
            }
            (Self::LVQ2x4x8, Euclidean) => Box::new(lvq::TwoLevelEuclideanDistance::<4, 8>),
            (Self::LVQ2x8x8, Dot) | (Self::LVQ2x8x8, Cosine) => {
                Box::new(lvq::TwoLevelDotProductDistance::<8, 8>)
            }
            (Self::LVQ2x8x8, Euclidean) => Box::new(lvq::TwoLevelEuclideanDistance::<8, 8>),
        }
    }
}

impl FromStr for F32VectorCoding {
    type Err = io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let input_err = |s| io::Error::new(io::ErrorKind::InvalidInput, s);
        match s {
            "raw" | "raw-l2-norm" | "f32" => Ok(Self::F32),
            s if s.starts_with("truncatedf32:") => {
                let s = s.strip_prefix("truncatedf32:").expect("prefix matched");
                s.parse::<usize>()
                    .map(Self::TruncatedF32)
                    .map_err(|_| input_err("could not parse dimension".into()))
            }
            "f16" => Ok(Self::F16),
            "binary" => Ok(Self::BinaryQuantized),
            "i8-scaled-uniform" => Ok(Self::I8ScaledUniformQuantized),
            "i4-scaled-uniform" => Ok(Self::I4ScaledUniformQuantized),
            "i16-scaled-uniform" => Ok(Self::I16ScaledUniformQuantized),
            "lvq1x1" => Ok(Self::LVQ1x1),
            "lvq1x4" => Ok(Self::LVQ1x4),
            "lvq1x8" => Ok(Self::LVQ1x8),
            "lvq2x1x8" => Ok(Self::LVQ2x1x8),
            "lvq2x1x16" => Ok(Self::LVQ2x1x16),
            "lvq2x4x4" => Ok(Self::LVQ2x4x4),
            "lvq2x4x8" => Ok(Self::LVQ2x4x8),
            "lvq2x8x8" => Ok(Self::LVQ2x8x8),
            _ => Err(input_err(format!("unknown vector coding {s}"))),
        }
    }
}

impl std::fmt::Display for F32VectorCoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::TruncatedF32(d) => write!(f, "truncatedf32:{}", *d),
            Self::F16 => write!(f, "f16"),
            Self::BinaryQuantized => write!(f, "binary"),
            Self::I8ScaledUniformQuantized => write!(f, "i8-scaled-uniform"),
            Self::I4ScaledUniformQuantized => write!(f, "i4-scaled-uniform"),
            Self::I16ScaledUniformQuantized => write!(f, "i16-scaled-uniform"),
            Self::LVQ1x1 => write!(f, "lvq1x1"),
            Self::LVQ1x4 => write!(f, "lvq1x4"),
            Self::LVQ1x8 => write!(f, "lvq1x8"),
            Self::LVQ2x1x8 => write!(f, "lvq2x1x8"),
            Self::LVQ2x1x16 => write!(f, "lvq2x1x16"),
            Self::LVQ2x4x4 => write!(f, "lvq2x4x4"),
            Self::LVQ2x4x8 => write!(f, "lvq2x4x8"),
            Self::LVQ2x8x8 => write!(f, "lvq2x8x8"),
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
    #[allow(unused_variables)]
    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
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
    fn new(distance_fn: D, query: impl Into<Cow<'a, [u8]>>) -> Self {
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
    query: impl Into<Cow<'a, [f32]>>,
    similarity: VectorSimilarity,
    coding: F32VectorCoding,
) -> Box<dyn QueryVectorDistance + 'a> {
    use VectorSimilarity::{Cosine, Dot, Euclidean};

    match (similarity, coding) {
        (_, F32VectorCoding::F32) => float32::new_query_vector_distance(similarity, query.into()),
        (_, F32VectorCoding::TruncatedF32(n)) => {
            let query = query.into();
            let truncated = match similarity {
                VectorSimilarity::Dot => l2_normalize(&query[..(n.min(query.len()))]),
                _ => Cow::from(&query[..(n.min(query.len()))]),
            }
            .to_vec();
            float32::new_query_vector_distance(similarity, truncated.into())
        }
        (Cosine, F32VectorCoding::F16) => Box::new(float16::DotProductQueryDistance::new(
            l2_normalize(query.into()),
        )),
        (Dot, F32VectorCoding::F16) => {
            Box::new(float16::DotProductQueryDistance::new(query.into()))
        }
        (Euclidean, F32VectorCoding::F16) => {
            Box::new(float16::EuclideanQueryDistance::new(query.into()))
        }
        (_, F32VectorCoding::BinaryQuantized) => Box::new(binary::I1DotProductQueryDistance::new(
            query.into().as_ref(),
        )),
        (Cosine, F32VectorCoding::I8ScaledUniformQuantized) => Box::new(
            scaled_uniform::I8DotProductQueryDistance::new(l2_normalize(query.into())),
        ),
        (Dot, F32VectorCoding::I8ScaledUniformQuantized) => {
            Box::new(scaled_uniform::I8DotProductQueryDistance::new(query.into()))
        }
        (Euclidean, F32VectorCoding::I8ScaledUniformQuantized) => {
            Box::new(scaled_uniform::I8EuclideanQueryDistance::new(query.into()))
        }
        (Cosine, F32VectorCoding::I16ScaledUniformQuantized) => Box::new(
            scaled_uniform::I16DotProductQueryDistance::new(l2_normalize(query.into())),
        ),
        (Dot, F32VectorCoding::I16ScaledUniformQuantized) => Box::new(
            scaled_uniform::I16DotProductQueryDistance::new(query.into()),
        ),
        (Euclidean, F32VectorCoding::I16ScaledUniformQuantized) => {
            Box::new(scaled_uniform::I16EuclideanQueryDistance::new(query.into()))
        }
        (Cosine, F32VectorCoding::I4ScaledUniformQuantized) => Box::new(
            scaled_uniform::I4PackedDotProductQueryDistance::new(l2_normalize(query.into())),
        ),
        (Dot, F32VectorCoding::I4ScaledUniformQuantized) => Box::new(
            scaled_uniform::I4PackedDotProductQueryDistance::new(query.into()),
        ),
        (Euclidean, F32VectorCoding::I4ScaledUniformQuantized) => Box::new(
            scaled_uniform::I4PackedEuclideanQueryDistance::new(query.into()),
        ),
        (Cosine, F32VectorCoding::LVQ1x1) => Box::new(
            lvq::PrimaryQueryDotProductDistance::<1>::new(l2_normalize(query.into())),
        ),
        (Dot, F32VectorCoding::LVQ1x1) => {
            Box::new(lvq::PrimaryQueryDotProductDistance::<1>::new(query.into()))
        }
        (Euclidean, F32VectorCoding::LVQ1x1) => {
            Box::new(lvq::PrimaryQueryEuclideanDistance::<1>::new(query.into()))
        }
        (Cosine, F32VectorCoding::LVQ1x4) => Box::new(
            lvq::PrimaryQueryDotProductDistance::<4>::new(l2_normalize(query.into())),
        ),
        (Dot, F32VectorCoding::LVQ1x4) => {
            Box::new(lvq::PrimaryQueryDotProductDistance::<4>::new(query.into()))
        }
        (Euclidean, F32VectorCoding::LVQ1x4) => {
            Box::new(lvq::PrimaryQueryEuclideanDistance::<4>::new(query.into()))
        }
        (Cosine, F32VectorCoding::LVQ1x8) => Box::new(
            lvq::PrimaryQueryDotProductDistance::<8>::new(l2_normalize(query.into())),
        ),
        (Dot, F32VectorCoding::LVQ1x8) => {
            Box::new(lvq::PrimaryQueryDotProductDistance::<8>::new(query.into()))
        }
        (Euclidean, F32VectorCoding::LVQ1x8) => {
            Box::new(lvq::PrimaryQueryEuclideanDistance::<8>::new(query.into()))
        }
        (Cosine, F32VectorCoding::LVQ2x1x8) => Box::new(
            lvq::TwoLevelQueryDotProductDistance::<1, 8>::new(l2_normalize(query.into())),
        ),
        (Dot, F32VectorCoding::LVQ2x1x8) => Box::new(
            lvq::TwoLevelQueryDotProductDistance::<1, 8>::new(query.into()),
        ),
        (Euclidean, F32VectorCoding::LVQ2x1x8) => Box::new(lvq::TwoLevelQueryEuclideanDistance::<
            1,
            8,
        >::new(query.into())),
        (Cosine, F32VectorCoding::LVQ2x1x16) => {
            Box::new(lvq::TwoLevelQueryDotProductDistance::<1, 16>::new(
                l2_normalize(query.into()),
            ))
        }
        (Dot, F32VectorCoding::LVQ2x1x16) => Box::new(
            lvq::TwoLevelQueryDotProductDistance::<1, 16>::new(query.into()),
        ),
        (Euclidean, F32VectorCoding::LVQ2x1x16) => Box::new(lvq::TwoLevelQueryEuclideanDistance::<
            1,
            16,
        >::new(query.into())),
        (Cosine, F32VectorCoding::LVQ2x4x4) => Box::new(
            lvq::TwoLevelQueryDotProductDistance::<4, 4>::new(l2_normalize(query.into())),
        ),
        (Dot, F32VectorCoding::LVQ2x4x4) => Box::new(
            lvq::TwoLevelQueryDotProductDistance::<4, 4>::new(query.into()),
        ),
        (Euclidean, F32VectorCoding::LVQ2x4x4) => Box::new(lvq::TwoLevelQueryEuclideanDistance::<
            4,
            4,
        >::new(query.into())),
        (Cosine, F32VectorCoding::LVQ2x4x8) => Box::new(
            lvq::TwoLevelQueryDotProductDistance::<4, 8>::new(l2_normalize(query.into())),
        ),
        (Dot, F32VectorCoding::LVQ2x4x8) => Box::new(
            lvq::TwoLevelQueryDotProductDistance::<4, 8>::new(query.into()),
        ),
        (Euclidean, F32VectorCoding::LVQ2x4x8) => Box::new(lvq::TwoLevelQueryEuclideanDistance::<
            4,
            8,
        >::new(query.into())),
        (Cosine, F32VectorCoding::LVQ2x8x8) => Box::new(
            lvq::TwoLevelQueryDotProductDistance::<8, 8>::new(l2_normalize(query.into())),
        ),
        (Dot, F32VectorCoding::LVQ2x8x8) => Box::new(
            lvq::TwoLevelQueryDotProductDistance::<8, 8>::new(query.into()),
        ),
        (Euclidean, F32VectorCoding::LVQ2x8x8) => Box::new(lvq::TwoLevelQueryEuclideanDistance::<
            8,
            8,
        >::new(query.into())),
    }
}

/// Create a new [QueryVectorDistance] for indexing that _requires_ symmetrical distance computation.
pub fn new_query_vector_distance_indexing<'a>(
    query: impl Into<Cow<'a, [u8]>>,
    similarity: VectorSimilarity,
    coding: F32VectorCoding,
) -> Box<dyn QueryVectorDistance + 'a> {
    use VectorSimilarity::{Cosine, Dot, Euclidean};
    macro_rules! quantized_qvd {
        ($dist_fn:expr, $query:ident) => {
            Box::new(QuantizedQueryVectorDistance::new($dist_fn, $query))
        };
    }
    match (similarity, coding) {
        (Cosine, F32VectorCoding::F32) => quantized_qvd!(float32::CosineDistance, query),
        (Dot, F32VectorCoding::F32) => quantized_qvd!(float32::DotProductDistance, query),
        (Euclidean, F32VectorCoding::F32) => quantized_qvd!(float32::EuclideanDistance, query),
        (_, F32VectorCoding::TruncatedF32(_)) => {
            new_query_vector_distance_indexing(query, similarity, F32VectorCoding::F32)
        }
        (Dot, F32VectorCoding::F16) => quantized_qvd!(float16::DotProductDistance, query),
        (Cosine, F32VectorCoding::F16) => quantized_qvd!(float16::DotProductDistance, query),
        (Euclidean, F32VectorCoding::F16) => quantized_qvd!(float16::EuclideanDistance, query),
        (_, F32VectorCoding::BinaryQuantized) => quantized_qvd!(binary::HammingDistance, query),
        (Dot, F32VectorCoding::I8ScaledUniformQuantized)
        | (Cosine, F32VectorCoding::I8ScaledUniformQuantized) => {
            quantized_qvd!(scaled_uniform::I8DotProductDistance, query)
        }
        (Euclidean, F32VectorCoding::I8ScaledUniformQuantized) => {
            quantized_qvd!(scaled_uniform::I8EuclideanDistance, query)
        }
        (Dot, F32VectorCoding::I4ScaledUniformQuantized)
        | (Cosine, F32VectorCoding::I4ScaledUniformQuantized) => {
            quantized_qvd!(scaled_uniform::I4PackedDotProductDistance, query)
        }
        (Euclidean, F32VectorCoding::I4ScaledUniformQuantized) => {
            quantized_qvd!(scaled_uniform::I4PackedEuclideanDistance, query)
        }
        (Dot, F32VectorCoding::I16ScaledUniformQuantized)
        | (Cosine, F32VectorCoding::I16ScaledUniformQuantized) => {
            quantized_qvd!(scaled_uniform::I16DotProductDistance, query)
        }
        (Euclidean, F32VectorCoding::I16ScaledUniformQuantized) => {
            quantized_qvd!(scaled_uniform::I16EuclideanDistance, query)
        }
        (Dot, F32VectorCoding::LVQ1x1) | (Cosine, F32VectorCoding::LVQ1x1) => {
            quantized_qvd!(lvq::PrimaryDotProductDistance::<1>, query)
        }
        (Euclidean, F32VectorCoding::LVQ1x1) => {
            quantized_qvd!(lvq::PrimaryEuclideanDistance::<1>, query)
        }
        (Dot, F32VectorCoding::LVQ1x4) | (Cosine, F32VectorCoding::LVQ1x4) => {
            quantized_qvd!(lvq::PrimaryDotProductDistance::<4>, query)
        }
        (Euclidean, F32VectorCoding::LVQ1x4) => {
            quantized_qvd!(lvq::PrimaryEuclideanDistance::<4>, query)
        }
        (Dot, F32VectorCoding::LVQ1x8) | (Cosine, F32VectorCoding::LVQ1x8) => {
            quantized_qvd!(lvq::PrimaryDotProductDistance::<8>, query)
        }
        (Euclidean, F32VectorCoding::LVQ1x8) => {
            quantized_qvd!(lvq::PrimaryEuclideanDistance::<8>, query)
        }
        (Dot, F32VectorCoding::LVQ2x1x8) | (Cosine, F32VectorCoding::LVQ2x1x8) => {
            quantized_qvd!(lvq::TwoLevelDotProductDistance::<1, 8>, query)
        }
        (Euclidean, F32VectorCoding::LVQ2x1x8) => {
            quantized_qvd!(lvq::TwoLevelEuclideanDistance::<1, 8>, query)
        }
        (Dot, F32VectorCoding::LVQ2x1x16) | (Cosine, F32VectorCoding::LVQ2x1x16) => {
            quantized_qvd!(lvq::TwoLevelDotProductDistance::<1, 16>, query)
        }
        (Euclidean, F32VectorCoding::LVQ2x1x16) => {
            quantized_qvd!(lvq::TwoLevelEuclideanDistance::<1, 16>, query)
        }
        (Dot, F32VectorCoding::LVQ2x4x4) | (Cosine, F32VectorCoding::LVQ2x4x4) => {
            quantized_qvd!(lvq::TwoLevelDotProductDistance::<4, 4>, query)
        }
        (Euclidean, F32VectorCoding::LVQ2x4x4) => {
            quantized_qvd!(lvq::TwoLevelEuclideanDistance::<4, 4>, query)
        }
        (Dot, F32VectorCoding::LVQ2x4x8) | (Cosine, F32VectorCoding::LVQ2x4x8) => {
            quantized_qvd!(lvq::TwoLevelDotProductDistance::<4, 8>, query)
        }
        (Euclidean, F32VectorCoding::LVQ2x4x8) => {
            quantized_qvd!(lvq::TwoLevelEuclideanDistance::<4, 8>, query)
        }
        (Dot, F32VectorCoding::LVQ2x8x8) | (Cosine, F32VectorCoding::LVQ2x8x8) => {
            quantized_qvd!(lvq::TwoLevelDotProductDistance::<8, 8>, query)
        }
        (Euclidean, F32VectorCoding::LVQ2x8x8) => {
            quantized_qvd!(lvq::TwoLevelEuclideanDistance::<8, 8>, query)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        distance::l2_normalize,
        vectors::{
            new_query_vector_distance_f32, F32VectorCoder, F32VectorCoding, VectorSimilarity,
        },
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
            // Encoders for Dot similarity may assume that any input vector is normalized.
            let vec = if similarity == VectorSimilarity::Dot {
                l2_normalize(vec)
            } else {
                vec.into()
            };
            let f32_coder = F32VectorCoding::F32.new_coder(similarity);
            let rvec = f32_coder
                .encode(&vec)
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
        let coder = format.new_coder(similarity);
        let a = TestVector::new(a, similarity, coder.as_ref());
        let b = TestVector::new(b, similarity, coder.as_ref());

        let f32_dist_fn = similarity.new_distance_function();
        let rf32_dist = f32_dist_fn.distance_f32(&a.rvec, &b.rvec);
        let ru8_dist =
            f32_dist_fn.distance(bytemuck::cast_slice(&a.rvec), bytemuck::cast_slice(&b.rvec));
        assert_float_near!(rf32_dist, ru8_dist, 0.00001, index);

        let dist_fn = format.new_vector_distance(similarity);
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
        let coder = format.new_coder(similarity);
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

    use F32VectorCoding::{
        I16ScaledUniformQuantized, I4ScaledUniformQuantized, I8ScaledUniformQuantized, F16,
    };
    use VectorSimilarity::{Cosine, Dot, Euclidean};

    #[test]
    fn f16_cosine() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Cosine, F16, i, &a, &b, 0.001);
            query_distance_compare(Cosine, F16, i, &a, &b, 0.001);
        }
    }

    #[test]
    fn f16_dot() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Dot, F16, i, &a, &b, 0.001);
            query_distance_compare(Dot, F16, i, &a, &b, 0.001);
        }
    }

    #[test]
    fn f16_l2() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Euclidean, F16, i, &a, &b, 0.001);
            query_distance_compare(Euclidean, F16, i, &a, &b, 0.001);
        }
    }

    #[test]
    fn i16_scaled_dot() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Dot, I16ScaledUniformQuantized, i, &a, &b, 0.001);
            query_distance_compare(Dot, I16ScaledUniformQuantized, i, &a, &b, 0.001);
        }
    }

    #[test]
    fn i16_scaled_l2() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Euclidean, I16ScaledUniformQuantized, i, &a, &b, 0.001);
            query_distance_compare(Euclidean, I16ScaledUniformQuantized, i, &a, &b, 0.001);
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

    #[test]
    fn lvq1x4_dot() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Dot, F32VectorCoding::LVQ1x4, i, &a, &b, 0.02);
            query_distance_compare(Dot, F32VectorCoding::LVQ1x4, i, &a, &b, 0.02);
        }
    }

    #[test]
    fn lvq1x4_l2() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Euclidean, F32VectorCoding::LVQ1x4, i, &a, &b, 0.02);
            query_distance_compare(Euclidean, F32VectorCoding::LVQ1x4, i, &a, &b, 0.02);
        }
    }

    #[test]
    fn lvq1x8_dot() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Dot, F32VectorCoding::LVQ1x8, i, &a, &b, 0.001);
            query_distance_compare(Dot, F32VectorCoding::LVQ1x8, i, &a, &b, 0.001);
        }
    }

    #[test]
    fn lvq1x8_l2() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Euclidean, F32VectorCoding::LVQ1x8, i, &a, &b, 0.001);
            query_distance_compare(Euclidean, F32VectorCoding::LVQ1x8, i, &a, &b, 0.001);
        }
    }

    #[test]
    fn lvq2x1x8_dot() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Dot, F32VectorCoding::LVQ2x1x8, i, &a, &b, 0.01);
            query_distance_compare(Dot, F32VectorCoding::LVQ2x1x8, i, &a, &b, 0.01);
        }
    }

    #[test]
    fn lvq2x1x8_l2() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Euclidean, F32VectorCoding::LVQ2x1x8, i, &a, &b, 0.01);
            query_distance_compare(Euclidean, F32VectorCoding::LVQ2x1x8, i, &a, &b, 0.01);
        }
    }

    #[test]
    fn lvq2x4x4_dot() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Dot, F32VectorCoding::LVQ2x4x4, i, &a, &b, 0.01);
            query_distance_compare(Dot, F32VectorCoding::LVQ2x4x4, i, &a, &b, 0.01);
        }
    }

    #[test]
    fn lvq2x4x4_l2() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Euclidean, F32VectorCoding::LVQ2x4x4, i, &a, &b, 0.01);
            query_distance_compare(Euclidean, F32VectorCoding::LVQ2x4x4, i, &a, &b, 0.01);
        }
    }

    #[test]
    fn lvq2x4x8_dot() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Dot, F32VectorCoding::LVQ2x4x8, i, &a, &b, 0.01);
            query_distance_compare(Dot, F32VectorCoding::LVQ2x4x8, i, &a, &b, 0.01);
        }
    }

    #[test]
    fn lvq2x4x8_l2() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Euclidean, F32VectorCoding::LVQ2x4x8, i, &a, &b, 0.01);
            query_distance_compare(Euclidean, F32VectorCoding::LVQ2x4x8, i, &a, &b, 0.01);
        }
    }

    #[test]
    fn lvq2x8x8_dot() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Dot, F32VectorCoding::LVQ2x8x8, i, &a, &b, 0.001);
            query_distance_compare(Dot, F32VectorCoding::LVQ2x8x8, i, &a, &b, 0.001);
        }
    }

    #[test]
    fn lvq2x8x8_l2() {
        for (i, (a, b)) in test_float_vectors().into_iter().enumerate() {
            distance_compare(Euclidean, F32VectorCoding::LVQ2x8x8, i, &a, &b, 0.001);
            query_distance_compare(Euclidean, F32VectorCoding::LVQ2x8x8, i, &a, &b, 0.001);
        }
    }
}
