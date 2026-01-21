//! Vector handling: formatting/quantization and distance computation.

use std::{borrow::Cow, fmt::Debug, io, str::FromStr};

mod binary;
mod float16;
mod float32;
mod lvq;
pub mod soar;

use serde::{Deserialize, Serialize};

pub use float32::{CosineDistance, DotProductDistance, EuclideanDistance, l2_norm, l2_normalize};

/// Functions used for to compute the distance between two vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorSimilarity {
    /// Euclidean (l2) distance, squared.
    ///
    /// True euclidean distance is the square root of this calculation, but computing the square
    /// root is expensive and would not alter the order of results.
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
            Self::Euclidean => Box::new(float32::EuclideanDistance::default()),
            Self::Dot => Box::new(float32::DotProductDistance::default()),
            Self::Cosine => Box::new(float32::CosineDistance::default()),
        }
    }

    /// Return true if vectors must be l2 normalized during encoding.
    pub fn l2_normalize(&self) -> bool {
        *self == Self::Cosine
    }

    /// Return an iterator over all similarity functions.
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
    /// Little-endian IEEE f16 encoding.
    F16,
    /// Single bit (sign bit) per dimension; positive or negative.
    ///
    /// This encoding is very compact and efficient for distance computation but also does not have
    /// high fidelity with distances computed between raw vectors.
    BinaryQuantized,
    /// Turbo LVQ; 1 bit primary vector.
    ///
    /// This encoding is optimized for cases where dimensionality is a multiple of 128.
    TLVQ1,
    /// Turbo LVQ; 2 bit primary vector.
    ///
    /// This encoding is optimized for cases where dimensionality is a multiple of 64.
    TLVQ2,
    /// Turbo LVQ; 4 bit primary vector.
    ///
    /// This encoding is optimized for cases where dimensionality is a multiple of 32.
    TLVQ4,
    /// Turbo LVQ; 8 bit primary vector.
    ///
    /// This encoding is optimized for cases where dimensionality is a multiple of 16.
    TLVQ8,
    /// Turbo LVQ; 1 bit primary vector and 8 bit residual vector.
    ///
    /// This encoding is optimized for cases where dimensionality is a multiple of 128.
    TLVQ1x8,
    /// Turbo LVQ; 2 bits primary vector and 8 bits residual vector.
    ///
    /// This encoding is optimized for cases where dimensionality is a multiple of 64.
    TLVQ2x8,
    /// Turbo LVQ; 4 bits primary vector and 8 bits residual vector.
    ///
    /// This encoding is optimized for cases where dimensionality is a multiple of 32.
    TLVQ4x8,
    /// Turbo LVQ; 8 bits primary vector and 8 bits residual vector.
    ///
    /// This encoding is optimized for cases where dimensionality is a multiple of 16.
    TLVQ8x8,
}

impl F32VectorCoding {
    /// Create a new coder for this format.
    pub fn new_coder(&self, similarity: VectorSimilarity) -> Box<dyn F32VectorCoder> {
        match self {
            Self::F32 => Box::new(float32::VectorCoder::new(similarity)),
            Self::F16 => Box::new(float16::VectorCoder::new(similarity)),
            Self::BinaryQuantized => Box::new(binary::BinaryQuantizedVectorCoder),
            Self::TLVQ1 => Box::new(lvq::TurboPrimaryCoder::<1>::default()),
            Self::TLVQ2 => Box::new(lvq::TurboPrimaryCoder::<2>::default()),
            Self::TLVQ4 => Box::new(lvq::TurboPrimaryCoder::<4>::default()),
            Self::TLVQ8 => Box::new(lvq::TurboPrimaryCoder::<8>::default()),
            Self::TLVQ1x8 => Box::new(lvq::TurboResidualCoder::<1>::default()),
            Self::TLVQ2x8 => Box::new(lvq::TurboResidualCoder::<2>::default()),
            Self::TLVQ4x8 => Box::new(lvq::TurboResidualCoder::<4>::default()),
            Self::TLVQ8x8 => Box::new(lvq::TurboResidualCoder::<8>::default()),
        }
    }

    /// Returns a [VectorDistance] between vectors encoded using this coder.
    pub fn new_vector_distance(&self, similarity: VectorSimilarity) -> Box<dyn VectorDistance> {
        use VectorSimilarity::{Cosine, Dot, Euclidean};

        match (self, similarity) {
            (Self::F32, Cosine) => Box::new(float32::CosineDistance::default()),
            (Self::F32, Dot) => Box::new(float32::DotProductDistance::default()),
            (Self::F32, Euclidean) => Box::new(float32::EuclideanDistance::default()),
            (Self::F16, Dot) | (Self::F16, Cosine) => {
                Box::new(float16::DotProductDistance::default())
            }
            (Self::F16, Euclidean) => Box::new(float16::EuclideanDistance::default()),
            (Self::BinaryQuantized, _) => Box::new(binary::HammingDistance),
            (Self::TLVQ1, _) => Box::new(lvq::TurboPrimaryDistance::<1>::new(similarity)),
            (Self::TLVQ2, _) => Box::new(lvq::TurboPrimaryDistance::<2>::new(similarity)),
            (Self::TLVQ4, _) => Box::new(lvq::TurboPrimaryDistance::<4>::new(similarity)),
            (Self::TLVQ8, _) => Box::new(lvq::TurboPrimaryDistance::<8>::new(similarity)),
            (Self::TLVQ1x8, _) => Box::new(lvq::TurboResidualDistance::<1>::new(similarity)),
            (Self::TLVQ2x8, _) => Box::new(lvq::TurboResidualDistance::<2>::new(similarity)),
            (Self::TLVQ4x8, _) => Box::new(lvq::TurboResidualDistance::<4>::new(similarity)),
            (Self::TLVQ8x8, _) => Box::new(lvq::TurboResidualDistance::<8>::new(similarity)),
        }
    }

    /// Create a new [QueryVectorDistance] given a query, similarity function, and vector coding.
    pub fn query_vector_distance_f32<'a>(
        &self,
        query: impl Into<Cow<'a, [f32]>>,
        similarity: VectorSimilarity,
    ) -> Box<dyn QueryVectorDistance + 'a> {
        use VectorSimilarity::{Cosine, Dot, Euclidean};

        match (similarity, *self) {
            (_, F32VectorCoding::F32) => {
                float32::new_query_vector_distance(similarity, query.into())
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
            (_, F32VectorCoding::BinaryQuantized) => Box::new(
                binary::I1DotProductQueryDistance::new(query.into().as_ref()),
            ),
            (_, F32VectorCoding::TLVQ1) => Box::new(lvq::TurboPrimaryQueryDistance::<1>::new(
                similarity,
                query.into(),
            )),
            (_, F32VectorCoding::TLVQ2) => Box::new(lvq::TurboPrimaryQueryDistance::<2>::new(
                similarity,
                query.into(),
            )),
            (_, F32VectorCoding::TLVQ4) => Box::new(lvq::TurboPrimaryQueryDistance::<4>::new(
                similarity,
                query.into(),
            )),
            (_, F32VectorCoding::TLVQ8) => Box::new(lvq::TurboPrimaryQueryDistance::<8>::new(
                similarity,
                query.into(),
            )),
            (_, F32VectorCoding::TLVQ1x8) => Box::new(lvq::TurboResidualQueryDistance::<1>::new(
                similarity,
                query.into(),
            )),
            (_, F32VectorCoding::TLVQ2x8) => Box::new(lvq::TurboResidualQueryDistance::<2>::new(
                similarity,
                query.into(),
            )),
            (_, F32VectorCoding::TLVQ4x8) => Box::new(lvq::TurboResidualQueryDistance::<4>::new(
                similarity,
                query.into(),
            )),
            (_, F32VectorCoding::TLVQ8x8) => Box::new(lvq::TurboResidualQueryDistance::<8>::new(
                similarity,
                query.into(),
            )),
        }
    }

    /// Create a new [QueryVectorDistance] between `query` and vectors formatted with this encoding
    /// by `similarity` distance metric, but trade fidelity for speed.
    /// * This is not implemented for all codings and similarities.
    /// * Distances are less accurate but this does not mean the distances are more or less than a
    ///   a higher fidelity distance computation. YMMV.
    pub fn query_vector_distance_f32_fast(
        &self,
        query: &[f32],
        similarity: VectorSimilarity,
    ) -> Option<Box<dyn QueryVectorDistance>> {
        match *self {
            F32VectorCoding::F32 | F32VectorCoding::F16 => None,
            F32VectorCoding::BinaryQuantized => Some(Box::new(QuantizedQueryVectorDistance::new(
                binary::HammingDistance,
                binary::BinaryQuantizedVectorCoder.encode(query),
            ))),
            // TODO: consider using asymmetric 8xN distance here.
            F32VectorCoding::TLVQ1 => Some(Box::new(QuantizedQueryVectorDistance::new(
                lvq::TurboPrimaryDistance::<1>::new(similarity),
                lvq::TurboPrimaryCoder::<1>::default().encode(query),
            ))),
            F32VectorCoding::TLVQ2 => Some(Box::new(QuantizedQueryVectorDistance::new(
                lvq::TurboPrimaryDistance::<2>::new(similarity),
                lvq::TurboPrimaryCoder::<2>::default().encode(query),
            ))),
            F32VectorCoding::TLVQ4 => Some(Box::new(QuantizedQueryVectorDistance::new(
                lvq::TurboPrimaryDistance::<4>::new(similarity),
                lvq::TurboPrimaryCoder::<4>::default().encode(query),
            ))),
            F32VectorCoding::TLVQ8 => Some(Box::new(QuantizedQueryVectorDistance::new(
                lvq::TurboPrimaryDistance::<8>::new(similarity),
                lvq::TurboPrimaryCoder::<8>::default().encode(query),
            ))),
            F32VectorCoding::TLVQ1x8 => Some(Box::new(
                lvq::TurboResidualFastQueryDistance::<1>::new(similarity, query),
            )),
            F32VectorCoding::TLVQ2x8 => Some(Box::new(
                lvq::TurboResidualFastQueryDistance::<2>::new(similarity, query),
            )),
            F32VectorCoding::TLVQ4x8 => Some(Box::new(
                lvq::TurboResidualFastQueryDistance::<4>::new(similarity, query),
            )),
            F32VectorCoding::TLVQ8x8 => Some(Box::new(
                lvq::TurboResidualFastQueryDistance::<8>::new(similarity, query),
            )),
        }
    }

    /// Create a new [QueryVectorDistance] for indexing that _requires_ symmetrical distance computation.
    pub fn query_vector_distance_indexing<'a>(
        &self,
        query: impl Into<Cow<'a, [u8]>>,
        similarity: VectorSimilarity,
    ) -> Box<dyn QueryVectorDistance + 'a> {
        use VectorSimilarity::{Cosine, Dot, Euclidean};
        macro_rules! quantized_qvd {
            ($dist_fn:expr, $query:ident) => {
                Box::new(QuantizedQueryVectorDistance::new($dist_fn, $query))
            };
        }
        match (similarity, *self) {
            (Cosine, F32VectorCoding::F32) => {
                quantized_qvd!(float32::CosineDistance::default(), query)
            }
            (Dot, F32VectorCoding::F32) => {
                quantized_qvd!(float32::DotProductDistance::default(), query)
            }
            (Euclidean, F32VectorCoding::F32) => {
                quantized_qvd!(float32::EuclideanDistance::default(), query)
            }
            (Dot, F32VectorCoding::F16) => {
                quantized_qvd!(float16::DotProductDistance::default(), query)
            }
            (Cosine, F32VectorCoding::F16) => {
                quantized_qvd!(float16::DotProductDistance::default(), query)
            }
            (Euclidean, F32VectorCoding::F16) => {
                quantized_qvd!(float16::EuclideanDistance::default(), query)
            }
            (_, F32VectorCoding::BinaryQuantized) => quantized_qvd!(binary::HammingDistance, query),
            (_, F32VectorCoding::TLVQ1) => {
                quantized_qvd!(lvq::TurboPrimaryDistance::<1>::new(similarity), query)
            }
            (_, F32VectorCoding::TLVQ2) => {
                quantized_qvd!(lvq::TurboPrimaryDistance::<2>::new(similarity), query)
            }
            (_, F32VectorCoding::TLVQ4) => {
                quantized_qvd!(lvq::TurboPrimaryDistance::<4>::new(similarity), query)
            }
            (_, F32VectorCoding::TLVQ8) => {
                quantized_qvd!(lvq::TurboPrimaryDistance::<8>::new(similarity), query)
            }
            (_, F32VectorCoding::TLVQ1x8) => {
                quantized_qvd!(lvq::TurboResidualDistance::<1>::new(similarity), query)
            }
            (_, F32VectorCoding::TLVQ2x8) => {
                quantized_qvd!(lvq::TurboResidualDistance::<2>::new(similarity), query)
            }
            (_, F32VectorCoding::TLVQ4x8) => {
                quantized_qvd!(lvq::TurboResidualDistance::<4>::new(similarity), query)
            }
            (_, F32VectorCoding::TLVQ8x8) => {
                quantized_qvd!(lvq::TurboResidualDistance::<8>::new(similarity), query)
            }
        }
    }
}

impl FromStr for F32VectorCoding {
    type Err = io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let input_err = |s| io::Error::new(io::ErrorKind::InvalidInput, s);
        match s {
            "raw" | "raw-l2-norm" | "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "binary" => Ok(Self::BinaryQuantized),
            "tlvq1" => Ok(Self::TLVQ1),
            "tlvq2" => Ok(Self::TLVQ2),
            "tlvq4" => Ok(Self::TLVQ4),
            "tlvq8" => Ok(Self::TLVQ8),
            "tlvq1x8" => Ok(Self::TLVQ1x8),
            "tlvq2x8" => Ok(Self::TLVQ2x8),
            "tlvq4x8" => Ok(Self::TLVQ4x8),
            "tlvq8x8" => Ok(Self::TLVQ8x8),
            _ => Err(input_err(format!("unknown vector coding {s}"))),
        }
    }
}

impl std::fmt::Display for F32VectorCoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::BinaryQuantized => write!(f, "binary"),
            Self::TLVQ1 => write!(f, "tlvq1"),
            Self::TLVQ2 => write!(f, "tlvq2"),
            Self::TLVQ4 => write!(f, "tlvq4"),
            Self::TLVQ8 => write!(f, "tlvq8"),
            Self::TLVQ1x8 => write!(f, "tlvq1x8"),
            Self::TLVQ2x8 => write!(f, "tlvq2x8"),
            Self::TLVQ4x8 => write!(f, "tlvq4x8"),
            Self::TLVQ8x8 => write!(f, "tlvq8x8"),
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
    fn decode(&self, encoded: &[u8]) -> Vec<f32> {
        let mut out = vec![0.0; self.dimensions(encoded.len())];
        self.decode_to(encoded, &mut out);
        out
    }

    /// Decode `encoded` to `out`.
    ///
    /// *Panics* if `out.len() < self.dimensions(encoded.len())`.
    fn decode_to(&self, encoded: &[u8], out: &mut [f32]);

    /// Return the number of dimensions that a vector of `byte_len` bytes will decode to.
    ///
    /// Some codecs may generate more dimensions than were originally specified due to sub-byte
    /// packing of dimensions.
    fn dimensions(&self, byte_len: usize) -> usize;
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

/// For a given similarity function, compute a distance score based on the unnormalized dot product
/// of two vectors and their l2 norms.
fn dot_unnormalized_to_distance(
    similarity: VectorSimilarity,
    dot_unnormalized: f64,
    l2_norm: (f64, f64),
) -> f64 {
    match similarity {
        VectorSimilarity::Cosine | VectorSimilarity::Dot => {
            let dot = dot_unnormalized / (l2_norm.0 * l2_norm.1);
            (-dot + 1.0) / 2.0
        }
        VectorSimilarity::Euclidean => {
            l2_norm.0 * l2_norm.0 + l2_norm.1 * l2_norm.1 - (2.0 * dot_unnormalized)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        F32VectorCoder, F32VectorCoding, VectorSimilarity, l2_normalize,
        lvq::{TurboPrimaryCoder, TurboResidualCoder},
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
        assert_float_near!(rf32_dist, ru8_dist, 0.0001, index);

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

        let query_dist_fn = format.query_vector_distance_f32(&a.rvec, similarity);
        let query_dist = query_dist_fn.distance(&b.qvec);

        assert_float_near!(f32_dist, query_dist, threshold, index);
    }

    use F32VectorCoding::{F16, TLVQ1, TLVQ1x8, TLVQ2, TLVQ2x8, TLVQ4, TLVQ4x8, TLVQ8, TLVQ8x8};
    use VectorSimilarity::{Cosine, Dot, Euclidean};
    use rand::{Rng, SeedableRng, TryRngCore, rngs::OsRng};

    macro_rules! distance_test {
        ($name:ident, $sim:path, $coder:path, $epsilon:literal) => {
            #[test]
            fn $name() {
                let seed = OsRng::default().try_next_u64().unwrap();
                println!("SEED {seed:#016x}");
                let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
                for i in 0..1024 {
                    let dim = rng.random_range(128..=256);
                    let a = (0..dim)
                        .map(|_| rng.random_range(-1.0f32..=1.0))
                        .collect::<Vec<_>>();
                    let b = (0..dim)
                        .map(|_| rng.random_range(-1.0f32..=1.0))
                        .collect::<Vec<_>>();

                    distance_compare($sim, $coder, i, &a, &b, $epsilon);
                    query_distance_compare($sim, $coder, i, &a, &b, $epsilon);
                }
            }
        };
    }

    distance_test!(f16_cosine_dist, Cosine, F16, 0.001);
    distance_test!(f16_dot_dist, Dot, F16, 0.001);
    distance_test!(f16_l2_dist, Euclidean, F16, 0.001);

    distance_test!(tlvq1_dot_dist, Dot, TLVQ1, 0.4);
    distance_test!(tlvq1_l2_dist, Euclidean, TLVQ1, 0.4);
    distance_test!(tlvq2_dot_dist, Dot, TLVQ2, 0.2);
    distance_test!(tlvq2_l2_dist, Euclidean, TLVQ2, 0.2);
    distance_test!(tlvq4_dot_dist, Dot, TLVQ4, 0.1);
    distance_test!(tlvq4_l2_dist, Euclidean, TLVQ4, 0.1);
    distance_test!(tlvq8_dot_dist, Dot, TLVQ8, 0.01);
    distance_test!(tlvq8_l2_dist, Euclidean, TLVQ8, 0.01);

    distance_test!(tlvq1x8_dot_dist, Dot, TLVQ1x8, 0.01);
    distance_test!(tlvq1x8_l2_dist, Euclidean, TLVQ1x8, 0.01);
    distance_test!(tlvq2x8_dot_dist, Dot, TLVQ2x8, 0.01);
    distance_test!(tlvq2x8_l2_dist, Euclidean, TLVQ2x8, 0.01);
    distance_test!(tlvq4x8_dot_dist, Dot, TLVQ4x8, 0.001);
    distance_test!(tlvq4x8_l2_dist, Euclidean, TLVQ4x8, 0.001);
    distance_test!(tlvq8x8_dot_dist, Dot, TLVQ8x8, 0.001);
    distance_test!(tlvq8x8_l2_dist, Euclidean, TLVQ8x8, 0.001);

    macro_rules! lvq_coding_simd_test {
        ($name:ident, $coder:ty) => {
            #[test]
            fn $name() {
                let seed = OsRng::default().try_next_u64().unwrap();
                println!("SEED {seed:#016x}");
                let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
                let scoder = <$coder>::scalar();
                let ocoder = <$coder>::default();
                // TODO: use randomly sized vectors like we do for distance tests.
                for i in 0..1024 {
                    let vec = l2_normalize(
                        (0..128)
                            .map(|_| rng.random_range(-1.0f32..=1.0))
                            .collect::<Vec<_>>(),
                    );
                    let svec = scoder.encode(&vec);
                    let ovec = ocoder.encode(&vec);
                    assert_eq!(
                        scoder.decode(&svec),
                        ocoder.decode(&ovec),
                        "index {i} input vector {vec:?}"
                    );
                }
            }
        };
    }

    lvq_coding_simd_test!(tlvq1_coding_simd, TurboPrimaryCoder::<1>);
    lvq_coding_simd_test!(tlvq2_coding_simd, TurboPrimaryCoder::<2>);
    lvq_coding_simd_test!(tlvq4_coding_simd, TurboPrimaryCoder::<4>);
    lvq_coding_simd_test!(tlvq8_coding_simd, TurboPrimaryCoder::<8>);
    lvq_coding_simd_test!(tlvq1x8_coding_simd, TurboResidualCoder::<1>);
    lvq_coding_simd_test!(tlvq2x8_coding_simd, TurboResidualCoder::<2>);
    lvq_coding_simd_test!(tlvq4x8_coding_simd, TurboResidualCoder::<4>);
    lvq_coding_simd_test!(tlvq8x8_coding_simd, TurboResidualCoder::<8>);
}
