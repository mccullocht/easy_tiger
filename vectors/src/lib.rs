//! Vector handling: formatting/quantization and distance computation.

use std::{borrow::Cow, fmt::Debug, io, str::FromStr};

mod binary;
pub mod float16;
pub mod float32;
mod lvq;
mod quiver;
mod rabitq;
pub mod rotate;

use serde::{Deserialize, Serialize};

pub use half::f16;

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
    pub fn distance_f32(&self) -> Box<dyn F32VectorDistance> {
        match self {
            Self::Euclidean => Box::new(float32::EuclideanDistance::default()),
            Self::Dot => Box::new(float32::DotProductDistance::default()),
            Self::Cosine => Box::new(float32::CosineDistance::default()),
        }
    }

    /// Return an [`F16VectorDistance`] for this similarity function.
    pub fn distance_f16(&self) -> Box<dyn F16VectorDistance> {
        match self {
            Self::Euclidean => Box::new(float16::EuclideanDistance::default()),
            Self::Dot => Box::new(float16::DotProductDistance::default()),
            Self::Cosine => Box::new(float16::CosineDistance::default()),
        }
    }

    /// Return true if vectors must be l2 normalized during encoding.
    pub fn l2_normalize(&self) -> bool {
        *self == Self::Cosine
    }

    /// Return true if this is an angular distance measure.
    pub fn angular(&self) -> bool {
        *self == Self::Cosine || *self == Self::Dot
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
    /// RaBitQ; 1 bit binary quantization with distance estimation.
    RaBitQ,
    /// QuIVer; 2 bit binary quantization with sign + magnitude.
    QuIVer,
}

impl F32VectorCoding {
    /// Create a new coder for this format and similarity function.
    ///
    /// If `center`` is present, the center vector will be subtracted from each vector before
    /// quantization occurs.  Centering reduces the dynamic range of the vectors and can reduce
    /// quantization loss substantially, particularly when using lower bit rates formats. This holds
    /// true even if the center is the mean vector over a large data set.
    pub fn coder(
        &self,
        similarity: VectorSimilarity,
        center: Option<Vec<f32>>,
    ) -> Box<dyn F32VectorCoder> {
        match (self, similarity) {
            (Self::F32, _) => Box::new(float32::VectorCoder::new(similarity)),
            (Self::F16, _) => Box::new(float16::VectorCoder::new(similarity)),
            (Self::BinaryQuantized, _) => Box::new(binary::BinaryQuantizedVectorCoder),
            (Self::TLVQ1, _) => Box::new(lvq::TurboPrimaryCoder::<1>::new(similarity, center)),
            (Self::TLVQ2, _) => Box::new(lvq::TurboPrimaryCoder::<2>::new(similarity, center)),
            (Self::TLVQ4, _) => Box::new(lvq::TurboPrimaryCoder::<4>::new(similarity, center)),
            (Self::TLVQ8, _) => Box::new(lvq::TurboPrimaryCoder::<8>::new(similarity, center)),
            (Self::RaBitQ, _) => Box::new(rabitq::Coder::new(center)),
            (Self::QuIVer, _) => quiver::new_coder(),
        }
    }

    /// Returns a [`VectorDistance`] between vectors encoded using this scheme.
    ///
    /// If `center` is present, it is assumed that all inputs vectors are centered with respect to
    /// this vector before encoding, and `center` may be used as part of distance corrections.
    pub fn distance_symmetric(&self, similarity: VectorSimilarity) -> Box<dyn VectorDistance> {
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
            (Self::RaBitQ, _) => Box::new(rabitq::Distance::new(similarity)),
            (Self::QuIVer, _) => quiver::new_symmetric_distance(),
        }
    }

    /// Create a new [`QueryVectorDistance`] that computes distance between a fixed float query and
    /// an arbitrary vector using this vector coding.
    ///
    /// If `center` is present then it will be accounted for in the distance calculation assuming
    /// all input vectors _also_ use the same center value.
    pub fn query_distance_asymmetric<'a>(
        &self,
        similarity: VectorSimilarity,
        query: impl Into<Cow<'a, [f32]>>,
        center: Option<&[f32]>,
    ) -> Box<dyn QueryVectorDistance + 'a> {
        match (*self, similarity) {
            (F32VectorCoding::F32, _) => {
                float32::new_query_vector_distance(similarity, query.into())
            }
            (F32VectorCoding::F16, VectorSimilarity::Cosine) => Box::new(
                float16::DotProductQueryDistance::new(float32::l2_normalize(query.into()).0),
            ),
            (F32VectorCoding::F16, VectorSimilarity::Dot) => {
                Box::new(float16::DotProductQueryDistance::new(query.into()))
            }
            (F32VectorCoding::F16, VectorSimilarity::Euclidean) => {
                Box::new(float16::EuclideanQueryDistance::new(query.into()))
            }
            (F32VectorCoding::BinaryQuantized, _) => Box::new(
                binary::I1DotProductQueryDistance::new(query.into().as_ref()),
            ),
            (F32VectorCoding::TLVQ1, _) => Box::new(lvq::TurboPrimaryQueryDistance1::new(
                similarity,
                query.into(),
                center,
            )),
            (F32VectorCoding::TLVQ2, _) => Box::new(lvq::TurboPrimaryQueryDistance::<2>::new(
                similarity,
                query.into(),
                center,
            )),
            (F32VectorCoding::TLVQ4, _) => Box::new(lvq::TurboPrimaryQueryDistance::<4>::new(
                similarity,
                query.into(),
                center,
            )),
            (F32VectorCoding::TLVQ8, _) => Box::new(lvq::TurboPrimaryQueryDistance::<8>::new(
                similarity,
                query.into(),
                center,
            )),
            (Self::RaBitQ, _) => Box::new(rabitq::QueryDistance::new(
                similarity,
                query.into().as_ref(),
                center,
            )),
            (Self::QuIVer, _) => quiver::new_asymmetric_distance(query.into().as_ref()),
        }
    }

    /// Create a new [`QueryVectorDistance`] that computes distance between a fixed query encoded
    /// in this format and other vectors that are also in this format.
    ///
    /// If `center` is present then it will be accounted for in the distance calculation assuming
    /// all input vectors _also_ use the same center value.
    pub fn query_distance_symmetric<'a>(
        &self,
        similarity: VectorSimilarity,
        query: impl Into<Cow<'a, [u8]>>,
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
            (_, Self::RaBitQ) => quantized_qvd!(rabitq::Distance::new(similarity), query),
            (_, Self::QuIVer) => quiver::new_symmetric_query_distance(query.into()),
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
            "RaBitQ" => Ok(Self::RaBitQ),
            "QuIVer" => Ok(Self::QuIVer),
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
            Self::RaBitQ => write!(f, "RaBitQ"),
            Self::QuIVer => write!(f, "QuIVer"),
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

    /// Compute the distance between the `query` vector and each of the `docs` vectors, writing
    /// the results to `out`.
    ///
    /// This function is not required to be commutative and may panic if one of the inputs is
    /// misshapen. It may also panic if `docs` and `out` are not the same length.
    fn bulk_distance(&self, query: &[u8], docs: &[&[u8]], out: &mut [f64]) {
        for (doc, out) in docs.iter().zip(out.iter_mut()) {
            *out = self.distance(query, doc);
        }
    }
}

/// Distance function for `f32` vectors.
pub trait F32VectorDistance: VectorDistance {
    /// Compute the distance between `a` and `b`; smaller values are better.
    ///
    /// Input vectors must be the same length or this function may panic.
    fn distance_f32(&self, a: &[f32], b: &[f32]) -> f64;
}

/// Distance function for `f16` vectors.
pub trait F16VectorDistance: VectorDistance {
    /// Compute the distance between `a` and `b`; smaller values are better.
    ///
    /// Input vectors must be the same length or this function may panic.
    fn distance_f16(&self, a: &[f16], b: &[f16]) -> f64;
}

/// Estimated distance between two vectors including an error bound.
///
/// The error bound is expected to be a statistical bound as opposed to an arithmetic bound.
/// If the input vector components have a Gaussian distribution then the error bounds should
/// correspond to a Z score of 1.0; callers may adjust the bound depending on their tolerance.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EstimatedDistance {
    /// Estimated distance.
    pub distance: f64,
    /// Error; actual distance is expected to be `distance +/- error`.
    pub error: f64,
}

/// Compute the distance between a fixed vector provided at creation time and other vectors.
/// This is often useful in query flows where everything references a specific point.
pub trait QueryVectorDistance: Send + Sync {
    /// Compute the distance between the bound query vector and `vector`.
    ///
    /// May panic if `vector` has an unexpected shape.
    fn distance(&self, vector: &[u8]) -> f64;

    /// Compute the distance between the bound query vector and `vectors`, writing the results to
    /// `out`.
    ///
    /// May panic if `vectors` and `out` are not the same length or if any of the vectors have an
    /// unexpected shape.
    fn bulk_distance(&self, vectors: &[&[u8]], out: &mut [f64]) {
        for (vector, out) in vectors.iter().zip(out.iter_mut()) {
            *out = self.distance(vector);
        }
    }

    /// Estimated distance between the bound query vector and `vector`.
    ///
    /// Note that not all distance functions will support this so callers should be prepared for the
    /// degenerate case where the error bound is 0.0.
    fn estimated_distance(&self, vector: &[u8]) -> EstimatedDistance {
        EstimatedDistance {
            distance: self.distance(vector),
            error: 0.0,
        }
    }
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

#[cfg(test)]
mod test {
    use crate::{F32VectorCoder, F32VectorCoding, VectorSimilarity, float32::l2_normalize};

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
                l2_normalize(vec).0
            } else {
                vec.into()
            };
            let f32_coder = F32VectorCoding::F32.coder(similarity, None);
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
        let coder = format.coder(similarity, None);
        let a = TestVector::new(a, similarity, coder.as_ref());
        let b = TestVector::new(b, similarity, coder.as_ref());

        let f32_dist_fn = similarity.distance_f32();
        let rf32_dist = f32_dist_fn.distance_f32(&a.rvec, &b.rvec);
        let ru8_dist =
            f32_dist_fn.distance(bytemuck::cast_slice(&a.rvec), bytemuck::cast_slice(&b.rvec));
        assert_float_near!(rf32_dist, ru8_dist, 0.0001, index);

        let dist_fn = format.distance_symmetric(similarity);
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
        let coder = format.coder(similarity, None);
        let a = TestVector::new(a, similarity, coder.as_ref());
        let b = TestVector::new(b, similarity, coder.as_ref());

        let f32_dist_fn = similarity.distance_f32();
        let f32_dist = f32_dist_fn.distance_f32(&a.rvec, &b.rvec);

        let query_dist_fn = format.query_distance_asymmetric(similarity, &a.rvec, None);
        let query_dist = query_dist_fn.distance(&b.qvec);

        assert_float_near!(f32_dist, query_dist, threshold, index);
    }

    use F32VectorCoding::{F16, TLVQ1, TLVQ2, TLVQ4, TLVQ8};
    use VectorSimilarity::{Cosine, Dot, Euclidean};
    use rand::{RngExt, SeedableRng, TryRng, rngs::SysRng};

    macro_rules! distance_test {
        ($name:ident, $sim:path, $coder:path, $epsilon:literal) => {
            #[test]
            fn $name() {
                let seed = SysRng::default().try_next_u64().unwrap();
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
}
