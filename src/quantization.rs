//! Utilities for vector quantization.
//!
//! Graph navigation during search uses these quantized vectors.

use std::{io, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::distance::{
    AsymmetricHammingDistance, HammingDistance, I8NaiveDistance, I8ScaledUniformDotProduct,
    I8ScaledUniformEuclidean, VectorDistance, VectorSimilarity,
};

/// Methods for quantizing vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorQuantizer {
    /// Reduces each dimension to a single bit around the origin point.
    Binary,
    /// Binary quantizes indexed vectors; produces an n-bit representation at
    /// query time to increase precision. This is also used during indexing.
    AsymmetricBinary { n: usize },
    /// Reduces each dimension to an 8-bit integer.
    /// This implementation does not train on any input to decide how to quantize.
    I8Naive,
    /// Reduces each dimension to an 8-bit integer but shaped to the input vector with uniform
    /// scaling across all dimensions.
    ///
    /// The maximum magnitude across all dimensions is used in scaling rather than l2 normalizing
    /// the vector and bounding by [-1,1], which greatly reduces loss. This quantization scheme
    /// works pretty well on transformer models where all values are in roughly the same range.
    /// Scoring can be performed directly on the quantized representation.
    ///
    /// This implementation does _not_ train on a sample of the dataset to decide parameters.
    I8ScaledUniform,
}

impl VectorQuantizer {
    /// Create a new distance function for this quantization method.
    pub fn new_distance_function(&self, similarity: &VectorSimilarity) -> Box<dyn VectorDistance> {
        match (self, similarity) {
            (Self::Binary, _) => Box::new(HammingDistance),
            (Self::AsymmetricBinary { n: _ }, _) => Box::new(AsymmetricHammingDistance),
            (Self::I8Naive, _) => Box::new(I8NaiveDistance(*similarity)),
            (Self::I8ScaledUniform, VectorSimilarity::Dot) => Box::new(I8ScaledUniformDotProduct),
            (Self::I8ScaledUniform, VectorSimilarity::Euclidean) => {
                Box::new(I8ScaledUniformEuclidean)
            }
        }
    }
}

impl Default for VectorQuantizer {
    fn default() -> Self {
        Self::Binary
    }
}

impl FromStr for VectorQuantizer {
    type Err = io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let input_err = |s| io::Error::new(io::ErrorKind::InvalidInput, s);
        match s {
            "binary" => Ok(Self::Binary),
            ab if ab.starts_with("asymmetric_binary:") => {
                let bits_str = ab
                    .strip_prefix("asymmetric_binary:")
                    .expect("prefix matched");
                bits_str
                    .parse::<usize>()
                    .ok()
                    .and_then(|b| if (1..=8).contains(&b) { Some(b) } else { None })
                    .map(|n| Self::AsymmetricBinary { n })
                    .ok_or_else(|| input_err(format!("invalid asymmetric_binary bits {bits_str}")))
            }
            "i8naive" => Ok(Self::I8Naive),
            "i8scaled-uniform" => Ok(Self::I8ScaledUniform),
            _ => Err(input_err(format!("unknown quantizer function {s}"))),
        }
    }
}
