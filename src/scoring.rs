use simsimd::{BinarySimilarity, SpatialSimilarity};

/// Trait type for a vector scorer.
pub trait VectorScorer {
    type Elem;

    /// Score vectors `a` and `b` against one another. Returns a score
    /// where higher values are better matches.
    ///
    /// Input vectors must be the same length or this function may panic.
    fn score(a: &[Self::Elem], b: &[Self::Elem]) -> f64;

    /// Normalize a vector for use with this scoring function.
    /// By default, does nothing.
    fn normalize(_vector: &mut [Self::Elem]) {}
}

/// Computes a score based on l2 distance.
#[derive(Debug, Copy, Clone)]
pub struct EuclideanScorer;

impl VectorScorer for EuclideanScorer {
    type Elem = f32;

    fn score(a: &[Self::Elem], b: &[Self::Elem]) -> f64 {
        1f64 / (1f64 + SpatialSimilarity::l2sq(a, b).unwrap())
    }
}

/// Computes a score based on the dot product.
#[derive(Debug, Copy, Clone)]
pub struct DotProductScorer;

impl VectorScorer for DotProductScorer {
    type Elem = f32;

    fn score(a: &[Self::Elem], b: &[Self::Elem]) -> f64 {
        // Assuming values are normalized, this will produce a score in [0,1]
        (1f64 + SpatialSimilarity::dot(a, b).unwrap()) / 2f64
    }

    fn normalize(vector: &mut [Self::Elem]) {
        let norm = SpatialSimilarity::dot(vector, vector).unwrap().sqrt() as f32;
        for d in vector.iter_mut() {
            *d /= norm;
        }
    }
}

/// Computes a score from two bitmaps using hamming distance.
#[derive(Debug, Copy, Clone)]
pub struct HammingScorer;

impl VectorScorer for HammingScorer {
    type Elem = u8;

    fn score(a: &[Self::Elem], b: &[Self::Elem]) -> f64 {
        let dim = (a.len() * 8) as f64;
        (dim - BinarySimilarity::hamming(a, b).unwrap()) / dim
    }
}
