//! Implementation of SOAR indexing distance for selecting replica centroids.
//!
//! https://arxiv.org/abs/2404.00774

use crate::{F32VectorDistance, float32::EuclideanDistance};

/// Compute SOAR distance between an input vector and another (centroid) vector, considering
/// orthogonality to the primary centroid.
///
/// This computation requires the input vectors to be single precision floats; callers are required
/// to decode any quantized vector format into floats before computing the value. It is recommended
/// that callers use a high precision format for this purpose: f32, f16, or lvq2x.
pub struct SoarQueryVectorDistance<'a> {
    // The vector to compute centroid distance against.
    vector: &'a [f32],
    // Residual of vector and primary centroid.
    residual: Vec<f32>,
    // Distance between vector and primary centroid.
    l2_dist_sq: f64,
    // Hyper parameter.
    lambda: f64,
}

impl<'a> SoarQueryVectorDistance<'a> {
    /// Default lambda value used when unspecified.
    const DEFAULT_LAMBDA: f64 = 1.0;

    /// Create a new soar vector from a vector reference and the closest centroid.
    pub fn new(vector: &'a [f32], centroid: &[f32]) -> Option<Self> {
        Self::with_lambda(vector, centroid, Self::DEFAULT_LAMBDA)
    }

    /// Create a new soar vector from a vector reference, the closest centroid, and lambda param.
    pub fn with_lambda(vector: &'a [f32], centroid: &[f32], lambda: f64) -> Option<Self> {
        assert_eq!(vector.len(), centroid.len());

        // If the vector and centroid are very close to each other then decline to provide SOAR
        // distance scoring; the caller should not select any replicas.
        let l2_dist_sq = EuclideanDistance::default().distance_f32(vector, centroid);
        if l2_dist_sq < 1e-10 {
            return None;
        }
        let residual = vector
            .iter()
            .zip(centroid.iter())
            .map(|(v, c)| *v - *c)
            .collect::<Vec<_>>();
        Some(Self {
            vector,
            residual,
            l2_dist_sq: l2_dist_sq.into(),
            lambda,
        })
    }

    /// Compute the SOAR distance between a fixed query and a new centroid vector.
    pub fn distance(&self, centroid: &[f32]) -> f64 {
        let mut centroid_l2_dist_sq = 0.0;
        let mut centroid_residual_projection = 0.0;
        for ((v, r), c) in self
            .vector
            .iter()
            .zip(self.residual.iter())
            .zip(centroid.iter())
        {
            let diff = *v - *c;
            centroid_l2_dist_sq = diff.mul_add(diff, centroid_l2_dist_sq);
            centroid_residual_projection = diff.mul_add(*r, centroid_residual_projection)
        }
        f64::from(centroid_l2_dist_sq)
            + self.lambda
                * f64::from(centroid_residual_projection)
                * f64::from(centroid_residual_projection)
                / self.l2_dist_sq
    }
}
