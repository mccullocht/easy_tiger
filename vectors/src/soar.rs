//! Implementation of SOAR indexing distance for selecting replica centroids.
//!
//! https://arxiv.org/pdf/2404.00774

use crate::{DotProductDistance, F32VectorDistance};

/// Compute SOAR distance between an input vector and another (centroid) vector, considering
/// orthogonality to the primary centroid.
///
/// This computation requires the input vectors to be single precision floats; callers are required
/// to decode any quantized vector format into floats before computing the value. It is recommended
/// that callers use a high precision format for this purpose: f32, f16, or lvq2x.
pub struct SoarQueryVectorDistance<'a> {
    /// The index vector to compute centroid distance against.
    vector: &'a [f32],
    /// Residual of vector and primary centroid.
    primary_residual: Vec<f32>,
    /// Dot of the residual vector against itself.
    primary_residual_dot: f64,
    /// Hyper parameter. Larger values select for orthogonality more aggressively.
    /// When combined with a geometric ratio filter larger values tend to result in fewer replicas.
    /// TODO: choose a lambda value based on expected MSE or the residual.
    lambda: f64,
}

impl<'a> SoarQueryVectorDistance<'a> {
    /// Default lambda value used when unspecified.
    const DEFAULT_LAMBDA: f64 = 1.0;

    /// Create a new soar vector from a vector reference and the closest centroid.
    pub fn new(vector: &'a [f32], centroid: &[f32]) -> Self {
        Self::with_lambda(vector, centroid, Self::DEFAULT_LAMBDA)
    }

    /// Create a new soar vector from a vector reference, the closest centroid, and lambda param.
    pub fn with_lambda(vector: &'a [f32], centroid: &[f32], lambda: f64) -> Self {
        assert_eq!(vector.len(), centroid.len());

        let primary_residual = vector
            .iter()
            .zip(centroid.iter())
            .map(|(v, c)| *v - *c)
            .collect::<Vec<_>>();
        let primary_residual_dot =
            DotProductDistance::get().distance_f32(&primary_residual, &primary_residual);
        Self {
            vector,
            primary_residual,
            primary_residual_dot,
            lambda,
        }
    }

    /// Compute loss for `centroid` as a secondary assignment. Lower values are preferred.
    pub fn loss(&self, centroid: &[f32]) -> f64 {
        let mut secondary_residual_dot = 0.0;
        let mut residual_dot = 0.0;
        for ((&v, &r), &c) in self
            .vector
            .iter()
            .zip(self.primary_residual.iter())
            .zip(centroid.iter())
        {
            let diff = v - c;
            secondary_residual_dot = diff.mul_add(diff, secondary_residual_dot);
            residual_dot = diff.mul_add(r, residual_dot)
        }
        f64::from(secondary_residual_dot)
            + self.lambda * f64::from(residual_dot).powi(2) / self.primary_residual_dot
    }
}
