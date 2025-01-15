// batch k-means
// * batch is sampled randomly from total input. probably best to shuffle the indices for this.
// * centroid initialization is done using only the first batch.
//   - run initialization n times and compute distances, then take the lowest sum-of-distances.
//   - kmeans++ is probably going to be unpleasant, maybe stick to random
// * iterate over batches
//   - compute centroid memberships and distances
//   - cluster count is propagated between batches
//   - update centroids incrementally
//     + propagate cluster counts between batches
//     + increment centroids, then add (sample - centroid) / count to the assigned centroid
//   - terminate if sum of distances between new and old centroids is within tolerances.
//     + make sure to use proper l2 distance.

use std::iter::Cycle;

use rand::seq::index;
use rand::{distributions::WeightedIndex, prelude::*};
use rayon::prelude::*;
use simsimd::SpatialSimilarity;

use crate::input::{SubsetViewVectorStore, VecVectorStore, VectorStore};

/// Centroid initialization method for k-means partitioning.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum InitializationMethod {
    /// Choose centers randomly from the data set.
    Random,
    /// Choose centers randomly from the data set weighted by distance to other centers.
    KMeansPlusPlus,
}

/// Parameters for k-means partitioning.
#[derive(Debug, PartialEq)]
pub struct Params {
    /// Maximum number of iterations to run before exiting, even if the centers have not converged.
    pub iters: usize,
    /// Convergence epsilon. Computation is considered to have converged if the sum of distances
    /// between two iterations of centroid is less than this amount.
    pub epsilon: f64,
    /// Algorithm for computing initial centroids.
    pub initialization: InitializationMethod,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            iters: 15,
            epsilon: 0.01,
            initialization: InitializationMethod::Random,
        }
    }
}

/// Compute batch k-means over `training_data`. `k` clusters will be produced and each batch will
/// contain exactly `batch_size` sample vectors.
///
/// Returns the computed centroids -- `Ok()` if the centroid computation converged and `Err()` if
/// we terminated by reaching max iterations.
pub fn batch_kmeans<V: VectorStore<Elem = f32> + Send + Sync>(
    training_data: &V,
    k: usize,
    batch_size: usize,
    params: &Params,
    rng: &mut impl Rng,
) -> Result<VecVectorStore<f32>, VecVectorStore<f32>> {
    let mut centroids = VecVectorStore::new(training_data.elem_stride());
    let mut centroid_counts = vec![0.0; k];
    for batch in BatchIter::new(training_data, batch_size, rng).take(params.iters) {
        if centroids.is_empty() {
            centroids = initialize_batch_centroids(training_data, k, params, rng);
        }

        let mut new_centroids = centroids.clone();
        for (vector, cluster) in batch.iter().zip(
            compute_assignments(training_data, &centroids)
                .into_iter()
                .map(|(c, _)| c),
        ) {
            centroid_counts[cluster] += 1.0;
            // Update means vectors in new_centroids.
            for (v, c) in vector.iter().zip(new_centroids[cluster].iter_mut()) {
                *c += (*v - *c) / centroid_counts[cluster];
            }
        }

        let centroid_distance_sum = compute_centroid_distance_sum(&centroids, &new_centroids);
        centroids = new_centroids;
        if centroid_distance_sum < params.epsilon {
            return Ok(centroids);
        }
    }

    Err(centroids)
}

fn initialize_batch_centroids<V: VectorStore<Elem = f32> + Send + Sync>(
    training_data: &V,
    k: usize,
    params: &Params,
    rng: &mut impl Rng,
) -> VecVectorStore<f32> {
    (0..params.iters)
        .map(|_| {
            let centroids = initialize_centroids(training_data, k, params.initialization, rng);
            let distances: f64 = compute_assignments(training_data, &centroids)
                .into_iter()
                .map(|a| a.1)
                .sum();
            (centroids, distances)
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("non-zero iters")
        .0
}

fn initialize_centroids<V: VectorStore<Elem = f32> + Send + Sync>(
    training_data: &V,
    k: usize,
    method: InitializationMethod,
    rng: &mut impl Rng,
) -> VecVectorStore<f32> {
    let mut centroids = VecVectorStore::with_capacity(training_data.elem_stride(), k);
    match method {
        InitializationMethod::Random => {
            for i in index::sample(rng, training_data.len(), k) {
                centroids.push(&training_data[i]);
            }
        }
        InitializationMethod::KMeansPlusPlus => {
            centroids.push(&training_data[rng.gen_range(0..training_data.len())]);
            while centroids.len() < k {
                let assignments = compute_assignments(training_data, &centroids);
                let index = WeightedIndex::new(assignments.into_iter().map(|a| a.1))
                    .unwrap()
                    .sample(rng);
                centroids.push(&training_data[index]);
            }
        }
    }
    centroids
}

fn compute_assignments<
    V: VectorStore<Elem = f32> + Send + Sync,
    C: VectorStore<Elem = f32> + Send + Sync,
>(
    training_data: &V,
    centroids: &C,
) -> Vec<(usize, f64)> {
    (0..training_data.len())
        .into_par_iter()
        .map(|i| {
            let v = &training_data[i];
            centroids
                .iter()
                .map(|c| SpatialSimilarity::l2(v, c).expect("same vector len"))
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .expect("at least one centroid")
        })
        .collect()
}

fn compute_centroid_distance_sum<
    C: VectorStore<Elem = f32> + Send + Sync,
    D: VectorStore<Elem = f32> + Send + Sync,
>(
    old: &C,
    new: &D,
) -> f64 {
    (0..old.len())
        .into_par_iter()
        .map(|i| SpatialSimilarity::l2(&old[i], &new[i]).expect("same vector len"))
        .sum()
}

struct BatchIter<'a, V> {
    training_data: &'a V,
    indices_iter: Cycle<std::vec::IntoIter<usize>>,
    batch_size: usize,
}

impl<'a, V: VectorStore<Elem = f32>> BatchIter<'a, V> {
    fn new(training_data: &'a V, batch_size: usize, rng: &mut impl Rng) -> Self {
        let mut shuffled_indices = (0..training_data.len()).collect::<Vec<_>>();
        shuffled_indices.shuffle(rng);
        Self {
            training_data,
            indices_iter: shuffled_indices.into_iter().cycle(),
            batch_size,
        }
    }
}

impl<'a, V: VectorStore<Elem = f32>> Iterator for BatchIter<'a, V> {
    type Item = SubsetViewVectorStore<'a, V>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut subset = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            subset.push(self.indices_iter.next().expect("cycle iter"));
        }
        Some(SubsetViewVectorStore::new(self.training_data, subset))
    }
}
