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

use rand::prelude::*;
use rand::seq::index;
use rayon::prelude::*;
use simsimd::SpatialSimilarity;

use crate::input::{SubsetViewVectorStore, VecVectorStore, VectorStore};

const ITERS: usize = 15;
const EPSILON: f64 = 0.01;

// XXX this could return Result<V, V> to indicate if we converged or not.
pub fn batch_kmeans<V: VectorStore<Elem = f32> + Send + Sync>(
    training_data: &V,
    k: usize,
    batch_size: usize,
    rng: &mut impl Rng,
) -> VecVectorStore<f32> {
    let mut batch_iter = BatchIter::new(training_data, batch_size, rng);

    let mut batch = batch_iter.next().expect("BatchIter is perpetual");
    // XXX initialize_batch_centroids computes assignment internally and discards them, but they
    // would be sufficient for the first iteration.
    let mut centroids = initialize_batch_centroids(&batch, k, rng);
    let mut centroid_counts = vec![0.0; centroids.len()];

    for _ in 0..ITERS {
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
        if centroid_distance_sum < EPSILON {
            break;
        }

        batch = batch_iter.next().expect("BatchIter is perpetual");
    }

    centroids
}

fn initialize_batch_centroids<V: VectorStore<Elem = f32> + Send + Sync>(
    training_data: &V,
    k: usize,
    rng: &mut impl Rng,
) -> VecVectorStore<f32> {
    (0..ITERS)
        .map(|_| {
            let centroids = initialize_centroids(training_data, k, rng);
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

fn initialize_centroids(
    training_data: &impl VectorStore<Elem = f32>,
    k: usize,
    rng: &mut impl Rng,
) -> VecVectorStore<f32> {
    // XXX kmeans++ initialization.
    let mut centroids = VecVectorStore::with_capacity(training_data.elem_stride(), k);
    for i in index::sample(rng, training_data.len(), k) {
        centroids.push(&training_data[i]);
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
