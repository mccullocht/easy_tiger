//! An implementation of k-means algorithms for clustering vectors.

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
    /// Maximum number of iterations when initializing centroids.
    pub init_iters: usize,
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
            init_iters: 3,
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
    let mut centroids = initialize_batch_centroids(training_data, k, batch_size, params, rng);
    let mut centroid_counts = vec![0.0; k];
    for (iter, batch) in BatchIter::new(training_data, batch_size, rng)
        .take(params.iters)
        .enumerate()
    {
        // XXX two balancing techniques:
        // * balance factor: allow injecting an update to score, do so based on factor * count.
        // * iterative partitioning: take members of largest centroid, divide it N ways with some
        //   target number of members involved. must globally recompute assignments afterward.
        // XXX assignment is a big problem when k is large (> 256). failure to converge, very slow.
        // * build a graph lol.
        // * quantize and re-score at the top.
        let mut new_centroids = centroids.clone();
        for (vector, (cluster, _)) in batch
            .iter()
            .zip(compute_assignments(&batch, &centroids).into_iter())
        {
            // XXX store centroid sums separately? needs extra state (3rd copy of centroids)
            centroid_counts[cluster] += 1.0;
            // Update means vectors in new_centroids.
            for (v, c) in vector.iter().zip(new_centroids[cluster].iter_mut()) {
                *c += (*v - *c) / centroid_counts[cluster];
            }
        }

        // XXX consider replacement of centroids? some centers have few/no vectors and we could
        // reassign these to maybe improve things?
        //
        // lambda addition to score based on current cluster count.

        let centroid_distance_max = compute_centroid_distance_max(&centroids, &new_centroids);
        centroids = new_centroids;
        // Terminate if _every_ centroid distance is less than epsilon.
        if centroid_distance_max < params.epsilon {
            println!("  converge at batch {}", iter);
            return Ok(centroids);
        }
    }

    Err(centroids)
}

fn initialize_batch_centroids<V: VectorStore<Elem = f32> + Send + Sync>(
    dataset: &V,
    k: usize,
    batch_size: usize,
    params: &Params,
    rng: &mut impl Rng,
) -> VecVectorStore<f32> {
    let training_data = SubsetViewVectorStore::new(
        dataset,
        index::sample(rng, dataset.len(), batch_size.min(dataset.len())).into_vec(),
    );
    (0..params.init_iters)
        .map(|_| initialize_centroids(dataset, &training_data, k, params.initialization, rng))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("non-zero iters")
        .0
}

fn initialize_centroids<
    D: VectorStore<Elem = f32> + Send + Sync,
    T: VectorStore<Elem = f32> + Send + Sync,
>(
    dataset: &D,
    training_data: &T,
    k: usize,
    method: InitializationMethod,
    rng: &mut impl Rng,
) -> (VecVectorStore<f32>, f64) {
    let mut centroids = VecVectorStore::with_capacity(dataset.elem_stride(), k);
    let assignments = match method {
        InitializationMethod::Random => {
            for i in index::sample(rng, dataset.len(), k) {
                centroids.push(&dataset[i]);
            }
            compute_assignments(training_data, &centroids)
        }
        InitializationMethod::KMeansPlusPlus => {
            centroids.push(&dataset[rng.gen_range(0..dataset.len())]);
            let mut assignments = compute_assignments(training_data, &centroids);
            while centroids.len() < k {
                let index = WeightedIndex::new(assignments.iter().map(|a| a.1))
                    .unwrap()
                    .sample(rng);

                let centroid = centroids.len();
                centroids.push(&training_data[index]);
                let centroid_vector = &centroids[centroid];
                let distances = (0..training_data.len())
                    .into_par_iter()
                    .map(|i| SpatialSimilarity::l2(&training_data[i], centroid_vector).unwrap())
                    .collect::<Vec<_>>();
                for ((cluster, distance), new_distance) in
                    assignments.iter_mut().zip(distances.into_iter())
                {
                    if new_distance < *distance {
                        *cluster = centroid;
                        *distance = new_distance;
                    }
                }
            }
            assignments
        }
    };
    let distance_sum = assignments.into_iter().map(|a| a.1).sum::<f64>();
    (centroids, distance_sum)
}

/// For each input vector compute the closest centroid and the distance to that centroid.
pub fn compute_assignments<
    V: VectorStore<Elem = f32> + Send + Sync,
    C: VectorStore<Elem = f32> + Send + Sync,
>(
    data: &V,
    centroids: &C,
) -> Vec<(usize, f64)> {
    (0..data.len())
        .into_par_iter()
        .map(|i| {
            let v = &data[i];
            centroids
                .iter()
                .map(|c| SpatialSimilarity::l2(v, c).expect("same vector len"))
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .expect("at least one centroid")
        })
        .collect()
}

/// Compute the maximum distance between new and old centroids.
fn compute_centroid_distance_max<
    C: VectorStore<Elem = f32> + Send + Sync,
    D: VectorStore<Elem = f32> + Send + Sync,
>(
    old: &C,
    new: &D,
) -> f64 {
    (0..old.len())
        .into_par_iter()
        .map(|i| SpatialSimilarity::l2(&old[i], &new[i]).expect("same vector len"))
        .max_by(|a, b| a.total_cmp(&b))
        .expect("non-zero k")
}

struct BatchIter<'a, V> {
    training_data: &'a V,
    indices_iter: Cycle<std::vec::IntoIter<usize>>,
    batch_size: usize,
}

impl<'a, V: VectorStore<Elem = f32>> BatchIter<'a, V> {
    fn new(training_data: &'a V, batch_size: usize, rng: &mut impl Rng) -> Self {
        let batch_size = batch_size.min(training_data.len());
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
