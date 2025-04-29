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
            epsilon: 0.0001,
            initialization: InitializationMethod::Random,
        }
    }
}

/// Iteratively compute k-means over `dataset`.
///
/// To do this we partition the dataset into `intermediate_k` partitions, then begin dividing
/// the largest partitions into no more than `intermediate_k` sub-partitions until we have `k`
/// total partitions. `balance_factor` is a number > 1.0 that tries to control max imbalance
/// relative to the average number of vectors you would expect in a single partition. Lower values
/// should reduce imbalance but at greater cost.
///
/// This is intended to be used when `k` is large (> 100) as it reduces the cost of re-computing
/// centroid assignments.
///
/// Returns a set of `k` centroids and the assignment of each vector in `dataset` to the centroids.
pub fn iterative_balanced_kmeans<V: VectorStore<Elem = f32> + Send + Sync>(
    dataset: &V,
    k: usize,
    intermediate_k: usize,
    balance_factor: f64,
    batch_size: usize,
    params: &Params,
    rng: &mut impl Rng,
) -> (VecVectorStore<f32>, Vec<(usize, f64)>) {
    assert!(intermediate_k < k);

    let mut centroids = match batch_kmeans(dataset, intermediate_k, batch_size, params, rng) {
        Ok(c) => c,
        Err(c) => {
            eprintln!("Initial iterative_balanced_kmeans step failed to converge!");
            c
        }
    };
    let mut assignments = compute_assignments(dataset, &centroids);

    // TODO: maybe also have a minimum size and try to combine small clusters with others?
    // in that case we need to know the closest centroid(s) for small centroids to do the
    // combination step.
    let max_centroid_size = ((dataset.len() as f64 * balance_factor) / k as f64) as usize;
    while centroids.len() < k {
        let mut centroid_to_vectors = assignments.iter().enumerate().fold(
            vec![vec![]; centroids.len()],
            |mut c2v, (i, (c, _))| {
                c2v[*c].push(i);
                c2v
            },
        );

        // Count the number of vectors in each centroid, then decide how pieces to divide each
        // centroid into. This is done greedily starting from the large centroid, and we terminate
        // once the splits would result in `k` centroids.
        let mut centroid_counts = centroid_to_vectors
            .iter()
            .map(|v| v.len())
            .enumerate()
            .collect::<Vec<_>>();
        centroid_counts.sort_unstable_by_key(|(i, c)| (*c, *i));
        let mut remaining_centroids = k - centroids.len();
        for (_, count) in centroid_counts.iter_mut().rev() {
            if remaining_centroids == 0 || *count <= max_centroid_size {
                *count = 1;
            } else {
                remaining_centroids += 1; // we're going to split this one.
                *count = remaining_centroids.min((*count / max_centroid_size) + 1);
                remaining_centroids -= *count;
            }
        }

        // Everything in centroid_count before unsplit_len will not be divided. This has
        // implications for re-computing assignment since these only need to compute centroid
        // assignment against added centroids.
        let unsplit_len = centroid_counts
            .iter()
            .position(|(_, p)| *p > 1)
            .unwrap_or(centroid_counts.len());

        // If all the centroids are too big to split based on our balance factor and we haven't
        // reached k we'll just terminate.
        if unsplit_len == centroid_counts.len() {
            break;
        }

        let mut new_centroids = VecVectorStore::with_capacity(
            centroids.elem_stride(),
            centroid_counts.iter().map(|(_, c)| *c).sum::<usize>(),
        );
        for (i, (c, nk)) in centroid_counts.iter().enumerate() {
            // Update assignment indexes for all the centroid vectors; assignment distances will be
            // recomputed once all centroids have been split.
            let centroid_vectors = std::mem::take(&mut centroid_to_vectors[*c]);
            for i in centroid_vectors.iter().copied() {
                assignments[i].0 = new_centroids.len();
            }

            if i < unsplit_len {
                new_centroids.push(&centroids[*c]);
            } else {
                let subset_centroids = match batch_kmeans(
                    &SubsetViewVectorStore::new(dataset, centroid_vectors),
                    *nk,
                    batch_size,
                    params,
                    rng,
                ) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("iterative_balanced_kmeans centroid split failed to converge!");
                        e
                    }
                };
                for c in subset_centroids.iter() {
                    new_centroids.push(c);
                }
            }
        }

        update_assignment_centroid_split(dataset, &new_centroids, unsplit_len, &mut assignments);
        centroids = new_centroids;
    }

    (centroids, assignments)
}

fn update_assignment_centroid_split<
    V: VectorStore<Elem = f32> + Send + Sync,
    C: VectorStore<Elem = f32> + Send + Sync,
>(
    dataset: &V,
    centroids: &C,
    unsplit_len: usize,
    assignments: &mut [(usize, f64)],
) {
    assignments
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, assignment)| {
            let v = &dataset[i];
            *assignment = if assignment.0 < unsplit_len {
                // This centroid was not split so we only need to compare to new centroids, making
                // sure to include the original assigned centroid in the mix.
                centroids
                    .iter()
                    .enumerate()
                    .skip(unsplit_len)
                    .map(|(i, c)| (i, SpatialSimilarity::l2(v, c).expect("same vector len")))
                    .chain(std::iter::once(*assignment))
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .expect("at least one centroid")
            } else {
                // This centroid was split so we need to completely recompute assignment.
                centroids
                    .iter()
                    .map(|c| SpatialSimilarity::l2(v, c).expect("same vector len"))
                    .enumerate()
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .expect("at least one centroid")
            };
        });
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
    for batch in BatchIter::new(training_data, batch_size, rng).take(params.iters) {
        let mut new_centroids = centroids.clone();
        for (vector, (cluster, _)) in batch
            .iter()
            .zip(compute_assignments(&batch, &centroids).into_iter())
        {
            centroid_counts[cluster] += 1.0;
            // Update means vectors in new_centroids.
            for (v, c) in vector.iter().zip(new_centroids[cluster].iter_mut()) {
                *c += (*v - *c) / centroid_counts[cluster];
            }
        }

        // TODO: consider replacement of centroids? some centers have few/no vectors and we could
        // reassign these to maybe improve things?

        let centroid_distance_max = compute_centroid_distance_max(&centroids, &new_centroids);
        centroids = new_centroids;
        // Terminate if _every_ centroid distance is less than epsilon.
        if centroid_distance_max < params.epsilon {
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
    dataset: &V,
    centroids: &C,
) -> Vec<(usize, f64)> {
    (0..dataset.len())
        .into_par_iter()
        .map(|i| {
            let v = &dataset[i];
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
        .max_by(|a, b| a.total_cmp(b))
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
