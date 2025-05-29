//! An implementation of k-means algorithms for clustering vectors.

use std::iter::Cycle;

use rand::seq::index;
use rand::{distributions::WeightedIndex, prelude::*};
use rayon::prelude::*;

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
#[derive(Debug, PartialEq, Clone)]
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
// TODO: guarantee that this terminates by giving up at some point, like if the number of centroids
// hasn't changed in some number of iterations.
// XXX should accept a range of cluster sizes, and max must be at least 2x min
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

    let min_centroid_size = ((dataset.len() as f64) * (1.0 / balance_factor) / k as f64) as usize;
    let max_centroid_size = ((dataset.len() as f64 * balance_factor) / k as f64) as usize;
    loop {
        eprintln!("do split centroids {} < {}", centroids.len(), k);
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
        for (_, count) in centroid_counts.iter_mut().rev() {
            *count = intermediate_k.min((*count / max_centroid_size) + 1);
        }

        let unsplit_len = centroid_counts
            .iter()
            .position(|(_, p)| *p > 1)
            .unwrap_or(centroid_counts.len());

        // Terminate if nothing can be split either because we've hit k or all the clusters are too
        // small to split.
        if unsplit_len == centroid_counts.len() {
            break;
        }

        let mut new_centroids = VecVectorStore::with_capacity(
            centroids.elem_stride(),
            centroid_counts.iter().map(|(_, c)| *c).sum::<usize>(),
        );
        // For centroids that are not being split, add the new centroid and update assignments.
        for (c, _) in centroid_counts.iter().take(unsplit_len) {
            let centroid_vectors = std::mem::take(&mut centroid_to_vectors[*c]);
            for i in centroid_vectors.iter().copied() {
                assignments[i].0 = new_centroids.len();
            }
            new_centroids.push(&centroids[*c]);
        }

        // For centroids that are being split, re-partition them and discard any smol clusters,
        // then update assignments for the next pass through the loop.
        for (c, nk) in centroid_counts.iter().skip(unsplit_len) {
            // XXX we should check that centroids_to_vectors doesn't contain any dupes.
            let subset_vectors =
                SubsetViewVectorStore::new(dataset, std::mem::take(&mut centroid_to_vectors[*c]));
            let (mut subset_centroids, mut subset_assignments) = match *nk {
                2 => bp_vectors(&subset_vectors, params.iters, min_centroid_size),
                _ => {
                    let subset_centroids =
                        match batch_kmeans(&subset_vectors, *nk, batch_size, params, rng) {
                            Ok(c) => c,
                            Err(e) => {
                                eprintln!(
                                    "iterative_balanced_kmeans centroid split failed to converge!"
                                );
                                e
                            }
                        };
                    let subset_assignments =
                        compute_assignments(&subset_vectors, &subset_centroids);
                    (subset_centroids, subset_assignments)
                }
            };
            let mut subset_centroids_to_vectors = subset_assignments.iter().enumerate().fold(
                vec![vec![]; subset_centroids.len()],
                |mut c2v, (i, (c, d))| {
                    c2v[*c].push((subset_vectors.original_index(i), *d));
                    c2v
                },
            );

            // XXX dump all of this into refine_centroids() or something. this function is like 200 lines long!
            // Remove any partitions that do not meet minimum size requirements. The partition count logic should ensure that at least
            // one centroid is above the size enforced by policy. This should never happen when *nk == 2 as binary partition will ensure
            // that the smaller cluster meets the minimum size.
            // This was large enough to partition so at least _one_ centroid should be above the size enforced by policy.
            let centroids_to_keep = subset_centroids_to_vectors
                .iter()
                .map(|v| v.len())
                .filter(|v| *v >= min_centroid_size)
                .count();
            assert_ne!(centroids_to_keep, 0);
            if centroids_to_keep < *nk {
                if centroids_to_keep == 1 {
                    // An N-way partition only produced one viable partition. Break the deadlock by binary partitioning.
                    (subset_centroids, subset_assignments) =
                        bp_vectors(&subset_vectors, params.iters, min_centroid_size);
                    subset_centroids_to_vectors.resize(2, vec![]);
                    subset_centroids_to_vectors[0].clear();
                    subset_centroids_to_vectors[1].clear();
                    for (i, (c, d)) in subset_assignments.iter().enumerate() {
                        subset_centroids_to_vectors[*c]
                            .push((subset_vectors.original_index(i), *d));
                    }
                } else {
                    // Iteratively remove the smallest partition and reassign vectors until all remaining centroids are larger
                    // than min_centroid_size.
                    let mut centroids_to_prune = subset_centroids_to_vectors
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| !v.is_empty() && v.len() < min_centroid_size)
                        .map(|(i, v)| (v.len(), i))
                        .collect::<Vec<_>>();
                    centroids_to_prune.sort_unstable();
                    for (_, pruned_centroid) in centroids_to_prune {
                        let reassign_vectors = SubsetViewVectorStore::new(
                            dataset,
                            std::mem::take(&mut subset_centroids_to_vectors[pruned_centroid])
                                .into_iter()
                                .map(|(i, _)| i)
                                .collect(),
                        );
                        let kept_centroids = SubsetViewVectorStore::new(
                            &subset_centroids,
                            subset_centroids_to_vectors
                                .iter()
                                .enumerate()
                                .filter(|(_, v)| !v.is_empty())
                                .map(|(i, _)| i)
                                .collect(),
                        );
                        for (i, (new_centroid, d)) in
                            compute_assignments(&reassign_vectors, &kept_centroids)
                                .into_iter()
                                .enumerate()
                        {
                            // XXX check that the vector to push doesn't appear in the target.
                            subset_centroids_to_vectors
                                [kept_centroids.original_index(new_centroid)]
                            .push((reassign_vectors.original_index(i), d));
                        }

                        if subset_centroids_to_vectors
                            .iter()
                            .all(|v| v.is_empty() || v.len() >= min_centroid_size)
                        {
                            break;
                        }
                    }
                }
            }

            for (c, v) in subset_centroids
                .iter()
                .zip(subset_centroids_to_vectors.iter())
                .filter(|(_, v)| !v.is_empty())
            {
                let centroid_id = new_centroids.len();
                for (i, d) in v {
                    assignments[*i] = (centroid_id, *d);
                }
                new_centroids.push(c);
            }
        }

        centroids = new_centroids;
    }

    (centroids, assignments)
}

// Build a binary partition of the vectors between two centroids.
//
// Split the dataset in half and produce two centroids, then measure the distance for each vector to both centroids
// and the weight of vectors to each centroid. Terminate if the centroids would produce two clusters of at least
// min_cluster_size, otherwise split the dataset into a new grouping and try again.
//
// Returns a "left" centroid, a "right" centroid, and cluster assignments for each vector in dataset.
// The returned assignments use 0 for left, 1 for right, and provide the distance.
// XXX this should return a Result to indicate when it failed to converge.
fn bp_vectors<V: VectorStore<Elem = f32> + Send + Sync>(
    dataset: &V,
    max_iters: usize,
    min_cluster_size: usize,
) -> (VecVectorStore<f32>, Vec<(usize, f64)>) {
    let acceptable_split = min_cluster_size..=(dataset.len() - min_cluster_size);
    assert!(!acceptable_split.is_empty());

    let split = dataset.len() / 2;
    let mut left_vectors = SubsetViewVectorStore::new(dataset, (0..split).collect());
    let mut right_vectors = SubsetViewVectorStore::new(dataset, (split..dataset.len()).collect());

    let mut left_centroid = vec![0.0; dataset.elem_stride()];
    let mut right_centroid = vec![0.0; dataset.elem_stride()];
    let mut distances = vec![(0usize, 0.0, 0.0, 0.0); dataset.len()];
    for _ in 0..max_iters {
        bp_update_centroid(&left_vectors, &mut left_centroid);
        bp_update_centroid(&right_vectors, &mut right_centroid);
        distances.par_iter_mut().enumerate().for_each(|(i, d)| {
            let ldist = crate::distance::l2sq(&dataset[i], &left_centroid);
            let rdist = crate::distance::l2sq(&dataset[i], &right_centroid);
            *d = (i, ldist, rdist, ldist - rdist)
        });
        distances.sort_unstable_by(|a, b| a.3.total_cmp(&b.3).then_with(|| a.0.cmp(&b.0)));
        // We want to produce two clusters of at least min_cluster_size and are happy to accept any such outcome.
        // TODO: consider iterating a couple more times after we are happy with the outcome to see if the split can get better.
        // TODO: exit early if we are not making progress.
        if acceptable_split.contains(
            &distances
                .iter()
                .position(|d| d.3 >= 0.0)
                .unwrap_or(dataset.len()),
        ) {
            break;
        }

        left_vectors = SubsetViewVectorStore::new(
            dataset,
            distances.iter().take(split).map(|d| d.0).collect(),
        );
        right_vectors = SubsetViewVectorStore::new(
            dataset,
            distances.iter().skip(split).map(|d| d.0).collect(),
        );
    }

    // XXX maybe left and right should always be in centroids???
    let mut centroids = VecVectorStore::with_capacity(dataset.elem_stride(), 2);
    centroids.push(&left_centroid);
    centroids.push(&right_centroid);

    let split_point = (*acceptable_split.end()).min(
        (*acceptable_split.start()).max(
            distances
                .iter()
                .position(|d| d.3 >= 0.0)
                .unwrap_or(dataset.len()),
        ),
    );
    assert!(acceptable_split.contains(&split_point));

    let mut assignments = vec![(0, 0.0); dataset.len()];
    for (i, (vidx, ldist, rdist, _)) in distances.iter().enumerate() {
        if i < split_point {
            assignments[*vidx] = (0, *ldist);
        } else {
            assignments[*vidx] = (1, *rdist);
        }
    }

    (centroids, assignments)
}

fn bp_update_centroid<V: VectorStore<Elem = f32>>(dataset: &V, centroid: &mut [f32]) {
    centroid.fill(0.0);
    for v in dataset.iter() {
        for (d, o) in v.iter().zip(centroid.iter_mut()) {
            *o += *d;
        }
    }
    for d in centroid.iter_mut() {
        *d /= dataset.len() as f32;
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
                // XXX this is unnecessary, we could use assignments.par_iter_mut().
                let distances = (0..training_data.len())
                    .into_par_iter()
                    .map(|i| crate::distance::l2(&training_data[i], centroid_vector))
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
                .map(|c| crate::distance::l2(v, c))
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
        .map(|i| crate::distance::l2(&old[i], &new[i]))
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
