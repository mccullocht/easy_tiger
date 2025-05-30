//! An implementation of k-means algorithms for clustering vectors.

use std::iter::Cycle;
use std::ops::RangeInclusive;

use rand::seq::index;
use rand::{distributions::WeightedIndex, prelude::*};
use rayon::prelude::*;
use tracing::debug;

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

/// Iteratively compute k-means over `dataset` in a way that produces balanced cluster sizes.
///
/// To do this we partition the dataset into up to `max_k` partitions, then divide each of those
/// sub partitions, etc, terminating when each partition has a vector count in centroid_size_bounds.
/// For each sub-partition we may reduce the `k` value to some number less than `max_k`, and use a
/// binary partitioning scheme when `k == 2`.
///
/// In general using a larger range for `centroid_size_bound` will cause this clustering to converge
/// faster.
///
/// Returns a centroid set and the assignment of each vector in `dataset` to the centroids.
///
/// *Panics* if `centroid_size_bounds.end() / 2 <= centroid_size_bounds.start()`. When we divide a
/// centroid that is just larger than the size bounds it must be possible to generate a valid split.
/// `centroid_size_bounds` must be set such that `end() / 2 > start()` `
pub fn iterative_balanced_kmeans<V: VectorStore<Elem = f32> + Send + Sync>(
    dataset: &V,
    centroid_size_bounds: RangeInclusive<usize>,
    max_k: usize,
    batch_size: usize,
    params: &Params,
    rng: &mut impl Rng,
) -> (VecVectorStore<f32>, Vec<(usize, f64)>) {
    assert!(
        !centroid_size_bounds.is_empty()
            && *centroid_size_bounds.end() / 2 > *centroid_size_bounds.start(),
        "maximum centroid size must be at least 2x minimum size to facilitate partitioning."
    );

    // TODO: if the input dataset is very smol what should we do?
    let mut centroids = match batch_kmeans(dataset, max_k, batch_size, params, rng) {
        Ok(c) => c,
        Err(c) => {
            debug!("iterative_balanced_kmeans initial partitioning failed to converge!");
            c
        }
    };
    let mut assignments = compute_assignments(dataset, &centroids);

    loop {
        let mut centroid_to_vectors = assignments.iter().enumerate().fold(
            vec![vec![]; centroids.len()],
            |mut c2v, (i, (c, _))| {
                c2v[*c].push(i);
                c2v
            },
        );

        // Count the number of vectors in each centroid, then decide how many pieces to divide each
        // centroid into targeting the maximum centroid size.
        let mut centroid_counts = centroid_to_vectors
            .iter()
            .map(|v| v.len())
            .enumerate()
            .collect::<Vec<_>>();
        centroid_counts.sort_unstable_by_key(|(i, c)| (*c, *i));
        for (_, count) in centroid_counts.iter_mut().rev() {
            *count = max_k.min((*count / *centroid_size_bounds.end()) + 1);
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
            let subset_vectors =
                SubsetViewVectorStore::new(dataset, std::mem::take(&mut centroid_to_vectors[*c]));
            let (mut subset_centroids, subset_assignments) = match *nk {
                2 => match bp_vectors(&subset_vectors, params.iters, *centroid_size_bounds.start())
                {
                    Ok(r) => r,
                    Err(r) => {
                        debug!("iterative_balanced_kmeans bp_vector partition failed to converge!");
                        r
                    }
                },
                _ => {
                    let subset_centroids = match batch_kmeans(
                        &subset_vectors,
                        *nk,
                        batch_size,
                        params,
                        rng,
                    ) {
                        Ok(c) => c,
                        Err(e) => {
                            debug!(
                                    "iterative_balanced_kmeans batch_kmeans partition failed to converge!"
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

            // Prune out any centroids that are too small. This may yield empty centroids that we
            // skip when updating new centroids and assignments.
            (subset_centroids, subset_centroids_to_vectors) = prune_iterative_centroids(
                *nk,
                *centroid_size_bounds.start(),
                params.iters,
                &subset_vectors,
                subset_centroids,
                subset_centroids_to_vectors,
            );

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

fn prune_iterative_centroids<V: VectorStore<Elem = f32> + Send + Sync>(
    k: usize,
    min_centroid_size: usize,
    iters: usize,
    vectors: &SubsetViewVectorStore<'_, V>,
    centroids: VecVectorStore<f32>,
    mut centroids_to_vectors: Vec<Vec<(usize, f64)>>,
) -> (VecVectorStore<f32>, Vec<Vec<(usize, f64)>>) {
    // Count centroids that meet minimum assignment size requirements. Initial clustering should
    // produce _at least_ one centroid that meets this requirement by splitting conservatively based
    // on maximum centroid size.
    let centroids_to_keep = centroids_to_vectors
        .iter()
        .map(|v| v.len())
        .filter(|v| *v >= min_centroid_size)
        .count();
    assert_ne!(centroids_to_keep, 0);
    if k == centroids_to_keep {
        // all vectors meet minimum size policy.
        return (centroids, centroids_to_vectors);
    }

    if centroids_to_keep == 1 {
        // Only one centroid qualified, so perform a binary partitioning to keep making progress
        // rather than teratively trying kmeans and hoping it converges.
        //
        // This typically only happens if k is 2 or 3.
        let (centroids, assignments) = match bp_vectors(vectors, iters, min_centroid_size) {
            Ok(r) => r,
            Err(r) => {
                debug!("iterative_balanced_kmeans prune_iterative_centroids bp_vector partition failed to converge!");
                r
            }
        };
        centroids_to_vectors.resize(2, vec![]);
        centroids_to_vectors[0].clear();
        centroids_to_vectors[1].clear();
        for (i, (c, d)) in assignments.iter().enumerate() {
            centroids_to_vectors[*c].push((vectors.original_index(i), *d));
        }
        return (centroids, centroids_to_vectors);
    }

    // Iteratively remove the smallest partition and reassign vectors until all remaining centroids
    // are larger than min_centroid_size.
    let mut centroids_to_prune = centroids_to_vectors
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_empty() && v.len() < min_centroid_size)
        .map(|(i, v)| (v.len(), i))
        .collect::<Vec<_>>();
    centroids_to_prune.sort_unstable();
    for (_, pruned_centroid) in centroids_to_prune {
        let reassign_vectors = SubsetViewVectorStore::new(
            vectors.parent(),
            std::mem::take(&mut centroids_to_vectors[pruned_centroid])
                .into_iter()
                .map(|(i, _)| i)
                .collect(),
        );
        let kept_centroids = SubsetViewVectorStore::new(
            &centroids,
            centroids_to_vectors
                .iter()
                .enumerate()
                .filter(|(_, v)| !v.is_empty())
                .map(|(i, _)| i)
                .collect(),
        );
        for (i, (new_centroid, d)) in compute_assignments(&reassign_vectors, &kept_centroids)
            .into_iter()
            .enumerate()
        {
            centroids_to_vectors[kept_centroids.original_index(new_centroid)]
                .push((reassign_vectors.original_index(i), d));
        }

        if centroids_to_vectors
            .iter()
            .all(|v| v.is_empty() || v.len() >= min_centroid_size)
        {
            break;
        }
    }

    (centroids, centroids_to_vectors)
}

type CentroidsAndAssignments = (VecVectorStore<f32>, Vec<(usize, f64)>);

/// Build a binary partition of the vectors between two centroids.
///
/// Split the dataset in half and produce two centroids, then measure the distance for each vector
/// to both centroids and the weight of vectors to each centroid. Terminate if the centroids would
/// produce two clusters of at least min_cluster_size, otherwise split the dataset into a new
/// grouping and try again.
///
/// Returns a vector store containing a "left" and a "right" centroid along with cluster assignment.
/// Note that these cluster assignments may not match the results of compute_assignments().
fn bp_vectors<V: VectorStore<Elem = f32> + Send + Sync>(
    dataset: &V,
    max_iters: usize,
    min_cluster_size: usize,
) -> Result<CentroidsAndAssignments, CentroidsAndAssignments> {
    let acceptable_split = min_cluster_size..=(dataset.len() - min_cluster_size);
    assert!(!acceptable_split.is_empty());

    let split = dataset.len() / 2;
    let mut left_vectors = SubsetViewVectorStore::new(dataset, (0..split).collect());
    let mut right_vectors = SubsetViewVectorStore::new(dataset, (split..dataset.len()).collect());

    let mut centroids = VecVectorStore::with_capacity(dataset.elem_stride(), 2);
    centroids.push(&vec![0.0; dataset.elem_stride()]);
    centroids.push(&vec![0.0; dataset.elem_stride()]);
    let mut distances = vec![(0usize, 0.0, 0.0, 0.0); dataset.len()];
    let mut converged = false;
    for _ in 0..max_iters {
        bp_update_centroid(&left_vectors, &mut centroids[0]);
        bp_update_centroid(&right_vectors, &mut centroids[1]);
        distances.par_iter_mut().enumerate().for_each(|(i, d)| {
            let ldist = crate::distance::l2sq(&dataset[i], &centroids[0]);
            let rdist = crate::distance::l2sq(&dataset[i], &centroids[1]);
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
            converged = true;
            break;
        }

        left_vectors =
            SubsetViewVectorStore::new(dataset, distances[..split].iter().map(|d| d.0).collect());
        right_vectors =
            SubsetViewVectorStore::new(dataset, distances[split..].iter().map(|d| d.0).collect());
    }

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
    for (i, d, _, _) in distances[..split_point].iter() {
        assignments[*i] = (0, *d);
    }
    for (i, _, d, _) in distances[split_point..].iter() {
        assignments[*i] = (1, *d);
    }

    if converged {
        Ok((centroids, assignments))
    } else {
        Err((centroids, assignments))
    }
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
                assignments.par_iter_mut().enumerate().for_each(|(i, a)| {
                    let d = crate::distance::l2(&training_data[i], centroid_vector);
                    if d < a.1 {
                        *a = (centroid, d);
                    }
                });
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
