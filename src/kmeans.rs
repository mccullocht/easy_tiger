//! An implementation of k-means algorithms for clustering vectors.

use std::collections::VecDeque;
use std::iter::Cycle;
use std::ops::RangeInclusive;

use rand::seq::index;
use rand::{distr::weighted::WeightedIndex, prelude::*};
use rayon::prelude::*;
use tracing::warn;
use vectors::{EuclideanDistance, F32VectorDistance};

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
    progress: impl Fn(u64),
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
            warn!("iterative_balanced_kmeans initial partitioning failed to converge!");
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
            progress(1);
        }

        // For centroids that are being split, re-partition them and discard any smol clusters,
        // then update assignments for the next pass through the loop.
        for (c, nk) in centroid_counts.iter().skip(unsplit_len) {
            let subset_vectors =
                SubsetViewVectorStore::new(dataset, std::mem::take(&mut centroid_to_vectors[*c]));
            let (mut subset_centroids, subset_assignments) = match *nk {
                2 => match binary_partition(
                    &subset_vectors,
                    params.iters,
                    *centroid_size_bounds.start(),
                ) {
                    Ok(r) => r,
                    Err(r) => {
                        warn!("iterative_balanced_kmeans binary_partition failed to converge!");
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
                            warn!(
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
            progress(1);
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
                progress(1);
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
        let (centroids, assignments) = match binary_partition(vectors, iters, min_centroid_size) {
            Ok(r) => r,
            Err(r) => {
                warn!("iterative_balanced_kmeans prune_iterative_centroids binary_partition failed to converge!");
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
            centroids.push(&dataset[rng.random_range(0..dataset.len())]);
            let mut assignments = compute_assignments(training_data, &centroids);
            let l2_dist = EuclideanDistance::default();
            while centroids.len() < k {
                let index = WeightedIndex::new(assignments.iter().map(|a| a.1))
                    .unwrap()
                    .sample(rng);

                let centroid = centroids.len();
                centroids.push(&training_data[index]);
                let centroid_vector = &centroids[centroid];
                assignments.par_iter_mut().enumerate().for_each(|(i, a)| {
                    let d = l2_dist.distance_f32(&training_data[i], centroid_vector);
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
    let l2_dist = EuclideanDistance::default();
    (0..dataset.len())
        .into_par_iter()
        .map(|i| {
            let v = &dataset[i];
            centroids
                .iter()
                .map(|c| l2_dist.distance_f32(v, c))
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
    let l2_dist = vectors::EuclideanDistance::default();
    (0..old.len())
        .into_par_iter()
        .map(|i| l2_dist.distance_f32(&old[i], &new[i]))
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

/// Split the dataset into two partitions with two representative centroids.
///
/// This functions aims to split the dataset into two partitions of at least `min_cluster_size`
/// vectors, with a goal of splitting the dataset as evenly as possible. On each iteration the
/// two centroids are updated, vectors are assigned to the closest centroid, and an inertia value
/// is computed. If the centroids are of `min_cluster_size` and inertia is positive we continue
/// iterating otherwise we terminate and use the last computed centroids and assignments.
///
/// Returns a vector store containing a "left" and a "right" centroid along with cluster assignment.
/// Note that these cluster assignments may not match the results of compute_assignments().
pub fn binary_partition(
    dataset: &(impl VectorStore<Elem = f32> + Send + Sync),
    max_iters: usize,
    min_cluster_size: usize,
) -> Result<CentroidsAndAssignments, CentroidsAndAssignments> {
    let acceptable_split = min_cluster_size..=(dataset.len() - min_cluster_size);
    assert!(!acceptable_split.is_empty());

    let half = dataset.len() / 2;
    let mut assignments = vec![(1usize, f64::MAX); dataset.len()];
    assignments[half..].fill((0, f64::MAX));

    let dist_fn = vectors::EuclideanDistance::default();
    let mut centroids = compute_centroids(dataset, &assignments);
    let mut distances = vec![(0usize, 0.0, 0.0, 0.0); dataset.len()];
    let mut prev_inertia = dataset.len();
    let mut converged = false;
    for _ in 0..max_iters {
        distances.par_iter_mut().enumerate().for_each(|(i, d)| {
            let ldist = dist_fn.distance_f32(&centroids[0], &dataset[i]);
            let rdist = dist_fn.distance_f32(&centroids[1], &dataset[i]);
            *d = (i, ldist, rdist, ldist - rdist)
        });
        distances.sort_unstable_by(|a, b| a.3.total_cmp(&b.3).then_with(|| a.0.cmp(&b.0)));
        let split_point = distances.iter().position(|d| d.3 >= 0.0).unwrap_or(0);
        let inertia = half.abs_diff(split_point);
        // We may terminate if the partition sizes are acceptable and we aren't improving the split.
        if acceptable_split.contains(&split_point) && (inertia == 0 || prev_inertia <= inertia) {
            converged = true;
            for (i, d) in distances.iter().enumerate() {
                assignments[d.0] = if i < split_point { (0, d.1) } else { (1, d.2) };
            }
            break;
        }

        prev_inertia = inertia;
        // Split evenly rather than using split_point. Using split_point may reinforce bias and
        // cause us to go further away from the the target split rather than converging.
        for (i, d) in distances.iter().enumerate() {
            assignments[d.0] = if i < half { (0, d.1) } else { (1, d.2) };
        }
        centroids = compute_centroids(dataset, &assignments);
    }

    if converged {
        Ok((centroids, assignments))
    } else {
        Err((centroids, assignments))
    }
}

fn compute_centroids(
    dataset: &(impl VectorStore<Elem = f32> + Send + Sync),
    assignments: &[(usize, f64)],
) -> VecVectorStore<f32> {
    let mut centroids: VecVectorStore<f32> =
        VecVectorStore::with_capacity(dataset.elem_stride(), 2);
    centroids.push(&vec![0.0; dataset.elem_stride()]);
    centroids.push(&vec![0.0; dataset.elem_stride()]);
    let mut counts = [0usize; 2];
    for (vector, i) in dataset.iter().zip(assignments.iter().map(|x| x.0)) {
        let centroid = &mut centroids[i];
        counts[i] += 1;
        for (d, o) in vector.iter().zip(centroid.iter_mut()) {
            *o += *d;
        }
    }
    for i in 0..2 {
        if counts[i] == 0 {
            continue;
        }
        let count = counts[i] as f32;
        for d in centroids[i].iter_mut() {
            *d /= count;
        }
    }
    centroids
}

/// Parameters for hierarchical k-means clustering.
pub struct HierarchicalKMeansParams {
    /// Maximum number of partitions at each level in the hierarchy.
    // XXX should I instead use max_cluster_len and buffer_len?
    pub max_k: usize,
    /// Maximum number of vectors per cluster. Any cluster larger than this will be partitioned.
    pub max_cluster_len: usize,
    /// Number of vectors to buffer at once.
    ///
    /// For any input larger than buffer_len we may sample vectors into the buffer to perform
    /// clustering, extrapolating the results to the full dataset.
    pub buffer_len: usize,
    /// K-means parameters.
    pub params: Params,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum VectorSource {
    Dataset,
    Buffer,
}

/// Cluster the input data set using hierarchical k-means.
///
/// This aims to cluster the input into clusters no larger than `max_cluster_len` vectors, but will
/// not exceed `max_k` when clustering any part of the data set. Instead, any clusters larger than
/// the maximum length will be partitioned again into smaller clusters.
///
/// To bound memory usage an IO no more than `buffer_len` vectors will be held in memory at once.
/// Larger clusters will be sampled into the buffer and extrapolated to the full dataset.
///
/// The progress callback will be called once for each final centroid that has been decided.
pub fn hierarchical_kmeans(
    dataset: &(impl VectorStore<Elem = f32> + Send + Sync),
    params: &HierarchicalKMeansParams,
    rng: &mut impl Rng,
    progress: impl Fn(u64),
) -> VecVectorStore<f32> {
    let mut buffer = VecVectorStore::with_capacity(dataset.elem_stride(), params.buffer_len);
    let mut centroids = VecVectorStore::new(dataset.elem_stride());
    let mut queue: VecDeque<(VectorSource, Vec<usize>)> = VecDeque::new();
    queue.push_back((VectorSource::Dataset, (0..dataset.len()).collect()));

    while let Some((source, subset)) = queue.pop_front() {
        let (iter_centroids, assignments) = match source {
            VectorSource::Dataset => {
                if subset.len() <= params.buffer_len {
                    // Buffer this subset and do any clustering below this point on the buffer only.
                    buffer.clear();
                    for i in &subset {
                        buffer.push(&dataset[*i]);
                    }
                    queue.push_front((VectorSource::Buffer, (0..subset.len()).collect()));
                    continue;
                }

                // Cluster on a sample of no more than buffer_len vectors.
                let cluster_dataset = SubsetViewVectorStore::new(
                    dataset,
                    index::sample(rng, subset.len(), params.buffer_len).into_vec(),
                );
                let c = hkmeans_unwrap(
                    kmeans(&cluster_dataset, params.max_k, &params.params, rng),
                    dataset.len(),
                    true,
                );
                // Assign globally on the full dataset to build the hierarchy.
                let assign_dataset = SubsetViewVectorStore::new(dataset, subset);
                let a = compute_assignments(&assign_dataset, &c);
                (c, a)
            }
            VectorSource::Buffer => {
                // Limit k based on the size of the input subset, then cluster and assign on all of
                // the vectors in the subset.
                let k = subset.len().div_ceil(params.max_cluster_len);
                let subset = SubsetViewVectorStore::new(&buffer, subset);
                let c =
                    hkmeans_unwrap(kmeans(&subset, k, &params.params, rng), subset.len(), false);
                let a = compute_assignments(&subset, &c);
                (c, a)
            }
        };

        let centroid_vectors = assignments.iter().enumerate().fold(
            vec![vec![]; iter_centroids.len()],
            |mut cv, (i, &(c, _))| {
                cv[c].push(i);
                cv
            },
        );
        for (centroid, subset) in iter_centroids.iter().zip(centroid_vectors.into_iter()) {
            if subset.len() <= params.max_cluster_len {
                if !subset.is_empty() {
                    progress(1);
                    centroids.push(centroid);
                }
                continue;
            }

            queue.push_front((source, subset));
        }
    }

    centroids
}

fn hkmeans_unwrap(
    r: Result<VecVectorStore<f32>, VecVectorStore<f32>>,
    len: usize,
    sampled: bool,
) -> VecVectorStore<f32> {
    match r {
        Ok(c) => c,
        Err(c) => {
            warn!("hierarchical_kmeans iteration failed to converge! len={len} sampled={sampled}");
            c
        }
    }
}

/// Cluster the input data set using k-means clustering.
///
/// Always returns the centroids generated by clustering, but returns them as an error if clustering
/// failed to converge.
pub fn kmeans(
    training_data: &(impl VectorStore<Elem = f32> + Send + Sync),
    k: usize,
    params: &Params,
    rng: &mut impl Rng,
) -> Result<VecVectorStore<f32>, VecVectorStore<f32>> {
    let mut centroids =
        initialize_centroids(training_data, training_data, k, params.initialization, rng).0;
    let mut next_centroids = centroids.clone();
    for _ in 0..params.iters {
        let assignments = compute_assignments(training_data, &centroids);

        // Use assignments to compute new updated centroids.
        let mut centroid_counts = vec![0usize; centroids.len()];
        for (i, (c, _)) in assignments.iter().enumerate() {
            if centroid_counts[*c] == 0 {
                next_centroids[*c].fill(0.0);
            }
            centroid_counts[*c] += 1;
            for (vd, cd) in training_data[i].iter().zip(next_centroids[*c].iter_mut()) {
                *cd += (*vd - *cd) / centroid_counts[*c] as f32;
            }
        }

        // If any centroid has no assigned vectors, replace it with a random vector.
        for (i, _) in centroid_counts.iter().enumerate().filter(|&(_, c)| *c == 0) {
            next_centroids[i]
                .copy_from_slice(&training_data[rng.random_range(0..training_data.len())]);
        }

        // If no centroid has moved more than epsilon, terminate.
        if compute_centroid_distance_max(&centroids, &next_centroids) < params.epsilon {
            return Ok(centroids);
        }

        // Update centroids.
        std::mem::swap(&mut centroids, &mut next_centroids);
    }

    Err(centroids)
}
