//! An implementation of k-means algorithms for clustering vectors.

use std::collections::VecDeque;
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
            let l2_dist = EuclideanDistance::get();
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
fn compute_assignments<
    V: VectorStore<Elem = f32> + Send + Sync,
    C: VectorStore<Elem = f32> + Send + Sync,
>(
    dataset: &V,
    centroids: &C,
) -> Vec<(usize, f64)> {
    let l2_dist = EuclideanDistance::get();
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
    let l2_dist = EuclideanDistance::get();
    old.iter()
        .zip(new.iter())
        .map(|(a, b)| l2_dist.distance_f32(a, b))
        .max_by(|a, b| a.total_cmp(b))
        .expect("non-zero k")
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
) -> Result<VecVectorStore<f32>, VecVectorStore<f32>> {
    let init_centroids = bp::simple_init(dataset);
    bp::bp_loop(dataset, init_centroids, max_iters, min_cluster_size)
}

mod bp {
    use rand::{distr::weighted::WeightedIndex, prelude::*};
    use rayon::prelude::*;
    use vectors::{EuclideanDistance, F32VectorDistance};

    use crate::{
        input::{VecVectorStore, VectorStore},
        kmeans::compute_assignments,
    };

    /// Run binary partitioning for up to `max_iters` iterations starting from `init_centroids`.
    pub fn bp_loop(
        dataset: &(impl VectorStore<Elem = f32> + Send + Sync),
        init_centroids: VecVectorStore<f32>,
        max_iters: usize,
        min_cluster_size: usize,
    ) -> Result<VecVectorStore<f32>, VecVectorStore<f32>> {
        let acceptable_split = min_cluster_size..=(dataset.len() - min_cluster_size);
        assert!(!acceptable_split.is_empty());

        let half = dataset.len() / 2;
        let dist_fn = EuclideanDistance::get();
        let mut centroids = init_centroids;
        let mut assignments = vec![0usize; dataset.len()];
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
            if acceptable_split.contains(&split_point) && (inertia == 0 || prev_inertia <= inertia)
            {
                converged = true;
                break;
            }

            // TODO: if inertia stops changing consider terminating early without converging.

            prev_inertia = inertia;
            // Split evenly rather than using split_point. Using split_point may reinforce bias and
            // cause us to go further away from the the target split rather than converging.
            for (i, d) in distances.iter().enumerate() {
                assignments[d.0] = if i < half { 0 } else { 1 };
            }
            centroids = compute_centroids(dataset, &assignments);
        }

        if converged {
            Ok(centroids)
        } else {
            Err(centroids)
        }
    }

    /// Simple centroid initialization: split the dataset in half and compute two means.
    pub fn simple_init(
        dataset: &(impl VectorStore<Elem = f32> + Send + Sync),
    ) -> VecVectorStore<f32> {
        let half = dataset.len() / 2;
        let mut assignments = vec![0usize; dataset.len()];
        assignments[half..].fill(1);
        compute_centroids(dataset, &assignments)
    }

    /// K-means++ style centroid initialization: choose a random point as the first centroid, then
    /// choose a second point weighted by distance to the first centroid.
    pub fn kmeanspp_init(
        dataset: &(impl VectorStore<Elem = f32> + Send + Sync),
        rng: &mut impl Rng,
    ) -> VecVectorStore<f32> {
        let mut centroids = VecVectorStore::with_capacity(dataset.elem_stride(), 2);
        centroids.push(&dataset[rng.random_range(0..dataset.len())]);
        let assignments = compute_assignments(dataset, &centroids);
        let index = WeightedIndex::new(assignments.iter().map(|a| a.1))
            .unwrap()
            .sample(rng);
        centroids.push(&dataset[index]);
        centroids
    }

    // TODO: consider choosing the point farthest from the mean vector as the first centroid and
    // then using k-means++ to choose the second centroid.

    /// Compute two new centroids from the dataset given assignments.
    pub fn compute_centroids(
        dataset: &(impl VectorStore<Elem = f32> + Send + Sync),
        assignments: &[usize],
    ) -> VecVectorStore<f32> {
        let mut centroids: VecVectorStore<f32> =
            VecVectorStore::with_capacity(dataset.elem_stride(), 2);
        centroids.push(&vec![0.0; dataset.elem_stride()]);
        centroids.push(&vec![0.0; dataset.elem_stride()]);
        let mut counts = [0usize; 2];
        for (vector, i) in dataset.iter().zip(assignments.iter().copied()) {
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
}

/// Parameters for hierarchical k-means clustering.
pub struct HierarchicalKMeansParams {
    /// Minimum and maximum number of vectors per cluster.
    ///
    /// This is _best effort_; hierarchical k-means does not guarantee these sizes will be met in
    /// the output but will use these bounds locally within the clustering hierarchy.
    ///
    /// Smaller clusters will be merged out; larger clusters will be split.
    pub cluster_size: RangeInclusive<usize>,
    /// Number of vectors to buffer at once.
    ///
    /// For any input larger than buffer_len we may sample vectors into the buffer to perform
    /// clustering, extrapolating the results to the full dataset.
    ///
    /// Larger values increase memory usage but may reduce clustering time and produce more balanced
    /// clusters.
    pub buffer_len: usize,
    /// Number of threads to use for clustering.
    ///
    /// After initial clustering the resulting centroids will be partitioned among this many threads
    /// to parallelize high level clustering work. This results in memory usage proportional to
    /// `buffer_len`. This work is still performed on rayon threads as are other clustering related
    /// workloads (namely vector -> centroid assignment).
    ///
    /// Larger values may reduce clustering time by better overlapping IO and compute.
    pub num_threads: usize,
    /// K-means parameters.
    pub params: Params,
}

/// Cluster the input data set using hierarchical k-means.
///
/// This aims to cluster the input into clusters no larger than `max_cluster_len` vectors. Rather
/// than running k-means on the entire dataset at once regardless of size, we cap to `buffer_len`
/// vectors at a time and produce no more than `buffer_len / max_cluster_len` centroids per run.
/// If the dataset is larger than the buffer size we randomly sample to fill the buffer and
/// extrapolate the results to the full dataset. The resulting centroids may then be iteratively
/// partitioned again if they are too larger.
///
/// A nice property of this scheme is that memory usage is roughly bound by `buffer_len` vectors,
/// and similarly IO is capped since we won't read any vectors during k-means iterations (only
/// buffer to fill the buffer and after to perform assignment).
///
/// The progress callback will be called once for each final centroid that has been decided.
pub fn hierarchical_kmeans(
    dataset: &(impl VectorStore<Elem = f32> + Send + Sync),
    params: &HierarchicalKMeansParams,
    rng: &mut (impl Rng + Clone + Send + Sync),
    progress: impl Fn(u64) + Send + Sync,
) -> VecVectorStore<f32> {
    // Partition the dataset into initial centroids.
    let (_, initial_partitions) = if dataset.len() <= params.buffer_len {
        hkmeans_step(dataset, dataset, params, rng)
    } else {
        let mut buffer = VecVectorStore::with_capacity(dataset.elem_stride(), params.buffer_len);
        for i in index::sample(rng, dataset.len(), params.buffer_len) {
            buffer.push(&dataset[i]);
        }
        hkmeans_step(&buffer, dataset, params, rng)
    };

    // Divide the generated centroids into groups and dispatch to separate threads.
    let chunks = initial_partitions.len().div_ceil(params.num_threads);
    let mut centroid_sets = initial_partitions
        .into_par_iter()
        .chunks(chunks)
        .map(|centroids| {
            let mut rng = rng.clone();
            hkmeans_group(dataset, params, centroids, &mut rng, &progress)
        })
        .collect::<Vec<_>>();

    let mut centroids = VecVectorStore::new(dataset.elem_stride());
    for set in centroid_sets.drain(..) {
        for c in set.iter() {
            centroids.push(c);
        }
    }

    // TODO: consider performing global rebalancing to bring assignments to cluster_size policy.

    centroids
}

/// Run a single step of hierarchical k-means.
///
/// Accepts a set of training data that is used to cluster and the whole data set for assignment.
/// These may be different data sets if the input is large enough.
///
/// Returns a set of centroids and the subsets of `assign_data` that are assigned to each centroid.
fn hkmeans_step(
    training_data: &(impl VectorStore<Elem = f32> + Send + Sync),
    assign_data: &(impl VectorStore<Elem = f32> + Send + Sync),
    params: &HierarchicalKMeansParams,
    rng: &mut impl Rng,
) -> (VecVectorStore<f32>, Vec<Vec<usize>>) {
    let centroids = hkmeans_unwrap(
        match training_data.len().div_ceil(*params.cluster_size.end()) {
            2 => bp::bp_loop(
                training_data,
                bp::kmeanspp_init(training_data, rng),
                params.params.iters,
                *params.cluster_size.start(),
            ),
            _ => kmeans(
                training_data,
                training_data.len().div_ceil(*params.cluster_size.end()),
                &params.params,
                rng,
            ),
        },
        assign_data.len(),
        training_data.len() < assign_data.len(),
    );
    prune_centroids(assign_data, centroids, *params.cluster_size.start())
}

fn hkmeans_group(
    dataset: &(impl VectorStore<Elem = f32> + Send + Sync),
    params: &HierarchicalKMeansParams,
    subsets: Vec<Vec<usize>>,
    rng: &mut impl Rng,
    progress: impl Fn(u64) + Send + Sync,
) -> VecVectorStore<f32> {
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    enum VectorSource {
        Dataset,
        Buffer,
    }

    let mut buffer = VecVectorStore::with_capacity(dataset.elem_stride(), params.buffer_len);
    let mut centroids = VecVectorStore::new(dataset.elem_stride());
    let mut queue = subsets
        .into_iter()
        .map(|subset| (VectorSource::Dataset, subset))
        .collect::<VecDeque<_>>();

    while let Some((mut source, mut subset)) = queue.pop_front() {
        if source == VectorSource::Dataset && subset.len() <= params.buffer_len {
            buffer.clear();
            // Buffer this subset and do any clustering below this point on the buffer only.
            for i in &subset {
                buffer.push(&dataset[*i]);
            }
            subset = (0..subset.len()).collect();
            source = VectorSource::Buffer;
        }

        let (iter_centroids, centroid_vectors) = match source {
            VectorSource::Dataset => {
                // Cluster on a buffered sample of vectors; assign on the full subset.
                buffer.clear();
                for i in index::sample(rng, subset.len(), params.buffer_len) {
                    buffer.push(&dataset[subset[i]]);
                }

                let assign_dataset = SubsetViewVectorStore::new(dataset, subset);
                hkmeans_step(&buffer, &assign_dataset, params, rng)
            }
            VectorSource::Buffer => {
                // Limit k based on the size of the input subset, then cluster and assign on all of
                // the vectors in the subset.
                let subset = SubsetViewVectorStore::new(&buffer, subset);
                hkmeans_step(&subset, &subset, params, rng)
            }
        };

        for (centroid, subset) in iter_centroids.iter().zip(centroid_vectors.into_iter()) {
            if subset.len() <= *params.cluster_size.end() {
                centroids.push(centroid);
                progress(subset.len() as u64);
            } else {
                queue.push_front((source, subset));
            }
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
        // This could benefit from parallelization/offloading to rayon, but there's no obvious
        // way to do so without using/allocating more memory.
        let mut centroid_counts = vec![0usize; centroids.len()];
        for (i, (c, _)) in assignments.iter().enumerate() {
            if centroid_counts[*c] == 0 {
                next_centroids[*c].fill(0.0);
            }
            centroid_counts[*c] += 1;
            for (vd, cd) in training_data[i].iter().zip(next_centroids[*c].iter_mut()) {
                *cd += *vd;
            }
        }

        for (count, centroid) in centroid_counts.iter().zip(next_centroids.iter_mut()) {
            if *count == 0 {
                // Pick a random vector to replace the centroid that isn't receiving assignments.
                centroid.copy_from_slice(&training_data[rng.random_range(0..training_data.len())]);
            } else {
                let recip = 1.0 / *count as f32;
                for cd in centroid.iter_mut() {
                    *cd *= recip;
                }
            }
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

/// Prune under-sized centroids.
fn prune_centroids(
    dataset: &(impl VectorStore<Elem = f32> + Send + Sync),
    mut centroids: VecVectorStore<f32>,
    min_cluster_len: usize,
) -> (VecVectorStore<f32>, Vec<Vec<usize>>) {
    let assignments = compute_assignments(dataset, &centroids);
    let mut centroid_vectors = assignments.iter().enumerate().fold(
        vec![vec![]; centroids.len()],
        |mut cv, (i, &(c, _))| {
            cv[c].push(i);
            cv
        },
    );

    while let Some((centroid, vectors)) = centroid_vectors
        .iter_mut()
        .enumerate()
        .min_by_key(|(_, c)| c.len())
    {
        // If all the centroids are large enough or there are only two centroids left, terminate.
        if vectors.len() >= min_cluster_len || centroids.len() == 2 {
            break;
        }

        // Remove the identified centroid from all related data structures.
        let merge_vectors = centroid_vectors.swap_remove(centroid);
        centroids.swap_remove(centroid);

        // Reassign all of the vectors to the other centroid.
        let merged_subset = SubsetViewVectorStore::new(dataset, merge_vectors);
        let assignments = compute_assignments(&merged_subset, &centroids);
        for (i, (c, _)) in assignments.iter().enumerate() {
            centroid_vectors[*c].push(merged_subset.original_index(i));
        }
    }

    (centroids, centroid_vectors)
}
