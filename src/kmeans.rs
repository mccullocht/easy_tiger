//! An implementation of k-means algorithms for clustering vectors.

use std::collections::VecDeque;
use std::ops::RangeInclusive;

use rand::seq::index;
use rand::{distr::weighted::WeightedIndex, prelude::*};
use rayon::prelude::*;
use tracing::warn;
use vectors::{EuclideanDistance, F32VectorDistance};

use crate::input::{SubsetViewVectorStore, VecVectorStore, VectorStore};
use crate::kmeans::bp::IterState;

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

pub fn bp_kmeans_pp(
    dataset: &(impl VectorStore<Elem = f32> + Send + Sync),
    max_iters: usize,
    min_cluster_size: usize,
    rng: &mut impl Rng,
) -> Result<VecVectorStore<f32>, VecVectorStore<f32>> {
    let mut state = IterState::new(dataset);
    let init_centroids = bp::kmeanspp_init(
        min_cluster_size..=(dataset.len() - min_cluster_size),
        &mut state,
        rng,
    );
    bp::bp_loop2(dataset, init_centroids, max_iters, min_cluster_size)
}

mod bp {
    use std::{cmp::Ordering, ops::RangeInclusive};

    use rand::{distr::weighted::WeightedIndex, prelude::*};
    use rayon::prelude::*;
    use vectors::{EuclideanDistance, F32VectorDistance};

    use crate::input::{VecVectorStore, VectorStore};

    /// Container for the dataset and reusable buffers for certain operations.
    pub struct IterState<'v, V> {
        dataset: &'v V,
        distances: Vec<(usize, f64, f64, f64)>,
        assignments: Vec<usize>,
    }

    impl<'v, V: VectorStore<Elem = f32> + Send + Sync> IterState<'v, V> {
        pub fn new(dataset: &'v V) -> Self {
            Self {
                dataset,
                distances: vec![(0usize, 0.0, 0.0, 0.0); dataset.len()],
                assignments: vec![0usize; dataset.len()],
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct Candidate {
        centroids: VecVectorStore<f32>,
        split: usize,
        distance_sum: f64,
    }

    impl Candidate {
        fn new(
            centroids: VecVectorStore<f32>,
            state: &mut IterState<'_, impl VectorStore<Elem = f32> + Send + Sync>,
        ) -> Self {
            let mut candidate = Self {
                centroids,
                split: 0,
                distance_sum: 0.0,
            };
            candidate.update_split(state);
            candidate
        }

        fn from_sampled_vectors(
            first: usize,
            second: usize,
            state: &mut IterState<'_, impl VectorStore<Elem = f32> + Send + Sync>,
        ) -> Self {
            // Take two sampled points from the dataset to use as initial centroids.
            let mut centroids = VecVectorStore::with_capacity(state.dataset.elem_stride(), 2);
            centroids.push(&state.dataset[first]);
            centroids.push(&state.dataset[second]);
            // Compute the initial split point, then update centroids and recompute the split.
            // This is essentially a single iteration of bp_loop compute actual mean vectors.
            let mut candidate = Self::new(centroids, state);
            candidate.update_centroids(candidate.split, state);
            candidate
        }

        fn update_centroids(
            &mut self,
            target_split: usize,
            state: &mut IterState<'_, impl VectorStore<Elem = f32> + Send + Sync>,
        ) {
            for (i, &(idx, _, _, _)) in state.distances.iter().enumerate() {
                state.assignments[idx] = if i < target_split { 0 } else { 1 };
            }
            self.centroids = {
                let assignments: &[usize] = &state.assignments;
                let mut centroids: VecVectorStore<f32> =
                    VecVectorStore::with_capacity(state.dataset.elem_stride(), 2);
                centroids.push(&vec![0.0; state.dataset.elem_stride()]);
                centroids.push(&vec![0.0; state.dataset.elem_stride()]);
                let mut counts = [0usize; 2];
                for (vector, i) in state.dataset.iter().zip(assignments.iter().copied()) {
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
            };
            self.update_split(state);
        }

        fn update_split(
            &mut self,
            state: &mut IterState<'_, impl VectorStore<Elem = f32> + Send + Sync>,
        ) {
            let dist_fn = EuclideanDistance::get();
            state
                .distances
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, d)| {
                    let ldist = dist_fn.distance_f32(&self.centroids[0], &state.dataset[i]);
                    let rdist = dist_fn.distance_f32(&self.centroids[1], &state.dataset[i]);
                    *d = (i, ldist, rdist, ldist - rdist)
                });
            state
                .distances
                .sort_unstable_by(|a, b| a.3.total_cmp(&b.3).then_with(|| a.0.cmp(&b.0)));
            self.split = state.distances.iter().position(|d| d.3 >= 0.0).unwrap_or(0);
            self.distance_sum = state.distances[..self.split]
                .iter()
                .map(|d| d.1)
                .sum::<f64>()
                + state.distances[self.split..]
                    .iter()
                    .map(|d| d.2)
                    .sum::<f64>();
        }
    }

    // XXX rename
    pub fn bp_loop2(
        dataset: &(impl VectorStore<Elem = f32> + Send + Sync),
        init_candidate: Candidate,
        max_iters: usize,
        min_cluster_size: usize,
    ) -> Result<VecVectorStore<f32>, VecVectorStore<f32>> {
        let acceptable_split = min_cluster_size..=(dataset.len() - min_cluster_size);
        assert!(!acceptable_split.is_empty());
        let half = dataset.len() / 2;

        let mut state = IterState::new(dataset);
        let mut current_candidate = init_candidate;
        let mut next_candidate = current_candidate.clone();
        let mut converged = false;
        let mut adjustment_mult = 1;
        for _ in 0..max_iters {
            let split_adjustment =
                (half.abs_diff(current_candidate.split) / 10 * adjustment_mult).max(1);
            // XXX there is risk here that adjustment_mult will pick a split that either exceeds the
            // lower of upper bound values of target_split (potentially underflowing).
            let target_split = if current_candidate.split < half {
                current_candidate.split + split_adjustment
            } else {
                current_candidate.split - split_adjustment
            };
            next_candidate.update_centroids(target_split, &mut state);

            let current_balance = half.abs_diff(current_candidate.split);
            let next_balance = half.abs_diff(next_candidate.split);
            if next_balance < current_balance {
                std::mem::swap(&mut current_candidate, &mut next_candidate);
                adjustment_mult = 1;
            } else if acceptable_split.contains(&current_candidate.split) {
                converged = true;
                break;
            } else {
                // Try harder to fix the split by taking elements from the "far" side of the split.
                // This improves the odds of convergence but also indicates that we probably chose
                // poor initial centroids.
                adjustment_mult *= 2;
            }
        }

        if converged {
            Ok(current_candidate.centroids)
        } else {
            Err(current_candidate.centroids)
        }
    }

    /// K-means++ style centroid initialization: choose a random point as the first centroid, then
    /// choose a second point weighted by distance to the first centroid.
    pub fn kmeanspp_init(
        acceptable_split: RangeInclusive<usize>,
        state: &mut IterState<'_, impl VectorStore<Elem = f32> + Send + Sync>,
        rng: &mut impl Rng,
    ) -> Candidate {
        (0..10)
            .map(|_| {
                // Select a centroid at random, then select a second centroid at random but weighted by
                // distance to the first centroid to prefer something farther away.
                let first = rng.random_range(0..state.dataset.len());
                let dist_fn = EuclideanDistance::get();
                let distances = (0..state.dataset.len())
                    .into_par_iter()
                    .map(|i| dist_fn.distance_f32(&state.dataset[i], &state.dataset[first]))
                    .collect::<Vec<_>>();
                let second = WeightedIndex::new(distances.iter().copied())
                    .unwrap()
                    .sample(rng);
                Candidate::from_sampled_vectors(first, second, state)
            })
            .min_by(|a, b| {
                let aok = acceptable_split.contains(&a.split);
                let bok = acceptable_split.contains(&b.split);
                if aok == bok {
                    a.distance_sum.total_cmp(&b.distance_sum)
                } else if aok {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .unwrap()
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
    let acceptable_split =
        *params.cluster_size.start()..=(training_data.len() - *params.cluster_size.start());
    let centroids = hkmeans_unwrap(
        match training_data.len().div_ceil(*params.cluster_size.end()) {
            2 => {
                let mut state = IterState::new(training_data);
                bp::bp_loop2(
                    training_data,
                    bp::kmeanspp_init(acceptable_split, &mut state, rng),
                    params.params.iters,
                    *params.cluster_size.start(),
                )
            }
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
