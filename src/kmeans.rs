//! An implementation of k-means algorithms for clustering vectors.

use std::collections::VecDeque;

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

type CentroidsAndAssignments = (VecVectorStore<f32>, Vec<(usize, f64)>);

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
pub fn compute_assignments<
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
    (0..old.len())
        .map(|i| l2_dist.distance_f32(&old[i], &new[i]))
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
) -> Result<CentroidsAndAssignments, CentroidsAndAssignments> {
    let acceptable_split = min_cluster_size..=(dataset.len() - min_cluster_size);
    assert!(!acceptable_split.is_empty());

    let half = dataset.len() / 2;
    let mut assignments = vec![(1usize, f64::MAX); dataset.len()];
    assignments[half..].fill((0, f64::MAX));

    let dist_fn = EuclideanDistance::get();
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
// XXX add min_cluster_len or make it a range
// XXX remove max_k -- derive from buffer_len and max_cluster_len
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

        // XXX this should prune centroids below a certain sizebut slightly differently:
        // * do not attempt to prune if there are only two clusters.
        // * do up to min_cluster_len vectors at a time.
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
// XXX this method is using >50% of the main CPU, the offloads are not doing very much but it also
// is not IO bound.
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
