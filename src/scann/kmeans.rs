use std::{
    num::NonZero,
    ops::{Index, IndexMut, Range},
};

use rand::{distributions::WeightedIndex, prelude::*};
use rayon::prelude::*;
use simsimd::SpatialSimilarity;

use crate::input::{DerefVectorStore, VectorStore};

/// How to initialize the centroids for k-means computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CentroidInitializationAlgorithm {
    /// Randomly choose points from the training data set to use as centroids.
    Random,
    /// Perform k-means++ centroid selection.
    KmeansPlusPlus,
}

/// Parameters for computing kmeans.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// Maximum number of iterations to run. May run converge in fewer iterations.
    pub max_iters: usize,
    /// How to initialize the centroids.
    pub initialization_algorithm: CentroidInitializationAlgorithm,
    /// Minimum number of samples in each cluster. If any clusters have fewer than this many samples
    /// the computation will not converge.
    pub min_cluster_size: usize,
    /// If the difference of cluster means between iterations is greater than epsilon the
    /// computation will not converge.
    pub epsilon: f64,
    /// Adjustment when reinitializing centroids for clusters that have too few samples.
    pub perturbation: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            max_iters: 10,
            initialization_algorithm: CentroidInitializationAlgorithm::KmeansPlusPlus,
            min_cluster_size: 1,
            epsilon: 0.000_01,
            perturbation: 0.000_000_1,
        }
    }
}

/// Compute `clusters` centroids over `training_data` using `params` configuration.
///
/// Returns the centroids as well as the set of samples in `training_data` that appear in each
/// cluster.
pub fn train<V: VectorStore<Elem = f32> + Send + Sync, R: Rng>(
    training_data: &V,
    clusters: NonZero<usize>,
    params: &Params,
    rng: &mut R,
) -> (DerefVectorStore<f32, Vec<f32>>, Vec<Vec<usize>>) {
    let mut centroids = initialize_centroids(
        training_data,
        clusters.get(),
        params.initialization_algorithm,
        rng,
    );

    let mut means = vec![0.0; clusters.get()];
    let mut cluster_sizes = vec![0usize; clusters.get()];
    let mut assignments: Vec<(usize, f64)> = vec![];

    for _ in 0..params.max_iters {
        assignments = compute_cluster_assignments(training_data, &centroids);
        let mut new_means = vec![0.0; clusters.get()];
        cluster_sizes.fill(0);
        for (cluster, distance) in assignments.iter() {
            new_means[*cluster] += *distance;
            cluster_sizes[*cluster] += 1;
        }
        for (m, c) in new_means.iter_mut().zip(cluster_sizes.iter_mut()) {
            if *c > 0 {
                *m /= *c as f64;
            }
        }

        // We've converged if none of the centers have moved substantially.
        if means
            .iter()
            .zip(new_means.iter())
            .zip(cluster_sizes.iter())
            .all(|((om, nm), s)| *s >= params.min_cluster_size && (nm - om).abs() <= params.epsilon)
        {
            break;
        }

        // Recompute centroids. Start by summing input vectors for each cluster and dividing by count.
        centroids.fill(0.0);
        for (i, (cluster, _)) in assignments.iter().enumerate() {
            for (c, v) in centroids[*cluster].iter_mut().zip(&training_data[i]) {
                *c += v;
            }
        }
        let min_cluster_size = std::cmp::min(
            params.min_cluster_size,
            training_data.len() / clusters.get(),
        );
        for (cluster, cluster_size) in cluster_sizes.iter().enumerate() {
            if *cluster_size >= min_cluster_size {
                for d in centroids[cluster].iter_mut() {
                    *d /= *cluster_size as f32;
                }
            } else {
                new_means[cluster] = -1.0;
                let (sample_index, sample_cluster) = loop {
                    let i = rng.gen_range(0..training_data.len());
                    let cluster = assignments[i].0;
                    if cluster_sizes[cluster] >= params.min_cluster_size {
                        break (i, cluster);
                    }
                };

                let sample_point = &training_data[sample_index];
                let sample_centroid = &centroids[sample_cluster];
                let new_centroid: Vec<f32> = sample_centroid
                    .iter()
                    .zip(sample_point.iter())
                    .map(|(c, s)| *c + params.perturbation * (*s - *c))
                    .collect();
                centroids[cluster].copy_from_slice(&new_centroid);
            }
        }
        means = new_means;
    }

    let mut partitions = cluster_sizes
        .into_iter()
        .map(Vec::with_capacity)
        .collect::<Vec<_>>();
    for (i, (c, _)) in assignments.into_iter().enumerate() {
        partitions[c].push(i);
    }

    (centroids.into(), partitions)
}

/// Create `clusters` initial centroids from `training_data` by the kmeans++ scheme.
fn initialize_centroids<V: VectorStore<Elem = f32> + Send + Sync, R: Rng>(
    training_data: &V,
    clusters: usize,
    algorithm: CentroidInitializationAlgorithm,
    rng: &mut R,
) -> MutableVectorStore<f32> {
    // Use kmeans++ initialization.
    let mut centroids = MutableVectorStore::with_capacity(training_data[0].len(), clusters);
    match algorithm {
        CentroidInitializationAlgorithm::Random => {
            let mut weights = vec![1.0; training_data.len()];
            while centroids.len() < clusters {
                let selected = WeightedIndex::new(weights.iter()).unwrap().sample(rng);
                centroids.push(&training_data[selected]);
                weights[selected] = 0.0;
            }
        }
        CentroidInitializationAlgorithm::KmeansPlusPlus => {
            centroids.push(&training_data[rng.gen_range(0..training_data.len())]);
            while centroids.len() < clusters {
                let assignments = compute_cluster_assignments(training_data, &centroids);
                let selected = WeightedIndex::new(assignments.iter().map(|(_, w)| w))
                    .unwrap()
                    .sample(rng);
                centroids.push(&training_data[selected]);
            }
        }
    }
    centroids
}

/// Compute the `centroid` that each sample in `training_data` is closest to as well as the distance
/// between the sample and the centroid.
fn compute_cluster_assignments<
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
                .enumerate()
                .map(|(ci, cv)| {
                    (
                        ci,
                        SpatialSimilarity::l2(v, cv).expect("same vector length"),
                    )
                })
                .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
                .expect("non-zero clusters")
        })
        .collect()
}

/// A mutable [crate::input::VectorStore] implementation where vector elements are of type `E`.
struct MutableVectorStore<E> {
    data: Vec<E>,
    elem_stride: usize,
}

impl<E: Clone> MutableVectorStore<E> {
    /// Create an empty MutableVectorStore with room for `capacity` vectors.
    pub fn with_capacity(elem_stride: usize, capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(elem_stride * capacity),
            elem_stride,
        }
    }

    /// Append `vector` to the store.
    ///
    /// *Panics* if `vector.len() != self.elem_stride()`.
    pub fn push(&mut self, vector: &[E]) {
        assert_eq!(vector.len(), self.elem_stride);
        self.data.extend_from_slice(vector);
    }

    /// Fill all elements of all vectors in the store with `value`.
    pub fn fill(&mut self, value: E) {
        self.data.fill(value);
    }

    fn range(&self, index: usize) -> Range<usize> {
        let start = index * self.elem_stride;
        start..(start + self.elem_stride)
    }
}

impl<E: Clone> VectorStore for MutableVectorStore<E> {
    type Elem = E;

    fn elem_stride(&self) -> usize {
        self.elem_stride
    }

    fn len(&self) -> usize {
        self.data.len() / self.elem_stride
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = &[Self::Elem]> {
        self.data.chunks(self.elem_stride)
    }
}

impl<E: Clone> Index<usize> for MutableVectorStore<E> {
    type Output = [E];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.range(index)]
    }
}

impl<E: Clone> IndexMut<usize> for MutableVectorStore<E> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let r = self.range(index);
        &mut self.data[r]
    }
}

impl<E> From<MutableVectorStore<E>> for DerefVectorStore<E, Vec<E>> {
    fn from(value: MutableVectorStore<E>) -> Self {
        DerefVectorStore::new_typed(value.data, value.elem_stride)
    }
}
