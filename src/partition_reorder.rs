use std::{
    collections::VecDeque,
    num::NonZero,
    ops::{Index, IndexMut, Range},
    sync::Mutex,
};

use rand::{distributions::WeightedIndex, prelude::*};
use rayon::prelude::*;
use simsimd::SpatialSimilarity;

use crate::input::VectorStore;

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
            max_iters: 15,
            initialization_algorithm: CentroidInitializationAlgorithm::Random,
            min_cluster_size: 1,
            epsilon: 0.000_01,
            perturbation: 0.000_000_1,
        }
    }
}

/// k-means clusters `vectors` recursively into a tree where each node contains `k` children.
/// Leaf nodes contain `m` or fewer vectors.
///
/// Returns a new order of the vector inputs that clusters by neighborhood.
pub fn partition_reorder<V: VectorStore<Elem = f32> + Send + Sync, R: Rng, P: Fn(), L: Fn()>(
    vectors: &V,
    k: usize,
    m: usize,
    params: &Params,
    rng: &mut R,
    progress: P,
    length: L,
) -> Vec<usize> {
    let sample_indices = if vectors.len() > 100_000 {
        rand::seq::index::sample(rng, vectors.len(), 100_000).into_vec()
    } else {
        (0..vectors.len()).collect()
    };
    let sample = SubsetViewVectorStore {
        parent: vectors,
        subset: sample_indices,
    };
    let root_k = std::cmp::max(k, ((vectors.len() as f32).sqrt() + 0.5).round() as usize);
    let (centroids, _) = train(&sample, NonZero::new(root_k).unwrap(), params, rng);
    let assignments = compute_cluster_assignments(vectors, &centroids);
    let mut root_subsets = vec![Vec::new(); centroids.len()];
    for (i, (cluster, _)) in assignments.into_iter().enumerate() {
        root_subsets[cluster].push(i);
    }

    let mut queue =
        VecDeque::from_iter(
            root_subsets
                .into_iter()
                .map(|subset| SubsetViewVectorStore {
                    parent: vectors,
                    subset,
                }),
        );
    for _ in 0..centroids.len() {
        length();
    }
    let mut reordered = Vec::with_capacity(vectors.len());

    while let Some(store) = queue.pop_front() {
        if store.len() <= m {
            reordered.extend_from_slice(store.indices());
            progress();
            continue;
        }

        // Append to the queue in reverse order. We want to emit things in the same order as the
        // leaves of the tree we are computing.
        let (_, subsets) = train(&store, NonZero::new(k).unwrap(), params, rng);
        for subset in subsets.into_iter().rev() {
            queue.push_front(store.subset_view(subset));
            length();
        }
        progress();
    }

    reordered
}

/// Compute `clusters` centroids over `training_data` using `params` configuration.
///
/// Returns the centroids as well as the set of samples in `training_data` that appear in each
/// cluster.
fn train<V: VectorStore<Elem = f32> + Send + Sync, R: Rng>(
    training_data: &V,
    clusters: NonZero<usize>,
    params: &Params,
    rng: &mut R,
) -> (MutableVectorStore<f32>, Vec<Vec<usize>>) {
    let mut centroids = initialize_centroids(
        training_data,
        clusters.get(),
        params.initialization_algorithm,
        rng,
    );

    let mut means = vec![0.0; clusters.get()];
    let mut cluster_sizes = vec![0usize; clusters.get()];
    let mut assignments: Vec<(usize, f64)> = vec![];
    let mut new_centroids;

    for _ in 0..params.max_iters {
        (assignments, new_centroids) =
            compute_cluster_assignments_and_update(training_data, &centroids);
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

        let min_cluster_size = std::cmp::min(
            params.min_cluster_size,
            training_data.len() / clusters.get(),
        );
        for (cluster, cluster_size) in cluster_sizes.iter().enumerate() {
            if *cluster_size < min_cluster_size {
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
                new_centroids[cluster].copy_from_slice(&new_centroid);
            }
        }
        centroids = new_centroids;
        means = new_means;
    }

    let mut partitions = cluster_sizes
        .into_iter()
        .map(Vec::with_capacity)
        .collect::<Vec<_>>();
    for (i, (c, _)) in assignments.into_iter().enumerate() {
        partitions[c].push(i);
    }

    (centroids, partitions)
}

/// Create `clusters` initial centroids from `training_data` by the kmeans++ scheme.
fn initialize_centroids<V: VectorStore<Elem = f32> + Send + Sync, R: Rng>(
    training_data: &V,
    clusters: usize,
    algorithm: CentroidInitializationAlgorithm,
    rng: &mut R,
) -> MutableVectorStore<f32> {
    // Use kmeans++ initialization.
    let mut centroids = MutableVectorStore::with_capacity(training_data.elem_stride(), clusters);
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

/// Compute the `centroid` that each sample in `training_data` is closest to as well as the distance
/// between the sample and the centroid.
fn compute_cluster_assignments_and_update<
    V: VectorStore<Elem = f32> + Send + Sync,
    C: VectorStore<Elem = f32> + Send + Sync,
>(
    training_data: &V,
    centroids: &C,
) -> (Vec<(usize, f64)>, MutableVectorStore<f32>) {
    let mut centroid_sums = Vec::with_capacity(centroids.len());
    centroid_sums.resize_with(centroids.len(), || {
        Mutex::new((vec![0.0f32; centroids.elem_stride()], 0usize))
    });

    let assignments = (0..training_data.len())
        .into_par_iter()
        .map(|i| {
            let v = &training_data[i];
            let assignment = centroids
                .iter()
                .enumerate()
                .map(|(ci, cv)| {
                    (
                        ci,
                        SpatialSimilarity::l2(v, cv).expect("same vector length"),
                    )
                })
                .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
                .expect("non-zero clusters");

            {
                let mut guard = centroid_sums[assignment.0].lock().unwrap();
                for (i, o) in v.iter().zip(guard.0.iter_mut()) {
                    *o += *i;
                }
                guard.1 += 1;
            }

            assignment
        })
        .collect();

    let mut new_centroids =
        MutableVectorStore::with_capacity(centroids.elem_stride(), centroids.len());
    for (mut sum, count) in centroid_sums.into_iter().map(|m| m.into_inner().unwrap()) {
        if count > 0 {
            for d in sum.iter_mut() {
                *d /= count as f32;
            }
        }
        new_centroids.push(&sum);
    }

    (assignments, new_centroids)
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

/// Implements a view over a list of indices in an underlying vector store. This view remaps the
/// vectors so that they are assigned dense indices.
///
/// This is useful when iteratively clustering an input data set.
struct SubsetViewVectorStore<'a, V> {
    parent: &'a V,
    subset: Vec<usize>,
}

impl<'a, V: VectorStore> SubsetViewVectorStore<'a, V> {
    /// Create a view from a subset of the vectors in this store.
    ///
    /// *Panics* if the input is not sorted or if any element is out of bounds.
    pub fn subset_view(&self, mut subset: Vec<usize>) -> Self {
        assert!(subset.is_sorted());
        assert!(subset.last().is_none_or(|i| *i < self.len()));

        for i in subset.iter_mut() {
            *i = self.subset[*i];
        }
        Self {
            parent: self.parent,
            subset,
        }
    }

    /// Indices against parent.
    pub fn indices(&self) -> &[usize] {
        &self.subset
    }
}

impl<'a, V: VectorStore> VectorStore for SubsetViewVectorStore<'a, V> {
    type Elem = V::Elem;

    fn elem_stride(&self) -> usize {
        self.parent.elem_stride()
    }

    fn len(&self) -> usize {
        self.subset.len()
    }

    fn is_empty(&self) -> bool {
        self.subset.is_empty()
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = &[Self::Elem]> {
        self.subset.iter().map(|i| &self.parent[*i])
    }
}

impl<V: VectorStore> Index<usize> for SubsetViewVectorStore<'_, V> {
    type Output = [V::Elem];

    fn index(&self, index: usize) -> &Self::Output {
        &self.parent[self.subset[index]]
    }
}
