//! Clustering to provide a new order for vectors in a graph based index.

use std::collections::VecDeque;

use rand::Rng;

use crate::{
    input::{SubsetViewVectorStore, VectorStore},
    kmeans::{self, batch_kmeans, compute_assignments},
};

pub fn cluster_for_reordering<
    V: VectorStore<Elem = f32> + Send + Sync,
    P: Fn(u64) + Send + Sync,
>(
    dataset: &V,
    m: usize,
    params: &kmeans::Params,
    rng: &mut impl Rng,
    progress: P,
) -> Vec<usize> {
    let batch_size = 1000;
    let mut new_order = Vec::with_capacity(dataset.len());
    let mut queue = VecDeque::new();
    queue.push_back((0..dataset.len()).collect::<Vec<_>>());
    while let Some(subset) = queue.pop_front() {
        if subset.len() <= m {
            new_order.extend_from_slice(&subset);
            progress(subset.len() as u64);
            continue;
        }

        let len = subset.len();
        let sm = m.min((len / m) + 1);
        let subset_vectors = SubsetViewVectorStore::new(dataset, subset);
        let centroids = match batch_kmeans(&subset_vectors, sm, batch_size, params, rng) {
            Ok(c) => c,
            Err(c) => {
                eprintln!(
                    "batch_kmeans did not converge on cluster of size {} k={} batch_size={}",
                    len, sm, batch_size
                );
                c
            }
        };

        let assignments = compute_assignments(&subset_vectors, &centroids);
        let centroid_to_vectors = assignments.iter().enumerate().fold(
            vec![vec![]; centroids.len()],
            |mut c2v, (i, (c, _))| {
                c2v[*c].push(subset_vectors.original_index(i));
                c2v
            },
        );

        for vectors in centroid_to_vectors {
            queue.push_front(vectors);
        }
    }
    new_order
}
