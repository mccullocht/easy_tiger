// TODO: integrate the changes here but switch back to k-means at the highest levels. Batch k-means
// is really the best bet for the root of the tree in terms of minimizing cost.

use std::collections::VecDeque;

use rayon::prelude::*;
use tracing::debug;
use vectors::F32VectorDistance;

use crate::input::{SubsetViewVectorStore, VecVectorStore, VectorStore};

pub struct ClusterIter<'a, V, P> {
    dataset: &'a V,
    queue: VecDeque<(Vec<f32>, SubsetViewVectorStore<'a, V>)>,
    max_cluster_len: usize,
    dist_fn: Box<dyn F32VectorDistance>,
    progress: P,
}

impl<'a, V: VectorStore<Elem = f32> + Send + Sync, P> ClusterIter<'a, V, P> {
    pub fn new(
        dataset: &'a V,
        max_cluster_len: usize,
        dist_fn: Box<dyn F32VectorDistance>,
        progress: P,
    ) -> Self {
        let mut queue = VecDeque::new();
        queue.push_front((
            dataset[0].to_vec(),
            SubsetViewVectorStore::new(dataset, (0..dataset.len()).collect()),
        ));
        Self {
            dataset,
            queue,
            max_cluster_len,
            dist_fn,
            progress,
        }
    }
}

impl<'a, V: VectorStore<Elem = f32> + Send + Sync, P: Fn(u64) + Send + Sync> Iterator
    for ClusterIter<'a, V, P>
{
    type Item = (Vec<f32>, Vec<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((centroid, vectors)) = self.queue.pop_front() {
            if vectors.len() <= self.max_cluster_len {
                return Some((centroid, vectors.into_subset()));
            }
            let (centroids, mut cluster0, mut cluster1) = match binary_partition(
                &vectors,
                15,
                self.max_cluster_len / 2,
                self.dist_fn.as_ref(),
                &self.progress,
            ) {
                Ok((c, a, b)) => (c, a, b),
                Err((c, a, b)) => {
                    debug!(
                        "binary partition failed to converge; input len {} max_cluster_len {}",
                        vectors.len(),
                        self.max_cluster_len
                    );
                    (c, a, b)
                }
            };

            for c in cluster0.iter_mut() {
                *c = vectors.original_index(*c);
            }
            for c in cluster1.iter_mut() {
                *c = vectors.original_index(*c);
            }

            self.queue.push_front((
                centroids[1].to_vec(),
                SubsetViewVectorStore::new(self.dataset, cluster1),
            ));
            self.queue.push_front((
                centroids[0].to_vec(),
                SubsetViewVectorStore::new(self.dataset, cluster0),
            ));
        }

        None
    }
}

type CentroidsAndAssignments = (VecVectorStore<f32>, Vec<usize>, Vec<usize>);

/// Build a binary partition of the vectors between two centroids.
///
/// Split the dataset in half and produce two centroids, then measure the distance for each vector
/// to both centroids and the weight of vectors to each centroid. Terminate if the centroids would
/// produce two clusters of at least min_cluster_size, otherwise split the dataset into a new
/// grouping and try again.
///
/// Returns a vector store containing a "left" and a "right" centroid along with cluster assignment.
/// Note that these cluster assignments may not match the results of compute_assignments().
fn binary_partition<V: VectorStore<Elem = f32> + Send + Sync, P: Fn(u64) + Send + Sync>(
    dataset: &V,
    max_iters: usize,
    min_cluster_size: usize,
    dist_fn: &dyn F32VectorDistance,
    progress: &P,
) -> Result<CentroidsAndAssignments, CentroidsAndAssignments> {
    let acceptable_split = min_cluster_size..=(dataset.len() - min_cluster_size);
    assert!(!acceptable_split.is_empty());

    let split = dataset.len() / 2;
    // TODO: this could be a bitmap and it would be much smaller.
    let mut assignments = vec![0u8; dataset.len()];
    assignments[split..].fill(1);

    progress(1);
    let mut centroids = compute_centroids(dataset, &assignments);
    progress(1);
    let mut distances = vec![(0usize, 0.0, 0.0, 0.0); dataset.len()];
    let mut prev_inertia = dataset.len();
    let mut converged = false;
    for _ in 0..max_iters {
        distances.par_iter_mut().enumerate().for_each(|(i, d)| {
            let ldist = dist_fn.distance_f32(&centroids[0], &dataset[i]);
            let rdist = dist_fn.distance_f32(&centroids[1], &dataset[i]);
            *d = (i, ldist, rdist, ldist - rdist)
        });
        progress(1);
        distances.sort_unstable_by(|a, b| a.3.total_cmp(&b.3).then_with(|| a.0.cmp(&b.0)));
        let split_point = distances.iter().position(|d| d.3 >= 0.0).unwrap_or(0);
        let inertia = split.abs_diff(split_point);
        // We may terminate if the partition sizes are acceptable and we aren't improving the split.
        if acceptable_split.contains(&split_point) && (inertia == 0 || prev_inertia <= inertia) {
            converged = true;
            assignments.fill(0);
            for i in distances[split_point..].iter().map(|d| d.0) {
                assignments[i] = 1;
            }
            break;
        }

        prev_inertia = inertia;
        // Split evenly rather than using split_point. Using split_point may reinforce bias and
        // cause us to go further away from the the target split rather than converging.
        assignments.fill(0);
        for i in distances[split..].iter().map(|d| d.0) {
            assignments[i] = 1;
        }
        centroids = compute_centroids(dataset, &assignments);
        progress(1);
    }

    let (left, right) =
        assignments
            .iter()
            .enumerate()
            .fold((vec![], vec![]), |(mut l, mut r), (i, a)| {
                if *a == 0 {
                    l.push(i)
                } else {
                    r.push(i)
                }
                (l, r)
            });
    progress(1);
    if converged {
        Ok((centroids, left, right))
    } else {
        Err((centroids, left, right))
    }
}

fn compute_centroids<V: VectorStore<Elem = f32> + Send + Sync>(
    dataset: &V,
    assignments: &[u8],
) -> VecVectorStore<f32> {
    let create_centroids = || {
        let mut store: VecVectorStore<f32> =
            VecVectorStore::with_capacity(dataset.elem_stride(), 2);
        store.push(&vec![0.0; dataset.elem_stride()]);
        store.push(&vec![0.0; dataset.elem_stride()]);
        store
    };
    let (mut centroids, counts) = (0..dataset.len())
        .into_par_iter()
        .fold(
            || (create_centroids(), [0usize; 2]),
            |mut state, i| {
                let assigned = assignments[i] as usize;
                let centroid = &mut state.0[assigned];
                for (d, o) in dataset[i].iter().zip(centroid.iter_mut()) {
                    *o += *d;
                }
                state.1[assigned] += 1;
                state
            },
        )
        .reduce(
            || (create_centroids(), [0usize; 2]),
            |mut a, b| {
                for i in 0..2 {
                    for (o, s) in a.0[i].iter_mut().zip(b.0[i].iter()) {
                        *o += *s;
                    }
                    a.1[i] += b.1[i];
                }
                a
            },
        );
    for i in 0..2 {
        for d in centroids[i].iter_mut() {
            *d /= counts[i] as f32;
        }
    }
    centroids
}
