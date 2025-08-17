use std::num::NonZero;

use rayon::prelude::*;

use crate::{input::VectorStore, vectors::VectorSimilarity, Neighbor};

// XXX delete me

/// An object that can compute recall metrics.
pub trait RecallComputer {
    /// Compute recall for the given query and actual result set.
    /// Returns a number in [0,1] where 1 is better.
    fn compute_recall(&self, query_index: usize, actual_results: &[Neighbor]) -> f64;
}

/// Compute recall by measuring the fraction of actual results that appear in the expected result
/// set. Notably, this does not consider rank or scores of the results.
pub struct SimpleRecallComputer {
    k: usize,
    expected: Vec<Vec<u32>>,
}

impl SimpleRecallComputer {
    pub fn new(k: NonZero<usize>, neighbors: &impl VectorStore<Elem = u32>) -> Self {
        assert!(k.get() <= neighbors.elem_stride());
        let expected = neighbors
            .iter()
            .map(|n| {
                let mut e = n[..k.get()].to_vec();
                e.sort_unstable();
                e
            })
            .collect();
        Self {
            k: k.get(),
            expected,
        }
    }
}

impl RecallComputer for SimpleRecallComputer {
    fn compute_recall(&self, query_index: usize, actual_results: &[Neighbor]) -> f64 {
        let expected = &self.expected[query_index];
        let count = actual_results
            .iter()
            .take(self.k)
            .filter(|n| expected.binary_search(&(n.vertex() as u32)).is_ok())
            .count();
        count as f64 / self.k as f64
    }
}

/// Compute Normalized Discounted Cumulative Gain recall. This metric takes into account ranks and
/// scores (~inverted distance) within each result set then normalizes into a [0,1] value.
pub struct NDCGRecallComputer {
    k: usize,
    expected: Vec<Vec<Neighbor>>,
    similarity: VectorSimilarity,
}

impl NDCGRecallComputer {
    pub fn new(
        k: NonZero<usize>,
        neighbors: &(impl VectorStore<Elem = u32> + Send + Sync),
        similarity: VectorSimilarity,
        queries: &(impl VectorStore<Elem = f32> + Send + Sync),
        vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    ) -> Self {
        assert!(neighbors.elem_stride() >= k.get());
        assert_eq!(neighbors.len(), queries.len());
        let expected = (0..neighbors.len())
            .into_par_iter()
            .map(|q| {
                let neighbors = &neighbors[q][..k.get()];
                let query = &queries[q];
                let distance_fn = similarity.new_distance_function();
                neighbors
                    .iter()
                    .map(|i| {
                        Neighbor::new(
                            (*i).into(),
                            distance_fn.distance_f32(query, &vectors[*i as usize]),
                        )
                    })
                    .collect()
            })
            .collect();
        Self {
            k: k.get(),
            expected,
            similarity,
        }
    }

    fn dcg(scores: impl Iterator<Item = f64>) -> f64 {
        scores
            .enumerate()
            .map(|(i, s)| s / (i as f64 + 1.0).log2())
            .sum()
    }

    fn distance_to_score(&self, distance: f64) -> f64 {
        match self.similarity {
            // Map distance to score the same way as Lucene. This normalizes perfect match to 0 but
            // otherwise creates a pretty strange looking curve.
            VectorSimilarity::Euclidean => 1.0 / (1.0 + distance.max(0.0)),
            // Angular distances are already in [0,1] so take the additive inverse
            VectorSimilarity::Cosine | VectorSimilarity::Dot => (1.0 - distance).clamp(0.0, 1.0),
        }
    }
}

impl RecallComputer for NDCGRecallComputer {
    fn compute_recall(&self, query_index: usize, actual_results: &[Neighbor]) -> f64 {
        let expected = &self.expected[query_index];
        let idcg = Self::dcg(
            expected
                .iter()
                .map(|n| self.distance_to_score(n.distance())),
        );
        // NB: this is N^2 but N will typically be small so we don't care too much.
        let dcg = Self::dcg(actual_results.iter().take(self.k).map(|n| {
            if expected.iter().any(|e| e.vertex() == n.vertex()) {
                self.distance_to_score(n.distance())
            } else {
                0.0
            }
        }));
        dcg / idcg
    }
}
