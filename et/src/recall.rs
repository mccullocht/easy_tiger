use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io,
    num::NonZero,
    path::PathBuf,
};

use clap::{Args, ValueEnum};
use easy_tiger::{
    input::{DerefVectorStore, VectorStore},
    Neighbor,
};
use memmap2::Mmap;
use vectors::VectorSimilarity;

/// Supported recall metrics.
#[derive(Default, Debug, Copy, Clone, ValueEnum)]
pub enum RecallMetric {
    /// Simple recall counts the number of common results between the expected and actual sets and
    /// returns a match ratio in [0,1]. This metric does not consider rank or distance values.
    #[default]
    Simple,
    /// Normalized Discounted Cumulative Gain recall. This metric takes into account ranks and
    /// scores (~inverted distance) within each result set then normalizes into a [0,1] value.
    ///
    /// When computing DCG for the actual result set, distances are replaced with values from the
    /// expected result set (or 0) to account for quantization error.
    Ndcg,
}

#[derive(Args)]
pub struct RecallArgs {
    /// Compute recall@k. Must be <= neighbors_len.
    #[arg(long)]
    recall_k: Option<NonZero<usize>>,
    /// Recall metric to compute.
    #[arg(long, value_enum, default_value_t = RecallMetric::Simple)]
    recall_metric: RecallMetric,
    /// Path buf to formatted [`Neighbor`] vectors.
    /// This should include one row of length neighbors_len for each vector in the query set.
    #[arg(long)]
    neighbors: Option<PathBuf>,
    /// Number of neighbors for each query in the neighbors file.
    #[arg(long, default_value_t = NonZero::new(100).unwrap())]
    neighbors_len: NonZero<usize>,
}

/// Computes the recall for a query from a golden file.
// TODO: add an option for NDGC recall computation.
pub struct RecallComputer {
    metric: RecallMetric,
    similarity: VectorSimilarity,
    k: usize,
    neighbors: DerefVectorStore<u8, Mmap>,
}

impl RecallComputer {
    const NEIGHBOR_LEN: usize = 16;

    pub fn from_args(args: RecallArgs, similarity: VectorSimilarity) -> io::Result<Option<Self>> {
        if let Some((neighbors, k)) = args.neighbors.zip(args.recall_k) {
            let elem_stride = Self::NEIGHBOR_LEN * args.neighbors_len.get();
            let neighbors: DerefVectorStore<u8, Mmap> = DerefVectorStore::<u8, _>::new(
                unsafe { Mmap::map(&File::open(neighbors)?)? },
                NonZero::new(elem_stride).unwrap(),
            )?;

            if k.get() <= args.neighbors_len.get() {
                Ok(Some(Self {
                    metric: args.recall_metric,
                    similarity,
                    k: k.get(),
                    neighbors,
                }))
            } else {
                Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "recall k must be <= neighbors_len",
                ))
            }
        } else {
            Ok(None)
        }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn label(&self) -> String {
        match self.metric {
            RecallMetric::Simple => format!("Recall@{}", self.k),
            RecallMetric::Ndcg => format!("NDCG@{}", self.k),
        }
    }

    pub fn neighbors_len(&self) -> usize {
        self.neighbors.len()
    }

    /// Compute the recall based on golden data for `query_index` given `query_results`.
    ///
    /// *Panics* if `query_index` is out of bounds in the golden file.
    pub fn compute_recall(&self, query_index: usize, query_results: &[Neighbor]) -> f64 {
        let expected = self.neighbors[query_index]
            .as_chunks::<{ Self::NEIGHBOR_LEN }>()
            .0
            .iter()
            .take(self.k)
            .map(|n| Neighbor::from(*n));
        let actual = query_results.iter().take(self.k).copied();
        match self.metric {
            RecallMetric::Simple => self.simple_recall(expected, actual),
            RecallMetric::Ndcg => self.ndcg_recall(expected, actual),
        }
    }

    fn simple_recall(
        &self,
        expected: impl Iterator<Item = Neighbor>,
        actual: impl Iterator<Item = Neighbor>,
    ) -> f64 {
        let expected = expected.map(|n| n.vertex()).collect::<HashSet<_>>();
        let count = actual.filter(|n| expected.contains(&n.vertex())).count();
        count as f64 / self.k as f64
    }

    fn ndcg_recall(
        &self,
        expected: impl Iterator<Item = Neighbor> + Clone,
        actual: impl Iterator<Item = Neighbor> + Clone,
    ) -> f64 {
        let ideal_scores = expected
            .clone()
            .map(|n| (n.vertex(), self.distance_to_score(n.distance())))
            .collect::<HashMap<_, _>>();
        let idcg = Self::dcg(expected.map(|n| self.distance_to_score(n.distance())));
        // Replace actual scores with ideal/expected scores, substituting zero when not found.
        // Quantization error may yield scores that are higher than the actual scores and may result
        // in a misleading recall figure (> 1.0).
        let dcg = Self::dcg(actual.map(|n| *ideal_scores.get(&n.vertex()).unwrap_or(&0.0)));
        dcg / idcg
    }

    fn dcg(scores: impl Iterator<Item = f64>) -> f64 {
        scores
            .enumerate()
            .map(|(i, s)| s / (i as f64 + 2.0).log2())
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
