use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io,
    num::NonZero,
    path::PathBuf,
};

use clap::{Args, ValueEnum};
use easy_tiger::{
    Neighbor,
    input::{DerefVectorStore, VectorStore},
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
    /// Estimates the depth into the actual result set required to recover the expected top-k
    /// results with full-fidelity reranking.
    ///
    /// Rather than a [0,1] ratio, this reports the minimum number of leading results from the
    /// actual set that must be re-ranked to surface all of the expected top-k neighbors. If any
    /// expected neighbor is missing from the actual set, the depth is counted as the length of the
    /// actual set. Averaged across queries this estimates how deep retrieval must go to achieve
    /// perfect recall@k.
    Depth,
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

/// Golden top-k neighbors, either mapped from a file or computed in process.
enum Golden {
    /// Flat rows of 16 byte [`Neighbor`] records, one row per query.
    Mapped(DerefVectorStore<u8, Mmap>),
    /// One entry per query, already decoded and ordered by distance.
    Memory(Vec<Vec<Neighbor>>),
}

impl Golden {
    fn len(&self) -> usize {
        match self {
            Self::Mapped(n) => n.len(),
            Self::Memory(n) => n.len(),
        }
    }

    /// The first `k` golden neighbors for `query_index`.
    ///
    /// *Panics* if `query_index` is out of bounds.
    fn row(&self, query_index: usize, k: usize) -> Vec<Neighbor> {
        match self {
            Self::Mapped(neighbors) => neighbors[query_index]
                .as_chunks::<{ RecallComputer::NEIGHBOR_LEN }>()
                .0
                .iter()
                .take(k)
                .map(|n| Neighbor::from(*n))
                .collect(),
            Self::Memory(neighbors) => neighbors[query_index].iter().copied().take(k).collect(),
        }
    }
}

/// Computes the recall for a query from a golden file.
// TODO: add an option for NDGC recall computation.
pub struct RecallComputer {
    metric: RecallMetric,
    similarity: VectorSimilarity,
    k: usize,
    golden: Golden,
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
                    golden: Golden::Mapped(neighbors),
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

    /// Build a computer over golden neighbors computed in process rather than read from a file.
    ///
    /// `golden` holds one row per query ordered by ascending distance; rows shorter than `k` simply
    /// yield fewer expected neighbors.
    pub fn in_memory(
        metric: RecallMetric,
        similarity: VectorSimilarity,
        k: NonZero<usize>,
        golden: Vec<Vec<Neighbor>>,
    ) -> Self {
        Self {
            metric,
            similarity,
            k: k.get(),
            golden: Golden::Memory(golden),
        }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn label(&self) -> String {
        match self.metric {
            RecallMetric::Simple => format!("Recall@{}", self.k),
            RecallMetric::Ndcg => format!("NDCG@{}", self.k),
            RecallMetric::Depth => format!("Depth@{}", self.k),
        }
    }

    pub fn neighbors_len(&self) -> usize {
        self.golden.len()
    }

    /// Format a summary line for the per-query results in `values`.
    ///
    /// For ratio metrics (Simple/Ndcg) this reports the mean. For [`RecallMetric::Depth`] it also
    /// reports the standard deviation and maximum depth across queries.
    pub fn summarize(&self, values: &[f64]) -> String {
        let n = values.len().max(1) as f64;
        let mean = values.iter().sum::<f64>() / n;
        match self.metric {
            RecallMetric::Simple | RecallMetric::Ndcg => {
                format!("{}: {:.6}", self.label(), mean)
            }
            RecallMetric::Depth => {
                let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
                let stddev = variance.sqrt();
                let max = values.iter().copied().fold(0.0f64, f64::max);
                format!(
                    "{}: mean {:.2} stddev {:.2} max {:.0}",
                    self.label(),
                    mean,
                    stddev,
                    max
                )
            }
        }
    }

    /// Compute the recall based on golden data for `query_index` given `query_results`.
    ///
    /// *Panics* if `query_index` is out of bounds in the golden file.
    pub fn compute_recall(&self, query_index: usize, query_results: &[Neighbor]) -> f64 {
        let expected = self.expected(query_index);
        let expected = expected.iter().copied();
        let actual = query_results.iter().copied();
        match self.metric {
            RecallMetric::Simple => self.simple_recall(expected, actual),
            RecallMetric::Ndcg => self.ndcg_recall(expected, actual),
            RecallMetric::Depth => Self::required_depth(expected, actual),
        }
    }

    /// The golden top-`k` neighbors for `query_index`.
    ///
    /// *Panics* if `query_index` is out of bounds in the golden set.
    fn expected(&self, query_index: usize) -> Vec<Neighbor> {
        self.golden.row(query_index, self.k)
    }

    /// Depth into `query_results` required to recover every golden top-`k` neighbor, or `None` if
    /// any of them is absent.
    ///
    /// Unlike [`RecallMetric::Depth`] a miss is reported as `None` rather than the length of
    /// `query_results`. Callers whose result set size is itself the variable under test cannot use
    /// that substitution: it ties the penalty for a miss to the quantity being measured, so misses
    /// saturate to whatever ceiling the caller happened to retain.
    pub fn realized_depth(&self, query_index: usize, query_results: &[Neighbor]) -> Option<usize> {
        let mut expected = self
            .expected(query_index)
            .iter()
            .map(|n| n.vertex())
            .collect::<HashSet<_>>();
        for (i, n) in query_results.iter().enumerate() {
            if expected.remove(&n.vertex()) && expected.is_empty() {
                return Some(i + 1);
            }
        }
        None
    }

    /// Compute the depth into `actual` required to recover every vertex in `expected`.
    ///
    /// Returns the 1-based position of the last expected vertex encountered while scanning `actual`
    /// in order. If any expected vertex is missing, returns the length of `actual`.
    fn required_depth(
        expected: impl Iterator<Item = Neighbor>,
        actual: impl Iterator<Item = Neighbor>,
    ) -> f64 {
        let mut expected = expected.map(|n| n.vertex()).collect::<HashSet<_>>();
        let mut depth = 0usize;
        for (i, n) in actual.enumerate() {
            depth = i + 1;
            if expected.remove(&n.vertex()) && expected.is_empty() {
                return (i + 1) as f64;
            }
        }
        // Not every expected vertex was found; count as the length of the actual set.
        depth as f64
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
