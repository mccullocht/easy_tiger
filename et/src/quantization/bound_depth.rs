use std::io;

use crate::{
    neighbor_util::{BoundedNeighbor, BoundedNeighbors},
    recall::RecallComputer,
    ui::progress_bar,
};
use clap::Args;
use easy_tiger::input::VectorStore;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

use super::exhaustive::ExhaustiveArgs;

#[derive(Args)]
pub struct BoundDepthArgs {
    #[command(flatten)]
    exhaustive: ExhaustiveArgs,

    #[command(flatten)]
    recall: crate::recall::RecallArgs,
}

/// Exhaustively score every doc against every query using quantized distance *bounds*, retaining
/// each candidate that could still enter the top k, then report how large that candidate set is.
///
/// Alongside the retained set size this reports the depth a reranker actually needed to recover the
/// golden top-k, the rate at which the bounds pruned a golden neighbor, and the ratio between the
/// two depths. Those three separate the two ways the bounds can be wrong: a large ratio means they
/// are merely loose, while a nonzero miss rate means they are unsound and no amount of tightening
/// will fix them.
pub fn bound_depth(
    args: BoundDepthArgs,
    doc_vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
) -> io::Result<()> {
    let exhaustive = args.exhaustive.setup(doc_vectors)?;
    let recall_computer =
        RecallComputer::from_args(args.recall, args.exhaustive.similarity)?.ok_or(
            io::Error::new(io::ErrorKind::InvalidInput, "must provide recall args"),
        )?;

    let k = recall_computer.k();
    let mut query_candidates = Vec::with_capacity(exhaustive.num_queries());
    query_candidates.resize_with(exhaustive.num_queries(), || BoundedNeighbors::new(k));
    (0..doc_vectors.len())
        .into_par_iter()
        .progress_with(progress_bar(doc_vectors.len(), "scoring"))
        .for_each(|d| {
            let (center, doc) = exhaustive.encode_doc(&doc_vectors[d]);
            for (q, candidates) in query_candidates.iter().enumerate() {
                let bounds = exhaustive.scorer(q, center).distance_bounds(&doc);
                candidates.add(BoundedNeighbor::new(d as i64, bounds));
            }
        });

    let results = query_candidates
        .into_iter()
        .map(|c| c.into_neighbors())
        .collect::<Vec<_>>();

    // Retained set size, and the depth a reranker needed within it. Realized depth is only defined
    // for queries where the bounds kept every golden neighbor, so slop is measured over those too.
    let mut queue_depths = Vec::with_capacity(results.len());
    let mut realized_depths = Vec::new();
    let mut slop = Vec::new();
    let mut misses = 0usize;
    for (i, r) in results.iter().enumerate() {
        let queue_depth = r.len() as f64;
        queue_depths.push(queue_depth);
        match recall_computer.realized_depth(i, r) {
            Some(depth) => {
                realized_depths.push(depth as f64);
                slop.push(queue_depth / depth as f64);
            }
            None => misses += 1,
        }
    }

    let recall_values = results
        .iter()
        .enumerate()
        .map(|(i, r)| recall_computer.compute_recall(i, r))
        .collect::<Vec<_>>();

    println!("{}", Distribution::new(queue_depths).summarize("Queue depth"));
    println!(
        "{}",
        Distribution::new(realized_depths).summarize("Realized depth")
    );
    println!("{}", Distribution::new(slop).summarize("Slop"));
    println!(
        "Bound misses: {}/{} ({:.4}%)",
        misses,
        results.len(),
        100.0 * misses as f64 / results.len().max(1) as f64
    );
    println!("{}", recall_computer.summarize(&recall_values));

    Ok(())
}

/// Per-query values summarized by mean and quantile.
struct Distribution(Vec<f64>);

impl Distribution {
    fn new(mut values: Vec<f64>) -> Self {
        values.sort_unstable_by(f64::total_cmp);
        Self(values)
    }

    /// Nearest-rank quantile of the sorted values.
    fn quantile(&self, p: f64) -> f64 {
        if self.0.is_empty() {
            return f64::NAN;
        }
        let rank = ((p * self.0.len() as f64).ceil() as usize).clamp(1, self.0.len());
        self.0[rank - 1]
    }

    /// Format the distribution, reporting quantiles rather than a standard deviation.
    ///
    /// Depth distributions are strongly right skewed, so a stddev around the mean does not describe
    /// the tail that sets a work budget.
    fn summarize(&self, label: &str) -> String {
        if self.0.is_empty() {
            return format!("{label}: no values");
        }
        let mean = self.0.iter().sum::<f64>() / self.0.len() as f64;
        format!(
            "{label}: n {} mean {:.2} p50 {:.2} p90 {:.2} p99 {:.2} max {:.2}",
            self.0.len(),
            mean,
            self.quantile(0.5),
            self.quantile(0.9),
            self.quantile(0.99),
            self.0[self.0.len() - 1],
        )
    }
}
