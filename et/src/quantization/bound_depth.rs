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
    let depths = results.iter().map(|r| r.len() as f64).collect::<Vec<_>>();
    let recall_values = results
        .iter()
        .enumerate()
        .map(|(i, r)| recall_computer.compute_recall(i, r))
        .collect::<Vec<_>>();

    let n = depths.len().max(1) as f64;
    let mean = depths.iter().sum::<f64>() / n;
    let stddev = (depths.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n).sqrt();
    let max = depths.iter().copied().fold(0.0f64, f64::max);
    println!("Queue depth: mean {mean:.2} stddev {stddev:.2} max {max:.0}");
    println!("{}", recall_computer.summarize(&recall_values));

    Ok(())
}
