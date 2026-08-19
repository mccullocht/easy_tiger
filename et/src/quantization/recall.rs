use std::io;

use crate::{neighbor_util::TopNeighbors, recall::RecallComputer, ui::progress_bar};
use clap::Args;
use easy_tiger::{Neighbor, input::VectorStore};
use half::slice::HalfFloatSliceExt;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;
use vectors::f16;

use super::exhaustive::ExhaustiveArgs;

#[derive(Args)]
pub struct QuantizationRecallArgs {
    #[command(flatten)]
    exhaustive: ExhaustiveArgs,

    #[command(flatten)]
    recall: crate::recall::RecallArgs,

    /// Multiplier for recall k.
    ///
    /// Collect k * k_mult neighbors for each query and compute recall using that set. This
    /// simulates recall with full-fidelity reranking where we over retrieve to get the final set.
    #[arg(long, default_value_t = 1.0)]
    k_mult: f64,
}

pub fn recall(
    args: QuantizationRecallArgs,
    doc_vectors: &(impl VectorStore<Elem = f16> + Send + Sync),
) -> io::Result<()> {
    let exhaustive = args.exhaustive.setup(doc_vectors)?;
    let recall_computer = RecallComputer::from_args(args.recall)?.ok_or(io::Error::new(
        io::ErrorKind::InvalidInput,
        "must provide recall args",
    ))?;

    let k = recall_computer.k();
    let result_len = (k as f64 * args.k_mult) as usize;
    let mut query_k = Vec::with_capacity(exhaustive.num_queries());
    query_k.resize_with(exhaustive.num_queries(), || TopNeighbors::new(result_len));
    let (total_scored, total_competitive) = (0..doc_vectors.len())
        .into_par_iter()
        .progress_with(progress_bar(doc_vectors.len(), "scoring"))
        .map(|d| {
            let (center, doc) = exhaustive.encode_doc(&doc_vectors[d].to_f32_vec());
            let mut total_scored = 0;
            let mut total_competitive = 0;
            for (q, results) in query_k.iter().enumerate() {
                let max_distance = results.max_distance();
                if let Some(distance) = exhaustive
                    .scorer(q, center)
                    .distance_with_bound(&doc, max_distance)
                {
                    results.add(Neighbor::new(d as i64, distance));
                    total_competitive += 1;
                }
                total_scored += 1;
            }
            (total_scored, total_competitive)
        })
        .reduce(|| (0usize, 0usize), |a, b| (a.0 + b.0, a.1 + b.1));

    let recall_values = query_k
        .into_iter()
        .enumerate()
        .map(|(i, r)| recall_computer.compute_recall(i, &r.into_neighbors()))
        .collect::<Vec<_>>();
    println!("{}", recall_computer.summarize(&recall_values));
    if total_competitive != total_scored {
        println!(
            "scored: {} competitive: {} ratio: {:.6}",
            total_scored,
            total_competitive,
            total_competitive as f64 / total_scored as f64
        );
    }

    Ok(())
}
