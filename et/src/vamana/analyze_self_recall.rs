use std::{io, ops::Add, sync::Arc};

use clap::Args;
use easy_tiger::vamana::{
    GraphSearchParams, GraphVectorIndex, GraphVectorStore,
    search::{GraphSearcher, Options, VertexIdTrace, VertexTrace},
    wt::{TableGraphVectorIndex, TransactionGraphVectorIndex},
};
use indicatif::{ParallelProgressIterator, ProgressBar};
use rayon::prelude::*;
use wt_mdb::{Connection, Error, Result};

use crate::ui::progress_bar;

#[derive(Args)]
pub struct AnalyzeSelfRecallArgs {
    /// Optional vertex IDs to analyze. Accepts comma-separated values.
    /// If not provided, all vectors in the highest fidelity table will be analyzed.
    ///
    /// Example: --ids 1,2,3 --ids 100,200
    #[arg(long, value_delimiter = ',')]
    ids: Vec<i64>,

    /// Number of candidates in the search beam width.
    #[arg(short, long, default_value_t = 128)]
    beam_width: usize,

    /// Number of results to re-rank.
    ///
    /// If not provided and the index has a rerank table, defaults to the beam width.
    /// If not provided and there is no rerank table, reranking is disabled.
    /// If set to a non-zero value but the index has no rerank table, exits with an error.
    #[arg(short, long)]
    num_rerank: Option<usize>,

    /// Add a new entry point as a seed.
    #[arg(long)]
    entry_point: Option<i64>,

    /// If true, print JSON search traces for every search run.
    #[arg(long, default_value_t = false)]
    trace: bool,
}

pub fn analyze_self_recall(
    connection: Arc<Connection>,
    index_name: &str,
    args: AnalyzeSelfRecallArgs,
) -> io::Result<()> {
    let index = Arc::new(TableGraphVectorIndex::from_db(&connection, index_name)?);
    let rerank_format = index.config().rerank_format;
    let beam_width = std::num::NonZero::new(args.beam_width)
        .unwrap_or_else(|| std::num::NonZero::new(1).unwrap());

    // Get the high-fidelity vector table (rerank if available, else nav).
    let hi_table = index.high_fidelity_table();
    let hi_format = hi_table.format();
    let hi_coder = hi_format.coder();

    // Build the list of vertex IDs to analyze.
    let vertex_ids: Vec<i64> = if args.ids.is_empty() {
        // Discover all vertex IDs by scanning the high-fidelity table.
        let txn = connection.begin_transaction(None)?;
        let mut cursor = txn.open_record_cursor(index.graph_table_name())?;
        cursor.set_bounds(0..)?;
        cursor
            .map(|r| r.map(|(id, _)| id))
            .collect::<Result<Vec<_>>>()?
    } else {
        args.ids.clone()
    };

    if vertex_ids.is_empty() {
        println!("No vectors found to analyze.");
        return Ok(());
    }

    if rerank_format.is_none() && args.num_rerank.unwrap_or(0) > 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "--num-rerank {} requested but the index has no rerank table",
                args.num_rerank.unwrap()
            ),
        ));
    }

    let num_rerank = rerank_format
        .map(|_| args.num_rerank.unwrap_or(args.beam_width))
        .unwrap_or(0);

    let progress = if vertex_ids.len() < 100 {
        ProgressBar::hidden()
    } else {
        progress_bar(vertex_ids.len(), "search vertices")
    };
    let summary = vertex_ids
        .into_par_iter()
        .progress_with(progress)
        .map_init(
            || {
                (
                    TransactionGraphVectorIndex::new(
                        Arc::clone(&index),
                        connection.begin_transaction(None).unwrap(),
                    ),
                    GraphSearcher::new(GraphSearchParams {
                        beam_width,
                        num_rerank,
                        patience: None,
                    }),
                )
            },
            |(txn_idx, searcher), vertex_id| {
                let mut vectors = txn_idx.high_fidelity_vectors()?;
                let vector = hi_coder.decode(
                    &vectors
                        .get(vertex_id)
                        .unwrap_or(Err(Error::not_found_error()))?,
                );

                let (_, trace) = searcher.search_with_options(
                    &vector,
                    Options::default()
                        .with_trace([vertex_id])
                        .with_seeds(args.entry_point),
                    txn_idx,
                )?;
                let trace = trace.unwrap();

                if args.trace {
                    println!(
                        "{}",
                        serde_json::to_string(&trace).expect("trace is serializeable")
                    );
                }

                Ok::<_, Error>(TraceSummary::from(trace.vectors[0]))
            },
        )
        .try_reduce(|| TraceSummary::default(), |a, b| Ok(a + b))?;

    if args.trace {
        return Ok(());
    }

    println!("{:<15} {:8}", "Not Found", summary.not_found);
    println!("{:<15} {:8}", "Unseen", summary.unseen);
    println!("{:<15} {:8}", "Seen", summary.seen);
    println!("{:<15} {:8}", "Rerank Dropped", summary.rerank_dropped);
    println!("{:<15} {:8}", "Found", summary.found);
    if summary.found > 0 {
        println!(
            "{:<15} {:8}",
            "Avg Found Rank",
            summary.found_rank_sum as f64 / summary.found as f64
        );
    }

    if !summary.notable.is_empty() {
        print!("Found {} notable traces", summary.notable.len());
        if summary.notable.len() > 10 {
            println!(", showing first 10");
        } else {
            println!();
        }
        for t in summary.notable.into_iter().take(10) {
            println!("  {:6} {:?}", t.id, t.trace);
        }
    }

    Ok(())
}

#[derive(Default, Debug, Clone)]
struct TraceSummary {
    not_found: usize,
    unseen: usize,
    seen: usize,
    rerank_dropped: usize,
    found: usize,
    found_rank_sum: usize,

    notable: Vec<VertexIdTrace>,
}

impl Add for TraceSummary {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut notable = self.notable;
        notable.extend_from_slice(&rhs.notable);
        Self {
            not_found: self.not_found + rhs.not_found,
            unseen: self.unseen + rhs.unseen,
            seen: self.seen + rhs.seen,
            rerank_dropped: self.rerank_dropped + rhs.rerank_dropped,
            found: self.found + rhs.found,
            found_rank_sum: self.found_rank_sum + rhs.found_rank_sum,
            notable,
        }
    }
}

impl From<VertexIdTrace> for TraceSummary {
    fn from(value: VertexIdTrace) -> Self {
        match value.trace {
            VertexTrace::NotFound => TraceSummary {
                not_found: 1,
                notable: vec![value],
                ..Default::default()
            },
            VertexTrace::Unseen => TraceSummary {
                unseen: 1,
                notable: vec![value],
                ..Default::default()
            },
            VertexTrace::Seen {
                rank: _,
                distance: _,
            } => TraceSummary {
                seen: 1,
                notable: vec![value],
                ..Default::default()
            },
            VertexTrace::RerankDropped {
                rank: _,
                distance: _,
            } => TraceSummary {
                rerank_dropped: 1,
                notable: vec![value],
                ..Default::default()
            },
            VertexTrace::Found { rank, distance: _ } => {
                let notable = if rank != 0 { vec![value] } else { vec![] };
                TraceSummary {
                    found: 1,
                    found_rank_sum: rank,
                    notable,
                    ..Default::default()
                }
            }
        }
    }
}
