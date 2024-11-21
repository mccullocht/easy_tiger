use std::{
    fs::File,
    io::{self},
    num::NonZero,
    path::PathBuf,
    sync::Arc,
};

use clap::Args;
use easy_tiger::{
    graph::GraphSearchParams,
    search::GraphSearcher,
    worker_pool::WorkerPool,
    wt::{WiredTigerGraphVectorIndex, WiredTigerGraphVectorIndexReader},
};
use indicatif::{ProgressBar, ProgressStyle};
use threadpool::ThreadPool;
use wt_mdb::Connection;

#[derive(Args)]
pub struct SearchArgs {
    /// Path to numpy formatted little-endian float vectors.
    #[arg(short, long)]
    query_vectors: PathBuf,
    /// Number candidates in the search list.
    #[arg(short, long)]
    candidates: NonZero<usize>,
    /// Number of results to re-rank at the end of each search.
    /// If unset, use the same figure as candidates.
    #[arg(short, long)]
    rerank_budget: Option<usize>,
    /// Maximum number of queries to run. If unset, run all queries in the vector file.
    #[arg(short, long)]
    limit: Option<usize>,
    /// If greater than 1, do up to this many concurrent reads during graph traversal.
    #[arg(long, default_value = "1")]
    concurrency: NonZero<usize>,
    // TODO: recall statistics
}

pub fn search(
    connection: Arc<Connection>,
    index: WiredTigerGraphVectorIndex,
    args: SearchArgs,
) -> io::Result<()> {
    let query_vectors = easy_tiger::input::NumpyF32VectorStore::new(
        unsafe { memmap2::Mmap::map(&File::open(args.query_vectors)?)? },
        index.metadata().dimensions,
    );
    let limit = std::cmp::min(
        query_vectors.len(),
        args.limit.unwrap_or(query_vectors.len()),
    );
    let mut searcher = GraphSearcher::new(GraphSearchParams {
        beam_width: args.candidates,
        num_rerank: args.rerank_budget.unwrap_or_else(|| args.candidates.get()),
    });
    let pool = if args.concurrency.get() > 1 {
        Some(WorkerPool::new(
            ThreadPool::new(args.concurrency.into()),
            connection.clone(),
        ))
    } else {
        None
    };

    let index = Arc::new(index);
    let mut session = connection.open_session()?;
    let progress = ProgressBar::new(limit as u64)
        .with_style(
            ProgressStyle::default_bar()
                .template("{wide_bar} {pos}/{len} ETA: {eta_precise} Elapsed: {elapsed_precise}")
                .unwrap(),
        )
        .with_finish(indicatif::ProgressFinish::AndLeave);
    for q in query_vectors.iter().take(limit) {
        session.begin_transaction(None)?;
        // XXX this is awkward as hell so internalize it as much as possible.
        let mut reader = if let Some(pool) = pool.as_ref() {
            WiredTigerGraphVectorIndexReader::with_worker_pool(index.clone(), session, pool.clone())
        } else {
            WiredTigerGraphVectorIndexReader::new(index.clone(), session)
        };
        let results = if pool.is_some() {
            searcher.search_concurrently(q, &mut reader, args.concurrency)
        } else {
            searcher.search(q, &mut reader)
        }?;
        assert_ne!(results.len(), 0);
        progress.inc(1);
        session = reader.into_session();
        session.rollback_transaction(None)?;
    }
    progress.finish_using_style();

    println!(
        "queries {} avg duration {:.3} ms",
        limit,
        progress.elapsed().div_f32(limit as f32).as_micros() as f64 / 1_000f64
    );

    Ok(())
}
