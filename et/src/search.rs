use std::{
    fs::File,
    io::{self},
    num::NonZero,
    path::PathBuf,
    sync::Arc,
};

use clap::Args;
use easy_tiger::{
    scoring::{DotProductScorer, HammingScorer},
    search::{GraphSearchParams, GraphSearcher},
    wt::{GraphMetadata, WiredTigerGraph, WiredTigerIndexParams, WiredTigerNavVectorStore},
};
use indicatif::{ProgressBar, ProgressStyle};
use wt_mdb::Connection;

#[derive(Args)]
pub struct SearchArgs {
    /// Number of vector dimensions for index and query vectors.
    // TODO: this should appear in the table, derive it from that!
    #[arg(short, long)]
    dimensions: NonZero<usize>,

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
    // TODO: recall statistics
}

pub fn search(
    connection: Arc<Connection>,
    index_params: WiredTigerIndexParams,
    args: SearchArgs,
) -> io::Result<()> {
    let query_vectors = easy_tiger::input::NumpyF32VectorStore::new(
        unsafe { memmap2::Mmap::map(&File::open(args.query_vectors)?)? },
        args.dimensions,
    );
    let limit = std::cmp::min(
        query_vectors.len(),
        args.limit.unwrap_or(query_vectors.len()),
    );

    // TODO: this should be read from the graph table.
    let metadata = GraphMetadata {
        dimensions: args.dimensions,
        max_edges: NonZero::new(1).unwrap(), // unused here.
        index_search_params: GraphSearchParams {
            //unused here.
            beam_width: NonZero::new(1).unwrap(),
            num_rerank: 0,
        },
    };
    let session = connection.open_session().map_err(io::Error::from)?;
    let mut graph = WiredTigerGraph::new(
        metadata,
        session.open_record_cursor(&index_params.graph_table_name)?,
    );
    let mut nav_vectors =
        WiredTigerNavVectorStore::new(session.open_record_cursor(&index_params.nav_table_name)?);
    let mut searcher = GraphSearcher::new(GraphSearchParams {
        beam_width: args.candidates,
        num_rerank: args.rerank_budget.unwrap_or_else(|| args.candidates.get()),
    });

    let progress = ProgressBar::new(limit as u64)
        .with_style(
            ProgressStyle::default_bar()
                .template("{wide_bar} {pos}/{len} ETA: {eta_precise} Elapsed: {elapsed_precise}")
                .unwrap(),
        )
        .with_finish(indicatif::ProgressFinish::AndLeave);
    for q in query_vectors.iter().take(limit) {
        let results = searcher.search(
            q,
            &mut graph,
            &DotProductScorer,
            &mut nav_vectors,
            &HammingScorer,
        )?;
        assert_ne!(results.len(), 0);
        progress.inc(1);
    }
    progress.finish_using_style();

    println!(
        "queries {} avg duration {:.3} ms",
        limit,
        progress.elapsed().div_f32(limit as f32).as_micros() as f64 / 1_000f64
    );

    Ok(())
}
