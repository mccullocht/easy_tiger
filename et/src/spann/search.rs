// TODO: factor out common components with vamana graph search.

use std::{
    fs::File,
    io::{self},
    num::NonZero,
    ops::Add,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use clap::Args;
use easy_tiger::{
    graph::GraphSearchParams,
    input::{DerefVectorStore, VectorStore},
    spann::{SessionIndexReader, SpannSearchParams, SpannSearchStats, SpannSearcher, TableIndex},
};
use memmap2::Mmap;
use wt_mdb::Connection;

use crate::{
    recall::{RecallArgs, RecallComputer},
    ui::progress_bar,
    wt_stats::WiredTigerConnectionStats,
};

#[derive(Args)]
pub struct SearchArgs {
    /// Path to numpy formatted little-endian float vectors.
    #[arg(short, long)]
    query_vectors: PathBuf,
    /// Number head (centroid) candidates in the search list.
    #[arg(long)]
    head_candidates: NonZero<usize>,
    /// Number of head (centroid) results to re-rank at the end of each search.
    /// If unset, use the same figure as --head-candidates.
    #[arg(long)]
    head_rerank_budget: Option<usize>,
    /// Number of centroids to search postings of. Must be >= than --head-rerank-budget
    #[arg(long)]
    posting_centroids: NonZero<usize>,
    /// Number of posting list candidates to keep, effectively a limit.
    #[arg(long)]
    posting_candidates: NonZero<usize>,
    /// Number of posting results to keep.
    /// If unset, use the same figure as --posting-candidates
    #[arg(long)]
    posting_rerank_budget: Option<usize>,
    /// Maximum number of queries to run. If unset, run all queries in the vector file.
    #[arg(short, long)]
    limit: Option<usize>,

    #[command(flatten)]
    recall: RecallArgs,

    #[arg(long, default_value = "1")]
    warmup_iters: usize,
    #[arg(long, default_value = "2")]
    test_iters: usize,
}

pub fn search(connection: Arc<Connection>, index_name: &str, args: SearchArgs) -> io::Result<()> {
    let index = Arc::new(TableIndex::from_db(&connection, index_name)?);
    let query_vectors = easy_tiger::input::DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.query_vectors)?)? },
        index.head_config().config().dimensions,
    )?;
    let limit = std::cmp::min(
        query_vectors.len(),
        args.limit.unwrap_or(query_vectors.len()),
    );
    let search_params = SpannSearchParams {
        head_params: GraphSearchParams {
            beam_width: args.head_candidates,
            num_rerank: args
                .head_rerank_budget
                .unwrap_or(args.head_candidates.get()),
        },
        num_centroids: args.posting_centroids,
        limit: args.posting_candidates,
        num_rerank: args
            .posting_rerank_budget
            .unwrap_or(args.posting_candidates.get()),
    };
    let recall_computer =
        RecallComputer::from_args(args.recall, index.head_config().config().similarity)?;
    if let Some(computer) = recall_computer.as_ref() {
        if computer.neighbors_len() != query_vectors.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "neighbors must have the same number of rows as query_vectors ({} vs {})",
                    computer.neighbors_len(),
                    query_vectors.len()
                ),
            ));
        }
    }

    if args.warmup_iters > 0 {
        search_phase(
            "warmup",
            args.warmup_iters,
            limit,
            &query_vectors,
            &index,
            &connection,
            search_params,
            recall_computer.as_ref(),
        )?;
    }

    if args.test_iters > 0 {
        let stats = search_phase(
            "test",
            args.test_iters,
            limit,
            &query_vectors,
            &index,
            &connection,
            search_params,
            recall_computer.as_ref(),
        )?;

        println!(
            "queries {} avg duration {:0.6}s max duration {:0.6}s",
            stats.count,
            stats.total_duration.as_secs_f64() / stats.count as f64,
            stats.max_duration.as_secs_f64(),
        );
        println!(
            "head search avg candidates {:.2} avg visited {:.2}",
            stats.total_stats.head.candidates as f64 / stats.count as f64,
            stats.total_stats.head.visited as f64 / stats.count as f64
        );
        println!(
            "tail search avg postings {:.2} avg read {:.2} avg scored {:.2}",
            stats.total_stats.postings_read as f64 / stats.count as f64,
            stats.total_stats.posting_entries_read as f64 / stats.count as f64,
            stats.total_stats.posting_entries_scored as f64 / stats.count as f64
        );

        let wt_stats = WiredTigerConnectionStats::try_from(&connection)?;
        println!(
            "WT {:15} bytes read on {:12} lookups",
            wt_stats.read_bytes, wt_stats.read_ios
        );

        if let Some((computer, mean_recall)) = recall_computer.zip(stats.mean_recall()) {
            println!("{}: {:0.6}", computer.label(), mean_recall);
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn search_phase<Q: Send + Sync>(
    name: &'static str,
    iters: usize,
    limit: usize,
    query_vectors: &DerefVectorStore<f32, Q>,
    index: &Arc<TableIndex>,
    connection: &Arc<Connection>,
    search_params: SpannSearchParams,
    recall_computer: Option<&RecallComputer>,
) -> io::Result<AggregateSearchStats> {
    let query_indices = (0..limit).cycle().take(iters * limit).collect::<Vec<_>>();
    let progress = progress_bar(query_indices.len(), name);
    #[cfg(feature = "serial_search")]
    let stats: AggregateSearchStats = {
        use indicatif::ProgressIterator;
        let mut searcher = SearcherState::new(&index, &connection, search_params).unwrap();
        query_indices
            .into_iter()
            .progress_with(progress.clone())
            .map(|index| searcher.query(index, &query_vectors[index], recall_computer))
            .reduce(|a, b| match (a, b) {
                (Ok(a), Ok(b)) => Ok(a + b),
                (Ok(_), Err(b)) => Err(b),
                (Err(a), _) => Err(a),
            })
            .expect("at least one query")?
    };
    #[cfg(not(feature = "serial_search"))]
    let stats: AggregateSearchStats = {
        use rayon::prelude::*;
        query_indices
            .into_par_iter()
            .map_init(
                || SearcherState::new(index, connection, search_params).unwrap(),
                |searcher, index| {
                    let stats = searcher.query(index, &query_vectors[index], recall_computer);
                    progress.inc(1);
                    stats
                },
            )
            .try_reduce(AggregateSearchStats::default, |a, b| Ok(a + b))?
    };
    // TODO: collect and return wt stats with search stats, reseting after collection.
    progress.finish_using_style();
    Ok(stats)
}

struct SearcherState {
    reader: SessionIndexReader,
    searcher: SpannSearcher,
}

impl SearcherState {
    fn new(
        index: &Arc<TableIndex>,
        connection: &Arc<Connection>,
        search_params: SpannSearchParams,
    ) -> io::Result<Self> {
        Ok(Self {
            reader: SessionIndexReader::new(index, connection.open_session()?),
            searcher: SpannSearcher::new(search_params),
        })
    }

    fn query(
        &mut self,
        index: usize,
        query: &[f32],
        recall_computer: Option<&RecallComputer>,
    ) -> io::Result<AggregateSearchStats> {
        self.reader.session().begin_transaction(None)?;
        let start = Instant::now();
        let results = self.searcher.search(query, &mut self.reader)?;
        let duration = Instant::now() - start;
        self.reader.session().rollback_transaction(None)?;
        Ok(AggregateSearchStats::new(
            duration,
            self.searcher.stats(),
            recall_computer.map(|r| r.compute_recall(index, &results)),
        ))
    }
}

#[derive(Default)]
struct AggregateSearchStats {
    count: usize,
    total_duration: Duration,
    max_duration: Duration,
    total_stats: SpannSearchStats,
    sum_recall: Option<f64>,
}

impl AggregateSearchStats {
    fn new(duration: Duration, total_stats: SpannSearchStats, recall: Option<f64>) -> Self {
        Self {
            count: 1,
            total_duration: duration,
            max_duration: duration,
            total_stats,
            sum_recall: recall,
        }
    }

    fn mean_recall(&self) -> Option<f64> {
        self.sum_recall.map(|s| s / self.count as f64)
    }
}

impl Add<AggregateSearchStats> for AggregateSearchStats {
    type Output = AggregateSearchStats;

    fn add(self, rhs: AggregateSearchStats) -> Self::Output {
        Self {
            count: self.count + rhs.count,
            total_duration: self.total_duration + rhs.total_duration,
            max_duration: std::cmp::max(self.max_duration, rhs.max_duration),
            total_stats: self.total_stats + rhs.total_stats,
            sum_recall: self
                .sum_recall
                .zip(rhs.sum_recall)
                .map(|(a, b)| a + b)
                .or(self.sum_recall)
                .or(rhs.sum_recall),
        }
    }
}
