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
    input::{DerefVectorStore, VectorStore},
    vamana::{
        search::{GraphSearchStats, GraphSearcher},
        wt::{SessionGraphVectorIndex, TableGraphVectorIndex},
        GraphSearchParams, PatienceParams,
    },
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
    /// Maximum record number to consider as a valid result. Used for filtering testing.
    #[arg(long)]
    record_limit: Option<usize>,

    /// Patience saturation threshold.
    ///
    /// During each search round fewer than this fraction of candidates must change. If this
    /// threshold is exceeded --patience-saturation-count consecutive times then the search will be
    /// terminated.
    #[arg(long, default_value_t = 0.995)]
    patience_saturation_threshold: f64,
    /// Patience saturation count.
    ///
    /// If unset, patience early termination will not be used.
    #[arg(long)]
    patience_saturation_count: Option<usize>,

    #[command(flatten)]
    recall: RecallArgs,

    #[arg(long, default_value = "1")]
    warmup_iters: usize,
    #[arg(long, default_value = "2")]
    test_iters: usize,
}

pub fn search(connection: Arc<Connection>, index_name: &str, args: SearchArgs) -> io::Result<()> {
    let index = Arc::new(TableGraphVectorIndex::from_db(&connection, index_name)?);
    let query_vectors = easy_tiger::input::DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.query_vectors)?)? },
        index.config().dimensions,
    )?;
    let limit = std::cmp::min(
        query_vectors.len(),
        args.limit.unwrap_or(query_vectors.len()),
    );
    let record_limit = args.record_limit.map(|l| l as i64).unwrap_or(i64::MAX);
    let patience = args.patience_saturation_count.map(|c| PatienceParams {
        saturation_threshold: args.patience_saturation_threshold,
        patience_count: c,
    });
    let search_params = GraphSearchParams {
        beam_width: args.candidates,
        num_rerank: args.rerank_budget.unwrap_or_else(|| args.candidates.get()),
        patience,
    };
    let recall_computer = RecallComputer::from_args(args.recall, index.config().similarity)?;
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
            record_limit,
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
            record_limit,
            &query_vectors,
            &index,
            &connection,
            search_params,
            recall_computer.as_ref(),
        )?;

        println!(
            "queries {} avg duration {:0.6}s max duration {:0.6}s  avg candidates {:.2} avg visited {:.2}",
            stats.count,
            stats.total_duration.as_secs_f64() / stats.count as f64,
            stats.max_duration.as_secs_f64(),
            stats.total_graph_stats.candidates as f64 / stats.count as f64,
            stats.total_graph_stats.visited as f64 / stats.count as f64,
        );

        let wt_stats = WiredTigerConnectionStats::try_from(&connection)?;
        println!(
            "WT cache hit rate {:5.2}% ({} reads, {} lookups); {:15} bytes read",
            (wt_stats.search_calls - wt_stats.read_ios) as f64 * 100.0
                / wt_stats.search_calls as f64,
            wt_stats.read_ios,
            wt_stats.search_calls,
            wt_stats.read_bytes,
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
    record_limit: i64,
    query_vectors: &DerefVectorStore<f32, Q>,
    index: &Arc<TableGraphVectorIndex>,
    connection: &Arc<Connection>,
    search_params: GraphSearchParams,
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
            .map(|index| {
                searcher.query(index, &query_vectors[index], record_limit, recall_computer)
            })
            .reduce(|a, b| match (a, b) {
                (Ok(a), Ok(b)) => Ok(a + b),
                (Err(e), _) => Err(e),
                (_, Err(e)) => Err(e),
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
                    let stats =
                        searcher.query(index, &query_vectors[index], record_limit, recall_computer);
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
    reader: SessionGraphVectorIndex,
    searcher: GraphSearcher,
}

impl SearcherState {
    fn new(
        index: &Arc<TableGraphVectorIndex>,
        connection: &Arc<Connection>,
        search_params: GraphSearchParams,
    ) -> io::Result<Self> {
        Ok(Self {
            reader: SessionGraphVectorIndex::new(index.clone(), connection.open_session()?),
            searcher: GraphSearcher::new(search_params),
        })
    }

    fn query(
        &mut self,
        index: usize,
        query: &[f32],
        record_limit: i64,
        recall_computer: Option<&RecallComputer>,
    ) -> io::Result<AggregateSearchStats> {
        self.reader.session().begin_transaction(None)?;
        let start = Instant::now();
        let results =
            self.searcher
                .search_with_filter(query, |i| i < record_limit, &mut self.reader)?;
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
    total_graph_stats: GraphSearchStats,
    sum_recall: Option<f64>,
}

impl AggregateSearchStats {
    fn new(duration: Duration, graph_stats: GraphSearchStats, recall: Option<f64>) -> Self {
        Self {
            count: 1,
            total_duration: duration,
            max_duration: duration,
            total_graph_stats: graph_stats,
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
            total_graph_stats: self.total_graph_stats + rhs.total_graph_stats,
            sum_recall: self
                .sum_recall
                .zip(rhs.sum_recall)
                .map(|(a, b)| a + b)
                .or(self.sum_recall)
                .or(rhs.sum_recall),
        }
    }
}
