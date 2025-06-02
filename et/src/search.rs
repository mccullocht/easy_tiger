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
    search::{GraphSearchStats, GraphSearcher},
    wt::{SessionGraphVectorIndexReader, TableGraphVectorIndex},
    Neighbor,
};
use memmap2::Mmap;
use wt_mdb::Connection;

use crate::{ui::progress_bar, wt_stats::WiredTigerConnectionStats};

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

    /// Path buf to numpy u32 formatted neighbors file.
    /// This should include one row of length neighbors_len for each vector in query_vectors.
    #[arg(long)]
    neighbors: Option<PathBuf>,
    /// Number of neighbors for each query in the neighbors file.
    #[arg(long, default_value = "100")]
    neighbors_len: NonZero<usize>,
    /// Compute recall@k. Must be <= neighbors_len.
    #[arg(long)]
    recall_k: Option<NonZero<usize>>,

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
    let search_params = GraphSearchParams {
        beam_width: args.candidates,
        num_rerank: args.rerank_budget.unwrap_or_else(|| args.candidates.get()),
    };
    let recall_computer = if let Some((neighbors, recall_k)) = args.neighbors.zip(args.recall_k) {
        let neighbors = DerefVectorStore::<u32, _>::new(
            unsafe { Mmap::map(&File::open(neighbors)?)? },
            args.neighbors_len,
        )?;
        if neighbors.len() != query_vectors.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "neighbors must have the same number of rows as query_vectors ({} vs {})",
                    neighbors.len(),
                    query_vectors.len()
                ),
            ));
        }
        Some(RecallComputer::new(recall_k, neighbors)?)
    } else {
        None
    };

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

        if let Some((computer, recalled_count)) = recall_computer.zip(stats.total_recall_results) {
            println!(
                "recall@{} {:0.6}",
                computer.k,
                recalled_count as f64 / (stats.count * computer.k) as f64
            );
        }
    }

    Ok(())
}

fn search_phase<Q: Send + Sync, N: Send + Sync>(
    name: &'static str,
    iters: usize,
    limit: usize,
    record_limit: i64,
    query_vectors: &DerefVectorStore<f32, Q>,
    index: &Arc<TableGraphVectorIndex>,
    connection: &Arc<Connection>,
    search_params: GraphSearchParams,
    recall_computer: Option<&RecallComputer<DerefVectorStore<u32, N>>>,
) -> io::Result<AggregateSearchStats> {
    let query_indices = (0..limit)
        .into_iter()
        .cycle()
        .take(iters * limit)
        .collect::<Vec<_>>();
    let progress = progress_bar(query_indices.len(), Some(name));
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
                (Err(e), _) => Err(e),
                (_, Err(e)) => Err(3),
            })
            .expect("at least one query")?
    };
    #[cfg(not(feature = "serial_search"))]
    let stats: AggregateSearchStats = {
        use rayon::prelude::*;
        query_indices
            .into_par_iter()
            .map_init(
                || SearcherState::new(&index, &connection, search_params).unwrap(),
                |searcher, index| {
                    let stats =
                        searcher.query(index, &query_vectors[index], record_limit, recall_computer);
                    progress.inc(1);
                    stats
                },
            )
            .try_reduce(|| AggregateSearchStats::default(), |a, b| Ok(a + b))?
    };
    // TODO: collect and return wt stats with search stats, reseting after collection.
    progress.finish_using_style();
    Ok(stats)
}

struct SearcherState {
    reader: SessionGraphVectorIndexReader,
    searcher: GraphSearcher,
}

impl SearcherState {
    fn new(
        index: &Arc<TableGraphVectorIndex>,
        connection: &Arc<Connection>,
        search_params: GraphSearchParams,
    ) -> io::Result<Self> {
        Ok(Self {
            reader: SessionGraphVectorIndexReader::new(index.clone(), connection.open_session()?),
            searcher: GraphSearcher::new(search_params),
        })
    }

    fn query<N: VectorStore<Elem = u32>>(
        &mut self,
        index: usize,
        query: &[f32],
        record_limit: i64,
        recall_computer: Option<&RecallComputer<N>>,
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

pub struct RecallComputer<N> {
    k: usize,
    neighbors: N,
}

impl<N> RecallComputer<N>
where
    N: VectorStore<Elem = u32>,
{
    fn new(k: NonZero<usize>, neighbors: N) -> io::Result<Self> {
        if k.get() <= neighbors.elem_stride() {
            Ok(Self {
                k: k.get(),
                neighbors,
            })
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "recall k must be <= neighbors_len",
            ))
        }
    }

    fn compute_recall(&self, query_index: usize, query_results: &[Neighbor]) -> usize {
        let mut expected = self.neighbors[query_index][..self.k].to_vec();
        expected.sort();
        query_results
            .iter()
            .take(self.k)
            .filter(|n| expected.binary_search(&(n.vertex() as u32)).is_ok())
            .count()
    }
}

#[derive(Default)]
struct AggregateSearchStats {
    count: usize,
    total_duration: Duration,
    max_duration: Duration,
    total_graph_stats: GraphSearchStats,
    total_recall_results: Option<usize>,
}

impl AggregateSearchStats {
    fn new(duration: Duration, graph_stats: GraphSearchStats, recall: Option<usize>) -> Self {
        Self {
            count: 1,
            total_duration: duration,
            max_duration: duration,
            total_graph_stats: graph_stats,
            total_recall_results: recall,
        }
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
            total_recall_results: self
                .total_recall_results
                .zip(rhs.total_recall_results)
                .map(|(a, b)| a + b)
                .or(self.total_recall_results)
                .or(rhs.total_recall_results),
        }
    }
}
