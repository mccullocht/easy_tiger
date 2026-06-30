use std::{
    fs::File,
    io,
    num::NonZero,
    ops::Add,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use clap::Args;
use easy_tiger::{
    flat::{self, search::exhaustive_search, FlatIndexConfig},
    input::{DerefVectorStore, VectorStore},
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
    /// Path to query vectors (little-endian f32).
    #[arg(short, long)]
    query_vectors: PathBuf,
    /// Number of nearest neighbors to return per query.
    #[arg(short, long)]
    k: NonZero<usize>,
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
    let config = flat::open_config(&connection, index_name)?;
    let table_name = flat::table_name(index_name);

    let query_vectors = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.query_vectors)?)? },
        config.dimensions,
    )?;
    let limit = args
        .limit
        .unwrap_or(query_vectors.len())
        .min(query_vectors.len());

    let recall_computer = RecallComputer::from_args(args.recall, config.similarity)?;
    if let Some(computer) = recall_computer.as_ref() {
        if computer.neighbors_len() < limit {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "neighbors must have at least as many rows as queries ({} vs {})",
                    computer.neighbors_len(),
                    limit,
                ),
            ));
        }
    }

    if args.warmup_iters > 0 {
        search_phase(
            "warmup",
            args.warmup_iters,
            limit,
            args.k,
            &query_vectors,
            &table_name,
            &connection,
            &config,
            recall_computer.as_ref(),
        )?;
    }

    if args.test_iters > 0 {
        let stats = search_phase(
            "test",
            args.test_iters,
            limit,
            args.k,
            &query_vectors,
            &table_name,
            &connection,
            &config,
            recall_computer.as_ref(),
        )?;

        println!(
            "queries {} avg duration {:0.6}s max duration {:0.6}s",
            stats.count,
            stats.total_duration.as_secs_f64() / stats.count as f64,
            stats.max_duration.as_secs_f64(),
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
    k: NonZero<usize>,
    query_vectors: &DerefVectorStore<f32, Q>,
    table_name: &str,
    connection: &Arc<Connection>,
    config: &FlatIndexConfig,
    recall_computer: Option<&RecallComputer>,
) -> io::Result<AggregateSearchStats> {
    use rayon::prelude::*;

    let vector_len = config
        .format
        .coder(config.similarity, None)
        .byte_len(config.dimensions.get());
    let query_indices = (0..limit).cycle().take(iters * limit).collect::<Vec<_>>();
    let progress = progress_bar(query_indices.len(), name);
    let stats = query_indices
        .into_par_iter()
        .map(|qi| {
            let query: &[f32] = &query_vectors[qi];
            let distance_fn =
                config
                    .format
                    .query_distance_asymmetric(config.similarity, query, None);
            let txn = connection.begin_transaction(None)?;
            let cursor = txn.open_record_cursor(table_name)?;

            let start = Instant::now();
            let results = exhaustive_search(cursor, vector_len, distance_fn.as_ref(), k)?;
            let duration = Instant::now() - start;
            progress.inc(1);
            Ok::<_, io::Error>(AggregateSearchStats::new(
                duration,
                recall_computer.map(|r| r.compute_recall(qi, &results)),
            ))
        })
        .try_reduce(AggregateSearchStats::default, |a, b| Ok(a + b))?;
    progress.finish_using_style();
    Ok(stats)
}

#[derive(Default)]
struct AggregateSearchStats {
    count: usize,
    total_duration: Duration,
    max_duration: Duration,
    sum_recall: Option<f64>,
}

impl AggregateSearchStats {
    fn new(duration: Duration, recall: Option<f64>) -> Self {
        Self {
            count: 1,
            total_duration: duration,
            max_duration: duration,
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
            max_duration: self.max_duration.max(rhs.max_duration),
            sum_recall: self
                .sum_recall
                .zip(rhs.sum_recall)
                .map(|(a, b)| a + b)
                .or(self.sum_recall)
                .or(rhs.sum_recall),
        }
    }
}
