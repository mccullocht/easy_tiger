use std::{
    fmt::Display,
    fs::File,
    io::{self},
    num::NonZero,
    path::PathBuf,
    sync::Arc,
};

use clap::Args;
use easy_tiger::{
    graph::GraphSearchParams,
    input::{DerefVectorStore, VectorStore},
    search::{GraphSearchStats, GraphSearcher},
    worker_pool::WorkerPool,
    wt::{SessionGraphVectorIndexReader, TableGraphVectorIndex},
    Neighbor,
};
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use threadpool::ThreadPool;
use wt_mdb::{Connection, Result, Session};
use wt_sys::{WT_STAT_CONN_CURSOR_SEARCH, WT_STAT_CONN_READ_IO};

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
}

pub fn search(connection: Arc<Connection>, index_name: &str, args: SearchArgs) -> io::Result<()> {
    let index = TableGraphVectorIndex::from_db(&connection, index_name)?;
    let query_vectors = easy_tiger::input::DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.query_vectors)?)? },
        index.config().dimensions,
    )?;
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
    let mut recall_computer = if let Some((neighbors, recall_k)) = args.neighbors.zip(args.recall_k)
    {
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

    let index = Arc::new(index);
    let mut session = connection.open_session()?;
    let mut search_stats = GraphSearchStats::default();
    let progress = ProgressBar::new(limit as u64)
        .with_style(
            ProgressStyle::default_bar()
                .template("{wide_bar} {pos}/{len} ETA: {eta_precise} Elapsed: {elapsed_precise}")
                .unwrap(),
        )
        .with_finish(indicatif::ProgressFinish::AndLeave);
    for (i, query) in query_vectors.iter().enumerate().take(limit) {
        session.begin_transaction(None)?;
        let mut reader = SessionGraphVectorIndexReader::new(
            index.clone(),
            session,
            pool.as_ref().map(WorkerPool::clone),
        );
        let results = searcher.search_with_concurrency(query, &mut reader, args.concurrency)?;
        assert_ne!(results.len(), 0);
        search_stats += searcher.stats();
        recall_computer.as_mut().map(|r| r.add_results(i, &results));

        progress.inc(1);
        session = reader.into_session();
        session.rollback_transaction(None)?;
    }
    progress.finish_using_style();

    println!(
        "queries {} avg duration {:.3}ms avg candidates {:.2} avg visited {:.2}",
        limit,
        progress.elapsed().div_f32(limit as f32).as_micros() as f64 / 1_000f64,
        search_stats.candidates as f64 / limit as f64,
        search_stats.visited as f64 / limit as f64,
    );

    let (search_calls, read_io) = cache_hit_stats(&session)?;
    println!(
        "cache hit rate {:.2}% ({} reads, {} lookups)",
        (search_calls - read_io) as f64 * 100.0 / search_calls as f64,
        read_io,
        search_calls,
    );

    if let Some(computer) = recall_computer {
        println!("{}", computer);
    }

    Ok(())
}

pub struct RecallComputer<N> {
    k: usize,
    neighbors: N,

    queries: usize,
    total: usize,
    matched: usize,
    expected_buf: Vec<u32>,
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
                queries: 0,
                total: 0,
                matched: 0,
                expected_buf: Vec::with_capacity(k.get()),
            })
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "recall k must be <= neighbors_len",
            ))
        }
    }

    fn add_results(&mut self, query_index: usize, query_results: &[Neighbor]) {
        self.expected_buf.clear();
        self.expected_buf
            .extend_from_slice(&self.neighbors[query_index][..self.k]);
        self.expected_buf.sort();

        self.queries += 1;
        for n in query_results.iter().take(self.k) {
            self.total += 1;
            if self
                .expected_buf
                .binary_search(&(n.vertex() as u32))
                .is_ok()
            {
                self.matched += 1;
            }
        }
    }
}

impl<N> Display for RecallComputer<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "recall@k {:.5}  ({} queries; {} matched; {} total)",
            self.matched as f64 / self.total as f64,
            self.queries,
            self.matched,
            self.total
        )
    }
}

/// Count lookup calls and read IOs. This can be used to estimate cache hit rate.
fn cache_hit_stats(session: &Session) -> Result<(i64, i64)> {
    let mut stat_cursor = session.new_stats_cursor(wt_mdb::options::Statistics::Fast, None)?;
    let search_calls = stat_cursor
        .seek_exact(WT_STAT_CONN_CURSOR_SEARCH)
        .expect("WT_STAT_CONN_CURSOR_SEARCH")?;
    let read_ios = stat_cursor
        .seek_exact(WT_STAT_CONN_READ_IO)
        .expect("WT_STAT_CONN_READ_IO")?;
    Ok((search_calls, read_ios))
}
