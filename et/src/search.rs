use std::{
    fmt::Display,
    fs::File,
    io::{self},
    num::NonZero,
    ops::Index,
    path::PathBuf,
    sync::Arc,
};

use clap::Args;
use easy_tiger::{
    graph::GraphSearchParams,
    search::GraphSearcher,
    worker_pool::WorkerPool,
    wt::{WiredTigerGraphVectorIndex, WiredTigerGraphVectorIndexReader},
    Neighbor,
};
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use stable_deref_trait::StableDeref;
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

pub fn search(
    connection: Arc<Connection>,
    index: WiredTigerGraphVectorIndex,
    args: SearchArgs,
) -> io::Result<()> {
    let query_vectors = easy_tiger::input::NumpyF32VectorStore::new(
        unsafe { Mmap::map(&File::open(args.query_vectors)?)? },
        index.metadata().dimensions,
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
        let neighbors = NumpyU32Neighbors::new(
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
    let progress = ProgressBar::new(limit as u64)
        .with_style(
            ProgressStyle::default_bar()
                .template("{wide_bar} {pos}/{len} ETA: {eta_precise} Elapsed: {elapsed_precise}")
                .unwrap(),
        )
        .with_finish(indicatif::ProgressFinish::AndLeave);
    for (i, query) in query_vectors.iter().enumerate().take(limit) {
        session.begin_transaction(None)?;
        let mut reader = WiredTigerGraphVectorIndexReader::new(
            index.clone(),
            session,
            pool.as_ref().map(WorkerPool::clone),
        );
        let results = searcher.search_with_concurrency(query, &mut reader, args.concurrency)?;
        assert_ne!(results.len(), 0);
        recall_computer.as_mut().map(|r| r.add_results(i, &results));
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

    if let Some(computer) = recall_computer {
        println!("{}", computer);
    }

    Ok(())
}

// XXX maybe also do vertex_visited/vertex_nav_scored counters?

pub struct RecallComputer {
    k: usize,
    neighbors: NumpyU32Neighbors<Mmap>,

    queries: usize,
    total: usize,
    matched: usize,
    expected_buf: Vec<u32>,
}

impl RecallComputer {
    fn new(k: NonZero<usize>, neighbors: NumpyU32Neighbors<Mmap>) -> io::Result<Self> {
        if k <= neighbors.neighbors_len() {
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

impl Display for RecallComputer {
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

/// Immutable store for numpy formatted u32 neighbor data.
///
/// In this format all i32 values are little endian coded and written end-to-end. The number of
/// neighbors contained in each row must be provided externally.
/// count must be provided externally.
// TODO: existing data files are u32 formatted, but maybe I should generate i64 value files?
// TODO: merge this with NumpyF32VectorStore -- could probably template on the elem type?
pub struct NumpyU32Neighbors<D> {
    // NB: the contents of data is referenced by vectors.
    #[allow(dead_code)]
    data: D,
    neighbors_len: NonZero<usize>,
    neighbors: &'static [u32],
}

impl<D> NumpyU32Neighbors<D>
where
    D: Send + Sync,
{
    /// Create a new store for numpy neighbor store with the given input and neighbors count.
    ///
    /// This will typically be used with a memory-mapped file.
    pub fn new(data: D, neighbors_len: NonZero<usize>) -> io::Result<Self>
    where
        D: StableDeref<Target = [u8]>,
    {
        let vectorp = data.as_ptr() as *const u32;
        if !vectorp.is_aligned() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "input neighbor data not aligned to u32".to_string(),
            ));
        }
        if data.len() % (std::mem::size_of::<u32>() * neighbors_len.get()) != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "input neighbor data does not divide evenly into stride length of {}",
                    std::mem::size_of::<f32>() * neighbors_len.get()
                ),
            ));
        }

        // Safety: StableDeref guarantees the pointer is stable even after a move.
        let neighbors: &'static [u32] =
            unsafe { std::slice::from_raw_parts(vectorp, data.len() / std::mem::size_of::<u32>()) };
        Ok(Self {
            data,
            neighbors_len,
            neighbors,
        })
    }

    /// Return number of neighbors in each row.
    pub fn neighbors_len(&self) -> NonZero<usize> {
        self.neighbors_len
    }

    /// Return the number of neighbors in the store.
    pub fn len(&self) -> usize {
        self.neighbors.len() / self.neighbors_len.get()
    }

    /// Return true if this store is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.neighbors.is_empty()
    }
}

impl<D> Index<usize> for NumpyU32Neighbors<D> {
    type Output = [u32];

    fn index(&self, index: usize) -> &Self::Output {
        let start = index * self.neighbors_len.get();
        let end = start + self.neighbors_len.get();
        &self.neighbors[start..end]
    }
}
