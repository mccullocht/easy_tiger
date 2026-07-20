//! Delete simulation for a SPANN index.
//!
//! For every rerank (raw) vector in the corpus this reads the vector, searches the head (centroid)
//! index with it, then reads posting lists in centroid order until it finds the posting that
//! contains the vector's own record id. The depth (number of postings that had to be read) is
//! recorded. This models the cost of locating a record for deletion, and in particular surfaces
//! records that cannot be found in any of the centroids returned by the head search.

use std::{
    io::{self},
    num::NonZero,
    sync::Arc,
};

use clap::Args;
use easy_tiger::posting_block::PostingBlock;
use easy_tiger::{
    Neighbor,
    spann::{TableIndex, TransactionIndex},
    vamana::{GraphSearchParams, PatienceParams, search::GraphSearcher},
};
use wt_mdb::Connection;

use crate::ui::progress_bar;

#[derive(Args)]
pub struct DeleteSimArgs {
    /// Number of head (centroid) candidates in the search list.
    #[arg(long)]
    head_candidates: NonZero<usize>,
    /// Number of head (centroid) results to re-rank at the end of each search.
    /// If unset, use the same figure as --head-candidates.
    #[arg(long)]
    head_rerank_budget: Option<usize>,
    /// Patience saturation threshold.
    #[arg(long, default_value_t = 0.995)]
    head_patience_saturation_threshold: f64,
    /// Patience saturation count. If unset, patience early termination will not be used.
    #[arg(long)]
    head_patience_saturation_count: Option<usize>,
    /// Maximum number of postings (centroids) to read while searching for each record.
    /// If unset, read every centroid returned by the head search.
    #[arg(long)]
    max_postings: Option<NonZero<usize>>,
    /// Maximum number of records to simulate. If unset, process every record in the corpus.
    #[arg(short, long)]
    limit: Option<usize>,
}

pub fn delete_sim(
    connection: Arc<Connection>,
    index_name: &str,
    args: DeleteSimArgs,
) -> io::Result<()> {
    let index = Arc::new(TableIndex::from_db(&connection, index_name)?);

    let head_params = GraphSearchParams {
        beam_width: args.head_candidates,
        num_rerank: args
            .head_rerank_budget
            .unwrap_or(args.head_candidates.get()),
        patience: args.head_patience_saturation_count.map(|c| PatienceParams {
            saturation_threshold: args.head_patience_saturation_threshold,
            patience_count: c,
        }),
    };
    let max_postings = args.max_postings.map(NonZero::get).unwrap_or(usize::MAX);

    // Rather than scan the whole rerank table up front, find the largest record id and derive the
    // inclusive range [0, max_key]. Record ids are assumed to be dense-ish and non-negative;
    // missing keys in the range are simply skipped (not counted as processed).
    let max_key = {
        let txn_idx = TransactionIndex::new(&index, connection.begin_transaction(None)?);
        let mut cursor = txn_idx
            .transaction()
            .open_cursor::<i64, Vec<u8>>(index.rerank_vectors_table_name())?;
        match cursor.largest_key() {
            Some(Ok(k)) => k,
            Some(Err(e)) => return Err(e.into()),
            None => {
                println!("rerank table is empty; nothing to simulate");
                return Ok(());
            }
        }
    };

    // Cap the number of candidate keys by --limit if provided.
    let last_key = match args.limit {
        Some(limit) if limit > 0 => std::cmp::min(max_key, limit as i64 - 1),
        Some(_) => return Ok(()),
        None => max_key,
    };
    let record_ids = 0..=last_key;

    let progress = progress_bar((last_key + 1) as usize, "delete-sim");

    let reduce = |a: SimStats, b: SimStats| a + b;

    let stats: SimStats = {
        use rayon::prelude::*;
        record_ids
            .into_par_iter()
            .map_init(
                || Worker::new(&index, &connection, head_params),
                |worker, record_id| {
                    let stats = worker.simulate(record_id, max_postings);
                    progress.inc(1);
                    stats
                },
            )
            .try_reduce(SimStats::default, |a, b| Ok(reduce(a, b)))?
    };

    progress.finish_using_style();
    stats.report();
    Ok(())
}

struct Worker {
    connection: Arc<Connection>,
    index: Arc<TableIndex>,
    searcher: GraphSearcher,
}

impl Worker {
    fn new(
        index: &Arc<TableIndex>,
        connection: &Arc<Connection>,
        head_params: GraphSearchParams,
    ) -> Self {
        Self {
            connection: Arc::clone(connection),
            index: Arc::clone(index),
            searcher: GraphSearcher::new(head_params),
        }
    }

    /// Search the head index for the vector belonging to `record_id`, then read postings in
    /// centroid order until the posting containing `record_id` is found. Returns the outcome as a
    /// [`SimStats`] contribution.
    fn simulate(&mut self, record_id: i64, max_postings: usize) -> io::Result<SimStats> {
        let reader = TransactionIndex::new(&self.index, self.connection.begin_transaction(None)?);

        // Read and decode this record's rerank vector to use as the query.
        let rerank_coder = self
            .index
            .config()
            .rerank_format
            .coder(self.index.head_config().config().similarity, None);
        let query: Vec<f32> = {
            let mut raw_cursor = reader
                .transaction()
                .open_record_cursor(self.index.rerank_vectors_table_name())?;
            // SAFETY: no other WT operations occur before the returned slice is decoded.
            match unsafe { raw_cursor.seek_exact_unsafe(record_id) } {
                Some(Ok(encoded)) => rerank_coder.decode(encoded),
                Some(Err(e)) => return Err(e.into()),
                // No record at this key in the range; skip it without counting it as processed.
                None => return Ok(SimStats::skipped()),
            }
        };

        let centroids: Vec<Neighbor> = self.searcher.search(&query, reader.head())?;
        if centroids.is_empty() {
            return Ok(SimStats::not_found());
        }

        let vector_len = self.index.posting_vector_len();
        let mut posting_cursor = reader
            .transaction()
            .open_cursor::<u32, Vec<u8>>(self.index.postings_table_name())?;

        for (rank, c) in centroids.iter().take(max_postings).enumerate() {
            let centroid_id: u32 = c.vertex().try_into().expect("centroid_id is a u32");
            // SAFETY: we are not performing any WT operations in between seeks.
            let data = match unsafe { posting_cursor.seek_exact_unsafe(centroid_id) } {
                Some(Ok(data)) => data,
                Some(Err(e)) => return Err(e.into()),
                None => continue,
            };
            let Some(block) = PostingBlock::new(data, vector_len) else {
                continue;
            };
            if block.lookup(record_id).is_some() {
                // depth is 1-based: number of postings read to find the record.
                return Ok(SimStats::found(rank + 1));
            }
        }

        Ok(SimStats::not_found())
    }
}

#[derive(Default, Clone)]
struct SimStats {
    /// Keys in the range that had no record and were skipped.
    skipped: usize,
    processed: usize,
    found: usize,
    not_found: usize,
    depth_sum: u64,
    max_depth: usize,
    /// Number of records found at each depth (index 0 == depth 1).
    depth_histogram: Vec<usize>,
}

impl SimStats {
    fn found(depth: usize) -> Self {
        let mut depth_histogram = vec![0; depth];
        depth_histogram[depth - 1] = 1;
        Self {
            processed: 1,
            found: 1,
            depth_sum: depth as u64,
            max_depth: depth,
            depth_histogram,
            ..Default::default()
        }
    }

    fn not_found() -> Self {
        Self {
            processed: 1,
            not_found: 1,
            ..Default::default()
        }
    }

    fn skipped() -> Self {
        Self {
            skipped: 1,
            ..Default::default()
        }
    }

    fn percentile(&self, p: f64) -> Option<usize> {
        if self.found == 0 {
            return None;
        }
        let target = (self.found as f64 * p).ceil() as usize;
        let target = target.max(1);
        let mut cumulative = 0;
        for (i, &count) in self.depth_histogram.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return Some(i + 1);
            }
        }
        Some(self.max_depth)
    }

    fn report(&self) {
        println!(
            "keys in range:     {} ({} skipped as missing)",
            self.processed + self.skipped,
            self.skipped
        );
        println!("records processed: {}", self.processed);
        println!(
            "found:             {} ({:.4}%)",
            self.found,
            100.0 * self.found as f64 / self.processed.max(1) as f64
        );
        println!(
            "NOT FOUND:         {} ({:.4}%)",
            self.not_found,
            100.0 * self.not_found as f64 / self.processed.max(1) as f64
        );
        if self.found > 0 {
            println!(
                "depth (postings read to locate record): mean {:.2} max {}",
                self.depth_sum as f64 / self.found as f64,
                self.max_depth,
            );
            for (label, p) in [("p50", 0.50), ("p90", 0.90), ("p99", 0.99)] {
                if let Some(d) = self.percentile(p) {
                    println!("  {label}: {d}");
                }
            }
        }
    }
}

impl std::ops::Add for SimStats {
    type Output = SimStats;

    fn add(mut self, rhs: SimStats) -> SimStats {
        self.skipped += rhs.skipped;
        self.processed += rhs.processed;
        self.found += rhs.found;
        self.not_found += rhs.not_found;
        self.depth_sum += rhs.depth_sum;
        self.max_depth = self.max_depth.max(rhs.max_depth);
        if self.depth_histogram.len() < rhs.depth_histogram.len() {
            self.depth_histogram.resize(rhs.depth_histogram.len(), 0);
        }
        for (slot, count) in self
            .depth_histogram
            .iter_mut()
            .zip(rhs.depth_histogram.iter())
        {
            *slot += count;
        }
        self
    }
}
