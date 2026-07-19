use std::{collections::HashMap, fs::File, io, num::NonZero, ops::Range, path::PathBuf, sync::Arc};

use clap::Args;
use easy_tiger::{
    input::{DerefVectorStore, VectorStore},
    spann::{
        centroid_stats::CentroidStats,
        postings::CentroidPostingsMut,
        rebalance::{BalanceSummary, RebalanceStats},
        TableIndex, TransactionIndex,
    },
    vamana::search::{GraphSearchStats, GraphSearcher, Options as GraphSearchOptions},
    Neighbor,
};
use indicatif::{ParallelProgressIterator, ProgressBar};
use rand::SeedableRng;
use rayon::prelude::*;
use vectors::F32VectorCoder;
use wt_mdb::{Connection, Error, Result};

use crate::{ui::progress_bar, wt_stats::WiredTigerWriteStats};

#[derive(Args)]
pub struct InsertVectorsArgs {
    /// Path to the input vectors to insert.
    #[arg(short, long)]
    f32_vectors: PathBuf,

    /// Index of the first vector to insert.
    #[arg(long, default_value_t = 0)]
    start: usize,

    /// Number of vectors to insert.
    #[arg(short, long)]
    count: NonZero<usize>,

    /// Number of vectors to insert in each transaction batch.
    #[arg(long, default_value_t = NonZero::new(256).unwrap())]
    batch_size: NonZero<usize>,

    /// Random seed used for clustering computations.
    /// Use a fixed value for repeatability.
    #[arg(long, default_value_t = 0x7774_7370414E4E)]
    seed: u64,
}

pub fn insert_vectors(
    connection: Arc<Connection>,
    index_name: &str,
    args: InsertVectorsArgs,
) -> io::Result<()> {
    let index = Arc::new(TableIndex::from_db(&connection, index_name)?);

    // Map the input vectors.
    let f32_vectors = DerefVectorStore::new(
        unsafe { memmap2::Mmap::map(&File::open(args.f32_vectors)?)? },
        index.head_config().config().dimensions,
    )?;
    // Advise random access since we might be jumping around (though sequentially in patches).
    // Actually, we are iterating sequentially, so Sequential is probably better for the main loop,
    // but the `SubsetView` might complicate things if we use it.
    // For now, let's stick with simple iteration.
    f32_vectors.data().advise(memmap2::Advice::Sequential)?;

    let end = args.start + args.count.get();
    if end > f32_vectors.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "requested range {}..{} exceeds vector file length {}",
                args.start,
                end,
                f32_vectors.len()
            ),
        ));
    }

    let posting_format = index.config().posting_coder;
    let similarity = index.head_config().config().similarity;
    let posting_coder = posting_format.coder(similarity, None);
    let rerank_coder = index
        .config()
        .rerank_format
        .map(|f| f.coder(similarity, None));

    let batch_size = args.batch_size.get();
    let main_progress = progress_bar(args.count.get(), "inserting vectors");

    let wt_stats_before = WiredTigerWriteStats::try_from(&connection)?;

    let mut batches: usize = 0;
    let mut total_batch_unique_centroids: usize = 0;
    let mut rebalance_stats = RebalanceStats::default();
    let mut rebalance_iters = 0;
    let mut search_stats = GraphSearchStats::default();
    let mut insert_time = std::time::Duration::ZERO;
    let mut rebalance_time = std::time::Duration::ZERO;
    let mut prepare_time = std::time::Duration::ZERO;
    let mut apply_time = std::time::Duration::ZERO;

    for batch_start in (args.start..end).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(end);

        let insert_start = std::time::Instant::now();
        let batch_result = insert_batch(
            &index,
            &connection,
            &f32_vectors,
            batch_start..batch_end,
            posting_coder.as_ref(),
            rerank_coder.as_ref().map(|c| c.as_ref()),
            &main_progress,
        )?;
        insert_time += insert_start.elapsed();
        total_batch_unique_centroids += batch_result.unique_centroids;
        search_stats += batch_result.search_stats;
        prepare_time += batch_result.prepare_time;
        apply_time += batch_result.apply_time;
        batches += 1;

        main_progress.set_message("rebalancing index");
        let rebalance_start = std::time::Instant::now();
        let (s, iters) = rebalance(&index, &connection, args.seed)?;
        rebalance_time += rebalance_start.elapsed();
        rebalance_stats += s;
        rebalance_iters += iters;
        main_progress.set_message("inserting vectors");
    }

    main_progress.finish();
    let queries = args.count.get() as f64;
    let total_time = insert_time + rebalance_time;
    println!("Wall time:");
    println!(
        "  Insert:           {:10.2} s ({:5.1}%)",
        insert_time.as_secs_f64(),
        if total_time.is_zero() {
            0.0
        } else {
            insert_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
        }
    );
    println!(
        "    Prepare:        {:10.2} s ({:5.1}%)",
        prepare_time.as_secs_f64(),
        if total_time.is_zero() {
            0.0
        } else {
            prepare_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
        }
    );
    println!(
        "    Apply:          {:10.2} s ({:5.1}%)",
        apply_time.as_secs_f64(),
        if total_time.is_zero() {
            0.0
        } else {
            apply_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
        }
    );
    println!(
        "  Rebalance:        {:10.2} s ({:5.1}%)",
        rebalance_time.as_secs_f64(),
        if total_time.is_zero() {
            0.0
        } else {
            rebalance_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
        }
    );
    let pd = rebalance_stats.phase_durations;
    let phase = |label: &str, d: std::time::Duration| {
        println!(
            "    {:<16}{:10.2} s ({:5.1}%)",
            label,
            d.as_secs_f64(),
            if total_time.is_zero() {
                0.0
            } else {
                d.as_secs_f64() / total_time.as_secs_f64() * 100.0
            }
        );
    };
    phase("Partition:", pd.split_update_head);
    phase("Insert:", pd.insert_split_centroids);
    phase("Reassign:", pd.posting_reassignments);
    phase("Apply Reassign:", pd.apply_posting_reassignments);
    phase("Remove:", pd.remove_source_centroids);
    phase("Nearby Find:", pd.select_nearby_centroids);
    phase("Nearby Select:", pd.compute_nearby_reassignments);
    phase("Nearby Apply:", pd.apply_nearby_reassignments);
    println!("Batches:        {:10}", batches);
    if batches > 0 {
        println!(
            "  Avg Unique:   {:10.1}",
            total_batch_unique_centroids as f64 / batches as f64
        );
        println!(
            "  Avg Balance:  {:10.1}",
            rebalance_iters as f64 / batches as f64
        );
    }
    println!("Insert Search:");
    println!("  Total:        {:10}", queries);
    println!(
        "  Candidates:   {:10.1}",
        search_stats.candidates as f64 / queries
    );
    println!(
        "  Cand added:   {:10.1}",
        search_stats.candidates_added as f64 / queries
    );
    println!(
        "  Visited:      {:10.1}",
        search_stats.visited as f64 / queries
    );
    println!(
        "  Filtered:     {:10.1}",
        search_stats.filtered as f64 / queries
    );
    println!(
        "  Skipped:      {:10.1}",
        search_stats.skipped as f64 / queries
    );
    println!("Merged:         {:10}", rebalance_stats.merged);
    if rebalance_stats.merged > 0 {
        println!(
            "  Moved:        {:10}",
            rebalance_stats.merge_stats.moved_vectors
        );
        println!(
            "  Avg Unique:   {:10.1}",
            rebalance_stats.merge_stats.unique_centroids as f64 / rebalance_stats.merged as f64
        );
    }
    println!("Split:          {:10}", rebalance_stats.split);
    if rebalance_stats.split > 0 {
        println!(
            "  Moved:        {:10}",
            rebalance_stats.split_stats.moved_vectors
        );
        println!(
            "  Searches:     {:10}",
            rebalance_stats.split_stats.searches
        );
        println!("Nearby:");
        println!(
            "  Seen:         {:10}",
            rebalance_stats.split_stats.nearby_seen
        );
        println!(
            "  Moved:        {:10}",
            rebalance_stats.split_stats.nearby_moved
        );
    }
    let rebalance_searches =
        (rebalance_stats.merge_stats.moved_vectors + rebalance_stats.split_stats.searches) as f64;
    if rebalance_searches > 0.0 {
        let s = rebalance_stats.search_stats;
        println!("Rebalance Search:");
        println!("  Total:        {:10}", rebalance_searches);
        println!(
            "  Candidates:   {:10.1}",
            s.candidates as f64 / rebalance_searches
        );
        println!(
            "  Cand added:   {:10.1}",
            s.candidates_added as f64 / rebalance_searches
        );
        println!(
            "  Visited:      {:10.1}",
            s.visited as f64 / rebalance_searches
        );
        println!(
            "  Filtered:     {:10.1}",
            s.filtered as f64 / rebalance_searches
        );
        println!(
            "  Skipped:      {:10.1}",
            s.skipped as f64 / rebalance_searches
        );
    }

    let wt_stats = WiredTigerWriteStats::try_from(&connection)? - wt_stats_before;
    println!("WiredTiger Stats");
    println!("  Log:          {:10} MB", wt_stats.log_bytes / (1 << 20));
    println!("  Data:         {:10} MB", wt_stats.data_bytes / (1 << 20));
    println!(
        "  Insert:       {:10} MB",
        wt_stats.insert_bytes / (1 << 20)
    );
    println!(
        "  Update:       {:10} MB",
        wt_stats.update_bytes / (1 << 20)
    );
    println!(
        "  Remove:       {:10} MB",
        wt_stats.remove_bytes / (1 << 20)
    );
    println!(
        "  Modify in:    {:10} MB",
        wt_stats.modify_bytes / (1 << 20)
    );
    println!(
        "  Modify out:   {:10} MB",
        wt_stats.modify_bytes_touch / (1 << 20)
    );
    println!("  CC conflict:  {:10}", wt_stats.txn_update_conflicts);

    Ok(())
}

/// A single insert
#[derive(Debug, Clone)]
struct InsertRecord {
    record_id: i64,
    assignment: u32,
    posting_vector: Vec<u8>,
    rerank_vector: Option<Vec<u8>>,
    stats: GraphSearchStats,
}

/// Batch of vectors for insertion complete with encoded vectors and assignments.
#[derive(Debug, Default, Clone)]
struct PreparedInsertBatch {
    postings: HashMap<u32, Vec<InsertRecord>>,
    stats: GraphSearchStats,
}

impl PreparedInsertBatch {
    fn add(&mut self, record: InsertRecord) {
        self.stats += record.stats;
        self.postings
            .entry(record.assignment)
            .or_default()
            .push(record);
    }

    fn merge(&mut self, other: Self) {
        self.stats += other.stats;
        for (c, p) in other.postings {
            self.postings.entry(c).or_default().extend(p.into_iter())
        }
    }
}

/// Result of inserting a single batch of vectors.
struct InsertBatchResult {
    unique_centroids: usize,
    search_stats: GraphSearchStats,
    /// Time spent producing the prepared batch (search + encoding).
    prepare_time: std::time::Duration,
    /// Time spent applying the prepared batch to the index.
    apply_time: std::time::Duration,
}

fn insert_batch(
    index: &Arc<TableIndex>,
    connection: &Arc<Connection>,
    f32_vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    batch: Range<usize>,
    posting_coder: &dyn F32VectorCoder,
    rerank_coder: Option<&dyn F32VectorCoder>,
    progress: &ProgressBar,
) -> Result<InsertBatchResult> {
    progress.set_message("inserting vectors");

    let prepare_start = std::time::Instant::now();
    let mut prepared_batch = batch
        .clone()
        .into_par_iter()
        .progress_with(progress.clone())
        .map_init(
            || {
                let txn_idx = TransactionIndex::new(
                    index,
                    connection
                        .begin_transaction(None)
                        .expect("begin transaction"),
                );
                let params = index.config().head_search_params;
                let result_scratch: Vec<Neighbor> = Vec::with_capacity(params.beam_width.get());
                let searcher = GraphSearcher::new(params);
                (txn_idx, searcher, result_scratch)
            },
            |(txn_idx, searcher, result_scratch), i| {
                let vector = &f32_vectors[i];

                // Search for centroid
                let mut candidates = searcher.search_with_options(
                    vector,
                    GraphSearchOptions::default()
                        .with_result_scratch(std::mem::take(result_scratch)),
                    txn_idx.head(),
                )?;
                assert!(!candidates.is_empty());
                let centroid_id = candidates[0].vertex() as u32;
                std::mem::swap(result_scratch, &mut candidates); // return scratch

                Ok::<_, Error>(InsertRecord {
                    record_id: i as i64,
                    assignment: centroid_id,
                    posting_vector: posting_coder.encode(vector),
                    rerank_vector: rerank_coder.map(|c| c.encode(vector)),
                    stats: searcher.stats(),
                })
            },
        )
        .try_fold(
            || PreparedInsertBatch::default(),
            |mut b, r| {
                b.add(r?);
                Ok::<_, Error>(b)
            },
        )
        .try_reduce(
            || PreparedInsertBatch::default(),
            |mut a, b| {
                a.merge(b);
                Ok(a)
            },
        )?;

    let prepare_time = prepare_start.elapsed();

    let unique_centroids = prepared_batch.postings.len();
    let search_stats = prepared_batch.stats;

    let apply_start = std::time::Instant::now();
    std::mem::take(&mut prepared_batch.postings)
        .into_par_iter()
        .try_for_each(|(centroid, postings)| {
            let txn_idx = TransactionIndex::new(index, connection.begin_transaction(None)?);
            let mut postings_mut = CentroidPostingsMut::from_txn(&txn_idx)?;
            let mut rerank_cursor = if rerank_coder.is_some() {
                Some(
                    txn_idx
                        .transaction()
                        .open_cursor::<i64, Vec<u8>>(index.raw_vectors_table_name())?,
                )
            } else {
                None
            };

            for r in postings {
                postings_mut.insert(centroid, r.record_id, &r.posting_vector)?;

                if let Some((cursor, vector)) = rerank_cursor.as_mut().zip(r.rerank_vector) {
                    cursor.set(r.record_id, &vector)?;
                }
            }

            postings_mut.flush()?;
            drop(postings_mut);
            drop(rerank_cursor);

            txn_idx.commit(None)
        })?;
    let apply_time = apply_start.elapsed();

    Ok(InsertBatchResult {
        unique_centroids,
        search_stats,
        prepare_time,
        apply_time,
    })
}

fn rebalance(
    index: &Arc<TableIndex>,
    connection: &Arc<Connection>,
    rng_seed: u64,
) -> Result<(RebalanceStats, usize)> {
    let mut rebalance_stats = RebalanceStats::default();
    let mut iters = 0;
    loop {
        // Need a new transaction for rebalancing steps
        let txn_idx = TransactionIndex::new(index, connection.begin_transaction(None)?);

        let stats = CentroidStats::from_index_stats(&txn_idx)?;
        let summary = BalanceSummary::new(&stats, index.config().centroid_len_range());

        if summary
            .below_exemplar()
            .or(summary.above_exemplar())
            .is_some()
        {
            rebalance_stats +=
                easy_tiger::spann::rebalance::parallel_rebalance(connection, index, &|| {
                    rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(rng_seed)
                })?;
            iters += 1;
        } else {
            break;
        }
    }

    Ok((rebalance_stats, iters))
}
