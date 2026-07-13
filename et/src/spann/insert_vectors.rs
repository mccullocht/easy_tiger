use std::{collections::HashSet, fs::File, io, num::NonZero, ops::Range, path::PathBuf, sync::Arc};

use clap::Args;
use easy_tiger::{
    input::{DerefVectorStore, VectorStore},
    spann::{
        CentroidAssignment, TableIndex, TransactionIndex,
        centroid_stats::{CentroidAssignmentUpdater, CentroidStats},
        postings::BlockPostingsMut,
        rebalance::{BalanceSummary, RebalanceStats},
    },
    vamana::search::{GraphSearchStats, GraphSearcher},
};
use indicatif::{ParallelProgressIterator, ProgressBar};
use rand::SeedableRng;
use rayon::prelude::*;
use vectors::F32VectorCoder;
use wt_mdb::{Connection, Result};

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

    for batch_start in (args.start..end).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(end);

        let insert_start = std::time::Instant::now();
        let (unique_centroids, batch_search_stats) = insert_batch(
            &index,
            &connection,
            &f32_vectors,
            batch_start..batch_end,
            posting_coder.as_ref(),
            rerank_coder.as_ref().map(|c| c.as_ref()),
            &main_progress,
        )?;
        insert_time += insert_start.elapsed();
        total_batch_unique_centroids += unique_centroids;
        search_stats += batch_search_stats;
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
        "  Insert:       {:10.2} s ({:5.1}%)",
        insert_time.as_secs_f64(),
        if total_time.is_zero() {
            0.0
        } else {
            insert_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
        }
    );
    println!(
        "  Rebalance:    {:10.2} s ({:5.1}%)",
        rebalance_time.as_secs_f64(),
        if total_time.is_zero() {
            0.0
        } else {
            rebalance_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
        }
    );
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

fn insert_batch(
    index: &Arc<TableIndex>,
    connection: &Arc<Connection>,
    f32_vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    batch: Range<usize>,
    posting_coder: &dyn F32VectorCoder,
    rerank_coder: Option<&dyn F32VectorCoder>,
    progress: &ProgressBar,
) -> Result<(usize, GraphSearchStats)> {
    progress.set_message("inserting vectors");

    let vector_state = batch
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
                let searcher = GraphSearcher::new(index.config().head_search_params);
                (txn_idx, searcher)
            },
            |(txn_idx, searcher), i| {
                let vector = &f32_vectors[i];

                // Search for centroid
                let candidates = searcher.search(vector, txn_idx.head())?;
                assert!(!candidates.is_empty());

                let centroid_id = candidates[0].vertex() as u32;
                Ok((
                    i,
                    CentroidAssignment::new(centroid_id),
                    posting_coder.encode(vector),
                    rerank_coder.map(|c| c.encode(vector)),
                    searcher.stats(),
                ))
            },
        )
        .collect::<Result<Vec<_>>>()?;

    let search_stats = vector_state
        .iter()
        .map(|(_, _, _, _, stats)| *stats)
        .fold(GraphSearchStats::default(), |acc, stats| acc + stats);

    let unique_centroids = vector_state
        .iter()
        .map(|(_, a, _, _, _)| a.primary_id)
        .collect::<HashSet<_>>()
        .len();

    let txn_idx = TransactionIndex::new(index, connection.begin_transaction(None)?);
    let mut assignment_updater = CentroidAssignmentUpdater::new(&txn_idx)?;
    let posting_cursor = txn_idx
        .transaction()
        .open_cursor::<u32, Vec<u8>>(index.postings_table_name())?;
    let mut postings = BlockPostingsMut::new(posting_cursor, index.posting_vector_len());
    let mut rerank_cursor = if rerank_coder.is_some() {
        Some(
            txn_idx
                .transaction()
                .open_cursor::<i64, Vec<u8>>(index.raw_vectors_table_name())?,
        )
    } else {
        None
    };

    for (i, assignment, posting_vector, rerank_vector, _) in vector_state.into_iter() {
        assignment_updater.insert(i as i64, assignment)?;
        postings.insert(assignment.primary_id, i as i64, &posting_vector)?;

        if let Some((cursor, vector)) = rerank_cursor.as_mut().zip(rerank_vector) {
            cursor.set(i as i64, &vector)?;
        }
    }

    postings.flush()?;
    assignment_updater.flush()?;
    drop(assignment_updater);
    drop(postings);
    drop(rerank_cursor);
    txn_idx.commit(None)?;
    Ok((unique_centroids, search_stats))
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
