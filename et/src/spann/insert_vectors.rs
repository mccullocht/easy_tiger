use std::{collections::HashSet, fs::File, io, num::NonZero, ops::Range, path::PathBuf, sync::Arc};

use clap::Args;
use easy_tiger::{
    input::{DerefVectorStore, VectorStore},
    spann::{
        CentroidAssignment, TableIndex, TransactionIndex,
        centroid_stats::{CentroidAssignmentUpdater, CentroidStats},
        postings::BlockPostingsMut,
        rebalance::{BalanceSummary, RebalanceStats, merge_centroid},
    },
    vamana::search::GraphSearcher,
};
use indicatif::{ParallelProgressIterator, ProgressBar};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
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

    /// Number of vectors to insert. If unset, insert all of the input vectors.
    #[arg(short, long)]
    count: Option<NonZero<usize>>,

    /// Number of vectors to insert in each transaction batch.
    #[arg(long, default_value_t = NonZero::new(256).unwrap())]
    batch_size: NonZero<usize>,

    /// If true, skip the "bottom half" of centroid split -- move some vectors out and perform
    /// reassignment on nearby centroids. This produces a higher quality index but is more
    /// expensive.
    #[arg(long, default_value_t = false)]
    split_no_bottom_half: bool,

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

    let end = args.start + args.count.map(|c| c.get()).unwrap_or(f32_vectors.len());
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

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(args.seed);

    let posting_format = index.config().posting_coder;
    let similarity = index.head_config().config().similarity;
    let posting_coder = posting_format.coder(similarity, None);
    let rerank_coder = index
        .config()
        .rerank_format
        .map(|f| f.coder(similarity, None));

    let batch_size = args.batch_size.get();
    let main_progress = progress_bar(end - args.start, "inserting vectors");

    let wt_stats_before = WiredTigerWriteStats::try_from(&connection)?;

    let mut rebalance_stats = RebalanceStats::default();
    let mut batches: usize = 0;
    let mut total_batch_unique_centroids: usize = 0;

    for batch_start in (args.start..end).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(end);

        total_batch_unique_centroids += insert_batch(
            &index,
            &connection,
            &f32_vectors,
            batch_start..batch_end,
            posting_coder.as_ref(),
            rerank_coder.as_ref().map(|c| c.as_ref()),
            &main_progress,
        )?;
        batches += 1;

        rebalance_stats += rebalance(
            &index,
            &connection,
            !args.split_no_bottom_half,
            &mut rng,
            &main_progress,
        )?;
    }

    main_progress.set_message("inserting vectors");
    main_progress.finish();
    println!("Batches:        {:10}", batches);
    if batches > 0 {
        println!(
            "  Avg Unique:   {:10.1}",
            total_batch_unique_centroids as f64 / batches as f64
        );
    }
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
        println!(
            "  Avg unique:   {:10.1}",
            rebalance_stats.split_stats.unique_centroids as f64 / rebalance_stats.split as f64
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

    let wt_stats = WiredTigerWriteStats::try_from(&connection)? - wt_stats_before;
    println!("WT log:         {:10} MB", wt_stats.log_bytes / (1 << 20));
    println!("WT data:        {:10} MB", wt_stats.data_bytes / (1 << 20));
    println!(
        "WT insert:      {:10} MB",
        wt_stats.insert_bytes / (1 << 20)
    );
    println!(
        "WT update:      {:10} MB",
        wt_stats.update_bytes / (1 << 20)
    );
    println!(
        "WT remove:      {:10} MB",
        wt_stats.remove_bytes / (1 << 20)
    );
    println!(
        "WT modify in:   {:10} MB",
        wt_stats.modify_bytes / (1 << 20)
    );
    println!(
        "WT modify out:  {:10} MB",
        wt_stats.modify_bytes_touch / (1 << 20)
    );

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
) -> Result<usize> {
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
                ))
            },
        )
        .collect::<Result<Vec<_>>>()?;

    let unique_centroids = vector_state
        .iter()
        .map(|(_, a, _, _)| a.primary_id)
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

    for (i, assignment, posting_vector, rerank_vector) in vector_state.into_iter() {
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
    Ok(unique_centroids)
}

fn rebalance(
    index: &Arc<TableIndex>,
    connection: &Arc<Connection>,
    split_bottom_half: bool,
    rng: &mut impl Rng,
    progress: &ProgressBar,
) -> Result<RebalanceStats> {
    let mut iter = 1;
    let mut rebalance_stats = RebalanceStats::default();
    loop {
        // Need a new transaction for rebalancing steps
        let txn_idx = TransactionIndex::new(index, connection.begin_transaction(None)?);

        let stats = CentroidStats::from_index_stats(&txn_idx)?;
        let summary = BalanceSummary::new(&stats, index.config().centroid_len_range());

        match (summary.below_exemplar(), summary.above_exemplar()) {
            (Some((to_merge, len)), _) if summary.total_clusters() > 1 => {
                progress.set_message(format!("merge {to_merge} of {len} ({iter})"));
                rebalance_stats += merge_centroid(&txn_idx, to_merge, len)?;
                txn_idx.commit(None)?;
            }
            (_, Some((to_split, len))) => {
                progress.set_message(format!("split {to_split} of {len} ({iter})"));
                drop(txn_idx);

                let mut centroid_split = easy_tiger::spann::rebalance::split_centroid_top_half(
                    connection,
                    index,
                    to_split as u32,
                    rng,
                )?;
                rebalance_stats += centroid_split.stats;
                if split_bottom_half {
                    rebalance_stats += easy_tiger::spann::rebalance::split_centroid_bottom_half(
                        connection,
                        index,
                        std::mem::take(&mut centroid_split.targets[0]),
                    )?;
                    rebalance_stats += easy_tiger::spann::rebalance::split_centroid_bottom_half(
                        connection,
                        index,
                        std::mem::take(&mut centroid_split.targets[1]),
                    )?;
                }
            }
            _ => break,
        }

        iter += 1;
    }

    Ok(rebalance_stats)
}
