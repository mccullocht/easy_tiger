use std::{
    fs::File,
    io,
    num::NonZero,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use clap::Args;
use easy_tiger::{
    input::{DerefVectorStore, VectorStore},
    spann::{
        CentroidAssignment, TableIndex, TransactionIndex,
        centroid_stats::CentroidAssignmentUpdater,
        postings::BlockPostingsMut,
        rebalance::{RebalanceStats, split_centroid_bottom_half, split_centroid_top_half},
    },
    vamana::search::GraphSearcher,
};
use indicatif::ParallelProgressIterator;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use vectors::F32VectorCoder;
use wt_mdb::{Connection, Error, Result, WiredTigerError, session::Formatted};

use crate::ui::progress_bar;

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

    let similarity = index.head_config().config().similarity;
    let posting_coder = index.config().posting_coder.coder(similarity, None);
    let rerank_coder = index
        .config()
        .rerank_format
        .map(|f| f.coder(similarity, None));
    let max_centroid_len = index.config().max_centroid_len;

    let main_progress = progress_bar(args.count.get(), "inserting vectors");

    // Each rayon worker gets its own rng seeded from a shared counter so splits use distinct
    // random state without requiring a Sync rng.
    let seed = AtomicU64::new(args.seed);
    let rebalance_stats = (args.start..end)
        .into_par_iter()
        .progress_with(main_progress.clone())
        .map_init(
            || {
                (
                    GraphSearcher::new(index.config().head_search_params),
                    Xoshiro256PlusPlus::seed_from_u64(seed.fetch_add(1, Ordering::Relaxed)),
                )
            },
            |(searcher, rng), i| {
                insert_one(
                    &connection,
                    &index,
                    &f32_vectors,
                    i,
                    posting_coder.as_ref(),
                    rerank_coder.as_deref(),
                    searcher,
                    rng,
                    max_centroid_len,
                )
            },
        )
        .try_reduce(RebalanceStats::default, |a, b| Ok(a + b))?;

    main_progress.set_message("inserting vectors");
    main_progress.finish();

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

    Ok(())
}

/// Insert a single vector, splitting overfull centroids as needed and retrying until the
/// vector lands in a centroid with room.
#[allow(clippy::too_many_arguments)]
fn insert_one(
    connection: &Arc<Connection>,
    index: &Arc<TableIndex>,
    f32_vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    i: usize,
    posting_coder: &dyn F32VectorCoder,
    rerank_coder: Option<&dyn F32VectorCoder>,
    searcher: &mut GraphSearcher,
    rng: &mut impl Rng,
    max_centroid_len: usize,
) -> Result<RebalanceStats> {
    let posting_vector = posting_coder.encode(&f32_vectors[i]);
    let rerank_vector = rerank_coder.map(|c| c.encode(&f32_vectors[i]));
    let mut stats = RebalanceStats::default();

    loop {
        match try_insert(
            connection,
            index,
            f32_vectors,
            i,
            &posting_vector,
            rerank_vector.as_deref(),
            searcher,
            max_centroid_len,
        )? {
            // The vector was inserted; we're done.
            None => return Ok(stats),
            // The chosen centroid is full; split it then retry the insert.
            Some(centroid_id) => {
                // NOT_FOUND from either split half means another worker has already
                // split/merged this centroid. That's fine: swallow it and retry the insert,
                // which will re-search for the now-current centroid.
                match split_centroid_top_half(connection, index, centroid_id, rng) {
                    Ok(split) => {
                        stats += split.stats;
                        for target in split.targets {
                            match split_centroid_bottom_half(connection, index, target) {
                                Ok(s) => stats.split_stats += s,
                                Err(e) if e == Error::not_found_error() => {}
                                Err(e) => return Err(e),
                            }
                        }
                    }
                    Err(e) if e == Error::not_found_error() => {}
                    Err(e) => return Err(e),
                }
            }
        }
    }
}

/// Attempt to insert vector `i` in a single transaction, retrying on OCC rollback.
///
/// Returns `Ok(None)` if the vector was inserted and the transaction committed. Returns
/// `Ok(Some(centroid_id))` if inserting would overflow `centroid_id`; in that case the
/// transaction is rolled back and the caller should split the centroid before retrying.
#[allow(clippy::too_many_arguments)]
fn try_insert(
    connection: &Arc<Connection>,
    index: &Arc<TableIndex>,
    f32_vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    i: usize,
    posting_vector: &[u8],
    rerank_vector: Option<&[u8]>,
    searcher: &mut GraphSearcher,
    max_centroid_len: usize,
) -> Result<Option<u32>> {
    loop {
        let txn_idx = TransactionIndex::new(index, connection.begin_transaction(None)?);
        // Perform all the reads and writes for this attempt; the transaction is committed
        // (or rolled back by drop) based on the result below.
        let result: Result<Option<u32>> = (|| {
            let candidates = searcher.search(&f32_vectors[i], txn_idx.head())?;
            assert!(!candidates.is_empty());

            // TODO: implement replica selection.
            let centroid_id = candidates[0].vertex() as u32;

            let mut postings = BlockPostingsMut::from_txn(&txn_idx)?;
            if postings.centroid_len(centroid_id)? >= max_centroid_len {
                return Ok(Some(centroid_id));
            }

            postings.insert(centroid_id, i as i64, posting_vector)?;

            let mut assignment_updater = CentroidAssignmentUpdater::new(&txn_idx)?;
            assignment_updater.insert(
                i as i64,
                CentroidAssignment::new(centroid_id, &[]).to_formatted_ref(),
            )?;

            if let Some(vector) = rerank_vector {
                txn_idx
                    .transaction()
                    .open_cursor::<i64, Vec<u8>>(index.raw_vectors_table_name())?
                    .set(i as i64, vector)?;
            }

            postings.flush()?;
            assignment_updater.flush()?;
            drop(assignment_updater);
            drop(postings);
            Ok(None)
        })();

        match result {
            // Retry the whole attempt on a fresh transaction.
            Err(Error::WiredTiger(WiredTigerError::Rollback)) => continue,
            // Centroid is full: drop (rollback) the transaction and signal a split.
            Ok(Some(centroid_id)) => return Ok(Some(centroid_id)),
            // Inserted: commit.
            Ok(None) => {
                txn_idx.commit(None)?;
                return Ok(None);
            }
            Err(e) => return Err(e),
        }
    }
}
