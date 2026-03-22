use std::{
    fs::File,
    io,
    num::NonZero,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, Sender},
    },
    thread,
    time::Duration,
};

use clap::Args;
use easy_tiger::{
    input::{DerefVectorStore, VectorStore},
    spann::{
        CentroidAssignment, PostingKey, TableIndex,
        centroid_stats::{CentroidAssignmentUpdater, CentroidStats},
        rebalance::{
            RebalanceStats, merge_centroid, partition_oversized_centroid,
            split_centroid_post_partition,
        },
    },
    vamana::{search::GraphSearcher, wt::SessionGraphVectorIndex},
};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::{ThreadPoolBuilder, prelude::*};
use tracing::{error, info};
use vectors::F32VectorCoder;
use wt_mdb::{
    Connection, Result,
    session::{Formatted, TransactionGuard},
};

use crate::ui::{progress_bar, progress_spinner};

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

    let rng = Mutex::new(Xoshiro256PlusPlus::seed_from_u64(args.seed));

    let posting_format = index.config().posting_coder;
    let similarity = index.head_config().config().similarity;
    let posting_coder = posting_format.new_coder(similarity);
    let rerank_coder = index
        .config()
        .rerank_format
        .map(|f| f.new_coder(similarity));

    let main_progress = progress_bar(args.count.get(), "inserting vectors");
    main_progress.enable_steady_tick(Duration::from_millis(100));
    let (tx, rx) = mpsc::channel();
    let rebalance_handle = {
        let index = Arc::clone(&index);
        let connection = Arc::clone(&connection);
        thread::spawn(move || rebalance_loop(index, connection, rx))
    };
    (args.start..end).into_par_iter().try_for_each_init(
        || {
            (
                SessionGraphVectorIndex::new(
                    Arc::clone(index.head_config()),
                    connection.open_session().expect("open session"),
                ),
                tx.clone(),
            )
        },
        |(head_index, tx), i| {
            insert_vector(
                index.as_ref(),
                head_index,
                i as i64,
                &f32_vectors[i],
                posting_coder.as_ref(),
                rerank_coder.as_ref().map(|c| c.as_ref()),
                &rng,
                &tx,
            )
            .inspect(|_| main_progress.inc(1))
        },
    )?;
    main_progress.finish();
    drop(tx);

    let rebalance_progress = progress_spinner("rebalancing");
    let rebalance_stats = rebalance_handle.join().expect("joined");
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
    rebalance_progress.finish();

    Ok(())
}

#[derive(Debug, Copy, Clone)]
enum RebalanceOp {
    Merge {
        centroid_id: u32,
    },
    Split {
        src_centroid_id: u32,
        dst_centroid_ids: (u32, u32),
    },
}

/// Insert a single vector into the index, retrying on `WT_ROLLBACK` conflicts.
///
/// Searches for the nearest centroid, encodes the vector, and writes assignment,
/// postings, and optional rerank data in a single transaction. Logs a warning if
/// the target centroid is out of policy (over- or under-sized) after insertion.
fn insert_vector(
    index: &TableIndex,
    head_index: &SessionGraphVectorIndex,
    record_id: i64,
    vector: &[f32],
    posting_coder: &dyn F32VectorCoder,
    rerank_coder: Option<&dyn F32VectorCoder>,
    rng: &Mutex<impl Rng>,
    tx: &Sender<Vec<RebalanceOp>>,
) -> Result<()> {
    let posting_vector = posting_coder.encode(vector);
    let rerank_vector = rerank_coder.map(|c| c.encode(vector));

    loop {
        let result = try_insert_vector(
            index,
            head_index,
            record_id,
            vector,
            &posting_vector,
            rerank_vector.as_deref(),
            rng,
        );
        match result {
            Ok(ops) => {
                if !ops.is_empty() {
                    tx.send(ops).expect("send rebalance ops");
                }
                return Ok(());
            }
            Err(e) if e.is_rollback() => continue,
            Err(e) => return Err(e),
        }
    }
}

fn try_insert_vector(
    index: &TableIndex,
    head_index: &SessionGraphVectorIndex,
    record_id: i64,
    vector: &[f32],
    posting_vector: &[u8],
    rerank_vector: Option<&[u8]>,
    rng: &Mutex<impl Rng>,
) -> Result<Vec<RebalanceOp>> {
    let txn = TransactionGuard::new(head_index.session(), None)?;

    let session = head_index.session();
    let mut assignment_updater = CentroidAssignmentUpdater::new(index, session)?;

    let mut searcher = GraphSearcher::new(index.config().head_search_params);
    let candidates = searcher.search(vector, head_index)?;
    assert!(!candidates.is_empty());

    // TODO: implement replica selection.
    let candidates = candidates
        .into_iter()
        .filter_map(|n| {
            let r = assignment_updater.lifecycle(n.vertex() as u32);
            if r.is_ok_and(|l| !l.is_alive()) {
                None
            } else {
                Some(r.map(|_| n))
            }
        })
        .take(1)
        .collect::<Result<Vec<_>>>()?;
    assert!(!candidates.is_empty());

    let mut posting_cursor =
        session.get_or_create_typed_cursor::<PostingKey, Vec<u8>>(index.postings_table_name())?;
    let mut rerank_cursor = if rerank_vector.is_some() {
        Some(session.get_record_cursor(index.raw_vectors_table_name())?)
    } else {
        None
    };

    let assignment = CentroidAssignment::new(candidates[0].vertex() as u32, &[]);
    assignment_updater.insert(record_id, assignment.to_formatted_ref())?;

    let mut ops = vec![];
    let policy = index.config().centroid_len_range();
    for (_, centroid_id) in assignment.iter() {
        let key = PostingKey {
            centroid_id,
            record_id,
        };
        posting_cursor.set(key, posting_vector)?;
        let count = assignment_updater.centroid_size(centroid_id)?;
        if count < *policy.start()
            && CentroidStats::from_index_stats(head_index.session(), index)?.centroid_count() > 64
        {
            assignment_updater.set_lifecycle(
                centroid_id,
                easy_tiger::spann::centroid_stats::Lifecycle::Tombstone,
            )?;
            ops.push(RebalanceOp::Merge { centroid_id });
        } else if count > *policy.end() {
            let dst = partition_oversized_centroid(
                index,
                head_index,
                centroid_id,
                &mut assignment_updater,
                rng,
            )?;
            ops.push(RebalanceOp::Split {
                src_centroid_id: centroid_id,
                dst_centroid_ids: dst,
            });
        }
    }
    if let Some((cursor, vector)) = rerank_cursor.as_mut().zip(rerank_vector) {
        cursor.set(record_id, vector)?;
    }
    assignment_updater.flush()?;
    txn.commit(None).map(|()| ops)
}

fn rebalance_loop(
    index: Arc<TableIndex>,
    connection: Arc<Connection>,
    rx: Receiver<Vec<RebalanceOp>>,
) -> RebalanceStats {
    let pool = ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("rebalance pool");
    pool.install(|| {
        let mut stats = RebalanceStats::default();
        for ops in rx {
            // XXX this doesn't reattempt rebalancing if it fails due to conflict.
            for op in ops {
                info!("Begin rebalance op {op:?}");
                // XXX this is failing in all cases with a "not found" error.
                match rebalance_op(index.as_ref(), &connection, op) {
                    Ok(op_stats) => stats += op_stats,
                    Err(e) => error!("Rebalance failed: {e}"),
                }
                info!("End rebalance op {op:?}");
            }
        }
        stats
    })
}

fn rebalance_op(
    index: &TableIndex,
    connection: &Arc<Connection>,
    op: RebalanceOp,
) -> Result<RebalanceStats> {
    let session = connection.open_session()?;
    let head_index = SessionGraphVectorIndex::new(Arc::clone(index.head_config()), session);
    let txn = head_index.session().transaction(None)?;
    let stats = match op {
        RebalanceOp::Merge { centroid_id } => {
            merge_centroid(index, &head_index, centroid_id as usize, 0).map(RebalanceStats::from)
        }
        RebalanceOp::Split {
            src_centroid_id,
            dst_centroid_ids,
        } => split_centroid_post_partition(
            index,
            &head_index,
            src_centroid_id as usize,
            (dst_centroid_ids.0 as usize, dst_centroid_ids.1 as usize),
        )
        .map(RebalanceStats::from),
    }?;
    txn.commit(None).map(|()| stats)
}
