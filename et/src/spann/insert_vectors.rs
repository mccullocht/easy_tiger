use std::{fs::File, io, num::NonZero, ops::Range, path::PathBuf, sync::Arc};

use clap::Args;
use easy_tiger::{
    input::{DerefVectorStore, VectorStore},
    spann::{
        centroid_stats::{CentroidAssignmentUpdater, CentroidStats},
        rebalance::{merge_centroid, split_centroid, BalanceSummary, RebalanceStats},
        CentroidAssignment, PostingKey, TableIndex,
    },
    vamana::{search::GraphSearcher, wt::SessionGraphVectorIndex},
};
use indicatif::{ParallelProgressIterator, ProgressBar};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use vectors::F32VectorCoder;
use wt_mdb::{
    session::{Formatted, TransactionGuard},
    Connection, Result,
};

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
    let session = connection.open_session()?;
    let head_index = SessionGraphVectorIndex::new(Arc::clone(index.head_config()), session);

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

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(args.seed);

    let posting_format = index.config().posting_coder;
    let similarity = index.head_config().config().similarity;
    let posting_coder = posting_format.new_coder(similarity);
    let rerank_coder = index
        .config()
        .rerank_format
        .map(|f| f.new_coder(similarity));

    let batch_size = args.batch_size.get();
    let main_progress = progress_bar(args.count.get() - args.start, "inserting vectors");

    let mut rebalance_stats = RebalanceStats::default();

    for batch_start in (args.start..end).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(end);

        insert_batch(
            &index,
            &head_index,
            &f32_vectors,
            batch_start..batch_end,
            posting_coder.as_ref(),
            rerank_coder.as_ref().map(|c| c.as_ref()),
            &main_progress,
        )?;

        rebalance_stats += rebalance(&index, &head_index, &mut rng, &main_progress)?;
    }

    main_progress.set_message("inserting vectors");
    main_progress.finish();

    println!("Merged:         {:10}", rebalance_stats.merged);
    if rebalance_stats.merged > 0 {
        println!(
            "  Moved:        {:10}",
            rebalance_stats.merge_stats.moved_vectors
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
            "  Nearby seen:  {:10}",
            rebalance_stats.split_stats.nearby_seen
        );
        println!(
            "  Nearby moved: {:10}",
            rebalance_stats.split_stats.nearby_moved
        );
    }

    Ok(())
}

fn insert_batch(
    index: &TableIndex,
    head_index: &SessionGraphVectorIndex,
    f32_vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    batch: Range<usize>,
    posting_coder: &dyn F32VectorCoder,
    rerank_coder: Option<&dyn F32VectorCoder>,
    progress: &ProgressBar,
) -> Result<()> {
    progress.set_message("inserting vectors");

    let connection = Arc::clone(head_index.session().connection());
    let vector_state = batch
        .clone()
        .into_par_iter()
        .progress_with(progress.clone())
        .map_init(
            || {
                let session = SessionGraphVectorIndex::new(
                    Arc::clone(index.head_config()),
                    connection.open_session().expect("open session"),
                );
                let searcher = GraphSearcher::new(index.config().head_search_params);
                (session, searcher)
            },
            |(head_index, searcher), i| {
                let vector = &f32_vectors[i];

                // Search for centroid
                let txn = head_index.session().transaction(None)?;
                let candidates = searcher.search(vector, head_index)?;
                assert!(!candidates.is_empty());
                txn.commit(None)?;

                // TODO: implement replica selection
                let centroid_id = candidates[0].vertex() as u32;
                Ok((
                    i,
                    CentroidAssignment::new(centroid_id, &[]),
                    posting_coder.encode(vector),
                    rerank_coder.map(|c| c.encode(vector)),
                ))
            },
        )
        .collect::<Result<Vec<_>>>()?;

    let txn = TransactionGuard::new(head_index.session(), None)?;
    progress.set_message("writing postings");
    let mut assignment_updater = CentroidAssignmentUpdater::new(index, head_index.session())?;
    let mut posting_cursor = head_index
        .session()
        .get_or_create_typed_cursor::<PostingKey, Vec<u8>>(index.postings_table_name())?;
    let mut rerank_cursor = if rerank_coder.is_some() {
        Some(
            head_index
                .session()
                .get_record_cursor(index.raw_vectors_table_name())?,
        )
    } else {
        None
    };

    for (i, assignment, posting_vector, rerank_vector) in vector_state.into_iter() {
        assignment_updater.insert(i as i64, assignment.to_formatted_ref())?;

        for (_, centroid_id) in assignment.iter() {
            let key = PostingKey {
                centroid_id,
                record_id: i as i64,
            };
            posting_cursor.set(key, &posting_vector)?;
        }

        if let Some((cursor, vector)) = rerank_cursor.as_mut().zip(rerank_vector) {
            cursor.set(i as i64, &vector)?;
        }
    }

    assignment_updater.flush()?;
    txn.commit(None)
}

fn rebalance(
    index: &TableIndex,
    head_index: &SessionGraphVectorIndex,
    rng: &mut impl Rng,
    progress: &ProgressBar,
) -> Result<RebalanceStats> {
    let mut iter = 1;
    let mut rebalance_stats = RebalanceStats::default();
    loop {
        // Need a new transaction for rebalancing steps
        let txn_guard = TransactionGuard::new(head_index.session(), None)?;

        let stats = CentroidStats::from_index_stats(head_index.session(), &index)?;
        let summary = BalanceSummary::new(&stats, index.config().centroid_len_range());

        match (summary.below_exemplar(), summary.above_exemplar()) {
            (Some((to_merge, len)), _) if summary.total_clusters() > 1 => {
                progress.set_message(format!("merge {to_merge} of {len} ({iter})"));
                rebalance_stats += merge_centroid(&index, &head_index, to_merge, len)?;
            }
            (_, Some((to_split, len))) => {
                progress.set_message(format!("split {to_split} of {len} ({iter})"));
                // TODO: split_centroid should allow splitting into multiple centroids.
                // This requires allocating an arbitrary number of ids and accommodating these
                // additional ids in the split of the centroid and updating of nearby centroids.
                let mut it = stats.available_centroid_ids();
                let target_centroid_ids = (it.next().unwrap(), it.next().unwrap());
                rebalance_stats +=
                    split_centroid(&index, &head_index, to_split, target_centroid_ids, len, rng)?;
            }
            _ => break,
        }

        txn_guard.commit(None)?;
        iter += 1;
    }

    Ok(rebalance_stats)
}
