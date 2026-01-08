use std::{fs::File, io, num::NonZero, path::PathBuf, sync::Arc};

use clap::Args;
use easy_tiger::{
    input::{DerefVectorStore, VectorStore},
    spann::{
        centroid_stats::{CentroidAssignmentUpdater, CentroidStats},
        rebalance::{merge_centroid, split_centroid, BalanceSummary},
        CentroidAssignment, PostingKey, TableIndex,
    },
    vamana::{search::GraphSearcher, wt::SessionGraphVectorIndex},
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use wt_mdb::{
    session::{Formatted, TransactionGuard},
    Connection,
};

use crate::ui::progress_bar;

#[derive(Args)]
pub struct InsertVectorsArgs {
    /// Path to the input vectors to insert.
    #[arg(short, long)]
    f32_vectors: PathBuf,
    // XXX this is unnecessary it comes from the index.
    /// Number of dimensions in input vectors.
    #[arg(short, long)]
    dimensions: NonZero<usize>,

    /// Index of the first vector to insert.
    #[arg(long, default_value_t = 0)]
    start: usize,

    /// Number of vectors to insert.
    #[arg(short, long)]
    count: NonZero<usize>,

    /// Number of vectors to insert in each transaction batch.
    #[arg(long, default_value_t = NonZero::new(1000).unwrap())]
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
        args.dimensions,
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

    let mut searcher = GraphSearcher::new(index.config().head_search_params);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(args.seed);

    let posting_format = index.config().posting_coder;
    let similarity = index.head_config().config().similarity;
    let posting_coder = posting_format.new_coder(similarity);

    let total_vectors = args.count.get();
    let batch_size = args.batch_size.get();

    let main_progress = progress_bar(total_vectors, "inserting vectors");

    for batch_start in (args.start..end).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(end);

        // XXX this shoudl be another function.
        // Transaction handling
        {
            let txn = TransactionGuard::new(head_index.session(), None)?;
            let mut posting_cursor = head_index
                .session()
                .get_or_create_typed_cursor::<PostingKey, Vec<u8>>(index.postings_table_name())?;
            let mut assignment_updater =
                CentroidAssignmentUpdater::new(&index, head_index.session())?;
            // Optional raw vectors table
            let mut raw_cursor = if index.config().rerank_format.is_some() {
                Some(
                    head_index
                        .session()
                        .get_or_create_typed_cursor::<i64, Vec<u8>>(
                            index.raw_vectors_table_name(),
                        )?,
                )
            } else {
                None
            };

            // XXX I want to do this in parallel, requires a thread-local session.
            for i in batch_start..batch_end {
                let vector = &f32_vectors[i];

                // Search for centroid
                let candidates = searcher.search(vector, &head_index)?;
                if candidates.is_empty() {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        "search returned no candidates",
                    ));
                }

                // TODO: Implement replica selection (SOAR/RNG) if needed.
                // For now, take the closest one as primary.
                // The rebalance logic assumes replica_count == 1, so we probably should too for now
                // or just pick the top 1.
                // XXX this is also wrong, but the function it needs is not public.
                let centroid_id = candidates[0].vertex() as u32;

                // Create posting
                let encoded_vector = posting_coder.encode(vector);
                let key = PostingKey {
                    centroid_id,
                    record_id: i as i64,
                };
                posting_cursor.set(key, &encoded_vector)?;

                // Update assignments
                assignment_updater.insert(
                    i as i64,
                    CentroidAssignment::new(centroid_id, &[]).to_formatted_ref(),
                )?;

                // Write raw vector if needed
                if let Some(ref mut cursor) = raw_cursor {
                    // XXX this is straight up wrong
                    // cast f32 slice to u8 slice
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            vector.as_ptr() as *const u8,
                            vector.len() * std::mem::size_of::<f32>(),
                        )
                    };
                    cursor.set(i as i64, bytes)?;
                }

                main_progress.inc(1);
            }

            // Flush stats
            assignment_updater.flush()?;

            // Rebalance loop
            // We need to commit the current transaction to make changes visible for rebalancing?
            // OR we can run rebalancing in the same transaction?
            // The rebalance logic in `rebalance.rs` creates its own transaction.
            // If we are in a transaction, we should probably run the rebalance logic here.
            // BUT `rebalance` function takes a connection and opens a session.
            // We want to rebalance using OUR session/transaction.
            // The `merge_centroid` and `split_centroid` functions take `&TableIndex` and `&SessionGraphVectorIndex`.
            // So we can call them directly!

            // HOWEVER, we need to loop until stable.
            // We should do this within the transaction to maintain consistency, OR commit the batch first.
            // Committing the batch first seems safer/easier to reason about, then run a rebalance pass.
            // If we run rebalance inside the batch transaction, it might get huge.
            txn.commit(None)?;
        }

        // XXX this should be another function

        // Rebalance pass
        // Running rebalancing after every batch.
        let mut rebalance_iterations = 0;
        loop {
            // Need a new transaction for rebalancing steps
            let txn_guard = TransactionGuard::new(head_index.session(), None)?;

            let stats = CentroidStats::from_index_stats(head_index.session(), &index)?;
            let summary = BalanceSummary::new(&stats, index.config().centroid_len_range());

            // If we are perfectly balanced, stop
            if summary.below_exemplar().is_none() && summary.above_exemplar().is_none() {
                break;
            }

            // Limit rebalance iterations per batch to avoid getting stuck?
            if rebalance_iterations > 10 {
                break;
            }

            match (summary.below_exemplar(), summary.above_exemplar()) {
                (Some((to_merge, _)), _) if summary.total_clusters() > 1 => {
                    merge_centroid(&index, &head_index, to_merge)?;
                }
                (_, Some((to_split, _))) => {
                    // split_centroid needs a new centroid ID.
                    let next_id = stats.available_centroid_ids().next().unwrap();
                    split_centroid(&index, &head_index, to_split, next_id, &mut rng)?;
                }
                _ => break,
            }

            txn_guard.commit(None)?;
            rebalance_iterations += 1;
        }
    }

    Ok(())
}
