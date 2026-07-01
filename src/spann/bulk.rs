use std::{cell::RefCell, sync::Arc};

use crate::{
    input::VectorStore,
    spann::{
        centroid_stats::CentroidCounts, postings::PostingsMut, select_centroids, CentroidAssignment,
        CentroidAssignmentType, TableIndex,
    },
    vamana::{search::GraphSearcher, wt::TransactionGraphVectorIndex},
};
use rayon::prelude::*;
use thread_local::ThreadLocal;
use wt_mdb::{session::Formatted, Connection, Result};

/// Assign all the vectors to one or more centroids in the head index. This performs the same search
/// and pruning as [`super::TransactionIndex`] does.
pub fn assign_to_centroids(
    index: &TableIndex,
    connection: &Arc<Connection>,
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    limit: usize,
    progress: impl Fn(u64) + Send + Sync,
) -> Result<Vec<CentroidAssignment>> {
    let tl_searcher = ThreadLocal::new();
    (0..limit)
        .into_par_iter()
        .map(|i| {
            let head_reader = TransactionGraphVectorIndex::new(
                Arc::clone(index.head_config()),
                connection.begin_transaction(None)?,
            );
            let mut searcher = tl_searcher
                .get_or(|| RefCell::new(GraphSearcher::new(index.config().head_search_params)))
                .borrow_mut();
            let candidates = searcher.search(&vectors[i], &head_reader)?;
            let selected = select_centroids(
                index.config().replica_selection,
                index.config().replica_count,
                candidates,
                &vectors[i],
                &head_reader,
            );
            progress(1);
            selected
        })
        .collect::<Result<Vec<_>>>()
}

/// Load all centroid assignments into a record id keyed table.
pub fn load_centroids(
    index: &TableIndex,
    connection: &Arc<Connection>,
    centroid_assignments: &[CentroidAssignment],
    progress: impl Fn(u64) + Send + Sync,
) -> Result<()> {
    let mut bulk_cursor = connection
        .new_bulk_load_cursor::<i64, CentroidAssignment>(&index.table_names.centroids, None)?;
    for (record_id, centroids) in centroid_assignments.iter().enumerate() {
        bulk_cursor.append(record_id as i64, centroids.to_formatted_ref())?;
        progress(1);
    }
    Ok(())
}

/// Bulk load centroid statistics into a stats table.
///
/// This creates a table mapping each centroid ID to the count of primary and secondary
/// assigned vectors for efficient statistics queries.
pub fn load_centroid_stats(
    index: &TableIndex,
    connection: &Arc<Connection>,
    centroid_assignments: &[CentroidAssignment],
    progress: impl Fn(u64) + Send + Sync,
) -> Result<()> {
    use std::collections::HashMap;

    // Count primary and secondary assignments for each centroid
    let mut stats: HashMap<u32, CentroidCounts> = HashMap::new();

    for centroids in centroid_assignments {
        for (assignment_type, centroid_id) in centroids.iter() {
            match assignment_type {
                CentroidAssignmentType::Primary => {
                    stats.entry(centroid_id).or_default().primary += 1;
                }
                CentroidAssignmentType::Secondary => {
                    stats.entry(centroid_id).or_default().secondary += 1;
                }
            }
        }
    }

    let mut stats = stats.into_iter().collect::<Vec<_>>();
    stats.sort_by_key(|(id, _)| *id);
    // Bulk load the stats
    let mut bulk_cursor = connection
        .new_bulk_load_cursor::<u32, CentroidCounts>(&index.table_names.centroid_stats, None)?;

    for (centroid_id, counts) in stats {
        bulk_cursor.append(centroid_id, counts)?;
        progress(1);
    }

    Ok(())
}

/// Load entries for each of the posting keys into `postings`.
///
/// Vectors are encoded in parallel batches and inserted in (centroid_id, record_id) order, which
/// allows implementations backed by sorted storage to place each centroid's entries contiguously.
/// Callers must call [`PostingsMut::flush`] (or ensure `postings` does so on drop) to commit
/// changes, though `load_postings` calls it internally before returning.
pub fn load_postings(
    index: &TableIndex,
    postings: &mut impl PostingsMut,
    centroid_assignments: &[CentroidAssignment],
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    progress: impl Fn(u64) + Send + Sync,
) -> Result<()> {
    let mut posting_keys: Vec<(u32, i64)> = centroid_assignments
        .into_par_iter()
        .enumerate()
        .flat_map_iter(|(i, a)| a.iter().map(move |(_, c)| (c, i as i64)))
        .collect();
    posting_keys.par_sort_unstable();

    let coder = index
        .config()
        .posting_coder
        .coder(index.head_config().config().similarity, None);
    // Encode in batches to avoid single-threading encoding work. If the vectors are backed by mmap
    // this will also allow us to parallelize IO.
    let mut encoded_buffer =
        vec![vec![0u8; coder.byte_len(index.head_config().config().dimensions.get())]; 1024];
    for batch in posting_keys.chunks(1024) {
        encoded_buffer
            .par_iter_mut()
            .zip(batch)
            .for_each(|(buf, &(_, record_id))| {
                coder.encode_to(&vectors[record_id as usize], buf);
            });
        for (&(centroid_id, record_id), buf) in batch.iter().zip(encoded_buffer.iter()) {
            postings.insert(centroid_id, record_id, buf)?;
        }
        progress(batch.len() as u64);
    }
    postings.flush()
}

/// Bulk load raw vector data into the raw vectors table for re-ranking.
pub fn load_raw_vectors(
    index: &TableIndex,
    connection: &Arc<Connection>,
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    limit: usize,
    progress: impl Fn(u64) + Send + Sync,
) -> Result<()> {
    let mut bulk_cursor =
        connection.new_bulk_load_cursor::<i64, Vec<u8>>(&index.table_names.raw_vectors, None)?;
    let coder = index
        .config()
        .rerank_format
        .unwrap()
        .coder(index.head_config().config().similarity, None);
    let mut encoded = vec![0u8; coder.byte_len(index.head_config().config().dimensions.get())];
    for (record_id, vector) in vectors.iter().enumerate().take(limit) {
        coder.encode_to(vector, &mut encoded);
        bulk_cursor.append(record_id as i64, &encoded)?;
        progress(1);
    }
    Ok(())
}
