use std::{cell::RefCell, collections::HashMap, sync::Arc};

use crate::{
    input::VectorStore,
    spann::{
        CentroidAssignment, TableIndex,
        centroid_stats::CentroidCounts,
        centroids::CentroidVectorSource,
        postings::BlockPostingsMut,
    },
    vamana::{search::GraphSearcher, wt::TransactionGraphVectorIndex},
};
use rayon::prelude::*;
use thread_local::ThreadLocal;
use wt_mdb::{Connection, Result};

/// Assign all the vectors to one centroid in the head index. This performs the same search
/// as [`super::TransactionIndex`] does.
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
            let selected = Ok(CentroidAssignment::new(candidates[0].vertex() as u32));
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
    for (record_id, assignment) in centroid_assignments.iter().enumerate() {
        bulk_cursor.append(record_id as i64, *assignment)?;
        progress(1);
    }
    Ok(())
}

/// Bulk load centroid statistics into a stats table.
///
/// This creates a table mapping each centroid ID to the count of assigned vectors for efficient
/// statistics queries.
pub fn load_centroid_stats(
    index: &TableIndex,
    connection: &Arc<Connection>,
    centroid_assignments: &[CentroidAssignment],
    progress: impl Fn(u64) + Send + Sync,
) -> Result<()> {
    let mut stats: HashMap<u32, CentroidCounts> = HashMap::new();
    for assignment in centroid_assignments {
        stats.entry(assignment.primary_id).or_default().primary += 1;
    }

    let mut stats = stats.into_iter().collect::<Vec<_>>();
    stats.sort_by_key(|(id, _)| *id);
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
/// When `index` is configured with `center_postings` each vector is encoded as a residual
/// (`v - c`) against its assigned centroid. Callers must call [`PostingsMut::flush`] (or ensure
/// `postings` does so on drop) to commit changes, though `load_postings` calls it internally
/// before returning.
pub fn load_postings(
    index: &TableIndex,
    connection: &Arc<Connection>,
    postings: &mut BlockPostingsMut<'_>,
    centroid_assignments: &[CentroidAssignment],
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    progress: impl Fn(u64) + Send + Sync,
) -> Result<()> {
    let mut posting_keys: Vec<(u32, i64)> = centroid_assignments
        .iter()
        .enumerate()
        .map(|(i, a)| (a.primary_id, i as i64))
        .collect();
    posting_keys.par_sort_unstable();

    let coder = index.config().posting_coder.coder();
    // Centroid vectors for residual encoding, read from the head index's high-fidelity store.
    let head_reader = index
        .config()
        .center_postings
        .then(|| -> Result<TransactionGraphVectorIndex> {
            Ok(TransactionGraphVectorIndex::new(
                Arc::clone(index.head_config()),
                connection.begin_transaction(None)?,
            ))
        })
        .transpose()?;
    let mut centroid_source = head_reader
        .as_ref()
        .map(CentroidVectorSource::new)
        .transpose()?;
    // Encode in batches to avoid single-threading encoding work. If the vectors are backed by mmap
    // this will also allow us to parallelize IO.
    let mut encoded_buffer =
        vec![vec![0u8; coder.byte_len(index.head_config().config().dimensions.get())]; 1024];
    for batch in posting_keys.chunks(1024) {
        // Keys are sorted by (centroid_id, record_id) so each batch touches only a handful of
        // distinct centroids; fetch each of their vectors once.
        let centroid_vectors = centroid_source
            .as_mut()
            .map(|source| {
                let mut centroid_vectors = HashMap::new();
                for &(centroid_id, _) in batch {
                    if let std::collections::hash_map::Entry::Vacant(e) =
                        centroid_vectors.entry(centroid_id)
                    {
                        e.insert(source.centroid_vector(centroid_id)?);
                    }
                }
                Ok::<_, wt_mdb::Error>(centroid_vectors)
            })
            .transpose()?;
        encoded_buffer
            .par_iter_mut()
            .zip(batch)
            .for_each(|(buf, &(centroid_id, record_id))| {
                if let Some(centroid_vector) =
                    centroid_vectors.as_ref().and_then(|m| m.get(&centroid_id))
                {
                    // Centered: encode the residual v - c.
                    let residual = vectors::prepare_vector(
                        &vectors[record_id as usize],
                        None,
                        false,
                        Some(centroid_vector),
                    );
                    coder.encode_to(&residual, buf);
                } else {
                    coder.encode_to(&vectors[record_id as usize], buf);
                }
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
    let coder = index.config().rerank_format.coder();
    let mut encoded = vec![0u8; coder.byte_len(index.head_config().config().dimensions.get())];
    for (record_id, vector) in vectors.iter().enumerate().take(limit) {
        coder.encode_to(vector, &mut encoded);
        bulk_cursor.append(record_id as i64, &encoded)?;
        progress(1);
    }
    Ok(())
}
