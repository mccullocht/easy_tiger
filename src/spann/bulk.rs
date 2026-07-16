use std::{cell::RefCell, collections::HashMap, sync::Arc};

use crate::{
    input::VectorStore,
    spann::{
        centroid_stats::CentroidCounts, postings::BlockPostingsMut, CentroidAssignment,
        TableIndex,
    },
    vamana::{search::GraphSearcher, wt::TransactionGraphVectorIndex, GraphVectorIndex, GraphVectorStore},
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
    use std::collections::HashMap;

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

/// Fetch the highest-fidelity decoded f32 vectors for a set of centroid IDs from the head index.
///
/// Used when `center_postings` is enabled to supply per-centroid coders during bulk loading.
pub fn fetch_centroid_vecs(
    index: &TableIndex,
    connection: &Arc<Connection>,
    centroid_assignments: &[CentroidAssignment],
) -> Result<HashMap<u32, Vec<f32>>> {
    let mut centroid_ids: Vec<u32> = centroid_assignments.iter().map(|a| a.primary_id).collect();
    centroid_ids.sort_unstable();
    centroid_ids.dedup();

    let txn = connection.begin_transaction(None)?;
    let head_reader = TransactionGraphVectorIndex::new(Arc::clone(index.head_config()), txn);
    let mut store = head_reader.high_fidelity_vectors()?;
    let coder = store.new_coder();
    let mut out = HashMap::with_capacity(centroid_ids.len());
    for id in centroid_ids {
        let raw = store
            .get(id as i64)
            .unwrap_or_else(|| Err(wt_mdb::Error::not_found_error()))?;
        out.insert(id, coder.decode(raw));
    }
    Ok(out)
}

/// Load entries for each of the posting keys into `postings`.
///
/// Vectors are encoded in parallel batches and inserted in (centroid_id, record_id) order, which
/// allows implementations backed by sorted storage to place each centroid's entries contiguously.
/// Callers must call [`PostingsMut::flush`] (or ensure `postings` does so on drop) to commit
/// changes, though `load_postings` calls it internally before returning.
///
/// When `centroid_vecs` is `Some`, each posting vector is encoded centered on its centroid's
/// decoded f32 vector. Pass `None` for standard uncentered encoding.
pub fn load_postings(
    index: &TableIndex,
    postings: &mut BlockPostingsMut<'_>,
    centroid_assignments: &[CentroidAssignment],
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    centroid_vecs: Option<&HashMap<u32, Vec<f32>>>,
    progress: impl Fn(u64) + Send + Sync,
) -> Result<()> {
    let mut posting_keys: Vec<(u32, i64)> = centroid_assignments
        .iter()
        .enumerate()
        .map(|(i, a)| (a.primary_id, i as i64))
        .collect();
    posting_keys.par_sort_unstable();

    let similarity = index.head_config().config().similarity;
    let encoded_len = index.posting_vector_len();

    // posting_keys is sorted by centroid_id; process one centroid group at a time so the
    // per-centroid coder is created only once per centroid rather than once per vector.
    let mut encoded_buffer = vec![vec![0u8; encoded_len]; 1024];
    let mut pos = 0;
    while pos < posting_keys.len() {
        let centroid_id = posting_keys[pos].0;
        let group_end = posting_keys[pos..]
            .iter()
            .position(|&(c, _)| c != centroid_id)
            .map(|n| pos + n)
            .unwrap_or(posting_keys.len());
        let group = &posting_keys[pos..group_end];

        let coder: Box<dyn vectors::F32VectorCoder> =
            match centroid_vecs.and_then(|m| m.get(&centroid_id)) {
                Some(center) => index.new_posting_coder_centered(center.clone()),
                None => index.config().posting_coder.coder(similarity, None),
            };

        for batch in group.chunks(1024) {
            encoded_buffer.resize(batch.len(), vec![0u8; encoded_len]);
            encoded_buffer
                .par_iter_mut()
                .zip(batch)
                .for_each(|(buf, &(_, record_id))| {
                    coder.encode_to(&vectors[record_id as usize], buf);
                });
            for (&(centroid, record_id), buf) in batch.iter().zip(encoded_buffer.iter()) {
                postings.insert(centroid, record_id, buf)?;
            }
            progress(batch.len() as u64);
        }

        pos = group_end;
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
