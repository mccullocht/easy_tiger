use std::{cell::RefCell, collections::HashMap, sync::Arc};

use crate::{
    input::VectorStore,
    spann::{centroid_stats::CentroidCounts, TableIndex},
    vamana::{search::GraphSearcher, wt::TransactionGraphVectorIndex},
};
use rayon::prelude::*;
use thread_local::ThreadLocal;
use wt_mdb::{connection::CreateOptionsBuilder, Connection, Result};

/// Assign all the vectors to one centroid in the head index. This performs the same search
/// as [`super::TransactionIndex`] does.
pub fn assign_to_centroids(
    index: &TableIndex,
    connection: &Arc<Connection>,
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    limit: usize,
    progress: impl Fn(u64) + Send + Sync,
) -> Result<Vec<u32>> {
    // TODO: use map_init() to avoid thread locals.
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
            let selected = Ok(candidates[0].vertex() as u32);
            progress(1);
            selected
        })
        .collect::<Result<Vec<_>>>()
}

/// Bulk load centroid statistics into a stats table.
///
/// This creates a table mapping each centroid ID to the count of assigned vectors for efficient
/// statistics queries.
pub fn load_centroid_stats(
    index: &TableIndex,
    connection: &Arc<Connection>,
    centroid_assignments: &[u32],
    progress: impl Fn(u64) + Send + Sync,
) -> Result<()> {
    use std::collections::HashMap;

    let mut stats: HashMap<u32, CentroidCounts> = HashMap::new();
    for &assignment in centroid_assignments {
        stats.entry(assignment).or_default().primary += 1;
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
/// Callers must call [`PostingsMut::flush`] (or ensure `postings` does so on drop) to commit
/// changes, though `load_postings` calls it internally before returning.
pub fn load_postings(
    index: &TableIndex,
    connection: &Arc<Connection>,
    centroid_assignments: &[u32],
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    progress: impl Fn(u64) + Send + Sync,
) -> Result<()> {
    let posting_keys: HashMap<u32, Vec<i64>> =
        centroid_assignments
            .iter()
            .enumerate()
            .fold(HashMap::new(), |mut acc, (i, &a)| {
                acc.entry(a).or_default().push(i as i64);
                acc
            });
    let max_centroid_id = *centroid_assignments.iter().max().unwrap();
    let coder = index.new_posting_coder();
    let leaf_page_size = crate::posting_block::leaf_page_max(
        index.config().max_centroid_len,
        coder.byte_len(index.head_config().config().dimensions.get()),
        4096,
    ) as u32;
    let mut bulk_cursor = connection.new_bulk_load_cursor::<u32, Vec<u8>>(
        &index.table_names.postings,
        Some(
            CreateOptionsBuilder::default()
                .key_format::<u32>()
                .value_format::<Vec<u8>>()
                .app_metadata(&serde_json::to_string(&index.config()).unwrap())
                .leaf_page_max(leaf_page_size)
                .leaf_value_max(leaf_page_size),
        ),
    )?;

    for i in (0..=max_centroid_id).step_by(64) {
        let batch = i..((max_centroid_id + 1).min(i + 64));
        let blocks = batch
            .clone()
            .into_par_iter()
            .map(|i| {
                let mut ids = posting_keys.get(&i).unwrap().clone();
                ids.sort_unstable();
                crate::posting_block::encode_f32(
                    ids.into_iter().map(|j| (j, &vectors[j as usize])),
                    coder.as_ref(),
                    index.head_config().config().dimensions.get(),
                )
            })
            .collect::<Vec<_>>();

        for (c, b) in batch.clone().zip(blocks) {
            bulk_cursor.append(c, &b)?;
        }
        progress(batch.len() as u64);
    }
    Ok(())
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
        connection.new_bulk_load_cursor::<i64, Vec<u8>>(&index.table_names.rerank_vectors, None)?;
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
