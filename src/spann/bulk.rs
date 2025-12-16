use std::{cell::RefCell, ops::DerefMut, sync::Arc};

use crate::{
    input::VectorStore,
    spann::{centroid_stats::CentroidCounts, select_centroids, PostingKey, TableIndex},
    vamana::{search::GraphSearcher, wt::SessionGraphVectorIndex},
};
use rayon::prelude::*;
use thread_local::ThreadLocal;
use wt_mdb::{options::CreateOptionsBuilder, Connection, Result, Session};

/// Assign all the vectors to one or more centroids in the head index. This performs the same search
/// and pruning as [`super::SessionIndexWriter`] does.
pub fn assign_to_centroids(
    index: &TableIndex,
    connection: &Arc<Connection>,
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    limit: usize,
    progress: impl Fn(u64) + Send + Sync,
) -> Result<Vec<Vec<u32>>> {
    let tl_head_reader = ThreadLocal::new();
    let tl_searcher = ThreadLocal::new();
    let distance_fn = index
        .head_config()
        .high_fidelity_table()
        .new_distance_function();
    (0..limit)
        .into_par_iter()
        .map(|i| {
            let mut head_reader = tl_head_reader
                .get_or_try(|| {
                    Ok::<_, wt_mdb::Error>(RefCell::new(SessionGraphVectorIndex::new(
                        Arc::clone(&index.head),
                        connection.open_session()?,
                    )))
                })?
                .borrow_mut();
            let mut searcher = tl_searcher
                .get_or(|| RefCell::new(GraphSearcher::new(index.config().head_search_params)))
                .borrow_mut();
            let candidates = searcher.search(&vectors[i], head_reader.deref_mut())?;
            let selected = select_centroids(
                index.config().replica_selection,
                index.config().replica_count,
                candidates,
                &vectors[i],
                head_reader.deref_mut(),
                distance_fn.as_ref(),
            );
            progress(1);
            selected
        })
        .collect::<Result<Vec<_>>>()
}

/// Load all centroid assignments into a record id keyed table.
pub fn bulk_load_centroids(
    index: &TableIndex,
    session: &Session,
    centroid_assignments: &[Vec<u32>],
    progress: impl Fn(u64) + Send + Sync,
) -> Result<()> {
    let mut bulk_cursor =
        session.new_bulk_load_cursor::<i64, Vec<u8>>(&index.table_names.centroids, None)?;
    let mut centroid_buf =
        Vec::with_capacity(std::mem::size_of::<u32>() * index.config().replica_count);
    for (record_id, centroids) in centroid_assignments.iter().enumerate() {
        centroid_buf.clear();
        for cid in centroids {
            centroid_buf.extend_from_slice(&cid.to_le_bytes());
        }
        bulk_cursor.insert(record_id as i64, &centroid_buf)?;
        progress(1);
    }
    Ok(())
}

// XXX fix the other names.

/// Bulk load centroid statistics into a stats table.
///
/// This creates a table mapping each centroid ID to the count of primary and secondary
/// assigned vectors for efficient statistics queries.
pub fn load_centroid_stats(
    index: &TableIndex,
    session: &Session,
    centroid_assignments: &[Vec<u32>],
    progress: impl Fn(u64) + Send + Sync,
) -> Result<()> {
    use std::collections::HashMap;

    // Count primary and secondary assignments for each centroid
    let mut stats: HashMap<u32, CentroidCounts> = HashMap::new();

    for centroids in centroid_assignments {
        if let Some(&primary_id) = centroids.first() {
            stats.entry(primary_id).or_default().primary += 1;

            for &secondary_id in centroids.iter().skip(1) {
                stats.entry(secondary_id).or_default().secondary += 1;
            }
        }
    }

    let mut stats = stats.into_iter().collect::<Vec<_>>();
    stats.sort_by_key(|(id, _)| *id);
    // Bulk load the stats
    let mut bulk_cursor = session
        .new_bulk_load_cursor::<u32, CentroidCounts>(&index.table_names.centroid_stats, None)?;

    for (centroid_id, counts) in stats {
        bulk_cursor.insert(centroid_id, counts)?;
        progress(1);
    }

    Ok(())
}

/// Bulk load entries for each of the posting keys into the database.
///
/// This runs in a single thread, reading the vector for each posting, quantizing it, and uploading
/// it into a table. Bulk upload ensures that all posting entries belonging to each centroid appear
/// contiguously on disk, whereas iterative insertion may "split up" a centroid as it splits leaf
/// pages. Bulk uploading also avoids checkpointing.
pub fn bulk_load_postings(
    index: &TableIndex,
    session: &Session,
    centroid_assignments: &[Vec<u32>],
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    progress: impl Fn(u64) + Send + Sync,
) -> Result<()> {
    let mut posting_keys: Vec<PostingKey> = centroid_assignments
        .into_par_iter()
        .enumerate()
        .flat_map_iter(|(i, a)| {
            a.iter().map(move |c| PostingKey {
                centroid_id: *c,
                record_id: i as i64,
            })
        })
        .collect();
    posting_keys.par_sort_unstable();

    let coder = index
        .config()
        .posting_coder
        .new_coder(index.head_config().config().similarity);
    let mut bulk_cursor = session.new_bulk_load_cursor::<PostingKey, Vec<u8>>(
        &index.table_names.postings,
        Some(
            CreateOptionsBuilder::default()
                .app_metadata(&serde_json::to_string(&index.config).unwrap()),
        ),
    )?;
    // Encode in batches to avoid single-threading encoding work. If the vectors are backed by mmap
    // this will also allow us to parallelize IO.
    let mut encoded_buffer =
        vec![vec![0u8; coder.byte_len(index.head_config().config().dimensions.get())]; 1024];
    for batch in posting_keys.chunks(1024) {
        encoded_buffer
            .par_iter_mut()
            .zip(batch)
            .for_each(|(buf, pk)| {
                coder.encode_to(&vectors[pk.record_id as usize], buf);
            });
        for (pk, buf) in batch.iter().zip(encoded_buffer.iter()) {
            bulk_cursor.insert(*pk, buf)?;
        }
        progress(batch.len() as u64);
    }
    Ok(())
}

/// Bulk load raw vector data into the raw vectors table for re-ranking.
pub fn bulk_load_raw_vectors(
    index: &TableIndex,
    session: &Session,
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    limit: usize,
    progress: impl Fn(u64) + Send + Sync,
) -> Result<()> {
    let mut bulk_cursor =
        session.new_bulk_load_cursor::<i64, Vec<u8>>(&index.table_names.raw_vectors, None)?;
    let coder = index
        .config()
        .rerank_format
        .unwrap()
        .new_coder(index.head_config().config().similarity);
    let mut encoded = vec![0u8; coder.byte_len(index.head_config().config().dimensions.get())];
    for (record_id, vector) in vectors.iter().enumerate().take(limit) {
        coder.encode_to(vector, &mut encoded);
        bulk_cursor.insert(record_id as i64, &encoded)?;
        progress(1);
    }
    Ok(())
}
