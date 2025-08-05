use std::{cell::RefCell, ops::DerefMut, sync::Arc};

use crate::{
    input::VectorStore,
    search::GraphSearcher,
    spann::{select_centroids, PostingKey, TableIndex},
    wt::SessionGraphVectorIndexReader,
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
    progress: (impl Fn(u64) + Send + Sync),
) -> Result<Vec<Vec<u32>>> {
    let tl_head_reader = ThreadLocal::new();
    let tl_searcher = ThreadLocal::new();
    let distance_fn = index.head_config().config().new_distance_function();
    (0..limit)
        .into_par_iter()
        .map(|i| {
            let mut head_reader = tl_head_reader
                .get_or_try(|| {
                    Ok::<_, wt_mdb::Error>(RefCell::new(SessionGraphVectorIndexReader::new(
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
                head_reader.deref_mut(),
                candidates,
                distance_fn.as_ref(),
                index.config().replica_count,
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
    progress: (impl Fn(u64) + Send + Sync),
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
    progress: (impl Fn(u64) + Send + Sync),
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
    for pk in posting_keys {
        let quantized = coder.encode(&vectors[pk.record_id as usize]);
        bulk_cursor.insert(pk, &quantized)?;
        progress(1);
    }
    Ok(())
}

/// Bulk load raw vector data into the raw vectors table for re-ranking.
pub fn bulk_load_raw_vectors(
    index: &TableIndex,
    session: &Session,
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    limit: usize,
    progress: (impl Fn(u64) + Send + Sync),
) -> Result<()> {
    let mut bulk_cursor = session.new_bulk_load_cursor::<i64, Vec<u8>>(
        &index.table_names.raw_vectors,
        Some(
            CreateOptionsBuilder::default()
                .app_metadata(&serde_json::to_string(&index.config).unwrap()),
        ),
    )?;
    let coder = index.head_config().config().new_rerank_coder();
    let mut encoded =
        Vec::with_capacity(coder.byte_len(index.head_config().config().dimensions.get()));
    for (record_id, vector) in vectors.iter().enumerate().take(limit) {
        coder.encode_to(vector, &mut encoded);
        bulk_cursor.insert(record_id as i64, &encoded)?;
        progress(1);
    }
    Ok(())
}
