use std::{cell::RefCell, ops::DerefMut, sync::Arc};

use crate::{
    input::VectorStore,
    search::GraphSearcher,
    spann::{select_centroids, PostingKey, TableIndex},
    wt::SessionGraphVectorIndexReader,
};
use rayon::prelude::*;
use thread_local::ThreadLocal;
use wt_mdb::{Connection, Result, Session};

/// Assign all the vectors to one or more centroids in the head index. This performs the same search
/// and pruning as [`super::SessionIndexWriter`] does.
pub fn assign_to_centroids(
    index: &TableIndex,
    connection: &Arc<Connection>,
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    limit: Option<usize>,
    progress: &(impl Fn(u64) + Send + Sync),
) -> Result<Vec<Vec<u32>>> {
    let tl_head_reader = ThreadLocal::new();
    let tl_searcher = ThreadLocal::new();
    let distance_fn = index.head_config().config().new_distance_function();
    (0..limit.unwrap_or(vectors.len()))
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

// XXX need to upload record -> centroid information

/// Compute the sorted list of [`super::PostingKey`]s from centroid assignments.
pub fn compute_posting_keys(assignments: Vec<Vec<u32>>) -> Vec<PostingKey> {
    let mut keys: Vec<PostingKey> = assignments
        .into_par_iter()
        .enumerate()
        .flat_map_iter(|(i, a)| {
            a.into_iter().map(move |c| PostingKey {
                centroid_id: c,
                record_id: i as i64,
            })
        })
        .collect();
    keys.par_sort_unstable();
    keys
}

// XXX move assignment stats computation here.

/// Bulk load entries for each of the posting keys into the database.
///
/// This runs in a single thread, reading the vector for each posting, quantizing it, and uploading
/// it into a table. Bulk upload ensures that all posting entries belonging to each centroid appear
/// contiguously on disk, whereas iterative insertion may "split up" a centroid as it splits leaf
/// pages. Bulk uploading also avoids checkpointing.
///
/// REQUIRES: `posting_keys.is_sorted()``
pub fn bulk_load_postings<V: VectorStore<Elem = f32> + Send + Sync>(
    index: &TableIndex,
    session: &Session,
    posting_keys: Vec<PostingKey>,
    vectors: &V,
    limit: Option<usize>,
) -> Result<()> {
    let quantizer = index.config().quantizer.new_quantizer();
    // XXX I can't bulk load an index table FML.
    todo!()
}
