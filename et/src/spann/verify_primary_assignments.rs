use std::{io, sync::Arc};

use easy_tiger::{
    input::{VecVectorStore, VectorStore},
    spann::{centroid_stats::CentroidStats, PostingKey, TableIndex},
    Neighbor,
};
use indicatif::ProgressIterator;
use rayon::prelude::*;
use wt_mdb::{session::TransactionGuard, Connection, Error, Result, Session};

use crate::ui::progress_bar;

pub fn verify_primary_assignments(connection: Arc<Connection>, index_name: &str) -> io::Result<()> {
    let index = Arc::new(TableIndex::from_db(&connection, index_name)?);
    let session = connection.open_session()?;
    let stats = CentroidStats::from_index_stats(&session, &index)?;

    // Read primary assignments and head vectors. Re-encode the head vectors in the same format as
    // the postings to make comparisons cheaper since we will be doing this exhaustively.
    let primary_assignments = read_primary_assignments(&session, &index, &stats)?;
    let (centroid_ids, head_vectors) = read_head_vectors(&session, &index)?;
    // Get the list of centroid ids with primary assignments. This will be our unit of parallelism
    // to avoid high overhead in rayon.
    let mut primary_centroid_ids = primary_assignments
        .iter()
        .map(|p| p.centroid_id)
        .collect::<Vec<_>>();
    primary_centroid_ids.dedup();

    let progress = progress_bar(primary_assignments.len(), "computing primary assignments");
    let (correct, total) = primary_centroid_ids
        .into_par_iter()
        .map_init(
            || connection.open_session().expect("failed to open session"),
            |session, centroid_id| {
                count_primary_assigned_vectors(
                    session,
                    &index,
                    centroid_id,
                    &primary_assignments,
                    &centroid_ids,
                    &head_vectors,
                )
                .inspect(|(_, t)| progress.inc(*t as u64))
            },
        )
        .reduce(
            || Ok((0, 0)),
            |a, b| a.and_then(|a| b.and_then(|b| Ok((a.0 + b.0, a.1 + b.1)))),
        )
        .expect("failed to count primary assigned vectors");

    progress.finish();

    println!(
        "Primary assignment correct {correct}/{total}; accuracy: {:.2}%",
        if total > 0 {
            (correct as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    );

    Ok(())
}

fn read_primary_assignments(
    session: &Session,
    index: &TableIndex,
    stats: &CentroidStats,
) -> Result<Vec<PostingKey>> {
    let cursor = session
        .get_or_create_typed_cursor::<i64, Vec<u8>>(&index.centroid_assignments_table_name())?;
    let primary_assignments = stats
        .primary_assignment_counts_iter()
        .map(|(_, c)| c as usize)
        .sum();
    let mut assignments = Vec::with_capacity(primary_assignments);
    for r in cursor.progress_with(progress_bar(
        primary_assignments,
        "reading primary assignments",
    )) {
        let (key, value) = r?;
        let (centroid_ids, tail) = value.as_chunks::<{ std::mem::size_of::<u32>() }>();
        assert!(!centroid_ids.is_empty() && tail.is_empty());
        assignments.push(PostingKey {
            centroid_id: u32::from_le_bytes(centroid_ids[0]),
            record_id: key,
        });
    }
    assignments.par_sort_unstable();
    Ok(assignments)
}

fn read_head_vectors(
    session: &Session,
    index: &TableIndex,
) -> Result<(Vec<u32>, VecVectorStore<u8>)> {
    let vector_table = index.head_config().high_fidelity_table();
    let dim = index.head_config().config().dimensions.get();
    let head_coder = vector_table.new_coder();
    let posting_coder = index
        .config()
        .posting_coder
        .new_coder(index.head_config().config().similarity);

    let mut cursor = session.get_or_create_typed_cursor::<i64, Vec<u8>>(&vector_table.name())?;
    let mut f32_buffer = vec![0.0f32; dim];
    let mut posting_buffer = vec![0u8; posting_coder.byte_len(dim)];
    let cap = cursor
        .largest_key()
        .unwrap_or(Err(Error::not_found_error()))? as usize
        + 1;
    let mut centroid_ids = Vec::with_capacity(cap);
    let mut vectors = VecVectorStore::<u8>::with_capacity(posting_buffer.len(), cap);
    for r in cursor.progress_with(progress_bar(cap, "read head vectors")) {
        let (key, value) = r?;
        head_coder.decode_to(&value, &mut f32_buffer);
        posting_coder.encode_to(&f32_buffer, &mut posting_buffer);
        centroid_ids.push(key as u32);
        vectors.push(&posting_buffer);
    }
    Ok((centroid_ids, vectors))
}

fn count_primary_assigned_vectors(
    session: &Session,
    index: &TableIndex,
    centroid_id: u32,
    posting_keys: &[PostingKey],
    head_centroid_ids: &[u32],
    head_vectors: &VecVectorStore<u8>,
) -> Result<(usize, usize)> {
    let posting_keys = &posting_keys[posting_keys.partition_point(|pk| pk.centroid_id < centroid_id)
        ..posting_keys.partition_point(|pk| pk.centroid_id < centroid_id + 1)];
    let mut pk_iter = posting_keys.iter().peekable();
    let _txn_guard = TransactionGuard::new(session, None)?;
    let mut postings_cursor =
        session.get_or_create_typed_cursor::<PostingKey, Vec<u8>>(&index.postings_table_name())?;
    postings_cursor.set_bounds(PostingKey::centroid_range(centroid_id))?;
    let dist_fn = index
        .config()
        .posting_coder
        .new_vector_distance(index.head_config().config().similarity);

    let mut correct = 0;
    let mut total = 0;
    for r in postings_cursor {
        let (key, vector) = r?;
        if pk_iter.next_if_eq(&&key).is_none() {
            continue;
        }

        let closest_centroid_id = head_centroid_ids
            .iter()
            .zip(head_vectors.iter())
            .map(|(id, v)| Neighbor::new(*id as i64, dist_fn.distance(&vector, v)))
            .min()
            .unwrap()
            .vertex() as u32;

        if closest_centroid_id == key.centroid_id {
            correct += 1;
        }
        total += 1;
    }
    Ok((correct, total))
}
