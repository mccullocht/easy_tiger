use std::{io, sync::Arc};

use clap::Args;
use easy_tiger::{
    spann::{centroid_stats::CentroidStats, PostingKey, SessionIndexReader, TableIndex},
    vamana::search::GraphSearcher,
};
use indicatif::ProgressIterator;
use rayon::prelude::*;
use wt_mdb::{Connection, Result, Session};

use crate::ui::progress_bar;

#[derive(Args)]
pub struct VerifyPrimaryAssignmentsArgs {
    /// Batch size for processing posting vectors.
    #[arg(long, default_value_t = 1000)]
    batch_size: usize,
}

pub fn verify_primary_assignments(
    connection: Arc<Connection>,
    index_name: &str,
    args: VerifyPrimaryAssignmentsArgs,
) -> io::Result<()> {
    let index = Arc::new(TableIndex::from_db(&connection, index_name)?);
    let session = connection.open_session()?;
    let stats = CentroidStats::from_index_stats(&session, &index)?;

    let mut primary_assignments = read_primary_assignments(&session, &index, &stats)?
        .into_iter()
        .peekable();

    let mut postings_cursor =
        session.get_or_create_typed_cursor::<PostingKey, Vec<u8>>(&index.postings_table_name())?;

    let mut total = 0;
    let mut correct = 0;

    let progress = progress_bar(primary_assignments.len(), "scanning postings");

    let mut have_input = true;
    let mut batch = Vec::with_capacity(args.batch_size);
    let coder = index
        .config()
        .posting_coder
        .new_coder(index.head_config().config().similarity);
    while have_input {
        batch.clear();
        while batch.len() < args.batch_size {
            match postings_cursor.next() {
                Some(Ok((key, value))) => {
                    if primary_assignments.next_if_eq(&key).is_some() {
                        batch.push((key.centroid_id, value));
                    }
                }
                Some(Err(e)) => return Err(e.into()),
                None => {
                    have_input = false;
                    break;
                }
            }
        }

        if batch.is_empty() {
            break;
        }

        let batch_len = batch.len();
        let batch_correct_count = batch
            .par_iter()
            .by_uniform_blocks(batch_len.div_ceil(rayon::current_num_threads()))
            .map_init(
                || {
                    let session = connection.open_session().expect("failed to open session");
                    let searcher = GraphSearcher::new(index.config().head_search_params);
                    let buffer = vec![0.0f32; coder.dimensions(batch[0].1.len())];
                    (SessionIndexReader::new(&index, session), searcher, buffer)
                },
                |(reader, searcher, ref mut vector), (assigned_centroid, vector_bytes)| {
                    coder.decode_to(&vector_bytes, vector);

                    // Search head
                    let results = searcher
                        .search(&vector, reader.head_reader())
                        .expect("search failed");

                    if results
                        .first()
                        .is_some_and(|n| n.vertex() as u32 == *assigned_centroid)
                    {
                        1
                    } else {
                        0
                    }
                },
            )
            .sum::<usize>();

        correct += batch_correct_count;
        total += batch_len;
        progress.inc(batch_len as u64);
    }

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
        assignments.push(PostingKey::new(u32::from_le_bytes(centroid_ids[0]), key));
    }
    assignments.par_sort_unstable();
    Ok(assignments)
}
