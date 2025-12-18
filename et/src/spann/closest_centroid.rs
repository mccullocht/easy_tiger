use std::{io, sync::Arc};

use clap::Args;
use easy_tiger::{
    spann::{centroid_stats::CentroidStats, PostingKey, SessionIndexReader, TableIndex},
    vamana::search::GraphSearcher,
};
use rayon::prelude::*;
use wt_mdb::Connection;

use crate::ui::progress_bar;

#[derive(Args)]
pub struct ClosestCentroidArgs {
    /// Batch size for processing posting vectors.
    #[arg(long, default_value_t = 1000)]
    batch_size: usize,
}

pub fn closest_centroid(
    connection: Arc<Connection>,
    index_name: &str,
    args: ClosestCentroidArgs,
) -> io::Result<()> {
    let index = Arc::new(TableIndex::from_db(&connection, index_name)?);
    let session = connection.open_session()?;
    let mut postings_cursor =
        session.get_or_create_typed_cursor::<PostingKey, Vec<u8>>(&index.table_names.postings)?;
    let stats = CentroidStats::from_index_stats(&session, &index)?;

    let mut total = 0;
    let mut correct = 0;

    let progress = progress_bar(stats.vector_count(), "scanning postings");

    // XXX as written this does not work correctly for secondary assignments.
    let mut have_input = true;
    let mut batch = Vec::with_capacity(args.batch_size);
    let coder = index
        .config
        .posting_coder
        .new_coder(index.head_config().config().similarity);
    while have_input {
        batch.clear();
        for _ in 0..args.batch_size {
            match postings_cursor.next() {
                Some(Ok((key, value))) => {
                    batch.push((key.centroid_id, value));
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
                    let searcher = GraphSearcher::new(index.config.head_search_params);
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

    println!("Total vectors checked: {total}");
    println!("Correctly assigned to closest centroid: {correct}");
    if total > 0 {
        println!("Accuracy: {:.2}%", (correct as f64 / total as f64) * 100.0);
    }

    Ok(())
}
