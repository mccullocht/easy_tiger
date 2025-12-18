use std::{
    io,
    sync::{atomic::AtomicUsize, Arc},
};

use clap::Args;
use easy_tiger::{
    spann::{PostingKey, SessionIndexReader, TableIndex},
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

    let total = AtomicUsize::new(0);
    let correct = AtomicUsize::new(0);

    let progress = progress_bar(0, "scanning postings");

    loop {
        let mut batch = Vec::with_capacity(args.batch_size);
        for _ in 0..args.batch_size {
            match postings_cursor.next() {
                Some(Ok((key, value))) => {
                    batch.push((key.centroid_id, value));
                }
                Some(Err(e)) => return Err(e.into()),
                None => break,
            }
        }

        if batch.is_empty() {
            break;
        }

        let batch_len = batch.len();
        let batch_correct_count = batch
            .par_iter()
            .map_init(
                || {
                    let session = connection.open_session().expect("failed to open session");
                    let searcher = GraphSearcher::new(index.config.head_search_params);
                    (SessionIndexReader::new(&index, session), searcher)
                },
                |(reader, searcher), (assigned_centroid, vector_bytes)| {
                    // Decode vector
                    // Optimization: Coder creation is cheap-ish?
                    let coder = index
                        .config
                        .posting_coder
                        .new_coder(index.head_config().config().similarity);

                    let vector = coder.decode(vector_bytes);

                    // Search head
                    // searcher needs to be mutable.
                    let results = searcher
                        .search(&vector, reader.head_reader())
                        .expect("search failed");

                    if let Some(best) = results.first() {
                        if best.vertex() as u32 == *assigned_centroid {
                            1
                        } else {
                            0
                        }
                    } else {
                        0
                    }
                },
            )
            .sum::<usize>();

        correct.fetch_add(batch_correct_count, std::sync::atomic::Ordering::Relaxed);
        total.fetch_add(batch_len, std::sync::atomic::Ordering::Relaxed);
        progress.inc(batch_len as u64);
    }

    progress.finish();

    let total_val = total.load(std::sync::atomic::Ordering::SeqCst);
    let correct_val = correct.load(std::sync::atomic::Ordering::SeqCst);

    println!("Total vectors checked: {}", total_val);
    println!("Correctly assigned to closest centroid: {}", correct_val);
    if total_val > 0 {
        println!(
            "Accuracy: {:.2}%",
            (correct_val as f64 / total_val as f64) * 100.0
        );
    }

    Ok(())
}
