use std::{
    io,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use clap::Args;
use easy_tiger::{
    posting_block::PostingBlock,
    spann::{TableIndex, TransactionIndex, centroid_stats::CentroidStats},
};
use histogram::Histogram;
use rayon::prelude::*;
use wt_mdb::Connection;

use crate::ui::progress_bar;

#[derive(Args)]
pub struct CentroidStatsArgs {
    /// Also compute posting-level stats: for each posting list, the sum of each posting vector's
    /// distance to its centroid, then distribution stats (min/max/mean/stddev) over those sums.
    #[arg(long, default_value_t = false)]
    posting_stats: bool,
}

pub fn centroid_stats(
    connection: Arc<Connection>,
    index_name: &str,
    args: CentroidStatsArgs,
) -> io::Result<()> {
    let index = Arc::new(TableIndex::from_db(&connection, index_name)?);
    let txn_idx = TransactionIndex::new(&index, connection.begin_transaction(None)?);
    let stats = CentroidStats::from_index_stats(&txn_idx)?;

    println!("Head contains {} centroids", stats.centroid_count());
    println!("{} tail posting entries", stats.vector_count());
    let max_value_power = max_value_power(&stats) + 1;
    println!("Assignments per centroid:");
    print_histogram(
        max_value_power,
        stats.assignment_counts_iter().map(|(_, c)| c),
    )?;

    if args.posting_stats {
        print_posting_stats(&txn_idx, &connection, &stats)?;
    }
    Ok(())
}

/// For every posting list, sum the distance of each posting vector to its centroid and report
/// distribution stats over those sums.
///
/// Distances use the index's posting coder against the centroid's high fidelity vector, matching
/// the (prepared) space search scores posting vectors in. Both the head and posting stores are
/// fully scanned, so this is considerably more expensive than the assignment stats above.
fn print_posting_stats(
    txn_idx: &TransactionIndex,
    connection: &Arc<Connection>,
    stats: &CentroidStats,
) -> io::Result<()> {
    // Load every centroid vector from the head index's high fidelity table, sorted by centroid id
    // for binary search.
    let hf_table = txn_idx.head().index().high_fidelity_table();
    let hf_coder = hf_table.new_coder();
    let mut centroids: Vec<(i64, Vec<f32>)> = Vec::new();
    {
        let cursor = txn_idx
            .head()
            .transaction()
            .open_record_cursor(hf_table.name())?;
        for item in cursor {
            let (key, encoded) = item?;
            if key < 0 {
                // Skip the entry point record stored at key -1.
                continue;
            }
            centroids.push((key, hf_coder.decode(&encoded)));
        }
    }
    centroids.sort_unstable_by_key(|(id, _)| *id);

    let index = txn_idx.index();
    // NB: override distance function to compute euclidean. This is not vector magnitude but rather
    // squared magnitude, which is perfect for what we are doing.
    let similarity = vectors::VectorSimilarity::Euclidean;
    let posting_coder = index.config().posting_coder;
    let vector_len = index.posting_vector_len();

    // The posting lists to process: every centroid with at least one assigned vector. Rayon
    // workers read each posting block through their own transaction, so the heavy block data is
    // pulled on demand and never held in memory for the whole index.
    let posting_lists: Vec<u32> = stats
        .assignment_counts_iter()
        .filter_map(|(id, count)| (count > 0).then_some(id as u32))
        .collect();

    /// Per-thread worker that reads posting blocks through a private transaction.
    struct Worker {
        reader: TransactionIndex,
    }

    impl Worker {
        fn new(connection: &Arc<Connection>, index: &Arc<TableIndex>) -> Self {
            Self {
                reader: TransactionIndex::new(
                    index,
                    connection
                        .begin_transaction(None)
                        .expect("failed to begin a read transaction"),
                ),
            }
        }

        /// Read the posting block for `centroid_id`, or `None` if it has no posting list.
        fn posting(&mut self, centroid_id: u32) -> io::Result<Option<Vec<u8>>> {
            let mut cursor = self
                .reader
                .transaction()
                .open_cursor::<u32, Vec<u8>>(self.reader.index().postings_table_name())?;
            match cursor.seek_exact(centroid_id) {
                Some(Ok(data)) => Ok(Some(data)),
                Some(Err(e)) => Err(e.into()),
                None => Ok(None),
            }
        }
    }

    let progress = progress_bar(posting_lists.len(), "sum posting distances");
    let empty_posting = AtomicUsize::new(0);
    let missing_centroid = AtomicUsize::new(0);
    let centroids = &centroids;
    let progress = &progress;
    let empty_posting = &empty_posting;
    let missing_centroid = &missing_centroid;
    let pending: Vec<Option<f64>> = posting_lists
        .into_par_iter()
        .map_init(
            || Worker::new(connection, index),
            move |worker, centroid_id| -> io::Result<Option<f64>> {
                // XXX just awful, awful garbage.
                let data = worker.posting(centroid_id)?;
                progress.inc(1);
                let Some(data) = data else {
                    return Ok(None);
                };
                let Some(block) = PostingBlock::new(&data, vector_len) else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("malformed posting block for centroid {centroid_id}"),
                    ));
                };
                if block.is_empty() {
                    empty_posting.fetch_add(1, Ordering::Relaxed);
                    return Ok(None);
                }
                let centroid =
                    match centroids.binary_search_by(|(id, _)| id.cmp(&(centroid_id as i64))) {
                        Ok(i) => centroids[i].1.as_slice(),
                        Err(_) => {
                            missing_centroid.fetch_add(1, Ordering::Relaxed);
                            return Ok(None);
                        }
                    };
                let dist_fn = posting_coder.query_distance_asymmetric(similarity, centroid);
                let sum = block.iter().map(|(_, v)| dist_fn.distance(v)).sum();
                Ok(Some(sum))
            },
        )
        .collect::<io::Result<Vec<Option<f64>>>>()?;
    progress.finish_using_style();

    let sums: Vec<f64> = pending.into_iter().flatten().collect();
    let empty_posting = empty_posting.load(Ordering::Relaxed);
    let missing_centroid = missing_centroid.load(Ordering::Relaxed);

    println!("\nPosting distance sums over {} posting lists", sums.len());
    if missing_centroid > 0 {
        println!("  {missing_centroid} posting lists had no centroid vector; skipped");
    }
    if empty_posting > 0 {
        println!("  {empty_posting} empty posting lists; skipped");
    }
    if sums.is_empty() {
        return Ok(());
    }

    let count = sums.len() as f64;
    let min = sums.iter().copied().fold(f64::INFINITY, f64::min);
    let max = sums.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = sums.iter().sum::<f64>() / count;
    let stddev = (sums.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / count).sqrt();
    let above_one_stddev = sums.iter().filter(|s| **s > mean + stddev).count();
    let above_two_stddev = sums.iter().filter(|s| **s > mean + 2.0 * stddev).count();

    println!("  min:    {min:.4}");
    println!("  max:    {max:.4}");
    println!("  mean:   {mean:.4}");
    println!("  stddev: {stddev:.4}");
    println!(
        "  centroids 1 stddev above mean: {above_one_stddev} ({:.1}%)",
        100.0 * above_one_stddev as f64 / count
    );
    println!(
        "  centroids 2 stddevs above mean: {above_two_stddev} ({:.1}%)",
        100.0 * above_two_stddev as f64 / count
    );
    Ok(())
}

fn max_value_power(stats: &CentroidStats) -> u8 {
    (stats
        .assignment_counts_iter()
        .map(|(_, c)| c)
        .max()
        .unwrap()
        .next_power_of_two()
        .ilog2() as u8)
        .max(3)
}

fn print_histogram(max_value_power: u8, input: impl Iterator<Item = u32>) -> io::Result<()> {
    let mut histogram = Histogram::new(2, max_value_power).unwrap();
    for c in input {
        histogram.add(c.into(), 1).unwrap();
    }
    use std::io::Write;
    let mut lock = std::io::stdout().lock();
    for b in histogram.into_iter().filter(|b| b.count() > 0) {
        writeln!(lock, "[{:5}..{:5}] {:7}", b.start(), b.end(), b.count())?;
    }
    Ok(())
}
