use std::{io, sync::Arc};

use clap::Args;
use easy_tiger::{
    posting_block::PostingBlock,
    spann::{TableIndex, TransactionIndex, centroid_stats::CentroidStats},
    vamana::{GraphVectorIndex, GraphVectorStore},
};
use histogram::Histogram;
use rayon::prelude::*;
use wt_mdb::{Connection, Error, Result};

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
    let max_value_power = max_value_power(&stats);
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
    // Collect all of the non-empty centroid ids and their representative vectors.
    let centroids: Vec<(i64, Vec<f32>)> = {
        let mut centroid_vectors = txn_idx.head().high_fidelity_vectors()?;
        let coder = centroid_vectors.format().coder();
        stats
            .assignment_counts_iter()
            .filter(|(_, c)| *c > 0)
            .map(|(ci, _)| {
                let ci = ci as i64;
                Ok((
                    ci,
                    coder.decode(
                        centroid_vectors
                            .get(ci)
                            .unwrap_or(Err(Error::not_found_error()))?,
                    ),
                ))
            })
            .collect::<Result<Vec<_>>>()?
    };

    let index = txn_idx.index();
    // NB: override distance function to compute euclidean. This is not vector magnitude but rather
    // squared magnitude, which is perfect for what we are doing.
    let similarity = vectors::VectorSimilarity::Euclidean;
    let posting_coder = index.config().posting_coder;
    let vector_len = index.posting_vector_len();

    /// Per-thread worker that reads posting blocks through a private transaction.
    struct Worker {
        reader: TransactionIndex,
    }

    // XXX this is still bullshit.
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
    }

    let progress = progress_bar(centroids.len(), "sum posting distances");
    let posting_stats: Vec<(i64, f64, usize)> = centroids
        .into_par_iter()
        .map_init(
            || Worker::new(connection, index),
            |worker, (centroid_id, centroid)| {
                let mut cursor = worker
                    .reader
                    .transaction()
                    .open_cursor::<u32, Vec<u8>>(worker.reader.index().postings_table_name())?;
                let data = unsafe { cursor.seek_exact_unsafe(centroid_id as u32) }
                    .unwrap_or(Err(Error::not_found_error()))
                    .unwrap();
                let block = PostingBlock::new(data, vector_len).unwrap();
                let dist_fn = posting_coder.query_distance_asymmetric(similarity, centroid);
                let sum: f64 = block.iter().map(|(_, v)| dist_fn.distance(v)).sum();
                progress.inc(1);
                Ok::<_, Error>((centroid_id, sum, block.len()))
            },
        )
        .collect::<Result<Vec<_>>>()?;
    progress.finish_using_style();

    let count = posting_stats.len() as f64;
    let min = posting_stats
        .iter()
        .copied()
        .fold(f64::INFINITY, |acc, x| acc.min(x.1));
    let max = posting_stats
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |acc, x| acc.max(x.1));
    let mean = posting_stats.iter().map(|x| x.1).sum::<f64>() / count;
    let stddev = (posting_stats
        .iter()
        .map(|x| (x.1 - mean).powi(2))
        .sum::<f64>()
        / count)
        .sqrt();
    let above_one_stddev = posting_stats
        .iter()
        .filter(|&&x| x.1 > mean + stddev)
        .count();
    let above_two_stddev = posting_stats
        .iter()
        .filter(|&&x| x.1 > mean + 2.0 * stddev)
        .count();

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

    if above_one_stddev > 0 {
        println!();
        println!("above one stddev histogram");
        print_histogram(
            max_value_power(stats),
            posting_stats
                .iter()
                .filter(|&&x| x.1 > mean + stddev)
                .map(|x| x.2 as u32),
        )?;
    }
    if above_two_stddev > 0 {
        println!();
        println!("above two stddev histogram");
        print_histogram(
            max_value_power(stats),
            posting_stats
                .iter()
                .filter(|&&x| x.1 > mean + stddev * 2.0)
                .map(|x| x.2 as u32),
        )?;
    }

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
        + 1
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
