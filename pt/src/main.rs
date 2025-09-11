use std::fs::File;
use std::io;
use std::num::NonZero;
use std::path::PathBuf;

use clap::Parser;
use easy_tiger::chrng;
use easy_tiger::input::DerefVectorStore;
use easy_tiger::vectors::VectorSimilarity;
use histogram::Histogram;
use indicatif::ProgressBar;
use memmap2::Mmap;

#[derive(Parser)]
#[command(version, about = "Tool for instrumenting vector partitioning techniques", long_about = None)]
struct Cli {
    /// Input vector file for partitioning.
    #[arg(short = 'v', long)]
    input_vectors: PathBuf,
    /// Vector dimensions in --input-vectors
    #[arg(short, long)]
    dimensions: NonZero<usize>,

    /// Maximum size of each centroid.
    #[arg(long)]
    max_centroid_len: NonZero<usize>,
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();

    let input_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(cli.input_vectors)?)? },
        cli.dimensions,
    )?;

    let progress = ProgressBar::new_spinner();

    let mut histogram = Histogram::new(2, 20).unwrap();
    let cluster_iter = chrng::clustering::ClusterIter::new(
        &input_vectors,
        cli.max_centroid_len.get(),
        VectorSimilarity::Euclidean.new_distance_function(),
        |i| progress.inc(i),
    );
    progress.inc(1);
    for (_, assignments) in cluster_iter {
        histogram.add(assignments.len() as u64, 1).unwrap();
    }

    for bucket in histogram.into_iter().filter(|b| b.count() > 0) {
        println!(
            "[{:5}..{:5}] {:4}",
            bucket.start(),
            bucket.end(),
            bucket.count()
        );
    }

    Ok(())
}
