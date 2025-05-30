use std::fs::File;
use std::io;
use std::num::NonZero;
use std::path::PathBuf;

use clap::Parser;
use easy_tiger::input::{DerefVectorStore, VectorStore};
use easy_tiger::kmeans::{Params, iterative_balanced_kmeans};
use histogram::Histogram;
use memmap2::Mmap;
use rand::thread_rng;

#[derive(Parser)]
#[command(version, about = "Tool for instrumenting vector partitioning techniques", long_about = None)]
struct Cli {
    /// Input vector file for partitioning.
    #[arg(short = 'v', long)]
    input_vectors: PathBuf,
    /// Vector dimensions in --input-vectors
    #[arg(short, long)]
    dimensions: NonZero<usize>,

    /// Minimum size of each centroid.
    #[arg(long)]
    min_centroid_size: NonZero<usize>,
    /// Maximum size of each centroid.
    #[arg(long)]
    max_centroid_size: NonZero<usize>,
    /// Run up to this many k-means iterations before terminating.
    #[arg(short, long, default_value_t = NonZero::new(15).unwrap())]
    iters: NonZero<usize>,
    /// Exit early from iteration if new centroids are within epsilon of the previous iteration.
    #[arg(long, default_value_t = 0.01)]
    epsilon: f64,
    /// Run this many initializations of the centroids before proceeding.
    #[arg(long, default_value_t = NonZero::new(3).unwrap())]
    init_iters: NonZero<usize>,
    /// Size of k-means batches. Larger numbers improve convergence rate but are more expensive.
    #[arg(short, long, default_value_t = NonZero::new(10_000).unwrap())]
    batch_size: NonZero<usize>,
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();

    let input_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(cli.input_vectors)?)? },
        cli.dimensions,
    )?;

    let centroid_size_bounds = cli.min_centroid_size.get()..=cli.max_centroid_size.get();
    println!(
        "Computing centroids size bounds=({:?})",
        centroid_size_bounds
    );
    let kmeans_params = Params {
        iters: cli.iters.get(),
        init_iters: cli.init_iters.get(),
        epsilon: cli.epsilon,
        ..Default::default()
    };

    let (centroids, assignments) = iterative_balanced_kmeans(
        &input_vectors,
        centroid_size_bounds,
        32,
        cli.batch_size.get(),
        &kmeans_params,
        &mut thread_rng(),
    );

    let centroid_counts =
        assignments
            .iter()
            .fold(vec![0usize; centroids.len()], |mut counts, (c, _)| {
                counts[*c] += 1;
                counts
            });

    let mut histogram = Histogram::new(2, 20).unwrap();
    for c in centroid_counts.iter() {
        histogram.add(*c as u64, 1).unwrap();
    }

    for bucket in histogram.into_iter().filter(|b| b.count() > 0) {
        println!(
            "[{:5}..{:5}] {:4}",
            bucket.start(),
            bucket.end(),
            bucket.count()
        );
    }

    let total_dist = assignments.iter().map(|(_, d)| *d).sum::<f64>();
    println!(
        "centroids {:5} vectors {:9} min {:9} max {:9} total dist {:8.3} avg dist {:8.6}",
        centroid_counts.len(),
        centroid_counts.iter().copied().sum::<usize>(),
        centroid_counts.iter().copied().min().unwrap(),
        centroid_counts.iter().copied().max().unwrap(),
        total_dist,
        total_dist / input_vectors.len() as f64
    );

    Ok(())
}
