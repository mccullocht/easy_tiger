use std::fs::File;
use std::io;
use std::num::NonZero;
use std::path::PathBuf;

use clap::Parser;
use easy_tiger::input::{DerefVectorStore, VectorStore};
use easy_tiger::kmeans::{Params, batch_kmeans, compute_assignments};
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

    /// Number of partitions to divide the input set into.
    #[arg(short = 'k', long)]
    num_partitions: NonZero<usize>,
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

    println!("Computing {} centroids", cli.num_partitions.get());
    let centroids = match batch_kmeans(
        &input_vectors,
        cli.num_partitions.get(),
        cli.batch_size.get(),
        &Params {
            iters: cli.iters.get(),
            init_iters: cli.init_iters.get(),
            epsilon: cli.epsilon,
            ..Default::default()
        },
        &mut thread_rng(),
    ) {
        Ok(c) => c,
        Err(e) => {
            println!("k-means failed to converge!");
            e
        }
    };

    println!(
        "Computing assignments for {} vectors into {} centroids",
        input_vectors.len(),
        cli.num_partitions.get()
    );
    let assignments = compute_assignments(&input_vectors, &centroids);

    let mut centroid_counts = vec![0usize; cli.num_partitions.get()];
    for (c, _) in assignments.iter() {
        centroid_counts[*c] += 1;
    }
    let mut histogram = Histogram::new(2, 18).unwrap();
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

    println!(
        "total {:9} min {:9} max {:9}",
        centroid_counts.iter().copied().sum::<usize>(),
        centroid_counts.iter().copied().min().unwrap(),
        centroid_counts.iter().copied().max().unwrap()
    );

    Ok(())
}
