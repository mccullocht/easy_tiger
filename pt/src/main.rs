use std::fs::File;
use std::io;
use std::num::NonZero;
use std::path::PathBuf;

use clap::Parser;
use easy_tiger::input::{DerefVectorStore, SubsetViewVectorStore, VecVectorStore, VectorStore};
use easy_tiger::kmeans::{self, Params, batch_kmeans, compute_assignments};
use histogram::Histogram;
use memmap2::Mmap;
use rand::{Rng, thread_rng};

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

    /// Keep splitting the largest centroid until no centroid is larger than this target size.
    #[arg(long)]
    iter_target_size: Option<usize>,
}

fn split_to_target_size<V: VectorStore<Elem = f32> + Send + Sync>(
    dataset: &V,
    mut centroids: VecVectorStore<f32>,
    mut assignments: Vec<(usize, f64)>,
    target_size: usize,
    batch_size: usize,
    params: &kmeans::Params,
    rng: &mut impl Rng,
) -> (VecVectorStore<f32>, Vec<(usize, f64)>) {
    loop {
        let centroid_assignments = assignments.iter().enumerate().fold(
            vec![vec![]; centroids.len()],
            |mut centroids, (i, (c, _))| {
                centroids[*c].push(i);
                centroids
            },
        );

        if centroid_assignments.iter().all(|c| c.len() <= target_size) {
            println!("  reached max target_size {}; terminating", target_size);
            break (centroids, assignments);
        }

        let mut new_centroids = VecVectorStore::new(centroids.elem_stride());
        for (c, centroid_vectors) in centroid_assignments.into_iter().enumerate() {
            if centroid_vectors.len() <= target_size {
                new_centroids.push(&centroids[c]);
                continue;
            }

            let k = centroid_vectors.len() / target_size + 1;
            println!(
                "  partitioning centroid {} of {} vectors into {} parts",
                c,
                centroid_vectors.len(),
                k,
            );
            let subset = SubsetViewVectorStore::new(dataset, centroid_vectors);
            let subset_centroids = match batch_kmeans(&subset, k, batch_size, params, rng) {
                Ok(c) => c,
                Err(e) => {
                    println!("k-means failed to converge!");
                    e
                }
            };

            for c in subset_centroids.iter() {
                new_centroids.push(c);
            }
        }

        // enumerate split centroids (old indices)
        // enumerate split generated centroids (new indices)
        // - if the previous assignment is in the first set need to totally recompute centroids.
        // - else compute against only the new centroids.

        centroids = new_centroids;
        println!("  recomputing assignments to {} centroids", centroids.len());
        // XXX I can make this less exhaustive -- if the assigned centroid was not split then I only
        // need to compute distances against the added centroids.
        assignments = compute_assignments(dataset, &centroids);
    }
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();

    let input_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(cli.input_vectors)?)? },
        cli.dimensions,
    )?;

    println!("Computing {} centroids", cli.num_partitions.get());
    let kemans_params = Params {
        iters: cli.iters.get(),
        init_iters: cli.init_iters.get(),
        epsilon: cli.epsilon,
        ..Default::default()
    };
    let mut centroids = match batch_kmeans(
        &input_vectors,
        cli.num_partitions.get(),
        cli.batch_size.get(),
        &kemans_params,
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
    let mut assignments = compute_assignments(&input_vectors, &centroids);

    if let Some(target_size) = cli.iter_target_size {
        (centroids, assignments) = split_to_target_size(
            &input_vectors,
            centroids,
            assignments,
            target_size,
            cli.batch_size.get(),
            &kemans_params,
            &mut thread_rng(),
        );
    }

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

    println!(
        "total {:9} min {:9} max {:9}",
        centroid_counts.iter().copied().sum::<usize>(),
        centroid_counts.iter().copied().min().unwrap(),
        centroid_counts.iter().copied().max().unwrap()
    );

    Ok(())
}
