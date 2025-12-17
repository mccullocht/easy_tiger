use std::fs::File;
use std::io;
use std::num::NonZero;
use std::path::PathBuf;

use clap::Args;
use easy_tiger::input::{DerefVectorStore, VectorStore};
use easy_tiger::kmeans::bp_kmeans_pp;
use memmap2::Mmap;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro128PlusPlus;

#[derive(Args)]
pub struct TestKmeansArgs {
    /// Dimension of the vectors.
    #[arg(long)]
    dimensions: usize,

    /// Path to input vectors.
    #[arg(long)]
    file: PathBuf,

    /// Maximum number of iterations.
    #[arg(long, default_value_t = 15)]
    max_iters: usize,

    /// Minimum cluster size.
    #[arg(long, default_value_t = 100)]
    min_cluster_size: usize,

    /// RNG seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

pub fn test_kmeans(args: TestKmeansArgs) -> io::Result<()> {
    let file = File::open(&args.file)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let stride = NonZero::new(args.dimensions)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "dimension must be > 0"))?;

    let store = DerefVectorStore::<f32, _>::new(mmap, stride)?;
    println!("Loaded {} vectors", store.len());

    let mut rng = Xoshiro128PlusPlus::seed_from_u64(args.seed);

    match bp_kmeans_pp(&store, args.max_iters, args.min_cluster_size, &mut rng) {
        Ok(centroids) => {
            println!("BP K-means converged. {} centroids found.", centroids.len());
        }
        Err(centroids) => {
            println!(
                "BP K-means failed to converge! {} centroids found.",
                centroids.len()
            );
        }
    }

    Ok(())
}
