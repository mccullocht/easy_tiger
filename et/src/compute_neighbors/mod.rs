mod cpu;
mod gpu;

use std::{io, num::NonZero, path::PathBuf};

use clap::Args;
use vectors::VectorSimilarity;

#[derive(Args)]
pub struct ComputeNeighborsArgs {
    /// Path to numpy formatted little-endian float vectors.
    #[arg(long)]
    query_vectors: PathBuf,
    /// Maximum number of query vectors to process.
    #[arg(long)]
    query_limit: Option<usize>,
    /// Path to numpy formatted little-endian float vectors.
    #[arg(long)]
    doc_vectors: PathBuf,
    /// Maximum number of doc vectors to process.
    #[arg(long)]
    doc_limit: Option<usize>,

    /// Number of dimensions for both query and doc vectors.
    #[arg(short, long)]
    dimensions: NonZero<usize>,
    /// Similarity function to use.
    #[arg(short, long)]
    similarity: VectorSimilarity,

    /// Path to neighbors file to write.
    ///
    /// The output file will contain one row for each vector in query_vectors. Within each row
    /// there will be neighbors_len entries of Neighbor, an (i64, f64) tuple.
    #[arg(short, long)]
    neighbors: PathBuf,
    /// Number of neighbors for each query in the neighbors file.
    #[arg(long, default_value_t = NonZero::new(100).unwrap())]
    neighbors_len: NonZero<usize>,

    /// If true, force the computation to run on the CPU even if a GPU adapter is available.
    #[arg(long, default_value_t = false)]
    force_cpu: bool,
}

pub fn compute_neighbors(args: ComputeNeighborsArgs) -> io::Result<()> {
    if let Some(adapter) = gpu::try_adapter()
        && !args.force_cpu
    {
        gpu::run(adapter, &args)
    } else {
        cpu::run(&args)
    }
}
