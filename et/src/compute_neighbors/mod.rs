mod cpu;
#[cfg(feature = "wgpu")]
mod gpu;

use std::{
    fs::File,
    io::{self, BufWriter, Write},
    num::NonZero,
    path::PathBuf,
};

use clap::Args;
use vectors::VectorSimilarity;

use crate::neighbor_util::TopNeighbors;

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
    /// The output file is written in BigANN format: a <count,neighbors_len> header followed by
    /// one row for each vector in query_vectors, each row containing neighbors_len entries of
    /// Neighbor, an (i64, f64) tuple.
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
    #[cfg(feature = "wgpu")]
    if let Some(adapter) = gpu::try_adapter()
        && !args.force_cpu
    {
        return gpu::run(adapter, &args);
    }
    cpu::run(&args)
}

/// Write `results` (one entry per query) to `args.neighbors` in BigANN format: a
/// `<count,neighbors_len>` header followed by up to `neighbors_len` [`Neighbor`] entries per row.
fn write_neighbors(args: &ComputeNeighborsArgs, results: Vec<TopNeighbors>) -> io::Result<()> {
    let k = args.neighbors_len.get();
    let mut writer = BufWriter::new(File::create(&args.neighbors)?);
    writer.write_all(&(results.len() as u32).to_le_bytes())?;
    writer.write_all(&(k as u32).to_le_bytes())?;
    for neighbors in results.into_iter().map(TopNeighbors::into_neighbors) {
        for n in neighbors.into_iter().take(k) {
            writer.write_all(&<[u8; 16]>::from(n))?;
        }
    }
    Ok(())
}
