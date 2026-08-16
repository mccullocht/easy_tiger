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
    /// Path to f16 vectors in BigANN format: an 8 byte `<len,dim>` header followed by
    /// little-endian f16 vector data.
    #[arg(long)]
    query_vectors: PathBuf,
    /// Maximum number of query vectors to process.
    #[arg(long)]
    query_limit: Option<usize>,
    /// Path to f16 vectors in BigANN format: an 8 byte `<len,dim>` header followed by
    /// little-endian f16 vector data.
    #[arg(long)]
    doc_vectors: PathBuf,
    /// Maximum number of doc vectors to process.
    #[arg(long)]
    doc_limit: Option<usize>,

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
        if gpu::supports_f16(&adapter) {
            return gpu::run(adapter, &args);
        }
        tracing::warn!(
            "GPU adapter {} does not support f16 shaders; falling back to CPU",
            adapter.get_info().name
        );
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
