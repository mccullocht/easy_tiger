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
    ///
    /// If this refers to the same file as `doc_vectors`, each query's own vector is filtered
    /// out of its neighbor list (it would otherwise always be its own nearest neighbor at
    /// distance 0).
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
    /// one row for each vector in query_vectors, each row containing neighbors_len vertex ids as
    /// little-endian u32.
    #[arg(short, long)]
    neighbors: PathBuf,
    /// Number of neighbors for each query in the neighbors file.
    #[arg(long, default_value_t = NonZero::new(100).unwrap())]
    neighbors_len: NonZero<usize>,

    /// If true, force the computation to run on the CPU even if a GPU adapter is available.
    #[arg(long, default_value_t = false)]
    force_cpu: bool,
}

/// Returns true if `query_vectors` and `doc_vectors` refer to the same file, in which case a
/// query vector will also appear among the doc vectors and should be excluded from its own
/// neighbor list.
pub(super) fn filters_self(args: &ComputeNeighborsArgs) -> bool {
    match (
        std::fs::canonicalize(&args.query_vectors),
        std::fs::canonicalize(&args.doc_vectors),
    ) {
        (Ok(q), Ok(d)) => q == d,
        _ => false,
    }
}

/// Number of neighbors to retain per query while accumulating results. When `query_vectors` and
/// `doc_vectors` are the same file, one extra slot is kept so a self-match can be filtered out in
/// [`write_neighbors`] while still leaving `neighbors_len` real neighbors behind.
pub(super) fn top_k(args: &ComputeNeighborsArgs) -> usize {
    args.neighbors_len.get() + if filters_self(args) { 1 } else { 0 }
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
/// `<count,neighbors_len>` header followed by up to `neighbors_len` little-endian `u32` vertex
/// ids per row.
fn write_neighbors(args: &ComputeNeighborsArgs, results: Vec<TopNeighbors>) -> io::Result<()> {
    let k = args.neighbors_len.get();
    let self_filter = filters_self(args);
    let mut writer = BufWriter::new(File::create(&args.neighbors)?);
    writer.write_all(&(results.len() as u32).to_le_bytes())?;
    writer.write_all(&(k as u32).to_le_bytes())?;
    for (q, neighbors) in results
        .into_iter()
        .map(TopNeighbors::into_neighbors)
        .enumerate()
    {
        let filtered = neighbors
            .into_iter()
            .filter(|n| !self_filter || n.vertex() != q as i64)
            .take(k);
        for n in filtered {
            writer.write_all(&(n.vertex() as u32).to_le_bytes())?;
        }
    }
    Ok(())
}
