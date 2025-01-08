use std::{
    fs::File,
    io::{self, BufWriter, Write},
    num::NonZero,
    path::PathBuf,
    time::Duration,
};

use clap::Args;
use easy_tiger::{
    input::{DerefVectorStore, VectorStore},
    partition_reorder,
};
use memmap2::Mmap;
use rand::thread_rng;

use crate::ui::progress_bar;

#[derive(Args)]
pub struct PartitionReorderArgs {
    /// Path to the input vectors to bulk ingest.
    #[arg(short, long)]
    f32_vectors: PathBuf,
    /// Number of dimensions in input vectors.
    #[arg(short, long)]
    dimensions: NonZero<usize>,
    /// Output reordered vectors.
    #[arg(short, long)]
    out_vectors: PathBuf,

    /// Number of children in each node of the partition tree. Must be >= 2.
    #[arg(long)]
    k: usize,
    /// Maximum numbers of samples in a leaf clustering node.
    #[arg(long)]
    m: usize,
}

pub fn partition_reorder(args: PartitionReorderArgs) -> io::Result<()> {
    let f32_vectors = DerefVectorStore::<f32, _>::new(
        unsafe { Mmap::map(&File::open(args.f32_vectors)?) }?,
        args.dimensions,
    )?;

    let progress = progress_bar(0, Some("partition"));
    progress.enable_steady_tick(Duration::from_secs(1));
    let reorder_map = partition_reorder::partition_reorder(
        &f32_vectors,
        args.k,
        args.m,
        &partition_reorder::Params::default(),
        &mut thread_rng(),
        || progress.inc(1),
        || progress.inc_length(1),
    );
    progress.finish();

    let progress = progress_bar(f32_vectors.len(), Some("write"));
    let mut w = BufWriter::with_capacity(1 << 20, File::create(args.out_vectors)?);
    for vector in reorder_map.into_iter().map(|i| &f32_vectors[i]) {
        for d in vector {
            w.write_all(&d.to_le_bytes())?;
        }
        progress.inc(1);
    }
    progress.finish();

    Ok(())
}
