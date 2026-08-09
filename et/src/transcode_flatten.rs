use std::{
    fs::File,
    io::{self, BufWriter, Write},
    num::NonZero,
    path::PathBuf,
};

use clap::Args;
use easy_tiger::input::{CompositeVectorStore, DerefVectorStore, VectorStore};
use memmap2::Mmap;
use rayon::prelude::*;
use vectors::{F32VectorCoding, VectorSimilarity};

use crate::ui::progress_bar;

const BATCH_SIZE: usize = 8192;

#[derive(Args)]
pub struct TranscodeFlattenArgs {
    /// Glob pattern matching BigANN-format (`<len,dim>` header) shard files containing vectors
    /// encoded with --format. Shards are concatenated in sorted path order.
    #[arg(short, long)]
    input: String,
    /// Encoding format of the input vectors. Also determines the dimensionality of the decoded
    /// output vectors.
    #[arg(short, long)]
    format: F32VectorCoding,
    /// Similarity function the input vectors were encoded with.
    #[arg(long, default_value = "l2")]
    similarity: VectorSimilarity,
    /// Path to a little-endian f32 center vector that was subtracted from each vector before
    /// encoding, if any. Required for formats that were originally quantized with a center.
    #[arg(long)]
    center: Option<PathBuf>,
    /// Output file to write flattened, decoded little-endian f32 vectors to.
    #[arg(short, long)]
    output: PathBuf,
}

pub fn transcode_flatten(args: TranscodeFlattenArgs) -> io::Result<()> {
    let mut paths = glob::glob(&args.input)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?
        .collect::<Result<Vec<PathBuf>, _>>()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
    paths.sort();
    if paths.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("no files matched glob {}", args.input),
        ));
    }

    let children = paths
        .into_iter()
        .map(DerefVectorStore::<u8, Mmap>::from_bigann_file)
        .collect::<io::Result<Vec<_>>>()?;
    let vectors = CompositeVectorStore::from_children(children).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "input shards are empty or do not share a common stride",
        )
    })?;

    let center = args.center.map(load_center).transpose()?;
    let coder = args.format.coder(args.similarity, center);

    let mut out = BufWriter::new(File::create(&args.output)?);
    let progress = progress_bar(vectors.len(), "transcode-flatten");

    for batch_start in (0..vectors.len()).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(vectors.len());
        let decoded: Vec<Vec<f32>> = (batch_start..batch_end)
            .into_par_iter()
            .map(|i| coder.decode(&vectors[i]))
            .collect();
        for v in &decoded {
            for &x in v {
                out.write_all(&x.to_le_bytes())?;
            }
        }
        progress.inc((batch_end - batch_start) as u64);
    }

    Ok(())
}

/// Load a single center vector from a file containing little-endian f32 values.
fn load_center(path: PathBuf) -> io::Result<Vec<f32>> {
    let mmap = unsafe { Mmap::map(&File::open(path)?)? };
    let len = NonZero::new(mmap.len() / std::mem::size_of::<f32>()).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "center file is empty")
    })?;
    let store = DerefVectorStore::<f32, Mmap>::new(mmap, len)?;
    Ok(store[0].to_vec())
}
