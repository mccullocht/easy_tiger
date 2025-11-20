mod loss;

use std::{fs::File, io, num::NonZero, path::PathBuf};

use clap::{Args, Subcommand};

use easy_tiger::input::{DerefVectorStore, VectorStore};
use indicatif::ProgressIterator;
use loss::{loss, LossArgs};
use memmap2::Mmap;

#[derive(Args)]
pub struct QuantizationArgs {
    #[command(subcommand)]
    command: Command,

    /// Input doc vector file for quantization.
    #[arg(short = 'v', long)]
    doc_vectors: PathBuf,
    /// Vector dimensions for --input-vectors
    #[arg(short, long)]
    dimensions: NonZero<usize>,
}

#[derive(Subcommand)]
pub enum Command {
    /// Compute loss resulting from quantization.
    Loss(LossArgs),
}

pub fn quantization(args: QuantizationArgs) -> io::Result<()> {
    let vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.doc_vectors)?)? },
        args.dimensions,
    )?;

    match args.command {
        Command::Loss(args) => loss(args, &vectors),
    }
}

fn compute_center(vectors: &impl VectorStore<Elem = f32>) -> Vec<f32> {
    let mut mean = vec![0.0; vectors.elem_stride()];
    for (i, v) in vectors
        .iter()
        .enumerate()
        .progress_count(vectors.len() as u64)
    {
        for (d, m) in v.iter().zip(mean.iter_mut()) {
            let delta = *d as f64 - *m;
            *m += delta / (i + 1) as f64;
        }
    }
    mean.into_iter().map(|m| m as f32).collect()
}
