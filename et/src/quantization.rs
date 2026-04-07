mod distance_loss;
mod loss;
mod recall;
mod rotate;

use std::{fs::File, io, num::NonZero, path::PathBuf};

use clap::{Args, Subcommand};

use easy_tiger::input::{DerefVectorStore, SubsetViewVectorStore, VectorStore};
use indicatif::ProgressIterator;
use memmap2::Mmap;

use distance_loss::{distance_loss, DistanceLossArgs};
use loss::{loss, LossArgs};
use recall::{recall, RecallArgs};
use rotate::{rotate, RotateArgs};

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

    /// Maximum number of vectors to process.
    #[arg(long)]
    doc_limit: Option<usize>,
}

#[derive(Subcommand)]
pub enum Command {
    /// Compute loss resulting from quantization.
    Loss(LossArgs),
    /// Compute loss in distance computation resulting from quantization.
    DistanceLoss(DistanceLossArgs),
    /// Compute recall difference with quantization using exhaustive search.
    Recall(RecallArgs),
    /// Apply an orthogonal rotation to each vector and write to an output file.
    Rotate(RotateArgs),
}

pub fn quantization(args: QuantizationArgs) -> io::Result<()> {
    let vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.doc_vectors)?)? },
        args.dimensions,
    )?;

    if let Some(limit) = args.doc_limit && limit < vectors.len() {
        let vectors = SubsetViewVectorStore::new(&vectors, (0..limit).collect());
        cmd(args.command, &vectors)
    } else {
        cmd(args.command, &vectors)
    }
}

fn cmd(cmd: Command, vectors: &(impl VectorStore<Elem = f32> + Send + Sync)) -> io::Result<()> {
    match cmd {
        Command::Loss(args) => loss(args, vectors),
        Command::DistanceLoss(args) => distance_loss(args, vectors),
        Command::Recall(args) => recall(args, vectors),
        Command::Rotate(args) => rotate(args, vectors),
    }
}

fn compute_center(vectors: &impl VectorStore<Elem = f32>) -> Vec<f32> {
    let mut mean = vec![0.0; vectors.elem_stride()];
    for (i, v) in vectors
        .iter()
        .enumerate()
        .progress_with(crate::ui::progress_bar(vectors.len(), "Computing center"))
    {
        for (d, m) in v.iter().zip(mean.iter_mut()) {
            let delta = *d as f64 - *m;
            *m += delta / (i + 1) as f64;
        }
    }
    mean.into_iter().map(|m| m as f32).collect()
}
