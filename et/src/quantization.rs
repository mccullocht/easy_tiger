mod distance_loss;
mod loss;
mod recall;
mod rotate;

use std::{io, path::PathBuf};

use clap::{Args, Subcommand};

use easy_tiger::input::{DerefVectorStore, SubsetViewVectorStore, VectorStore};
use half::slice::HalfFloatSliceExt;
use indicatif::ProgressIterator;
use memmap2::Mmap;
use vectors::f16;

use distance_loss::{DistanceLossArgs, distance_loss};
use loss::{LossArgs, loss};
use recall::{QuantizationRecallArgs, recall};
use rotate::{RotateArgs, rotate};

#[derive(Args)]
pub struct QuantizationArgs {
    #[command(subcommand)]
    command: Command,

    /// Input doc vector file: f16 vectors in BigANN format (an 8 byte `<len,dim>` header
    /// followed by little-endian f16 vector data).
    #[arg(short = 'v', long)]
    doc_vectors: PathBuf,

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
    Recall(QuantizationRecallArgs),
    /// Apply an orthogonal rotation to each vector and write to an output file.
    Rotate(RotateArgs),
}

pub fn quantization(args: QuantizationArgs) -> io::Result<()> {
    let vectors: DerefVectorStore<f16, Mmap> = DerefVectorStore::from_file(args.doc_vectors)?;

    if let Some(limit) = args.doc_limit
        && limit < vectors.len()
    {
        let vectors = SubsetViewVectorStore::new(&vectors, (0..limit).collect());
        cmd(args.command, &vectors)
    } else {
        cmd(args.command, &vectors)
    }
}

fn cmd(cmd: Command, vectors: &(impl VectorStore<Elem = f16> + Send + Sync)) -> io::Result<()> {
    match cmd {
        Command::Loss(args) => loss(args, vectors),
        Command::DistanceLoss(args) => distance_loss(args, vectors),
        Command::Recall(args) => recall(args, vectors),
        Command::Rotate(args) => rotate(args, vectors),
    }
}

fn compute_center(vectors: &impl VectorStore<Elem = f16>) -> Vec<f32> {
    let mut mean = vec![0.0; vectors.elem_stride()];
    let mut widened = vec![0.0f32; vectors.elem_stride()];
    for (i, v) in vectors
        .iter()
        .enumerate()
        .progress_with(crate::ui::progress_bar(vectors.len(), "Computing center"))
    {
        v.convert_to_f32_slice(&mut widened);
        for (d, m) in widened.iter().zip(mean.iter_mut()) {
            let delta = *d as f64 - *m;
            *m += delta / (i + 1) as f64;
        }
    }
    mean.into_iter().map(|m| m as f32).collect()
}
