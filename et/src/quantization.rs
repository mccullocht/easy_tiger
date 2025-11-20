mod loss;

use std::io;

use clap::{Args, Subcommand};

use easy_tiger::input::VectorStore;
use indicatif::ProgressIterator;
use loss::{loss, LossArgs};

#[derive(Args)]
pub struct QuantizationArgs {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Compute loss resulting from quantization.
    Loss(LossArgs),
}

pub fn quantization(args: QuantizationArgs) -> io::Result<()> {
    match args.command {
        Command::Loss(args) => loss(args),
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
