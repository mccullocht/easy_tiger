use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::PathBuf,
};

use clap::Args;
use easy_tiger::input::VectorStore;
use rayon::prelude::*;
use vectors::rotate::Rotator;

use crate::ui::progress_bar;

const BATCH_SIZE: usize = 8192;

#[derive(Args)]
pub struct RotateArgs {
    /// Random seed for the rotation. Must remain fixed for all vectors that will be compared.
    #[arg(long)]
    seed: u64,
    /// Output file to write rotated vectors to as little-endian f32 values.
    #[arg(short, long)]
    output: PathBuf,
}

pub fn rotate(
    args: RotateArgs,
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
) -> io::Result<()> {
    let rotator = Rotator::new(vectors.elem_stride(), args.seed);
    let mut out = BufWriter::new(File::create(&args.output)?);
    let progress = progress_bar(vectors.len(), "rotate");

    for batch_start in (0..vectors.len()).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(vectors.len());
        let rotated: Vec<Vec<f32>> = (batch_start..batch_end)
            .into_par_iter()
            .map(|i| rotator.forward(&vectors[i]))
            .collect();
        for v in &rotated {
            for &x in v {
                out.write_all(&x.to_le_bytes())?;
            }
        }
        progress.inc((batch_end - batch_start) as u64);
    }

    Ok(())
}
