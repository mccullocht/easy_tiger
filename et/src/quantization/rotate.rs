use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::PathBuf,
};

use clap::Args;
use easy_tiger::input::VectorStore;
use half::slice::HalfFloatSliceExt;
use vectors::{f16, rotate::Rotator};

use crate::ui::progress_bar;

#[derive(Args)]
pub struct RotateArgs {
    /// Random seed for the rotation. Must remain fixed for all vectors that will be compared.
    #[arg(long)]
    seed: u64,
    /// Output file to write rotated f16 vectors to in BigANN format.
    #[arg(short, long)]
    output: PathBuf,
}

pub fn rotate(
    args: RotateArgs,
    vectors: &(impl VectorStore<Elem = f16> + Send + Sync),
) -> io::Result<()> {
    let rotator = Rotator::new(vectors.elem_stride(), args.seed);
    let mut out = BufWriter::new(File::create(&args.output)?);
    out.write_all(&(vectors.len() as u32).to_le_bytes())?;
    out.write_all(&(vectors.elem_stride() as u32).to_le_bytes())?;

    let progress = progress_bar(vectors.len(), "rotate");
    let mut widened = vec![0.0f32; vectors.elem_stride()];
    let mut narrowed = vec![f16::ZERO; vectors.elem_stride()];
    for i in 0..vectors.len() {
        vectors[i].convert_to_f32_slice(&mut widened);
        let rotated = rotator.forward(&widened);
        narrowed.convert_from_f32_slice(&rotated);
        for &x in narrowed.iter() {
            out.write_all(&x.to_le_bytes())?;
        }
        progress.inc(1);
    }

    Ok(())
}
