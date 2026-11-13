use std::{
    fs::File,
    io::{self, BufWriter, Write},
    num::NonZero,
    path::PathBuf,
};

use clap::Args;
use easy_tiger::input::{DerefVectorStore, VectorStore};
use half::{f16, slice::HalfFloatSliceExt};
use memmap2::Mmap;

use crate::ui::progress_bar;

#[derive(Args)]
pub struct ConvertF16Args {
    /// Input file containing raw little-endian f32 vectors with no header.
    #[arg(short, long)]
    input: PathBuf,
    /// Vector dimensionality of the input vectors.
    #[arg(short, long)]
    dimensions: NonZero<usize>,
    /// Output file to write f16 vectors in BigANN format (<len,dim> header followed by data).
    #[arg(short, long)]
    output: PathBuf,
}

pub fn convert_f16(args: ConvertF16Args) -> io::Result<()> {
    let vectors: DerefVectorStore<f32, Mmap> =
        DerefVectorStore::from_file_with_stride(args.input, args.dimensions)?;

    let mut out = BufWriter::new(File::create(&args.output)?);
    out.write_all(&(vectors.len() as u32).to_le_bytes())?;
    out.write_all(&(args.dimensions.get() as u32).to_le_bytes())?;

    let progress = progress_bar(vectors.len(), "convert");
    let mut buf = vec![f16::ZERO; args.dimensions.get()];
    for v in vectors.iter() {
        buf.convert_from_f32_slice(v);
        for &x in buf.iter() {
            out.write_all(&x.to_le_bytes())?;
        }
        progress.inc(1);
    }

    Ok(())
}
