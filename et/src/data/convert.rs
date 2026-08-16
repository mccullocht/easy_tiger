use std::{
    fs::File,
    io::{self, BufWriter, Write},
    num::NonZero,
    path::PathBuf,
};

use clap::Args;
use easy_tiger::input::{DerefVectorStore, VectorStore};
use memmap2::Mmap;
use vectors::{F32VectorCoding, VectorSimilarity};

use crate::ui::progress_bar;

#[derive(Args)]
pub struct ConvertArgs {
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

pub fn convert(args: ConvertArgs) -> io::Result<()> {
    let vectors: DerefVectorStore<f32, Mmap> =
        DerefVectorStore::from_file_with_stride(args.input, args.dimensions)?;
    let coder = F32VectorCoding::F16.coder(VectorSimilarity::Euclidean, None);

    let mut out = BufWriter::new(File::create(&args.output)?);
    out.write_all(&(vectors.len() as u32).to_le_bytes())?;
    out.write_all(&(args.dimensions.get() as u32).to_le_bytes())?;

    let progress = progress_bar(vectors.len(), "convert");
    let mut buf = vec![0u8; coder.byte_len(args.dimensions.get())];
    for v in vectors.iter() {
        coder.encode_to(v, &mut buf);
        out.write_all(&buf)?;
        progress.inc(1);
    }

    Ok(())
}
