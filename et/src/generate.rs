use std::{
    fs::File,
    io::{self, BufWriter, Write},
    num::NonZero,
    path::PathBuf,
};

use clap::Args;
use half::{f16, slice::HalfFloatSliceExt};
use indicatif::ProgressIterator;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::ui::progress_bar;

#[derive(Args)]
pub struct GenerateArgs {
    /// Output file to write generated vectors as little-endian f32 values.
    #[arg(short, long)]
    output: PathBuf,
    /// Number of dimensions per vector.
    #[arg(short, long)]
    dimensions: NonZero<usize>,
    /// Number of vectors to generate.
    #[arg(short, long)]
    count: NonZero<usize>,
    /// Random seed for reproducible generation.
    #[arg(short, long)]
    seed: u64,
}

pub fn generate(args: GenerateArgs) -> io::Result<()> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(args.seed);
    let dims = args.dimensions.get();
    let count = args.count.get();

    let mut out = BufWriter::new(File::create(&args.output)?);
    out.write_all(&(count as u32).to_le_bytes())?;
    out.write_all(&(dims as u32).to_le_bytes())?;

    let mut v = vec![0.0f32; dims];
    let mut v16 = vec![f16::ZERO; dims];
    for _ in (0..count).progress_with(progress_bar(count, "generate")) {
        for x in &mut v {
            *x = StandardNormal.sample(&mut rng);
        }
        v = vectors::float32::l2_normalize(v).0.into_owned();
        v16.convert_from_f32_slice(&v);
        for &x in &v16 {
            out.write_all(&x.to_le_bytes())?;
        }
    }

    Ok(())
}
