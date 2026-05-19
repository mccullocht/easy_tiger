use std::{
    fs::File,
    io::{self, BufWriter, Write},
    num::NonZero,
    path::PathBuf,
};

use clap::{Args, ValueEnum};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::ui::progress_bar;

#[derive(Clone, ValueEnum)]
pub enum Distribution {
    /// Generate vectors with each dimension in [-1, 1], then unit normalize.
    F32Unit,
    /// Generate vectors where each dimension is a random i8, then convert to f32.
    I8,
    /// Generate vectors where each dimension is a random u8, then convert to f32.
    U8,
}

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
    /// Distribution used to generate vector values.
    #[arg(long, default_value = "f32-unit")]
    distribution: Distribution,
}

pub fn generate(args: GenerateArgs) -> io::Result<()> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(args.seed);
    let mut out = BufWriter::new(File::create(&args.output)?);
    let dims = args.dimensions.get();
    let count = args.count.get();
    let progress = progress_bar(count, "generate");

    let mut v = vec![0.0f32; dims];

    for _ in 0..count {
        match args.distribution {
            Distribution::F32Unit => {
                for x in &mut v {
                    *x = rng.random_range(-1.0f32..=1.0f32);
                }
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in &mut v {
                        *x /= norm;
                    }
                }
            }
            Distribution::I8 => {
                for x in &mut v {
                    *x = rng.random::<i8>() as f32;
                }
            }
            Distribution::U8 => {
                for x in &mut v {
                    *x = rng.random::<u8>() as f32;
                }
            }
        }
        for &x in &v {
            out.write_all(&x.to_le_bytes())?;
        }
        progress.inc(1);
    }

    Ok(())
}
