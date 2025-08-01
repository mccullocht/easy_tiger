use std::{fs::File, io, num::NonZero, path::PathBuf};

use clap::{Args, Parser, Subcommand};
use easy_tiger::{
    input::{DerefVectorStore, VectorStore},
    vectors::F32VectorCoding,
};
use memmap2::Mmap;
use rayon::prelude::*;

#[derive(Parser)]
#[command(version, about = "Tool for instrumenting vector quantization techniques", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Input vector file for partitioning.
    #[arg(short = 'v', long)]
    input_vectors: PathBuf,
    /// Vector dimensions for --input-vectors
    #[arg(short, long)]
    dimensions: NonZero<usize>,
}

#[derive(Subcommand)]
enum Command {
    QuantizationLoss(QuantizationLossArgs),
}

#[derive(Args)]
struct QuantizationLossArgs {
    /// Target format to measure the quantization loss of.
    #[arg(short, long)]
    format: F32VectorCoding,
}

fn quantization_loss(
    args: QuantizationLossArgs,
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
) -> io::Result<()> {
    let coder = args.format.new_coder();
    assert!(
        coder.decode(&coder.encode(&vectors[0])).is_some(),
        "specified vector format doesn't support decoding"
    );
    let (abs_error, sq_error) = (0..vectors.len())
        .into_par_iter()
        .map(|i| {
            let v = &vectors[i];
            let encoded = coder.encode(v);
            let q = coder.decode(&encoded).unwrap();
            let error = v
                .iter()
                .zip(q.iter())
                .map(|(d, q)| (*d - *q).abs() as f64)
                .sum::<f64>();
            (error, error * error)
        })
        .reduce(|| (0.0f64, 0.0f64), |a, b| (a.0 + b.0, a.1 + b.1));
    println!("Vectors: {}", vectors.len());
    println!(
        "Sum of absolute error: {:.6} squared error: {:.6}",
        abs_error, sq_error
    );
    println!(
        "Per vector absolute error: {:.6} squared error: {:.6}",
        abs_error / vectors.len() as f64,
        sq_error / vectors.len() as f64
    );
    Ok(())
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();

    let vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(cli.input_vectors)?)? },
        cli.dimensions,
    )?;
    match cli.command {
        Command::QuantizationLoss(args) => quantization_loss(args, &vectors),
    }
}
