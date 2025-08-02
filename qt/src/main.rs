use std::{fs::File, io, num::NonZero, path::PathBuf};

use clap::{Args, Parser, Subcommand};
use easy_tiger::{
    input::{DerefVectorStore, SubsetViewVectorStore, VectorStore},
    vectors::{F32VectorCoding, lucene::ScalarQuantizerVectorCoder},
};
use memmap2::Mmap;
use rand::SeedableRng;
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
    /// Compute vector quantization loss.
    QuantizationLoss(QuantizationLossArgs),
    /// Train Lucene ScalarQuantizer parameters.
    TrainLuceneScalarQuantizer(TrainLuceneScalarQuantizerArgs),
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

#[derive(Args)]
struct TrainLuceneScalarQuantizerArgs {
    #[arg(long, default_value_t = NonZero::new(25_000).unwrap())]
    sample_size: NonZero<usize>,
}

fn train_lucene_scalar_quantizer(
    args: TrainLuceneScalarQuantizerArgs,
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
) -> io::Result<()> {
    let sample_indexes = if vectors.len() > args.sample_size.get() {
        let mut rng = rand_xoshiro::Xoroshiro128PlusPlus::seed_from_u64(0x455A_5469676572);
        let mut indexes =
            rand::seq::index::sample(&mut rng, vectors.len(), args.sample_size.get()).into_vec();
        indexes.sort_unstable();
        indexes
    } else {
        (0..vectors.len()).collect()
    };
    let sample = SubsetViewVectorStore::new(vectors, sample_indexes);
    let (min_quantile, max_quantile) =
        ScalarQuantizerVectorCoder::train(NonZero::new(vectors.len()).unwrap(), sample.iter());
    println!(
        "vectors: {}, min_quantile: {} max_quantile: {}",
        sample.len(),
        min_quantile,
        max_quantile
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
        Command::TrainLuceneScalarQuantizer(args) => train_lucene_scalar_quantizer(args, &vectors),
    }
}
