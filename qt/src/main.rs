use std::{
    borrow::Cow,
    fs::File,
    io::{self},
    num::NonZero,
    path::PathBuf,
    sync::Arc,
};

use clap::{Args, Parser, Subcommand};
use easy_tiger::{
    input::{DerefVectorStore, VectorStore},
    vectors::{
        F32VectorCoding, VectorSimilarity, new_query_vector_distance_f32,
        new_query_vector_distance_indexing,
    },
};
use indicatif::{ParallelProgressIterator, ProgressIterator};
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
    /// Compute precision loss in vector quantization.
    QuantizationLoss(QuantizationLossArgs),
    /// Compute precision loss in distance computation resulting from vector quantization.
    DistanceLoss(DistanceLossArgs),
}

#[derive(Args)]
struct QuantizationLossArgs {
    /// Target format to measure the quantization loss of.
    #[arg(short, long)]
    format: F32VectorCoding,
    /// If set, compute the center of the dataset and apply before quantizing.
    #[arg(long, default_value_t = false)]
    center: bool,
}

fn quantization_loss(
    args: QuantizationLossArgs,
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
) -> io::Result<()> {
    let mean = if args.center {
        Some(compute_center(vectors))
    } else {
        None
    };

    // Assume Euclidean. It might be best to make this configurable as some encodings might perform
    // better when the inputs are l2 normalized.
    let coder = args.format.new_coder(VectorSimilarity::Euclidean);
    assert!(
        coder.decode(&coder.encode(&vectors[0])).is_some(),
        "specified vector format doesn't support decoding"
    );
    let (abs_error, sq_error) = (0..vectors.len())
        .into_par_iter()
        .progress_count(vectors.len() as u64)
        .map(|i| {
            let v = mean
                .as_ref()
                .map(|m| {
                    Cow::from(
                        vectors[i]
                            .iter()
                            .zip(m.iter())
                            .map(|(d, m)| *d - *m)
                            .collect::<Vec<_>>(),
                    )
                })
                .unwrap_or(Cow::from(&vectors[i]));
            let encoded = coder.encode(&v);
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

#[derive(Args)]
struct DistanceLossArgs {
    /// Little-endian f32 vectors of some dimensionality as input vectors.
    #[arg(long)]
    query_vectors: PathBuf,
    /// If true, quantize queries before computing loss, bypassing any f32 x quantized query
    /// vector distance implementation.
    #[arg(long)]
    quantize_query: bool,

    /// Limit on the number of documents. If unset, use all input vectors as docs.
    #[arg(long)]
    doc_limit: Option<usize>,

    /// Similarity function to use.
    #[arg(long)]
    similarity: VectorSimilarity,
    /// Format to compare against f32 distance.
    #[arg(long)]
    format: F32VectorCoding,

    /// If set, compute the center of the dataset and apply before computing distances.
    #[arg(long, default_value_t = false)]
    center: bool,
}

fn distance_loss(
    args: DistanceLossArgs,
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
) -> io::Result<()> {
    let query_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.query_vectors)?)? },
        NonZero::new(vectors.elem_stride()).unwrap(),
    )?;

    let center = if args.center {
        Some(compute_center(vectors))
    } else {
        None
    };

    let coder = args.format.new_coder(args.similarity);
    let query_scorers = (0..query_vectors.len())
        .into_par_iter()
        .map(|i| {
            let mut query = Cow::from(&query_vectors[i]);
            if let Some(center) = center.as_ref() {
                for (q, c) in query.to_mut().iter_mut().zip(center.iter()) {
                    *q -= *c;
                }
            }
            let qdist = if args.quantize_query {
                new_query_vector_distance_indexing(
                    coder.encode(&query),
                    args.similarity,
                    args.format,
                )
            } else {
                new_query_vector_distance_f32(query.to_vec(), args.similarity, args.format)
            };
            let f32_dist = new_query_vector_distance_f32(
                query.into_owned(),
                args.similarity,
                F32VectorCoding::F32,
            );
            (f32_dist, qdist)
        })
        .collect::<Vec<_>>();

    let doc_limit = args.doc_limit.unwrap_or(vectors.len());

    let (count, error_sum, error_sq_sum) = (0..doc_limit)
        .into_par_iter()
        .flat_map(|d| {
            let mut doc_f32 = Cow::from(&vectors[d]);
            if let Some(center) = center.as_ref() {
                for (d, c) in doc_f32.to_mut().iter_mut().zip(center.iter()) {
                    *d -= *c;
                }
            }
            let doc = Arc::new(coder.encode(&doc_f32));
            (0..query_vectors.len())
                .into_par_iter()
                .map(move |q| (q, d, Arc::clone(&doc)))
        })
        .progress_count((query_scorers.len() * doc_limit) as u64)
        .map(|(q, d, doc)| {
            let (f32_dist, qdist) = &query_scorers[q];
            let diff = f32_dist
                .as_ref()
                .distance(bytemuck::cast_slice(&vectors[d]))
                - qdist.as_ref().distance(doc.as_ref());
            (1, diff.abs(), diff * diff)
        })
        .reduce(
            || (0, 0.0f64, 0.0f64),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        );

    println!(
        "Vectors: {} mean abs error: {} mean square error: {}",
        count,
        error_sum / count as f64,
        error_sq_sum / count as f64
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
        Command::DistanceLoss(args) => distance_loss(args, &vectors),
    }
}
