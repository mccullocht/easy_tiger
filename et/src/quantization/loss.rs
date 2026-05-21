use std::io;

use clap::Args;
use easy_tiger::input::VectorStore;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;
use vectors::{F32VectorCoding, VectorSimilarity};

use crate::ui::progress_bar;

#[derive(Args)]
pub struct LossArgs {
    /// Target format to measure the quantization loss of.
    #[arg(short, long)]
    format: F32VectorCoding,
    /// If set, compute the center of the dataset and apply before quantizing.
    #[arg(long, default_value_t = false)]
    center: bool,
}

pub fn loss(
    args: LossArgs,
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
) -> io::Result<()> {
    let mean = if args.center {
        Some(super::compute_center(vectors))
    } else {
        None
    };

    // Assume Euclidean. It might be best to make this configurable as some encodings might perform
    // better when the inputs are l2 normalized.
    let coder = args.format.coder(VectorSimilarity::Euclidean, mean);
    let (abs_error, sq_error, magnitude) = (0..vectors.len())
        .into_par_iter()
        .progress_with(progress_bar(vectors.len(), "loss"))
        .map(|i| {
            let v = &vectors[i];
            let encoded = coder.encode(&v);
            let q = coder.decode(&encoded);
            let (abs_error, sq_error, sq_magnitude) = v
                .iter()
                .zip(q.iter())
                .map(|(d, q)| {
                    let diff = *d - *q;
                    (diff.abs(), diff * diff, *d * *d)
                })
                .reduce(|a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
                .unwrap();
            (abs_error, sq_error.sqrt(), sq_magnitude.sqrt())
        })
        .reduce(
            || (0.0f32, 0.0f32, 0.0f32),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        );
    let n = vectors.len() as f32;
    let mean_magnitude = magnitude / n;
    println!("Vectors: {}", vectors.len());
    println!(
        "Sum of absolute error: {:.6} error l2 norm: {:.6}",
        abs_error, sq_error
    );
    println!(
        "Per vector absolute error: {:.6} error l2 norm: {:.6}",
        abs_error / n,
        sq_error / n
    );
    println!("Mean vector magnitude: {:.6}", mean_magnitude);
    println!(
        "Error ratio (abs/magnitude): {:.6} error ratio (l2/magnitude): {:.6}",
        abs_error / n / mean_magnitude,
        sq_error / n / mean_magnitude
    );
    Ok(())
}
