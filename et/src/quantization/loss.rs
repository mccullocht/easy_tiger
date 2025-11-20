use std::{borrow::Cow, io};

use clap::Args;
use easy_tiger::input::VectorStore;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;
use vectors::{F32VectorCoding, VectorSimilarity};

#[derive(Args)]
pub struct LossArgs {
    /// Target format to measure the quantization loss of.
    #[arg(short, long)]
    format: F32VectorCoding,
    /// If set, compute the center of the dataset and apply before quantizing.
    #[arg(long, default_value_t = false)]
    center: bool,
}

pub fn loss<V: VectorStore<Elem = f32> + Send + Sync>(
    args: LossArgs,
    vectors: &V,
) -> io::Result<()> {
    let mean = if args.center {
        Some(super::compute_center(vectors))
    } else {
        None
    };

    // Assume Euclidean. It might be best to make this configurable as some encodings might perform
    // better when the inputs are l2 normalized.
    let coder = args.format.new_coder(VectorSimilarity::Euclidean);
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
            let q = coder.decode(&encoded);
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
