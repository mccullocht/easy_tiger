use std::io;

use clap::Args;
use easy_tiger::input::{SubsetViewVectorStore, VectorStore};
use indicatif::ProgressIterator;
use vectors::{F32VectorCoding, VectorSimilarity};

#[derive(Args)]
pub struct ResidualsArgs {
    /// Vector coding format.
    #[arg(short, long)]
    format: F32VectorCoding,

    /// Number of vectors to process.
    #[arg(short, long)]
    limit: Option<usize>,
}

pub fn residuals(args: ResidualsArgs, vectors: &impl VectorStore<Elem = f32>) -> io::Result<()> {
    let limit = args.limit.unwrap_or(vectors.len());
    let vectors = SubsetViewVectorStore::new(vectors, (0..limit).collect());
    let center = super::compute_center(&vectors);

    let coder = args.format.new_coder(VectorSimilarity::Euclidean);

    let mut sum_l2_norm = 0.0;
    let mut sum_residual_l2_norm = 0.0;
    let mut sum_dot_decoded = 0.0;

    // XXX I'm curious what these numbers look like if I don't center.
    // 0.56195149
    // 0.43967578
    // +/- 0.01846406 which really isn't that bad.

    // l2norm * epsilon (1.9) * sqrt((((<r, r> * <r_, r_> / (<r, r_>^2) - 1 / dim - 1))))

    for (_, v) in vectors
        .iter()
        .enumerate()
        .take(limit)
        .progress_count(limit as u64)
    {
        sum_l2_norm += v.iter().map(|x| x * x).sum::<f32>().sqrt() as f64;
        let r: Vec<f32> = v.iter().zip(center.iter()).map(|(v, c)| *v - *c).collect();

        // <r, r>
        sum_residual_l2_norm += r.iter().map(|x| x * x).sum::<f32>().sqrt() as f64;

        // <r, decoded>
        let encoded = coder.encode(&r);
        let decoded = coder.decode(&encoded);
        sum_dot_decoded += r
            .iter()
            .zip(decoded.iter())
            .map(|(r, d)| r * d)
            .sum::<f32>()
            .sqrt() as f64;
    }

    println!("Mean L2 norm: {}", sum_l2_norm / limit as f64);
    println!(
        "Mean L2 norm of residual: {}",
        sum_residual_l2_norm / limit as f64
    );
    println!(
        "Mean dot product <r, decoded>: {}",
        sum_dot_decoded / limit as f64
    );

    Ok(())
}
