use std::{borrow::Cow, fs::File, io, num::NonZero, path::PathBuf, sync::Arc};

use clap::Args;
use easy_tiger::input::{DerefVectorStore, VectorStore};
use indicatif::ParallelProgressIterator;
use memmap2::Mmap;
use rayon::prelude::*;
use vectors::{F32VectorCoding, VectorSimilarity};

#[derive(Args)]
pub struct DistanceLossArgs {
    /// Little-endian f32 vectors of some dimensionality as input vectors.
    #[arg(long)]
    query_vectors: PathBuf,
    /// If true, quantize queries before computing loss, bypassing any f32 x quantized query
    /// vector distance implementation.
    #[arg(long)]
    quantize_query: bool,
    /// Limit on the number of queries. If unset, use all input queries.
    #[arg(long)]
    query_limit: Option<usize>,

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

pub fn distance_loss(
    args: DistanceLossArgs,
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
) -> io::Result<()> {
    let query_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.query_vectors)?)? },
        NonZero::new(vectors.elem_stride()).unwrap(),
    )?;
    let query_limit = args
        .query_limit
        .unwrap_or(query_vectors.len())
        .min(query_vectors.len());
    let doc_limit = args.doc_limit.unwrap_or(vectors.len()).min(vectors.len());

    let center = if args.center {
        Some(super::compute_center(vectors))
    } else {
        None
    };

    let coder = args.format.coder(args.similarity, None);
    let query_scorers = (0..query_limit)
        .into_par_iter()
        .map(|i| {
            let mut query = Cow::from(&query_vectors[i]);
            if let Some(center) = center.as_ref() {
                for (q, c) in query.to_mut().iter_mut().zip(center.iter()) {
                    *q -= *c;
                }
            }
            let qdist = if args.quantize_query {
                args.format
                    .query_distance_symmetric(args.similarity, coder.encode(&query), None)
            } else {
                args.format
                    .query_distance_asymmetric(args.similarity, query.to_vec(), None)
            };
            let f32_dist = F32VectorCoding::F32.query_distance_asymmetric(
                args.similarity,
                query.into_owned(),
                None,
            );
            (f32_dist, qdist)
        })
        .collect::<Vec<_>>();

    let (count, error_sum, error_sq_sum, in_range_count) = (0..doc_limit)
        .into_par_iter()
        .progress_count(doc_limit as u64)
        .flat_map(|d| {
            let mut doc_f32 = Cow::from(&vectors[d]);
            if let Some(center) = center.as_ref() {
                for (d, c) in doc_f32.to_mut().iter_mut().zip(center.iter()) {
                    *d -= *c;
                }
            }
            let doc_f32 = Arc::new(doc_f32.into_owned());
            let doc = Arc::new(coder.encode(&doc_f32));
            let doc_decoded = coder.decode(&doc);
            let error_term = (doc_f32
                .iter()
                .zip(doc_decoded.iter())
                .map(|(a, b)| {
                    let diff = a - b;
                    diff * diff
                })
                .sum::<f32>()
                / doc_f32.len() as f32)
                .sqrt();
            (0..query_limit)
                .into_par_iter()
                .map(move |q| (q, Arc::clone(&doc), Arc::clone(&doc_f32), error_term))
        })
        .map(|(q, doc, doc_f32, error_term)| {
            let (f32_dist, qdist) = &query_scorers[q];
            let expected = f32_dist.as_ref().distance(bytemuck::cast_slice(&doc_f32));
            let actual = qdist.as_ref().distance(doc.as_ref());
            let actual_est_error = 1.96 * error_term as f64 * 2.0;
            let actual_range = (actual - actual_est_error)..=(actual + actual_est_error);
            let diff = expected - actual;
            (
                1,
                diff.abs(),
                diff * diff,
                if actual_range.contains(&expected) {
                    1
                } else {
                    0
                },
            )
        })
        .reduce(
            || (0, 0.0f64, 0.0f64, 0),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3),
        );

    println!(
        "Vectors: {count} mean abs error: {:.6} mean square error: {:.6} in range: {in_range_count} ({:.2}%)",
        error_sum / count as f64,
        error_sq_sum / count as f64,
        in_range_count as f64 / count as f64 * 100.0
    );
    Ok(())
}
