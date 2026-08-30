use std::{io, path::PathBuf};

use crate::{neighbor_util::TopNeighbors, recall::RecallComputer, ui::progress_bar};
use clap::Args;
use easy_tiger::{
    input::{DerefVectorStore, SubsetViewVectorStore, VecVectorStore, VectorStore},
    kmeans::{Params, kmeans},
};
use half::slice::HalfFloatSliceExt;
use indicatif::ParallelProgressIterator;
use memmap2::Mmap;
use rand::SeedableRng;
use rayon::prelude::*;
use tdigests::TDigest;
use vectors::{F32VectorCoding, VectorSimilarity, f16};

#[derive(Args)]
pub struct QuantizationRecallArgs {
    /// Query vectors: f16 vectors in BigANN format (an 8 byte `<len,dim>` header followed by
    /// little-endian f16 vector data).
    #[arg(long)]
    query_vectors: PathBuf,
    /// If set, only process this many input queries.
    #[arg(long)]
    query_limit: Option<usize>,
    /// If true, quantize the query before scoring.
    ///
    /// Some format implement f32 x quantized scoring which is more accurate but slower.
    #[arg(long, default_value_t = false)]
    quantize_query: bool,

    /// Vector coding to test.
    #[arg(long)]
    format: F32VectorCoding,
    /// Similarity function to use.
    #[arg(long)]
    similarity: VectorSimilarity,

    #[command(flatten)]
    recall: crate::recall::RecallArgs,

    /// Multiplier for recall k.
    ///
    /// Collect k * k_mult neighbors for each query and compute recall using that set. This
    /// simulates recall with full-fidelity reranking where we over retrieve to get the final set.
    #[arg(long, default_value_t = 1.0)]
    k_mult: f64,

    /// Z-Score applied to error bounds.
    ///
    /// Higher values increase confidence that recall will be accurate at the cost of higher rerank
    /// depth.
    #[arg(long, default_value_t = 1.0)]
    z_score: f64,

    /// Number of centers to compute and use.
    ///
    /// If 0, the data set will be uncentered.
    ///
    /// If 1, a mean vector will be computed and used as the center for all queries and docs.
    ///
    /// If >1, k-means will be used to compute centers. Each comparison will happen relative to
    /// the closest center for each doc.
    #[arg(long, default_value_t = 0)]
    centers: usize,

    /// When computing 2 or more centers, sample the data set to at most this many vectors.
    #[arg(long, default_value_t = 100_000)]
    center_sample_size: usize,

    /// Random seed used for clustering computations.
    /// Use a fixed value for repeatability.
    #[arg(long, default_value_t = 0x7774_7370414E4E)]
    seed: u64,
}

pub fn recall(
    args: QuantizationRecallArgs,
    doc_vectors: &(impl VectorStore<Elem = f16> + Send + Sync),
) -> io::Result<()> {
    let query_vectors: DerefVectorStore<f16, Mmap> =
        DerefVectorStore::from_file(args.query_vectors)?;
    if query_vectors.elem_stride() != doc_vectors.elem_stride() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "query and doc vectors must have the same dimensionality ({} vs {})",
                query_vectors.elem_stride(),
                doc_vectors.elem_stride()
            ),
        ));
    }
    let query_limit = args
        .query_limit
        .unwrap_or(query_vectors.len())
        .min(query_vectors.len());

    let recall_computer = RecallComputer::from_args(args.recall)?.ok_or(io::Error::new(
        io::ErrorKind::InvalidInput,
        "must provide recall args",
    ))?;

    let centers = match args.centers {
        0 => None,
        1 => {
            let vectors = SubsetViewVectorStore::new(doc_vectors, (0..doc_vectors.len()).collect());
            let mean = super::compute_center(&vectors);
            let mut centers = VecVectorStore::with_capacity(doc_vectors.elem_stride(), 1);
            centers.push(&mean);
            Some(centers)
        }
        _ => {
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(args.seed);
            let sample_size = args.center_sample_size.min(doc_vectors.len());
            let sample_vectors = if sample_size < doc_vectors.len() {
                let indices = rand::seq::index::sample(&mut rng, doc_vectors.len(), sample_size);
                SubsetViewVectorStore::new(doc_vectors, indices.into_vec())
            } else {
                SubsetViewVectorStore::new(doc_vectors, (0..doc_vectors.len()).collect())
            };
            println!(
                "Computing {} centers from a sample of {} vectors",
                args.centers,
                sample_vectors.len()
            );
            let mut widened_sample =
                VecVectorStore::with_capacity(doc_vectors.elem_stride(), sample_vectors.len());
            let mut buf = vec![0.0f32; doc_vectors.elem_stride()];
            for v in sample_vectors.iter() {
                v.convert_to_f32_slice(&mut buf);
                widened_sample.push(&buf);
            }
            let centers = kmeans(
                &widened_sample,
                args.centers,
                &Params {
                    iters: 100,
                    epsilon: 0.0001,
                    ..Params::default()
                },
                &mut rng,
            );
            Some(centers.unwrap_or_else(|e| e))
        }
    };

    let coders: Vec<Box<dyn vectors::F32VectorCoder>> = match centers.as_ref() {
        None => vec![args.format.coder(args.similarity, None)],
        Some(cs) => cs
            .iter()
            .map(|c| args.format.coder(args.similarity, Some(c.to_vec())))
            .collect(),
    };

    let query_scorers = (0..query_limit)
        .into_par_iter()
        .map(|i| {
            let mut query = vec![0.0f32; query_vectors.elem_stride()];
            query_vectors[i].convert_to_f32_slice(&mut query);
            coders
                .iter()
                .enumerate()
                .map(|(ci, coder)| {
                    let center = centers.as_ref().map(|cs| &cs[ci]);
                    if args.quantize_query {
                        args.format
                            .query_distance_symmetric(args.similarity, coder.encode(&query))
                    } else {
                        args.format.query_distance_asymmetric(
                            args.similarity,
                            query.clone(),
                            center,
                        )
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let k = recall_computer.k();
    let result_len = (k as f64 * args.k_mult) as usize;
    let mut query_k = Vec::with_capacity(query_limit);
    query_k.resize_with(query_limit, || TopNeighbors::new(result_len));
    (0..doc_vectors.len())
        .into_par_iter()
        .progress_with(progress_bar(doc_vectors.len(), "scoring"))
        .for_each(|d| {
            let doc_f32 = doc_vectors[d].to_f32_vec();
            let center = select_center_for_doc(&doc_f32, centers.as_ref(), args.similarity);
            let doc = coders[center].encode(&doc_f32);
            for (q, s) in query_scorers.iter().enumerate() {
                let mut estimate = s[center].estimated_distance(&doc);
                estimate.error *= args.z_score;
                query_k[q].add_estimate(d as i64, estimate);
            }
        });

    let mut depth = Vec::with_capacity(query_k.len());
    let recall_values = query_k
        .into_iter()
        .enumerate()
        .map(|(i, r)| {
            let results = r.into_neighbors();
            depth.push(results.len() as f64);
            recall_computer.compute_recall(i, &results)
        })
        .collect::<Vec<_>>();
    println!("{}", recall_computer.summarize(&recall_values));
    let mean_depth = depth.iter().copied().sum::<f64>() / depth.len() as f64;
    let digest = TDigest::from_values(depth);
    println!(
        "Queue depth mean {mean_depth:<6.1} p50 {:<6.1} p75 {:<6.1} p90 {:<6.1} p95 {:<6.1} p99 {:<6.1} p99.9 {:<6.1}",
        digest.estimate_quantile(0.5),
        digest.estimate_quantile(0.75),
        digest.estimate_quantile(0.9),
        digest.estimate_quantile(0.95),
        digest.estimate_quantile(0.99),
        digest.estimate_quantile(0.999)
    );

    Ok(())
}

fn select_center_for_doc(
    doc: &[f32],
    centers: Option<&VecVectorStore<f32>>,
    similarity: VectorSimilarity,
) -> usize {
    if let Some(centers) = centers {
        if centers.len() == 1 {
            0
        } else {
            let dist = similarity.distance_f32();
            centers
                .iter()
                .enumerate()
                .map(|(i, c)| (i, dist.distance_f32(doc, &c)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i)
                .unwrap()
        }
    } else {
        0
    }
}
