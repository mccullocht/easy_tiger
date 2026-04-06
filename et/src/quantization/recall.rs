use std::{fs::File, io, num::NonZero, path::PathBuf};

use crate::{neighbor_util::TopNeighbors, recall::RecallComputer, ui::progress_bar};
use clap::Args;
use easy_tiger::{
    input::{DerefVectorStore, SubsetViewVectorStore, VecVectorStore, VectorStore},
    kmeans::{kmeans, Params},
    Neighbor,
};
use indicatif::ParallelProgressIterator;
use memmap2::Mmap;
use rand::SeedableRng;
use rayon::prelude::*;
use vectors::{F32VectorCoding, VectorSimilarity};

#[derive(Args)]
pub struct RecallArgs {
    /// Little-endian f32 vectors as a flat file where each vector has --dimensions
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

    /// If set, only process this many input doc vectors.
    #[arg(long)]
    doc_limit: Option<usize>,

    /// Vector coding to test.
    #[arg(long)]
    format: F32VectorCoding,
    /// Similarity function to use.
    #[arg(long)]
    similarity: VectorSimilarity,

    #[command(flatten)]
    recall: crate::recall::RecallArgs,

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
    args: RecallArgs,
    doc_vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
) -> io::Result<()> {
    let query_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.query_vectors)?)? },
        NonZero::new(doc_vectors.elem_stride()).unwrap(),
    )?;
    let query_limit = args
        .query_limit
        .unwrap_or(query_vectors.len())
        .min(query_vectors.len());
    let doc_limit = args
        .doc_limit
        .unwrap_or(doc_vectors.len())
        .min(doc_vectors.len());

    let recall_computer = RecallComputer::from_args(args.recall, args.similarity)?.ok_or(
        io::Error::new(io::ErrorKind::InvalidInput, "must provide recall args"),
    )?;

    let centers = match args.centers {
        0 => None,
        1 => {
            let vectors = SubsetViewVectorStore::new(doc_vectors, (0..doc_limit).collect());
            let mean = super::compute_center(&vectors);
            let mut centers = VecVectorStore::with_capacity(doc_vectors.elem_stride(), 1);
            centers.push(&mean);
            Some(centers)
        }
        _ => {
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(args.seed);
            let sample_size = args.center_sample_size.min(doc_limit);
            let sample_vectors = if sample_size < doc_limit {
                let indices = rand::seq::index::sample(&mut rng, doc_limit, sample_size);
                SubsetViewVectorStore::new(doc_vectors, indices.into_vec())
            } else {
                SubsetViewVectorStore::new(doc_vectors, (0..doc_limit).collect())
            };
            println!(
                "Computing {} centers from a sample of {} vectors",
                args.centers,
                sample_vectors.len()
            );
            let centers = kmeans(
                &sample_vectors,
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

    let coder = args.format.coder(args.similarity, None);
    let query_scorers = (0..query_limit)
        .into_par_iter()
        .map(|i| match centers.as_ref() {
            None => {
                let qdist = if args.quantize_query {
                    args.format.query_distance_symmetric(
                        args.similarity,
                        coder.encode(&query_vectors[i]),
                        None,
                    )
                } else {
                    args.format
                        .query_distance_asymmetric(args.similarity, query_vectors[i].to_vec(), None)
                };
                vec![qdist]
            }
            Some(centers) => centers
                .iter()
                .map(|c| {
                    let centered = query_vectors[i]
                        .iter()
                        .zip(c.iter())
                        .map(|(q, c)| q - c)
                        .collect::<Vec<_>>();
                    if args.quantize_query {
                        args.format.query_distance_symmetric(
                            args.similarity,
                            coder.encode(&centered),
                            None,
                        )
                    } else {
                        args.format
                            .query_distance_asymmetric(args.similarity, centered, None)
                    }
                })
                .collect::<Vec<_>>(),
        })
        .collect::<Vec<_>>();

    let k = recall_computer.k();
    let mut query_k = Vec::with_capacity(query_limit);
    query_k.resize_with(query_limit, || TopNeighbors::new(k));
    let (total_scored, total_competitive) = (0..doc_limit)
        .into_par_iter()
        .progress_with(progress_bar(doc_limit, "scoring"))
        .map(|d| {
            let center = select_center_for_doc(&doc_vectors[d], centers.as_ref(), args.similarity);
            let doc = if let Some(centers) = centers.as_ref() {
                coder.encode(
                    &doc_vectors[d]
                        .iter()
                        .zip(centers[center].iter())
                        .map(|(d, c)| d - c)
                        .collect::<Vec<_>>(),
                )
            } else {
                coder.encode(&doc_vectors[d])
            };
            let mut total_scored = 0;
            let mut total_competitive = 0;
            for (q, s) in query_scorers.iter().enumerate() {
                let max_distance = query_k[q].max_distance();
                if let Some(distance) = s[center].distance_with_bound(&doc, max_distance) {
                    query_k[q].add(Neighbor::new(d as i64, distance));
                    total_competitive += 1;
                }
                total_scored += 1;
            }
            (total_scored, total_competitive)
        })
        .reduce(|| (0usize, 0usize), |a, b| (a.0 + b.0, a.1 + b.1));

    // TODO: add analysis for re-scoring depth. For simple recall this amount to using a larger set
    // on the "actual" side, but may be more complicated for NDCG.
    let sum_recall = query_k
        .into_iter()
        .enumerate()
        .map(|(i, r)| recall_computer.compute_recall(i, &r.into_neighbors()))
        .sum::<f64>();
    println!(
        "{}: {:.6}",
        recall_computer.label(),
        sum_recall / query_limit as f64
    );
    println!(
        "scored: {} competitive: {} ratio: {:.6}",
        total_scored,
        total_competitive,
        total_competitive as f64 / total_scored as f64
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
            let dist = similarity.new_distance_function();
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
