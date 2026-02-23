use std::{fs::File, io, num::NonZero, path::PathBuf};

use crate::{neighbor_util::TopNeighbors, recall::RecallComputer};
use clap::Args;
use easy_tiger::{
    input::{DerefVectorStore, SubsetViewVectorStore, VecVectorStore, VectorStore},
    Neighbor,
};
use indicatif::ParallelProgressIterator;
use memmap2::Mmap;
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
        _ => unimplemented!(),
    };

    let coder = args.format.new_coder(args.similarity);
    let query_scorers = (0..query_limit)
        .into_par_iter()
        .map(|i| match centers.as_ref() {
            None => {
                let qdist = if args.quantize_query {
                    args.format.query_vector_distance_indexing(
                        coder.encode(&query_vectors[i]),
                        args.similarity,
                    )
                } else {
                    args.format
                        .query_vector_distance_f32(query_vectors[i].to_vec(), args.similarity)
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
                        args.format.query_vector_distance_indexing(
                            coder.encode(&centered),
                            args.similarity,
                        )
                    } else {
                        args.format
                            .query_vector_distance_f32(centered, args.similarity)
                    }
                })
                .collect::<Vec<_>>(),
        })
        .collect::<Vec<_>>();

    let k = recall_computer.k();
    let mut query_k = Vec::with_capacity(query_limit);
    query_k.resize_with(query_limit, || TopNeighbors::new(k));
    (0..doc_limit)
        .into_par_iter()
        .progress_count(doc_limit as u64)
        .for_each(|d| {
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
            for (q, s) in query_scorers.iter().enumerate() {
                query_k[q].add(Neighbor::new(d as i64, s[center].distance(&doc)));
            }
        });

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
