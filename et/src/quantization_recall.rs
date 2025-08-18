use std::{fs::File, io, num::NonZero, path::PathBuf};

use crate::{
    neighbor_util::TopNeighbors,
    recall::{RecallArgs, RecallComputer},
};
use clap::Args;
use easy_tiger::{
    input::{DerefVectorStore, VectorStore},
    vectors::{
        new_query_vector_distance_f32, new_query_vector_distance_indexing, F32VectorCoding,
        VectorSimilarity,
    },
    Neighbor,
};
use indicatif::ParallelProgressIterator;
use memmap2::Mmap;
use rayon::prelude::*;

#[derive(Args)]
pub struct QuantizationRecallArgs {
    /// Number of dimensions for input vectors.
    #[arg(short, long)]
    dimensions: NonZero<usize>,

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

    /// Little-endian f32 vectors as a flat file where each vector has --dimensions
    #[arg(long)]
    doc_vectors: PathBuf,
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
    recall: RecallArgs,
}

pub fn quantization_recall(args: QuantizationRecallArgs) -> io::Result<()> {
    let query_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.query_vectors)?)? },
        args.dimensions,
    )?;
    let query_limit = args
        .query_limit
        .unwrap_or(query_vectors.len())
        .min(query_vectors.len());

    let doc_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.doc_vectors)?)? },
        args.dimensions,
    )?;
    let doc_limit = args
        .doc_limit
        .unwrap_or(doc_vectors.len())
        .min(doc_vectors.len());

    let recall_computer = RecallComputer::from_args(args.recall, args.similarity)?.ok_or(
        io::Error::new(io::ErrorKind::InvalidInput, "must provide recall args"),
    )?;

    let coder = args.format.new_coder(args.similarity);
    let query_scorers = (0..query_limit)
        .into_par_iter()
        .map(|i| {
            let qdist = if args.quantize_query {
                new_query_vector_distance_indexing(
                    coder.encode(&query_vectors[i]),
                    args.similarity,
                    args.format,
                )
            } else {
                new_query_vector_distance_f32(
                    query_vectors[i].to_vec(),
                    args.similarity,
                    args.format,
                )
            };
            qdist
        })
        .collect::<Vec<_>>();

    let k = recall_computer.k();
    let mut query_k = Vec::with_capacity(query_limit);
    query_k.resize_with(query_limit, || TopNeighbors::new(k));
    (0..doc_limit)
        .into_par_iter()
        .progress_count(doc_limit as u64)
        .for_each(|d| {
            let doc = coder.encode(&doc_vectors[d]);
            for (q, s) in query_scorers.iter().enumerate() {
                query_k[q].add(Neighbor::new(d as i64, s.distance(&doc)));
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
