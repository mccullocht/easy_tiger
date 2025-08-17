use std::{
    fs::File,
    io,
    num::NonZero,
    path::PathBuf,
    sync::{atomic::AtomicU64, Mutex},
};

use crate::recall::{RecallArgs, RecallComputer};
use clap::Args;
use crossbeam_utils::CachePadded;
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

    // Keep the top neighbors, along with a maximum distance. The distance is read atomically as
    // to filter non-competitive results; competitive results are added to the neighbor list and
    // periodically pruned to update the competitive value.
    struct TopNeighbors {
        rep: CachePadded<Mutex<Vec<Neighbor>>>,
        threshold: CachePadded<AtomicU64>,
    }
    impl TopNeighbors {
        fn new(n: usize) -> Self {
            Self {
                rep: CachePadded::new(Mutex::new(Vec::with_capacity(n * 2))),
                threshold: CachePadded::new(AtomicU64::new(f64::MAX.to_bits())),
            }
        }

        fn push(&self, neighbor: Neighbor, n: usize) {
            use std::sync::atomic::Ordering;

            if f64::from_bits(self.threshold.load(Ordering::Relaxed)) < neighbor.distance() {
                return; // neighbor is not competitive.
            }

            let mut neighbors = self.rep.lock().unwrap();
            neighbors.push(neighbor);
            if neighbors.len() == n * 2 {
                let (_, t, _) = neighbors.select_nth_unstable(n - 1);
                self.threshold
                    .store(t.distance().to_bits(), Ordering::Relaxed);
                neighbors.truncate(n);
            }
        }
    }

    let k = recall_computer.k();
    let mut query_k = Vec::with_capacity(query_limit);
    query_k.resize_with(query_limit, || TopNeighbors::new(k));
    (0..doc_limit)
        .into_par_iter()
        .progress_count(doc_limit as u64)
        .for_each(|d| {
            let doc = coder.encode(&doc_vectors[d]);
            for (q, s) in query_scorers.iter().enumerate() {
                query_k[q].push(Neighbor::new(d as i64, s.distance(&doc)), k);
            }
        });

    // XXX add back the ability to compare a deeper query set to a smaller truth set.
    // not useful for ndcg but good for measuring the upside of depth in re-scoring.
    let sum_recall = query_k
        .into_iter()
        .map(|r| r.rep.into_inner().into_inner().unwrap())
        .enumerate()
        .map(|(i, mut r)| {
            // XXX this should happen somewhere else, maybe a finish method?
            r.sort_unstable();
            recall_computer.compute_recall(i, &r)
        })
        .sum::<f64>();
    // XXX print the metric type!
    println!(
        "Recall@{}: {:.6}",
        recall_computer.k(),
        sum_recall / query_limit as f64
    );

    Ok(())
}
