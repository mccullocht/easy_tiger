use std::{
    borrow::Cow,
    fs::File,
    io::{self, BufWriter, Write},
    num::NonZero,
    path::PathBuf,
    sync::{Arc, Mutex, atomic::AtomicU64},
};

use clap::{Args, Parser, Subcommand};
use crossbeam_utils::CachePadded;
use easy_tiger::{
    Neighbor,
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
    /// Compute recall against a ground truth set using quantized vectors.
    QuantizationRecall(QuantizationRecallArgs),
    /// Compute ground truth set for float32 vectors.
    ExhaustiveSearch(ExhaustiveSearchArgs),
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
        .progress_count(doc_limit as u64)
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

#[derive(Args)]
pub struct QuantizationRecallArgs {
    /// List of little-endian float32 queries.
    #[arg(long)]
    query_vectors: PathBuf,
    /// Neighbors list to use as ground truth for each query.
    #[arg(long)]
    neighbors: PathBuf,
    /// Number of neighbors per-query.
    #[arg(long, default_value_t = NonZero::new(100).unwrap())]
    neighbors_len: NonZero<usize>,
    /// If true, quantize the query before scoring.
    ///
    /// Some format implement f32 x quantized scoring which is more accurate but slower.
    #[arg(long, default_value_t = false)]
    quantize_query: bool,
    /// If set, only process this many input queries.
    #[arg(long)]
    query_limit: Option<usize>,

    /// Vector coding to test.
    #[arg(long)]
    format: F32VectorCoding,
    /// Similarity function to use.
    #[arg(long)]
    similarity: VectorSimilarity,

    /// Number of results to compare.
    #[arg(long)]
    recall_k: NonZero<usize>,
    /// If specified, consider any in the top rerank_budget as matches.
    ///
    /// This mimics the behavior of re-ranking with exact vectors.
    #[arg(long)]
    rerank_budget: Option<NonZero<usize>>,

    #[arg(long)]
    doc_limit: Option<usize>,
}

fn quantization_recall(
    args: QuantizationRecallArgs,
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
) -> io::Result<()> {
    let query_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.query_vectors)?)? },
        NonZero::new(vectors.elem_stride()).unwrap(),
    )?;
    let neighbors: DerefVectorStore<u32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.neighbors)?)? },
        args.neighbors_len,
    )?;
    let query_limit = args
        .query_limit
        .unwrap_or(query_vectors.len())
        .min(query_vectors.len());

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

    let topn = args.rerank_budget.unwrap_or(args.recall_k).get();
    let mut query_topn = Vec::with_capacity(query_limit);
    query_topn.resize_with(query_limit, || TopNeighbors::new(topn));
    (0..vectors.len())
        .into_par_iter()
        .progress_count((vectors.len()) as u64)
        .for_each(|d| {
            let doc = coder.encode(&vectors[d]);
            for (q, s) in query_scorers.iter().enumerate() {
                query_topn[q].push(Neighbor::new(d as i64, s.distance(&doc)), topn);
            }
        });

    let matching = neighbors
        .iter()
        .zip(
            query_topn
                .into_iter()
                .map(|r| r.rep.into_inner().into_inner().unwrap()),
        )
        .map(|(neighbors, results)| {
            results
                .iter()
                .filter(|n| neighbors[..args.recall_k.get()].contains(&(n.vertex() as u32)))
                .count()
        })
        .sum::<usize>();
    println!(
        "Recall@{}: {:.6}",
        args.recall_k.get(),
        matching as f64 / (args.recall_k.get() * query_scorers.len()) as f64
    );

    Ok(())
}

#[derive(Args)]
pub struct ExhaustiveSearchArgs {
    /// Path to numpy formatted little-endian float vectors.
    #[arg(short, long)]
    query_vectors: PathBuf,
    /// Path buf to numpy u32 formatted neighbors file to write.
    /// This should include one row of length neighbors_len for each vector in query_vectors.
    #[arg(long)]
    neighbors: PathBuf,
    /// Number of neighbors for each query in the neighbors file.
    #[arg(long, default_value_t = NonZero::new(100).unwrap())]
    neighbors_len: NonZero<usize>,

    /// Maximum number of records to search for query vectors.
    #[arg(long)]
    record_limit: Option<usize>,

    /// Similarity function to use.
    #[arg(long)]
    similarity: VectorSimilarity,
}

pub fn exhaustive_search(
    args: ExhaustiveSearchArgs,
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
) -> io::Result<()> {
    let query_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.query_vectors)?)? },
        NonZero::new(vectors.elem_stride()).unwrap(),
    )?;

    let mut results = Vec::with_capacity(query_vectors.len());
    let k = args.neighbors_len.get();
    results.resize_with(query_vectors.len(), || Vec::with_capacity(k * 2));
    let distance_fn = args.similarity.new_distance_function();

    let limit = args
        .record_limit
        .unwrap_or(vectors.len())
        .min(vectors.len());
    for (i, doc) in vectors
        .iter()
        .enumerate()
        .take(limit)
        .progress_count(limit as u64)
    {
        results.par_iter_mut().enumerate().for_each(|(q, r)| {
            let n = Neighbor::new(i as i64, distance_fn.distance_f32(&query_vectors[q], doc));
            if r.len() <= k || n < r[k] {
                r.push(n);
                if r.len() == r.capacity() {
                    r.select_nth_unstable(k);
                    r.truncate(k);
                }
            }
        });
    }

    let mut writer = BufWriter::new(File::create(args.neighbors)?);
    for mut neighbors in results.into_iter() {
        neighbors.sort_unstable();
        for n in neighbors.into_iter().take(k) {
            writer.write_all(&(n.vertex() as u32).to_le_bytes())?;
        }
    }

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
        Command::QuantizationRecall(args) => quantization_recall(args, &vectors),
        Command::ExhaustiveSearch(args) => exhaustive_search(args, &vectors),
    }
}
