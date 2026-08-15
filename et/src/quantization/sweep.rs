//! Pick a quantizer by sweeping formats through the bound-depth measurement.
//!
//! This is a self-contained version of `bound-depth`: it draws a small doc and query sample,
//! computes exact nearest neighbors over that sample in process, and then runs every requested
//! format against the same samples and golden data. Because the samples are small the whole sweep
//! runs in seconds and needs no precomputed neighbors file.

use std::{fs::File, io, num::NonZero, path::PathBuf};

use clap::Args;
use easy_tiger::{
    Neighbor,
    input::{DerefVectorStore, SubsetViewVectorStore, VecVectorStore, VectorStore},
};
use indicatif::ParallelProgressIterator;
use memmap2::Mmap;
use rand::SeedableRng;
use rayon::prelude::*;
use vectors::{F32VectorCoding, VectorSimilarity};

use crate::{
    neighbor_util::TopNeighbors,
    recall::{RecallComputer, RecallMetric},
    ui::progress_bar,
};

use super::{bound_depth, exhaustive::Exhaustive};

#[derive(Args)]
pub struct SweepArgs {
    /// Number of docs to sample from the doc vector file to serve as the index.
    #[arg(long, default_value_t = 10_000)]
    doc_sample_size: usize,
    /// Number of queries to sample.
    #[arg(long, default_value_t = 1_000)]
    query_sample_size: usize,
    /// Little-endian f32 vectors as a flat file where each vector has --dimensions.
    ///
    /// If unset, queries are sampled from the doc vector file, disjoint from the doc sample.
    #[arg(long)]
    query_vectors: Option<PathBuf>,

    /// Similarity function to use.
    #[arg(long)]
    similarity: VectorSimilarity,
    /// Vector codings to evaluate, in the order they should be reported.
    #[arg(long, value_delimiter = ',', default_value = "tlvq1,tlvq2,tlvq4,tlvq8")]
    formats: Vec<F32VectorCoding>,
    /// If true, quantize the query before scoring.
    #[arg(long, default_value_t = false)]
    quantize_query: bool,

    /// Number of centers to compute and use.
    ///
    /// If 0, the data set will be uncentered. If 1, a mean vector is used for all queries and docs.
    /// If >1, k-means centers are computed and each doc is encoded against its closest center.
    #[arg(long, default_value_t = 0)]
    centers: usize,
    /// When computing 2 or more centers, sample the doc sample down to at most this many vectors.
    #[arg(long, default_value_t = 100_000)]
    center_sample_size: usize,

    /// Depth of the golden result set to recover.
    #[arg(long, default_value_t = NonZero::new(10).unwrap())]
    k: NonZero<usize>,
    /// Recall metric to compute.
    #[arg(long, value_enum, default_value_t = RecallMetric::Simple)]
    recall_metric: RecallMetric,

    /// Random seed used for sampling and clustering. Use a fixed value for repeatability.
    #[arg(long, default_value_t = 0x7774_7370414E4E)]
    seed: u64,

    /// Number of docs in the full corpus, used to project the storage cost of each format.
    ///
    /// The projected cost is the byte length of a vector in the given format times this count,
    /// plus --rerank-byte-len times the average query depth (the rerank cost). When unset, no
    /// cost column is reported.
    #[arg(long)]
    cost_doc_count: Option<u64>,
    /// Bytes read per candidate when reranking, used to compute the rerank cost.
    ///
    /// This should reflect the cost of reading a block of rerank vectors off disk (e.g. a page or
    /// other storage unit), not just the byte length of a single rerank vector, since reranking
    /// reads are rarely tightly packed at the single-vector granularity.
    #[arg(long)]
    rerank_byte_len: Option<u64>,
}

/// Sample docs and queries, compute exact neighbors over the sample, then report bound depth for
/// each requested format.
///
/// Every format sees the same docs, queries, and golden neighbors, so the rows of the output table
/// are directly comparable: the format with the smallest queue depth at an acceptable miss rate is
/// the cheapest quantizer that still supports exact reranking to the golden top-k.
pub fn sweep(
    args: SweepArgs,
    doc_vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
) -> io::Result<()> {
    if args.formats.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "must provide at least one format",
        ));
    }

    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(args.seed);
    let query_file = args
        .query_vectors
        .as_ref()
        .map(|p| -> io::Result<DerefVectorStore<f32, Mmap>> {
            DerefVectorStore::new(
                unsafe { Mmap::map(&File::open(p)?)? },
                NonZero::new(doc_vectors.elem_stride()).unwrap(),
            )
        })
        .transpose()?;

    // Sample docs first; when queries come from the same file they are drawn from what is left so
    // that no query is also a doc. A query that is present in the index is its own nearest neighbor
    // at distance zero, which flatters every quantizer equally and wastes a slot in the top k.
    let (doc_indices, query_indices) = match query_file.as_ref() {
        Some(queries) => (
            sample(&mut rng, doc_vectors.len(), args.doc_sample_size),
            sample(&mut rng, queries.len(), args.query_sample_size),
        ),
        None => {
            let total = args.doc_sample_size.saturating_add(args.query_sample_size);
            if total > doc_vectors.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "doc sample ({}) + query sample ({}) exceeds {} available vectors; pass --query-vectors or reduce the samples",
                        args.doc_sample_size,
                        args.query_sample_size,
                        doc_vectors.len()
                    ),
                ));
            }
            let mut both = sample(&mut rng, doc_vectors.len(), total);
            let queries = both.split_off(args.doc_sample_size);
            (both, queries)
        }
    };

    let docs = SubsetViewVectorStore::new(doc_vectors, doc_indices);
    // The query sample is small and comes from one of two differently typed stores, so copy it into
    // an owned store rather than carrying the source type through the sweep.
    let mut queries = VecVectorStore::with_capacity(doc_vectors.elem_stride(), query_indices.len());
    for i in query_indices {
        match query_file.as_ref() {
            Some(f) => queries.push(&f[i]),
            None => queries.push(&doc_vectors[i]),
        }
    }
    println!(
        "Sampled {} docs and {} queries of {} dimensions",
        docs.len(),
        queries.len(),
        docs.elem_stride()
    );

    if args.k.get() > docs.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "k must be <= the doc sample size",
        ));
    }

    let golden = exact_neighbors(&docs, &queries, args.similarity, args.k);
    let recall_computer =
        RecallComputer::in_memory(args.recall_metric, args.similarity, args.k, golden);

    // Centers depend only on the docs and the similarity, so compute them once and share them
    // across formats. This also keeps the comparison fair when k-means is in play.
    let centers =
        super::exhaustive::compute_centers(&docs, args.centers, args.center_sample_size, args.seed);

    let cost_params = args.cost_doc_count.zip(args.rerank_byte_len);

    let mut rows = Vec::with_capacity(args.formats.len());
    for format in args.formats.iter().copied() {
        let exhaustive = Exhaustive::new(
            args.similarity,
            format,
            args.quantize_query,
            centers.clone(),
            &queries,
        );
        let report = bound_depth::measure(&exhaustive, &docs, &recall_computer);
        let format_byte_len = format.coder(args.similarity, None).byte_len(docs.elem_stride());
        let cost_mib = cost_params.map(|(doc_count, rerank_byte_len)| {
            let cost_bytes = format_byte_len as f64 * doc_count as f64
                + rerank_byte_len as f64 * report.queue_depth.mean();
            cost_bytes / (1024.0 * 1024.0)
        });
        rows.push((format, report, cost_mib));
    }

    println!();
    if cost_params.is_some() {
        println!(
            "{:<10} {:>10} {:>10} {:>10} {:>10} {:>14} {:>10} {:>12} {:>12} {:>14}",
            "format",
            "depth p50",
            "depth p90",
            "depth p99",
            "depth max",
            "rerank total",
            "slop p50",
            "miss rate",
            recall_computer.label(),
            "cost (MiB)",
        );
    } else {
        println!(
            "{:<10} {:>10} {:>10} {:>10} {:>10} {:>14} {:>10} {:>12} {:>12}",
            "format",
            "depth p50",
            "depth p90",
            "depth p99",
            "depth max",
            "rerank total",
            "slop p50",
            "miss rate",
            recall_computer.label(),
        );
    }
    for (format, report, cost_mib) in &rows {
        print!(
            "{:<10} {:>10.1} {:>10.1} {:>10.1} {:>10.0} {:>14.0} {:>10.2} {:>11.4}% {:>12.4}",
            format.to_string(),
            report.queue_depth.quantile(0.5),
            report.queue_depth.quantile(0.9),
            report.queue_depth.quantile(0.99),
            report.queue_depth.quantile(1.0),
            report.queue_depth.sum(),
            report.slop.quantile(0.5),
            report.miss_rate() * 100.0,
            report.mean_recall(),
        );
        if let Some(cost_mib) = cost_mib {
            print!(" {:>14.2}", cost_mib);
        }
        println!();
    }

    Ok(())
}

/// Draw `n` distinct indices from `[0, len)`, or all of them when `n >= len`.
fn sample(rng: &mut impl rand::Rng, len: usize, n: usize) -> Vec<usize> {
    if n >= len {
        (0..len).collect()
    } else {
        rand::seq::index::sample(rng, len, n).into_vec()
    }
}

/// Brute force the exact top-`k` docs for each query using full fidelity f32 distances.
fn exact_neighbors(
    docs: &(impl VectorStore<Elem = f32> + Send + Sync),
    queries: &(impl VectorStore<Elem = f32> + Send + Sync),
    similarity: VectorSimilarity,
    k: NonZero<usize>,
) -> Vec<Vec<Neighbor>> {
    let distance_fn = similarity.new_distance_function();
    let mut results = Vec::with_capacity(queries.len());
    results.resize_with(queries.len(), || TopNeighbors::new(k.get()));
    (0..docs.len())
        .into_par_iter()
        .progress_with(progress_bar(docs.len(), "exact neighbors"))
        .for_each(|d| {
            for (q, result) in results.iter().enumerate() {
                result.add(Neighbor::new(
                    d as i64,
                    distance_fn.distance_f32(&queries[q], &docs[d]),
                ));
            }
        });
    results.into_iter().map(|r| r.into_neighbors()).collect()
}
