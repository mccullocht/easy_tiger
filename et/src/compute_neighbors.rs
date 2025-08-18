use std::{
    fs::File,
    io::{self, BufWriter, Write},
    num::NonZero,
    path::PathBuf,
};

use clap::Args;
use easy_tiger::{
    input::{DerefVectorStore, VectorStore},
    vectors::VectorSimilarity,
    Neighbor,
};
use indicatif::ParallelProgressIterator;
use memmap2::Mmap;
use rayon::prelude::*;

use crate::neighbor_util::TopNeighbors;

#[derive(Args)]
pub struct ComputeNeighborsArgs {
    /// Path to numpy formatted little-endian float vectors.
    #[arg(long)]
    query_vectors: PathBuf,
    /// Maximum number of query vectors to search for query vectors.
    #[arg(long)]
    query_limit: Option<usize>,
    /// Path to numpy formatted little-endian float vectors.
    #[arg(long)]
    doc_vectors: PathBuf,
    /// Maximum number of doc vectors to search for query vectors.
    #[arg(long)]
    doc_limit: Option<usize>,

    /// Number of dimensions for both query and doc vectors.
    #[arg(short, long)]
    dimensions: NonZero<usize>,
    /// Similarity function to use.
    #[arg(short, long)]
    similarity: VectorSimilarity,

    /// Path to neighbors file to write.
    ///
    /// The output file will contain one row for each vector in query_vectors. Within each row there
    /// will be neighbors_len entries of Neighbor, an (i64,f64) tuple.
    #[arg(short, long)]
    neighbors: PathBuf,
    /// Number of neighbors for each query in the neighbors file.
    #[arg(long, default_value_t = NonZero::new(100).unwrap())]
    neighbors_len: NonZero<usize>,
}

pub fn compute_neighbors(args: ComputeNeighborsArgs) -> io::Result<()> {
    let query_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.query_vectors)?)? },
        args.dimensions,
    )?;
    let query_limit = args.query_limit.unwrap_or(query_vectors.len());
    let doc_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.doc_vectors)?)? },
        args.dimensions,
    )?;
    let doc_limit = args
        .doc_limit
        .unwrap_or(doc_vectors.len())
        .min(doc_vectors.len());

    let distance_fn = args.similarity.new_distance_function();
    let k = args.neighbors_len.get();
    let mut results = Vec::with_capacity(query_limit);
    results.resize_with(query_limit, || TopNeighbors::new(args.neighbors_len.get()));
    (0..doc_limit)
        .into_par_iter()
        .progress_count(doc_limit as u64)
        .for_each(|d| {
            for q in 0..query_limit {
                results[q].add(Neighbor::new(
                    d as i64,
                    distance_fn.distance_f32(&query_vectors[q], &doc_vectors[d]),
                ));
            }
        });

    let mut writer = BufWriter::new(File::create(args.neighbors)?);
    for neighbors in results.into_iter().map(|r| r.into_neighbors()) {
        for n in neighbors.into_iter().take(k) {
            writer.write_all(&<[u8; 16]>::from(n))?;
        }
    }

    Ok(())
}
