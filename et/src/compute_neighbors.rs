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
use indicatif::ProgressIterator;
use memmap2::Mmap;
use rayon::prelude::*;

#[derive(Args)]
pub struct ComputeNeighborsArgs {
    /// Path to numpy formatted little-endian float vectors.
    #[arg(long)]
    query_vectors: PathBuf,
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
    let doc_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.doc_vectors)?)? },
        args.dimensions,
    )?;

    let mut results = Vec::with_capacity(query_vectors.len());
    let k = args.neighbors_len.get();
    results.resize_with(query_vectors.len(), || Vec::with_capacity(k * 2));
    let distance_fn = args.similarity.new_distance_function();

    let limit = args
        .doc_limit
        .unwrap_or(doc_vectors.len())
        .min(doc_vectors.len());
    for (i, doc) in doc_vectors
        .iter()
        .enumerate()
        .take(limit)
        .progress_count(limit as u64)
    {
        results.par_iter_mut().enumerate().for_each(|(q, r)| {
            let n = Neighbor::new(i as i64, distance_fn.distance_f32(&query_vectors[q], doc));
            if r.len() < k || n < r[k - 1] {
                r.push(n);
                if r.len() == k * 2 {
                    r.select_nth_unstable(k - 1);
                    r.truncate(k);
                }
            }
        });
    }

    let mut writer = BufWriter::new(File::create(args.neighbors)?);
    for mut neighbors in results.into_iter() {
        neighbors.sort_unstable();
        for n in neighbors.into_iter().take(k) {
            writer.write_all(&<[u8; 16]>::from(n))?;
        }
    }

    Ok(())
}
