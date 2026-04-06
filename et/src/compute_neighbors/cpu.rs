use std::{
    fs::File,
    io::{self, BufWriter, Write},
};

use easy_tiger::{
    input::{DerefVectorStore, VectorStore},
    Neighbor,
};
use indicatif::ParallelProgressIterator;
use memmap2::Mmap;
use rayon::prelude::*;

use crate::neighbor_util::TopNeighbors;

use super::ComputeNeighborsArgs;

pub fn run(args: &ComputeNeighborsArgs) -> io::Result<()> {
    let query_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(&args.query_vectors)?)? },
        args.dimensions,
    )?;
    let query_limit = args.query_limit.unwrap_or(query_vectors.len());
    let doc_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(&args.doc_vectors)?)? },
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

    let mut writer = BufWriter::new(File::create(&args.neighbors)?);
    for neighbors in results.into_iter().map(|r| r.into_neighbors()) {
        for n in neighbors.into_iter().take(k) {
            writer.write_all(&<[u8; 16]>::from(n))?;
        }
    }

    Ok(())
}
