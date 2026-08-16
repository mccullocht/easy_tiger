use std::io;

use easy_tiger::{
    Neighbor,
    input::{DerefVectorStore, VectorStore},
};
use indicatif::ParallelProgressIterator;
use memmap2::Mmap;
use rayon::prelude::*;

use crate::neighbor_util::TopNeighbors;

use super::{ComputeNeighborsArgs, write_neighbors};

pub fn run(args: &ComputeNeighborsArgs) -> io::Result<()> {
    let query_vectors: DerefVectorStore<f32, Mmap> =
        DerefVectorStore::from_file_with_stride(&args.query_vectors, args.dimensions)?;
    let query_limit = args.query_limit.unwrap_or(query_vectors.len());
    let doc_vectors: DerefVectorStore<f32, Mmap> =
        DerefVectorStore::from_file_with_stride(&args.doc_vectors, args.dimensions)?;
    let doc_limit = args
        .doc_limit
        .unwrap_or(doc_vectors.len())
        .min(doc_vectors.len());

    let distance_fn = args.similarity.distance_f32();
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

    write_neighbors(args, results)
}
