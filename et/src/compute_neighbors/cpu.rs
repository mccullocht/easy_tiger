use std::io;

use easy_tiger::{
    Neighbor,
    input::{DerefVectorStore, VectorStore},
};
use indicatif::ParallelProgressIterator;
use memmap2::Mmap;
use rayon::prelude::*;
use vectors::f16;

use crate::neighbor_util::TopNeighbors;

use super::{ComputeNeighborsArgs, write_neighbors};

pub fn run(args: &ComputeNeighborsArgs) -> io::Result<()> {
    let query_vectors: DerefVectorStore<f16, Mmap> =
        DerefVectorStore::from_file(&args.query_vectors)?;
    let query_limit = args.query_limit.unwrap_or(query_vectors.len());
    let doc_vectors: DerefVectorStore<f16, Mmap> = DerefVectorStore::from_file(&args.doc_vectors)?;
    let doc_limit = args
        .doc_limit
        .unwrap_or(doc_vectors.len())
        .min(doc_vectors.len());
    if query_vectors.elem_stride() != doc_vectors.elem_stride() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "query and doc vectors must have the same dimensionality ({} vs {})",
                query_vectors.elem_stride(),
                doc_vectors.elem_stride()
            ),
        ));
    }

    let distance_fn = args.similarity.distance_f16();
    let mut results = Vec::with_capacity(query_limit);
    results.resize_with(query_limit, || TopNeighbors::new(args.neighbors_len.get()));
    (0..doc_limit)
        .into_par_iter()
        .progress_count(doc_limit as u64)
        .for_each(|d| {
            for q in 0..query_limit {
                results[q].add(Neighbor::new(
                    d as i64,
                    distance_fn.distance_f16(&query_vectors[q], &doc_vectors[d]),
                ));
            }
        });

    write_neighbors(args, results)
}
