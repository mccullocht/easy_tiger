//! Hierarchical clustered relative neighbor graph.
//!
//! Using binary partitioning, the input data set is clustered into groups of no more than M'
//! vectors, where M' is some small multiple of M, the maximum number of edges at each vertex in
//! the graph. The cluster means are saved and an RNG is built over them to use as a "head" graph
//! that we use to select an entry point into the "tail" graph over the data set.
//!
//! Lucene binary vector partitioning uses a similar algorithm to reorder the docid space within a
//! segment which dramatically reduces index and search latency over a graph based index. This
//! appears to be locality related -- edges tend to be to nodes "nearby" and the same effect does
//! not hold if I simplify modify the insertion order rather than reordering identifiers. This is
//! an attempt to build a mutable index with similar properties.
//!
//! For the head graph index we will use assigned cluster ids as table keys and edge identifiers; in
//! the tail index we will use <cluster_id, vector_id> tuples as keys and edge identifiers. The
//! table containing the original full fidelity vectors and filtering metadata will be keyed by
//! the input vector id and contain a link to the associated cluster id.
//!
//! To search this index we will search the head graph for an entry point cluster id (k=1), then
//! initialize the candidate queue with the first entry in the cluster and search using the regular
//! RNG search algorithm that HNSW and Vamana use. At this point we are relying on the clustering
//! of the graph to reduce the need for IO -- traversal should pull us towards a handful of clusters
//! and those clusters and vectors within those clusters should be physically contiguous or nearly
//! so.
//!
//! To insert into this index we will search the head index, select the nearest centroid, and add
//! the vector to this cluster. Graph edges will be built using a full search of the tail graph in
//! much the same way as they would using a regular graph index. To delete a vector we will locate
//! its cluster and remove vector and graph edges, reconnecting any missing nodes. Upserts will
//! perform a compound version of insert + delete. If clusters become imbalanced in size according
//! to policy (too small or too large) we will gather the vectors from nearby clusters and
//! repartition to add or remove a cluster.
//!
//! Bulk builds can use a path like Vamana and it will likely be faster due to the clustering. We
//! may also want to experiment with limiting the scope of the search, pre-seeding links within
//! each cluster and then searching "nearby" clusters for neighbors. This may be easy to parallelize
//! than the regular Vamana implementation.
//!
//! This data structure borrows inspiration from SPANN, HNSW, and the Lucene experiment with using
//! binary partitioning to order the neighbor graph.

use std::{
    fs::File,
    io::{self, BufWriter, Write},
    num::NonZero,
};

use memmap2::Mmap;
use tempfile::tempfile;

use crate::{
    distance::l2_normalize,
    hcrng::clustering::ClusterIter,
    input::{DerefVectorStore, VectorStore},
    vectors::VectorSimilarity,
};

// XXX should not be pub.
pub mod clustering;

#[derive(Debug, Clone)]
pub struct VectorOrdinalMapping {
    original_ids: Vec<usize>,
    cluster_ends: Vec<usize>,
}

impl VectorOrdinalMapping {
    /// Maps an ordinal in the clustered id space back to the original data set ordinal.
    pub fn to_original_id(&self, clustered_id: usize) -> usize {
        self.original_ids[clustered_id]
    }

    /// Identify the cluster that a given clustered_id is in.
    pub fn identify_cluster_id(&self, clustered_id: usize) -> usize {
        match self.cluster_ends.binary_search(&clustered_id) {
            Ok(x) => x + 1,
            Err(x) => x,
        }
    }
}

/// Create clusters from dataset using the given similiarity and max cluster length.
///
/// Returns the input dataset reordered by the cluster, the set of centroids, and an ordinal mapper
/// to help associate the new vectors with their original ordinals.
pub fn create_clusters(
    dataset: &(impl VectorStore<Elem = f32> + Send + Sync),
    similarity: VectorSimilarity,
    max_cluster_len: usize,
    progress: &(impl Fn(u64) + Send + Sync),
) -> io::Result<(
    DerefVectorStore<f32, Mmap>,
    DerefVectorStore<f32, Mmap>,
    VectorOrdinalMapping,
)> {
    // Vector file containing the cluster centroids.
    let mut cluster_writer = BufWriter::with_capacity(128 << 10, tempfile()?);
    // Vector file containing the reordered vectors.
    let mut vector_writer = BufWriter::with_capacity(128 << 10, tempfile()?);
    // Indexed by cluster id, contains the index of the first vector in the next cluster.
    // This is used to identify the cluster given an ordinal in the clustered space.
    let mut cluster_ends = vec![];
    // Original id for each vector in the clustered list.
    let mut original_ids = Vec::with_capacity(dataset.len());

    let cluster_it = ClusterIter::new(
        dataset,
        max_cluster_len,
        similarity.new_distance_function(),
        progress,
    );

    for (centroid, original_vector_ids) in cluster_it {
        // Make sure the centroids are normalized for dot product similarity.
        let centroid = match similarity {
            VectorSimilarity::Dot => l2_normalize(centroid),
            _ => centroid.into(),
        };
        for d in centroid.iter() {
            cluster_writer.write_all(&d.to_le_bytes())?;
        }
        for vector in original_vector_ids.iter().map(|i| &dataset[*i]) {
            for d in vector {
                vector_writer.write_all(&d.to_le_bytes())?;
            }
        }
        cluster_ends.push(cluster_ends.last().unwrap_or(&0) + original_vector_ids.len());
        original_ids.extend_from_slice(&original_vector_ids);
    }

    let clusters = writer_to_vector_store(cluster_writer, dataset.elem_stride())?;
    let vectors = writer_to_vector_store(vector_writer, dataset.elem_stride())?;
    let mapping = VectorOrdinalMapping {
        original_ids,
        cluster_ends,
    };

    Ok((vectors, clusters, mapping))
}

fn writer_to_vector_store(
    writer: BufWriter<File>,
    stride: usize,
) -> io::Result<DerefVectorStore<f32, Mmap>> {
    let file = writer.into_inner()?;
    file.sync_all()?;
    DerefVectorStore::<f32, _>::new(
        unsafe { memmap2::Mmap::map(&file) }?,
        NonZero::new(stride).unwrap(),
    )
}

// XXX I want to re-use as much as I possibly can
// * bp yields batches as (centroid, assigned) tuples
// * centroids are saved and use with a batch loader to write the head index.
// * assignment used to quantized the vectors into a new order and write them into a temporary location
// * assignment is also used to generate an ordinal -> (cluster_id, vector_id) tuple
// * feed temporary vectors to vamana bulk ingestion
// * extract the graph information, write with full keys.
