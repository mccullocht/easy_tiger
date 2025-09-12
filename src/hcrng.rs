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

use crate::{hcrng::clustering::ClusterIter, input::VectorStore, vectors::VectorSimilarity};

// XXX should not be pub.
pub mod clustering;

pub struct VectorOrdinalMapping;

impl VectorOrdinalMapping {
    /// Maps an ordinal in the clustered id space back to the original data set ordinal.
    pub fn to_original_id(&self, clustered_id: usize) -> usize {
        todo!()
    }

    /// Identify the cluster that a given clustered_id is in.
    pub fn identify_cluster_id(&self, clustered_id: usize) -> usize {
        // XXX binary_search will return the index that this element would be inserted at to
        // maintain order OR an exact match so this index would be sufficient to identify cluster.
        todo!()
    }
}

pub fn create_clusters(
    dataset: &(impl VectorStore<Elem = f32> + Send + Sync),
    similarity: VectorSimilarity,
    max_cluster_len: usize,
    progress: &(impl Fn(u64) + Send + Sync),
) {
    // XXX enumerate to generate cluster ids.
    // XXX to build the final graph I will take clustered_id keys (in tables and edges) to process.
    // XXX for each i will identify the cluster id and original ordinal. don't need original -> clustered.
    let mut cluster_it = ClusterIter::new(
        dataset,
        max_cluster_len,
        similarity.new_distance_function(),
        progress,
    );

    // XXX we have to return a DerefVectorStore pointing to a temp file.
    // XXX we also have to flush this file before we deref it.
    todo!()
}

// XXX I want to re-use as much as I possibly can
// * bp yields batches as (centroid, assigned) tuples
// * centroids are saved and use with a batch loader to write the head index.
// * assignment used to quantized the vectors into a new order and write them into a temporary location
// * assignment is also used to generate an ordinal -> (cluster_id, vector_id) tuple
// * feed temporary vectors to vamana bulk ingestion
// * extract the graph information, write with full keys.
