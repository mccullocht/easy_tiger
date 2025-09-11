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
