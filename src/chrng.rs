//! # Clustered hierarchical relative neighbor graph (CHRNG).
//!
//! Partition the input data set into clusters of no more than M' vectors, where M' is some small
//! multiple of M, the maximum number of edges at each vertex in the graph. Cluster centroids are
//! saved and an RNG is built over them to use as a "head" graph that we use to select an entry
//! point into the "tail" graph over the data set.
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
//! To search this index:
//! * Search the head graph for an entry point cluster id E.
//! * Score every vector in E and add them to the candidate queue.
//! * Examine the top entry in the candidate queue and score every vector in every cluster it has
//!   an outbound ege to.
//! * Continue with the standard graph vector search algorithm.
//!
//! In practice most of the results comes from E and its neighboring clusters with only a small
//! number of candidates coming from outside these clusters.
//!
//! To insert into this index we will search the head index, select the nearest centroid, and add
//! the vector to this cluster. Graph edges will be built using a full search of the tail graph in
//! much the same way as they would using a regular graph index. To delete a vector we will locate
//! its cluster and remove vector and graph edges, reconnecting any missing nodes. Upserts will
//! perform a compound version of insert + delete. If clusters become imbalanced in size according
//! to policy (too small or too large) we will gather the vectors from nearby clusters and
//! repartition to add or remove a cluster as is suggested in the SPFresh paper.
//!
//! Bulk builds can use a path like Vamana and it will likely be faster due to the clustering. We
//! may also want to experiment with limiting the scope of the search, pre-seeding links within
//! each cluster and then searching "nearby" clusters for neighbors. This may be easy to parallelize
//! than the regular Vamana implementation.
//!
//! This data structure borrows inspiration from SPANN, HNSW, and the Lucene experiment with using
//! binary partitioning to order the neighbor graph.
//!
//! ## Performance
//! All tests performed on a 1.9M voyage 3.5 dbpedia data set with 2048 dimensions and lvq1x8.
//!
//! When there is enough space for everything to be memory resident this performs about the same as
//! Vamana in terms of latency but scores about 70% more vectors and gets a 2.4% recall boost.
//! Improved vector locality and sequential scoring allow us to score more vectors without a
//! substantial increase in latency, since we aren't stalling on last-level cache misses or reads as
//! often as in a typical graph search.
//!
//! When memory is artifically restricted (by setting WT cache size to 256MB) latency is 10% lower
//! and 20% fewer bytes are read into WT cache.
//!
//! ## Future work
//!
//! ### Add a normal vamana switch
//!
//! This can be searched like a normal Vamana index, so add an option in the searcher to do that.
//!
//! ### Measure pointer chase length.
//!
//! Logically speaking:
//! * Search the head graph.
//! * Score the entry cluster.
//! * Score first neighboring clusters.
//!
//! Is a sequence of 3 sets of reads that can largely be done in parallel. After that we score
//! vertex-by-vertex and it would nice to be know how many rounds of scoring we do after that point
//! as it would influence what our latency looks like. If the vectors are on object storage it would
//! not be possible to hit a 500ms latency target in any case but <5s would be interesting.
//!
//! ### Head graph improvements
//!
//! The head graph could use mediods instead of centroids and pivot directly into the graph in the
//! same way HNSW does today.
//!
//! ### Bulk builds
//!
//! Bulk builds are implemented by clustering and building a regular Vamana graph, then re-keying
//! the data set. After clustering completes builds are faster than regular Vamana owing to locality
//! improvements, but there may be more gains from seeding in-cluster graph edges and bulk scoring
//! against other clusters.
//!
//! ### Clustering Improvements
//!
//! We are using binary partitioning but would probably be better off with batch k-means at the
//! beginning of clustering to reduce costs.
//!
//! ### Filtered search
//!
//! Figure out a story here. ACORN style approaches don't map as well to this approach, but the bulk
//! scoring step is also very cheap due to locality so it might not matter?
//!

use std::{
    array::TryFromSliceError,
    fs::File,
    io::{self, BufWriter, Write},
    num::NonZero,
    sync::Arc,
};

use memmap2::Mmap;
use tempfile::tempfile;
use vectors::{l2_normalize, VectorSimilarity};
use wt_mdb::{options::CreateOptionsBuilder, Connection, Result};

use crate::{
    chrng::clustering::ClusterIter,
    input::{DerefVectorStore, VectorStore},
    vamana::wt::{read_app_metadata, Leb128EdgeIterator, ENTRY_POINT_KEY},
};

// XXX should not be pub.
pub mod clustering;
pub mod search;
pub mod wt;

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
        VectorSimilarity::Euclidean.new_distance_function(),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClusterKey {
    pub cluster_id: u32,
    pub vector_id: i64,
}

impl ClusterKey {
    fn from_mapping(clustered_id: i64, mapping: &VectorOrdinalMapping) -> Self {
        Self {
            cluster_id: mapping.identify_cluster_id(clustered_id as usize) as u32,
            vector_id: mapping.to_original_id(clustered_id as usize) as i64,
        }
    }

    fn to_key_bytes(self) -> [u8; 12] {
        let mut key = [0u8; 12];
        key[..4].copy_from_slice(&self.cluster_id.to_be_bytes());
        key[4..].copy_from_slice(&self.vector_id.to_be_bytes());
        key
    }
}

impl From<[u8; 12]> for ClusterKey {
    fn from(value: [u8; 12]) -> Self {
        let cluster_id = u32::from_be_bytes(value[..4].try_into().unwrap());
        let vector_id = i64::from_be_bytes(value[4..].try_into().unwrap());
        Self {
            cluster_id,
            vector_id,
        }
    }
}

impl TryFrom<&[u8]> for ClusterKey {
    type Error = TryFromSliceError;

    fn try_from(value: &[u8]) -> std::result::Result<Self, Self::Error> {
        let rep: [u8; 12] = value.try_into()?;
        Ok(ClusterKey::from(rep))
    }
}

/// Rewrite the table containing a Vamana graph to have hierarchical keys, using the given key
/// mapping to write the hierarchy.
pub fn rewrite_graph_table(
    connection: &Arc<Connection>,
    table_name: &str,
    mapping: &VectorOrdinalMapping,
    progress: impl Fn(u64),
) -> Result<()> {
    let session = connection.open_session()?;
    let app_metadata = read_app_metadata(&session, table_name).unwrap()?;
    let input_cursor = session.open_record_cursor(table_name)?;
    let tmp_table_name = format!("{table_name}.tmp");
    let mut output_cursor = session.new_bulk_load_cursor::<Vec<u8>, Vec<u8>>(
        &tmp_table_name,
        Some(CreateOptionsBuilder::default().app_metadata(&app_metadata)),
    )?;
    let mut out_edges = vec![];
    for e in input_cursor {
        let (record_id, edges) = e?;
        if record_id == ENTRY_POINT_KEY {
            // we don't need an entry point. the head graph will be used for this.
            continue;
        }
        let key = ClusterKey::from_mapping(record_id, mapping);

        out_edges.clear();
        let mut last_edge = ClusterKey {
            cluster_id: 0,
            vector_id: 0,
        };
        for edge in Leb128EdgeIterator::new(&edges) {
            let out_edge = ClusterKey::from_mapping(edge, mapping);
            let cluster_delta = out_edge.cluster_id - last_edge.cluster_id;
            let vector_delta = if cluster_delta == 0 {
                out_edge.vector_id - last_edge.vector_id
            } else {
                out_edge.vector_id
            };
            leb128::write::unsigned(&mut out_edges, cluster_delta.into()).expect("write to vec");
            leb128::write::signed(&mut out_edges, vector_delta).expect("write to vec");
            last_edge = out_edge;
        }

        output_cursor.insert(&key.to_key_bytes(), &out_edges)?;
        progress(1);
    }
    drop(output_cursor);

    session.drop_table(table_name, None)?;
    session.rename_table(&tmp_table_name, table_name)?;

    Ok(())
}

/// Rewrite a table to have hierarchical keys, using the given key mapping to write the hierarchy.
pub fn rewrite_table(
    connection: &Arc<Connection>,
    table_name: &str,
    mapping: &VectorOrdinalMapping,
    progress: impl Fn(u64),
) -> Result<()> {
    let session = connection.open_session()?;
    let input_cursor = session.open_record_cursor(table_name)?;
    let tmp_table_name = format!("{table_name}.tmp");
    let mut output_cursor =
        session.new_bulk_load_cursor::<Vec<u8>, Vec<u8>>(&tmp_table_name, None)?;
    for e in input_cursor {
        let (record_id, payload) = e?;
        let key = ClusterKey::from_mapping(record_id, mapping);
        output_cursor.insert(&key.to_key_bytes(), &payload)?;
        progress(1);
    }
    drop(output_cursor);

    session.drop_table(table_name, None)?;
    session.rename_table(&tmp_table_name, table_name)?;

    Ok(())
}

/// Cursor over the head graph, providing access to the outbound edges from each vertex.
pub trait HeadGraphCursor {
    /// Iterator over a list of vector ordinals.
    type EdgeIterator<'a>: Iterator<Item = i64>
    where
        Self: 'a;

    /// Returns the entry point to the graph.
    fn entry_point(&mut self) -> Result<i64>;

    /// Return an iterator over the outbound edges from `vertex_id`.
    fn edges(&mut self, vertex_id: i64) -> Result<Self::EdgeIterator<'_>>;
}

/// Cursor over head vectors, returning the distance between a fixed query and the named vertex.
pub trait HeadVectorDistanceCursor {
    /// Compute the distance between the fixed query and vertex_id.
    ///
    /// Returns a not found error if `vertex_id` cannot be found.
    fn distance(&mut self, vertex_id: i64) -> Result<f64>;
}

/// Cursor over the tail graph, providing access to the outbound edges from each vertex.
pub trait TailGraphCursor {
    /// Iterator over a sequence of `ClusterKey`
    type EdgeIterator<'a>: Iterator<Item = ClusterKey>
    where
        Self: 'a;

    /// Return an iterator over the outbound edges from `vertex_id`.
    fn edges(&mut self, vertex_id: ClusterKey) -> Result<Self::EdgeIterator<'_>>;
}

/// Cursor over tail vectors accessible by [`ClusterKey`] or a `cluster_id`, computing the distance
/// against a fixed input query vector.
pub trait TailVectorDistanceCursor {
    type VectorDistanceIter<'a, F: FnMut(ClusterKey) -> bool>: Iterator<
        Item = Result<(ClusterKey, f64)>,
    >
    where
        Self: 'a;

    /// Compute the distance to the given vertex.
    ///
    /// Returns a not found error if `vertex_id` cannot be found.
    fn distance(&mut self, vertex_id: ClusterKey) -> Result<f64>;

    /// Compute the distance to the vectors in `cluster_id`, after imposing `filter` on the input.
    fn cluster_distance<F: FnMut(ClusterKey) -> bool>(
        &mut self,
        cluster_id: u32,
        filter: F,
    ) -> Self::VectorDistanceIter<'_, F>;
}

pub trait IndexReader {
    type HeadGraphCursor<'a>: HeadGraphCursor + 'a
    where
        Self: 'a;
    type HeadVectorDistanceCursor<'a>: HeadVectorDistanceCursor + 'a
    where
        Self: 'a;
    type TailGraphCursor<'a>: TailGraphCursor + 'a
    where
        Self: 'a;
    type TailVectorDistanceCursor<'a>: TailVectorDistanceCursor + 'a
    where
        Self: 'a;

    /// Return a cursor over graph data for head index.
    fn head_graph_cursor(&self) -> Result<Self::HeadGraphCursor<'_>>;

    /// Return a cursor over head vectors that will yield the distance between `query` and arbitrary
    /// vectors in the index.
    fn head_vector_distance_cursor(
        &self,
        query: impl Into<Vec<f32>>,
    ) -> Result<Self::HeadVectorDistanceCursor<'_>>;

    /// Return a cursor over graph data for the tail index.
    fn tail_graph_cursor(&self) -> Result<Self::TailGraphCursor<'_>>;

    /// Return a cursor over tail vectors that will yield the distance between `query` and arbitrary
    /// vectors in the index.
    fn tail_vector_distance_cursor(
        &self,
        query: impl Into<Vec<f32>>,
    ) -> Result<Self::TailVectorDistanceCursor<'_>>;
}
