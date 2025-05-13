//! Implement SPANN-like vector indexing on top of WiredTiger.
//!
//! This implemented by clustering the input dataset and building a graph-based index over the
//! select centroids. This index is used to build and navigate a posting index.

use std::{borrow::Cow, sync::Arc};

use rand::Rng;
use wt_mdb::{
    Connection, Error, IndexCursorGuard, IndexRecordView, Record, RecordCursorGuard, Result,
};

use crate::{
    bulk::{self, BulkLoadBuilder},
    graph::{GraphSearchParams, GraphVectorIndexReader, RawVectorStore},
    input::{VecVectorStore, VectorStore},
    kmeans::{self, iterative_balanced_kmeans},
    quantization::VectorQuantizer,
    search::GraphSearcher,
    wt::{SessionGraphVectorIndexReader, TableGraphVectorIndex},
    Neighbor,
};

// XXX high level method structure:
// * cluster input vector, build graph
// * iterate input vectors, assign to centroid(s) and quantize to build pls
// * search

/// Build the head of the SPANN index.
///
/// Select or compute roughly `dataset.len() * ratio` vectors from `dataset` to use as the head.
/// A graph-based index will be built over these vectors and used to build and search the rest of
/// the index.
pub fn build_head<V: VectorStore<Elem = f32> + Send + Sync>(
    dataset: &V,
    ratio: f64,
    kmeans_params: kmeans::Params,
    connection: Arc<Connection>,
    index: TableGraphVectorIndex,
    rng: &mut impl Rng,
) -> Result<()> {
    let head_len = (dataset.len() as f64 * ratio + 0.5).round() as usize;
    // TODO: consider cluster ordering the centroids to make graph search faster.
    let (centroids, _) = iterative_balanced_kmeans(
        dataset,
        head_len,
        head_len.min(32),
        1.5,
        1000,
        &kmeans_params,
        rng,
    );
    let mut loader = BulkLoadBuilder::new(
        connection,
        index,
        centroids,
        bulk::Options {
            memory_quantized_vectors: false,
            wt_vector_store: true,
            cluster_ordered_insert: false,
        },
        head_len,
    );
    for phase in loader.phases() {
        // XXX gotta find a way to jam progress in here.
        loader.execute_phase(phase, |_| {})?;
    }
    Ok(())
}

pub struct IndexConfig {
    pub replica_count: usize,
    pub head_search_params: GraphSearchParams,
    pub quantizer: VectorQuantizer,
}

pub struct TableIndex {
    // Table that maps (centroid_id,record_id) -> quantized vector.
    // Ranges of this table are searched based on the outcome of searching the head.
    posting_table_name: String,
    // Table that maps record_id -> centroid_id*.
    // This table is used when deleting a vector; it allows us to locate rows to delete.
    centroid_table_name: String,
    config: IndexConfig,
}

/// A key in the posting table.
///
/// Serialized posting keys should result in entries ordered by centroid_id and then record_id,
/// allowing each centroid to be read as a contiguous range.
struct PostingKey {
    centroid_id: u32,
    record_id: i64,
}

impl From<[u8; 12]> for PostingKey {
    fn from(value: [u8; 12]) -> Self {
        let (c, r) = value.as_ref().split_at(std::mem::size_of::<u32>());
        PostingKey {
            centroid_id: u32::from_be_bytes(c.try_into().expect("centroid_id")),
            record_id: i64::from_be_bytes(r.try_into().expect("record_id")),
        }
    }
}

impl From<PostingKey> for [u8; 12] {
    fn from(value: PostingKey) -> Self {
        let mut bytes = [0u8; 12];
        bytes[..std::mem::size_of::<u32>()].copy_from_slice(&value.centroid_id.to_be_bytes());
        bytes[std::mem::size_of::<u32>()..].copy_from_slice(&value.record_id.to_be_bytes());
        bytes
    }
}

pub struct SessionIndexWriter {
    index: Arc<TableIndex>,
    head_reader: SessionGraphVectorIndexReader,
    head_searcher: GraphSearcher,
}

impl SessionIndexWriter {
    pub fn new(index: Arc<TableIndex>, head_reader: SessionGraphVectorIndexReader) -> Self {
        let head_searcher = GraphSearcher::new(index.config.head_search_params);
        Self {
            index,
            head_reader,
            head_searcher,
        }
    }

    pub fn upsert(&mut self, record_id: i64, vector: &[f32]) -> Result<()> {
        let candidates = self.head_searcher.search(vector, &mut self.head_reader)?;
        let centroid_ids = self.select_centroids(candidates)?;

        let mut centroid_cursor = self.centroid_cursor()?;
        let mut posting_cursor = self.posting_cursor()?;
        if let Some(centroid_ids) = Self::read_centroid_ids(record_id, &mut centroid_cursor)? {
            Self::remove_postings(centroid_ids, record_id, &mut posting_cursor)?;
        }

        centroid_cursor.set(&Record::new(
            record_id,
            Cow::from(
                centroid_ids
                    .iter()
                    .flat_map(|i| i.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
        ))?;
        let quantized = self
            .head_reader
            .index()
            .config()
            .new_quantizer()
            .for_doc(vector);
        // TODO: try centering vector on each centroid before quantizing.
        for centroid_id in centroid_ids {
            let key: [u8; 12] = PostingKey {
                centroid_id,
                record_id,
            }
            .into();
            posting_cursor.set(&IndexRecordView::new(&key, &quantized))?;
        }

        Ok(())
    }

    pub fn delete(&self, record_id: i64) -> Result<()> {
        let mut centroid_cursor = self.centroid_cursor()?;
        if let Some(centroid_ids) = Self::read_centroid_ids(record_id, &mut centroid_cursor)? {
            let mut posting_cursor = self.posting_cursor()?;
            Self::remove_postings(centroid_ids, record_id, &mut posting_cursor)?;
            centroid_cursor.remove(record_id)?;
        }
        Ok(())
    }

    fn posting_cursor(&self) -> Result<IndexCursorGuard<'_>> {
        self.head_reader
            .session()
            .get_index_cursor(&self.index.posting_table_name)
    }

    fn centroid_cursor(&self) -> Result<RecordCursorGuard<'_>> {
        self.head_reader
            .session()
            .get_record_cursor(&self.index.centroid_table_name)
    }

    fn select_centroids(&self, candidates: Vec<Neighbor>) -> Result<Vec<u32>> {
        assert!(!candidates.is_empty());
        let replica_count = self.index.config.replica_count;
        let mut raw_vectors = self.head_reader.raw_vectors()?;
        let distance_fn = self.head_reader.index().config().new_distance_function();

        let mut centroid_ids: Vec<u32> = Vec::with_capacity(replica_count);
        let mut centroids = VecVectorStore::with_capacity(
            self.head_reader.index().config().dimensions.get(),
            replica_count,
        );
        for candidate in candidates {
            if centroid_ids.len() >= replica_count {
                break;
            }

            let v = raw_vectors
                .get_raw_vector(candidate.vertex())
                .expect("returned vector should exist")?;
            if !centroids
                .iter()
                .any(|c| distance_fn.distance(c, &v) < candidate.distance())
            {
                centroid_ids.push(
                    candidate
                        .vertex()
                        .try_into()
                        .expect("centroid_ids <= u32::MAX"),
                );
                centroids.push(&v);
            }
        }
        Ok(centroid_ids)
    }

    fn read_centroid_ids(
        record_id: i64,
        cursor: &mut RecordCursorGuard<'_>,
    ) -> Result<Option<Vec<u32>>> {
        Ok(cursor.seek_exact(record_id).transpose()?.map(|r| {
            r.into_inner_value()
                .as_ref()
                .chunks(4)
                .map(|c| u32::from_be_bytes(c.try_into().expect("u32 centroid")))
                .collect()
        }))
    }

    fn remove_postings(
        centroid_ids: Vec<u32>,
        record_id: i64,
        cursor: &mut IndexCursorGuard<'_>,
    ) -> Result<()> {
        for centroid_id in centroid_ids {
            let key: [u8; 12] = PostingKey {
                centroid_id,
                record_id,
            }
            .into();
            cursor.remove(&key).or_else(|e| {
                if e == Error::not_found_error() {
                    Ok(())
                } else {
                    Err(e)
                }
            })?;
        }
        todo!()
    }
}

// XXX configure automatic checkpointing, something like checkpoint=(log_size=67108864,wait=15)
// XXX also explicitly checkpoint when closing a db?
