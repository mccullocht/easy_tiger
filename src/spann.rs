//! Implement SPANN-like vector indexing on top of WiredTiger.
//!
//! This implemented by clustering the input dataset and building a graph-based index over the
//! select centroids. This index is used to build and navigate a posting index.

use std::{
    borrow::Cow,
    collections::{BinaryHeap, HashSet},
    io,
    iter::FusedIterator,
    num::NonZero,
    ops::{Add, AddAssign},
    sync::Arc,
};

use bytemuck::try_cast_slice;
use serde::{Deserialize, Serialize};
use tracing::warn;
use wt_mdb::{
    options::{CreateOptionsBuilder, DropOptions},
    Connection, Error, IndexCursorGuard, IndexRecordView, Record, RecordCursorGuard, Result,
    Session, WiredTigerError,
};

use crate::{
    distance::{F32VectorDistance, VectorDistance},
    graph::{GraphConfig, GraphSearchParams, GraphVectorIndexReader, RawVectorStore},
    input::{VecVectorStore, VectorStore},
    quantization::{Quantizer, VectorQuantizer},
    search::{GraphSearchStats, GraphSearcher},
    wt::{SessionGraphVectorIndexReader, TableGraphVectorIndex},
    Neighbor,
};

#[derive(Serialize, Deserialize, Clone)]
pub struct IndexConfig {
    pub replica_count: usize,
    pub head_search_params: GraphSearchParams,
    pub quantizer: VectorQuantizer,
}

#[derive(Clone)]
struct TableNames {
    // Table that maps (centroid_id,record_id) -> quantized vector.
    // Ranges of this table are searched based on the outcome of searching the head.
    postings: String,
    // Table that maps record_id -> centroid_id*.
    // This table is necessary when deleting a vector to locate rows posting rows to delete.
    // It may also be useful for determining matching centroids in a filtered search.
    centroids: String,
    // Table that maps record_id -> raw vector.
    // This is used for re-scoring after a SPANN search.
    raw_vectors: String,
}

impl TableNames {
    fn from_index_name(index_name: &str) -> Self {
        TableNames {
            postings: format!("{}.postings", index_name),
            centroids: format!("{}.centroids", index_name),
            raw_vectors: format!("{}.raw_vectors", index_name),
        }
    }

    fn record_table_names(&self) -> impl Iterator<Item = &str> {
        [self.centroids.as_str(), self.raw_vectors.as_str()].into_iter()
    }

    fn index_table_names(&self) -> impl Iterator<Item = &str> {
        [self.postings.as_str()].into_iter()
    }

    fn all_names(&self) -> impl Iterator<Item = &str> {
        [
            self.postings.as_str(),
            self.centroids.as_str(),
            self.raw_vectors.as_str(),
        ]
        .into_iter()
    }
}

#[derive(Clone)]
pub struct TableIndex {
    // Head vector index containing the centroids.
    head: Arc<TableGraphVectorIndex>,
    table_names: TableNames,
    config: IndexConfig,
}

impl TableIndex {
    pub fn head_config(&self) -> &Arc<TableGraphVectorIndex> {
        &self.head
    }

    pub fn config(&self) -> &IndexConfig {
        &self.config
    }

    pub fn from_db(connection: &Arc<Connection>, index_name: &str) -> io::Result<Self> {
        let head = Arc::new(TableGraphVectorIndex::from_db(
            connection,
            &Self::head_name(index_name),
        )?);

        let table_names = TableNames::from_index_name(index_name);
        let session = connection.open_session()?;
        let mut cursor = session.open_record_cursor(&table_names.centroids)?;
        let config_json = unsafe { cursor.seek_exact_unsafe(-1) }
            .unwrap_or(Err(Error::WiredTiger(WiredTigerError::NotFound)))?;
        let config: IndexConfig = serde_json::from_slice(config_json.value())?;

        Ok(Self {
            head,
            table_names,
            config,
        })
    }

    pub fn from_init(
        index_name: &str,
        head_config: GraphConfig,
        spann_config: IndexConfig,
    ) -> Self {
        let head = Arc::new(
            TableGraphVectorIndex::from_init(head_config, &Self::head_name(index_name)).unwrap(),
        );
        Self {
            head,
            table_names: TableNames::from_index_name(index_name),
            config: spann_config,
        }
    }

    pub fn init_index(
        connection: &Arc<Connection>,
        index_name: &str,
        head_config: GraphConfig,
        spann_config: IndexConfig,
    ) -> io::Result<Self> {
        let head = Arc::new(TableGraphVectorIndex::init_index(
            connection,
            None,
            head_config,
            &Self::head_name(index_name),
        )?);
        let table_names = TableNames::from_index_name(index_name);
        let session = connection.open_session()?;
        for table_name in table_names.record_table_names() {
            session.create_table(
                table_name,
                Some(
                    CreateOptionsBuilder::default()
                        .table_type(wt_mdb::options::TableType::Record)
                        .into(),
                ),
            )?;
        }
        for table_name in table_names.index_table_names() {
            session.create_table(
                table_name,
                Some(
                    CreateOptionsBuilder::default()
                        .table_type(wt_mdb::options::TableType::Index)
                        .into(),
                ),
            )?;
        }
        let mut cursor = session.open_record_cursor(&table_names.centroids)?;
        cursor.set(&Record::new(-1, serde_json::to_vec(&spann_config)?))?;
        Ok(Self {
            head,
            table_names,
            config: spann_config,
        })
    }

    pub fn drop_tables(
        session: &Session,
        index_name: &str,
        options: &Option<DropOptions>,
    ) -> Result<()> {
        TableGraphVectorIndex::drop_tables(session, &Self::head_name(index_name), options)?;
        for table_name in TableNames::from_index_name(index_name).all_names() {
            session.drop_table(table_name, options.clone())?;
        }
        Ok(())
    }

    fn head_name(index_name: &str) -> String {
        format!("{}.head", index_name)
    }
}

/// A key in the posting table.
///
/// Serialized posting keys should result in entries ordered by centroid_id and then record_id,
/// allowing each centroid to be read as a contiguous range.
struct PostingKey {
    centroid_id: u32,
    record_id: i64,
}

impl PostingKey {
    fn for_centroid(centroid_id: u32) -> Self {
        Self {
            centroid_id,
            record_id: 0,
        }
    }

    fn into_key(self) -> [u8; 12] {
        <[u8; 12]>::from(self)
    }
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
    distance_fn: Box<dyn F32VectorDistance + 'static>,
    posting_quantizer: Box<dyn Quantizer + 'static>,

    head_reader: SessionGraphVectorIndexReader,
    head_searcher: GraphSearcher,
}

impl SessionIndexWriter {
    pub fn new(index: Arc<TableIndex>, session: Session) -> Self {
        let distance_fn = index.head.config().new_distance_function();
        let posting_quantizer = index.config.quantizer.new_quantizer();
        let head_reader = SessionGraphVectorIndexReader::new(index.head.clone(), session);
        let head_searcher = GraphSearcher::new(index.config.head_search_params);
        Self {
            index,
            distance_fn,
            posting_quantizer,
            head_reader,
            head_searcher,
        }
    }

    pub fn session(&self) -> &Session {
        self.head_reader.session()
    }

    pub fn upsert(&mut self, record_id: i64, vector: &[f32]) -> Result<Vec<u32>> {
        let vector = self.distance_fn.normalize(vector.into());
        let candidates = self
            .head_searcher
            .search(vector.as_ref(), &mut self.head_reader)?;
        let centroid_ids = self.select_centroids(candidates)?;

        let mut centroid_cursor = self.centroid_cursor()?;
        let mut posting_cursor = self.posting_cursor()?;
        if let Some(centroid_ids) = Self::read_centroid_ids(record_id, &mut centroid_cursor)? {
            Self::remove_postings(centroid_ids, record_id, &mut posting_cursor)?;
        }

        let mut raw_vector_cursor = self.raw_vector_cursor()?;
        let vector = self.distance_fn.normalize(vector);
        // TODO: factor out handling of high fidelity vector tables.
        raw_vector_cursor.set(&Record::new(
            record_id,
            vector
                .iter()
                .flat_map(|d| d.to_le_bytes())
                .collect::<Vec<_>>(),
        ))?;
        centroid_cursor.set(&Record::new(
            record_id,
            Cow::from(
                centroid_ids
                    .iter()
                    .flat_map(|i| i.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
        ))?;
        let quantized = self.posting_quantizer.for_doc(vector.as_ref());
        // TODO: try centering vector on each centroid before quantizing. This would likely reduce
        // error but would also require quantizing the vector for each centroid during search and
        // would complicate de-duplication (would have to accept best score).
        for centroid_id in centroid_ids.iter().copied() {
            let key: [u8; 12] = PostingKey {
                centroid_id,
                record_id,
            }
            .into();
            posting_cursor.set(&IndexRecordView::new(&key, &quantized))?;
        }

        Ok(centroid_ids)
    }

    pub fn delete(&self, record_id: i64) -> Result<()> {
        let mut centroid_cursor = self.centroid_cursor()?;
        if let Some(centroid_ids) = Self::read_centroid_ids(record_id, &mut centroid_cursor)? {
            let mut posting_cursor = self.posting_cursor()?;
            Self::remove_postings(centroid_ids, record_id, &mut posting_cursor)?;
            centroid_cursor.remove(record_id)?;
            let mut raw_vector_cursor = self.raw_vector_cursor()?;
            raw_vector_cursor.remove(record_id)?;
        }
        Ok(())
    }

    fn posting_cursor(&self) -> Result<IndexCursorGuard<'_>> {
        self.head_reader
            .session()
            .get_index_cursor(&self.index.table_names.postings)
    }

    fn centroid_cursor(&self) -> Result<RecordCursorGuard<'_>> {
        self.head_reader
            .session()
            .get_record_cursor(&self.index.table_names.centroids)
    }

    fn raw_vector_cursor(&self) -> Result<RecordCursorGuard<'_>> {
        self.head_reader
            .session()
            .get_record_cursor(&self.index.table_names.raw_vectors)
    }

    fn select_centroids(&self, candidates: Vec<Neighbor>) -> Result<Vec<u32>> {
        assert!(!candidates.is_empty());
        let replica_count = self.index.config.replica_count;
        let mut raw_vectors = self.head_reader.raw_vectors()?;

        let mut centroid_ids: Vec<u32> = Vec::with_capacity(replica_count);
        let mut centroids = VecVectorStore::with_capacity(
            self.head_reader.index().config().dimensions.get() * 4,
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
                .any(|c| self.distance_fn.distance(c, &v) < candidate.distance())
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

struct PostingIter<'a, 'b, 'c, 'd> {
    cursor: IndexCursorGuard<'a>,
    seen: &'b mut HashSet<i64>,
    query: &'c [u8],
    distance_fn: &'d dyn VectorDistance,

    read: usize,
}

impl<'a, 'b, 'c, 'd> PostingIter<'a, 'b, 'c, 'd> {
    fn new(
        reader: &'a SessionIndexReader,
        centroid_id: u32,
        seen: &'b mut HashSet<i64>,
        query: &'c [u8],
        distance_fn: &'d dyn VectorDistance,
    ) -> Result<Self> {
        // I _think_ wt copies bounds so we should be cool with temporaries here.
        let mut cursor = reader
            .session()
            .get_index_cursor(&reader.index().table_names.postings)?;
        cursor.set_bounds(
            PostingKey::for_centroid(centroid_id).into_key().as_slice()
                ..PostingKey::for_centroid(centroid_id + 1)
                    .into_key()
                    .as_slice(),
        )?;
        Ok(Self {
            cursor,
            seen,
            query,
            distance_fn,
            read: 0,
        })
    }

    fn read(&self) -> usize {
        self.read
    }
}

impl Iterator for PostingIter<'_, '_, '_, '_> {
    type Item = Result<Neighbor>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(record_result) = unsafe { self.cursor.next_unsafe() } {
            self.read += 1;
            let (raw_key, vector) = match record_result {
                Ok(r) => r.into_inner(),
                Err(e) => return Some(Err(e)),
            };
            let record_id = PostingKey::from(
                <[u8; 12]>::try_from(raw_key.as_ref()).expect("12-byte posting key"),
            )
            .record_id;
            if self.seen.insert(record_id) {
                let dist = self.distance_fn.distance(self.query, &vector);
                return Some(Ok(Neighbor::new(record_id, dist)));
            }
        }
        None
    }
}

impl FusedIterator for PostingIter<'_, '_, '_, '_> {}

pub struct SessionIndexReader {
    index: Arc<TableIndex>,
    head_reader: SessionGraphVectorIndexReader,
}

impl SessionIndexReader {
    pub fn new(index: &Arc<TableIndex>, session: Session) -> Self {
        let head_reader = SessionGraphVectorIndexReader::new(index.head_config().clone(), session);
        Self {
            index: index.clone(),
            head_reader,
        }
    }

    pub fn session(&self) -> &Session {
        self.head_reader.session()
    }

    pub fn index(&self) -> &TableIndex {
        self.index.as_ref()
    }
}

/// Parameters for SPANN searches.
#[derive(Debug, Clone, Copy)]
pub struct SpannSearchParams {
    /// Parameters for searching the head graph.
    /// NB: `head_params.beam_width` should be at least as large as `num_centroids`
    pub head_params: GraphSearchParams,
    /// The number of centroids to search.
    pub num_centroids: NonZero<usize>,
    /// The number of vectors to rerank using raw vectors.
    pub num_rerank: usize,
    /// The number of results to return.
    pub limit: NonZero<usize>,
}

/// Statistics for SPANN searches.
#[derive(Debug, Copy, Clone, Default)]
pub struct SpannSearchStats {
    /// Stats from the search of the head graph.
    pub head: GraphSearchStats,
    /// Number of posting lists read.
    pub postings_read: usize,
    /// Number of posting entries read.
    pub posting_entries_read: usize,
    /// Number of posting entries scored.
    pub posting_entries_scored: usize,
}

impl Add for SpannSearchStats {
    type Output = SpannSearchStats;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            head: self.head + rhs.head,
            postings_read: self.postings_read + rhs.postings_read,
            posting_entries_read: self.posting_entries_read + rhs.posting_entries_read,
            posting_entries_scored: self.posting_entries_scored + rhs.posting_entries_scored,
        }
    }
}

impl AddAssign for SpannSearchStats {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

pub struct SpannSearcher {
    params: SpannSearchParams,
    head_searcher: GraphSearcher,
    seen: HashSet<i64>,
    stats: SpannSearchStats,
}

impl SpannSearcher {
    pub fn new(params: SpannSearchParams) -> Self {
        Self {
            params,
            head_searcher: GraphSearcher::new(params.head_params),
            seen: HashSet::new(),
            stats: SpannSearchStats::default(),
        }
    }

    pub fn stats(&self) -> SpannSearchStats {
        self.stats
    }

    pub fn search(
        &mut self,
        query: &[f32],
        reader: &mut SessionIndexReader,
    ) -> Result<Vec<Neighbor>> {
        self.stats = SpannSearchStats::default();

        let mut centroids = self.head_searcher.search(query, &mut reader.head_reader)?;
        self.stats.head = self.head_searcher.stats();
        // TODO: be clever about choosing centroids.
        centroids.truncate(self.params.num_centroids.get());
        self.stats.postings_read = centroids.len();
        if centroids.is_empty() {
            return Ok(vec![]);
        }

        self.seen.clear();
        let quantizer = reader.index().config().quantizer.new_quantizer();
        let quantized_query = quantizer.for_query(query);
        let distance_fn = reader
            .index
            .config()
            .quantizer
            .new_distance_function(&reader.index().head_config().config().similarity);
        // TODO: replace the heap, try https://quickwit.io/blog/top-k-complexity
        let mut results = BinaryHeap::with_capacity(self.params.limit.get());
        for c in centroids {
            let centroid_id: u32 = c.vertex().try_into().expect("centroid_id is a u32");
            // TODO: consider structuring as a single iterator over centroids to avoid cursor freelisting.
            // TODO: if I can't read a posting list then skip and warn rather than exiting early.
            let mut it = PostingIter::new(
                reader,
                centroid_id,
                &mut self.seen,
                &quantized_query,
                distance_fn.as_ref(),
            )?;
            for candidate_result in &mut it {
                match candidate_result {
                    Ok(candidate) => {
                        if results.len() < self.params.limit.get() {
                            results.push(candidate);
                        } else {
                            let mut top = results.peek_mut().expect("pq full");
                            if candidate < *top {
                                *top = candidate;
                            }
                        }
                        self.stats.posting_entries_scored += 1;
                    }
                    Err(e) => warn!("Failed to read posting in centroid {}: {}", centroid_id, e),
                }
            }
            self.stats.posting_entries_read += it.read();
        }

        if self.params.num_rerank == 0 {
            return Ok(results.into_sorted_vec());
        }

        let distance_fn = reader.head_reader.config().new_distance_function();
        let mut raw_cursor = reader
            .session()
            .open_record_cursor(&reader.index().table_names.raw_vectors)?;
        let mut reranked = results
            .into_sorted_vec()
            .into_iter()
            .take(self.params.num_rerank)
            .map(|n| {
                let raw_vector_bytes = unsafe {
                    raw_cursor
                        .seek_exact_unsafe(n.vertex())
                        .expect("raw vector for candidate")
                }?
                .into_inner_value();
                let raw_vector: Cow<'_, [f32]> = try_cast_slice(raw_vector_bytes.as_ref())
                    .map(Cow::from)
                    .unwrap_or_else(|_| {
                        raw_vector_bytes
                            .chunks(4)
                            .map(|d| f32::from_le_bytes(d.try_into().expect("chunk size 4")))
                            .collect::<Vec<_>>()
                            .into()
                    });
                Ok(Neighbor::new(
                    n.vertex(),
                    distance_fn.distance_f32(query, &raw_vector),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        reranked.sort_unstable();

        Ok(reranked)
    }
}
