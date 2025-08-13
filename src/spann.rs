//! Implement SPANN-like vector indexing on top of WiredTiger.
//!
//! This implemented by clustering the input dataset and building a graph-based index over the
//! select centroids. This index is used to build and navigate a posting index.

pub mod bulk;

use std::{
    collections::{BinaryHeap, HashSet},
    io,
    iter::FusedIterator,
    num::NonZero,
    ops::{Add, AddAssign},
    sync::Arc,
};

use serde::{Deserialize, Serialize};
use tracing::warn;
use wt_mdb::{
    options::{CreateOptionsBuilder, DropOptions},
    session::{FormatString, FormatWriter, Formatted, PackedFormatReader},
    Connection, Error, RecordCursorGuard, Result, Session, TypedCursorGuard,
};

use crate::{
    graph::{GraphConfig, GraphSearchParams, GraphVectorIndexReader, GraphVectorStore},
    input::{VecVectorStore, VectorStore},
    search::{GraphSearchStats, GraphSearcher},
    vectors::{
        new_query_vector_distance_f32, F32VectorCoder, F32VectorCoding, QueryVectorDistance,
        VectorDistance,
    },
    wt::{read_app_metadata, SessionGraphVectorIndexReader, TableGraphVectorIndex},
    Neighbor,
};

#[derive(Serialize, Deserialize, Clone)]
pub struct IndexConfig {
    pub replica_count: usize,
    pub head_search_params: GraphSearchParams,
    pub posting_coder: F32VectorCoding,
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
            postings: format!("{index_name}.postings"),
            centroids: format!("{index_name}.centroids"),
            raw_vectors: format!("{index_name}.raw_vectors"),
        }
    }

    fn record_table_names(&self) -> impl Iterator<Item = &str> {
        [self.centroids.as_str(), self.raw_vectors.as_str()].into_iter()
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
        let raw_config_json =
            read_app_metadata(&session, &table_names.postings).ok_or(Error::not_found_error())??;
        Ok(Self {
            head,
            table_names,
            config: serde_json::from_str(&raw_config_json)?,
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
                        .key_format::<i64>()
                        .value_format::<Vec<u8>>()
                        .into(),
                ),
            )?;
        }
        session.create_table(
            &table_names.postings,
            Some(
                CreateOptionsBuilder::default()
                    .key_format::<PostingKey>()
                    .value_format::<Vec<u8>>()
                    .app_metadata(&serde_json::to_string(&spann_config)?)
                    .into(),
            ),
        )?;
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
        format!("{index_name}.head")
    }
}

/// A key in the posting table.
///
/// Serialized posting keys should result in entries ordered by centroid_id and then record_id,
/// allowing each centroid to be read as a contiguous range.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PostingKey {
    centroid_id: u32,
    record_id: i64,
}

impl PostingKey {
    fn new(centroid_id: u32, record_id: i64) -> Self {
        Self {
            centroid_id,
            record_id,
        }
    }

    fn for_centroid(centroid_id: u32) -> Self {
        Self {
            centroid_id,
            record_id: 0,
        }
    }
}

impl Formatted for PostingKey {
    const FORMAT: FormatString = FormatString::new(c"Iq");

    type Ref<'a> = Self;

    fn to_formatted_ref(&self) -> Self::Ref<'_> {
        *self
    }

    fn pack(writer: &mut impl FormatWriter, value: &PostingKey) -> Result<()> {
        writer.pack(value.centroid_id)?;
        writer.pack(value.record_id)
    }

    fn unpack<'b>(reader: &mut PackedFormatReader<'b>) -> Result<Self::Ref<'b>> {
        let centroid_id = reader.unpack()?;
        let record_id = reader.unpack()?;
        Ok(Self {
            centroid_id,
            record_id,
        })
    }
}

pub struct SessionIndexWriter {
    index: Arc<TableIndex>,
    distance_fn: Box<dyn VectorDistance + 'static>,
    posting_coder: Box<dyn F32VectorCoder + 'static>,
    raw_coder: Option<Box<dyn F32VectorCoder + 'static>>,

    head_reader: SessionGraphVectorIndexReader,
    head_searcher: GraphSearcher,
}

impl SessionIndexWriter {
    pub fn new(index: Arc<TableIndex>, session: Session) -> Self {
        let distance_fn = index.head.high_fidelity_table().new_distance_function();
        let posting_coder = index
            .config
            .posting_coder
            .new_coder(index.head.config().similarity);
        let raw_coder = index.head.rerank_table().map(|t| t.new_coder());
        let head_reader = SessionGraphVectorIndexReader::new(index.head.clone(), session);
        let head_searcher = GraphSearcher::new(index.config.head_search_params);
        Self {
            index,
            distance_fn,
            posting_coder,
            raw_coder,
            head_reader,
            head_searcher,
        }
    }

    pub fn session(&self) -> &Session {
        self.head_reader.session()
    }

    pub fn upsert(&mut self, record_id: i64, vector: &[f32]) -> Result<Vec<u32>> {
        let candidates = self
            .head_searcher
            .search(vector.as_ref(), &mut self.head_reader)?;
        let centroid_ids = select_centroids(
            &self.head_reader,
            candidates,
            self.distance_fn.as_ref(),
            self.index.config.replica_count,
        )?;

        let mut centroid_cursor = self.centroid_cursor()?;
        let mut posting_cursor = self.posting_cursor()?;
        if let Some(centroid_ids) = Self::read_centroid_ids(record_id, &mut centroid_cursor)? {
            Self::remove_postings(centroid_ids, record_id, &mut posting_cursor)?;
        }

        if let Some(raw_coder) = self.raw_coder.as_ref() {
            // XXX separate head coder from tail coder for raw vectors.
            let mut raw_vector_cursor = self.raw_vector_cursor()?;
            raw_vector_cursor.set(record_id, &raw_coder.encode(vector))?;
        }
        centroid_cursor.set(
            record_id,
            &centroid_ids
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;
        let quantized = self.posting_coder.encode(vector.as_ref());
        // TODO: try centering vector on each centroid before quantizing. This would likely reduce
        // error but would also require quantizing the vector for each centroid during search and
        // would complicate de-duplication (would have to accept best score).
        for centroid_id in centroid_ids.iter().copied() {
            posting_cursor.set(
                PostingKey::new(centroid_id, record_id),
                quantized.as_slice(),
            )?;
        }

        Ok(centroid_ids)
    }

    pub fn delete(&self, record_id: i64) -> Result<()> {
        let mut centroid_cursor = self.centroid_cursor()?;
        if let Some(centroid_ids) = Self::read_centroid_ids(record_id, &mut centroid_cursor)? {
            let mut posting_cursor = self.posting_cursor()?;
            Self::remove_postings(centroid_ids, record_id, &mut posting_cursor)?;
            centroid_cursor.remove(record_id)?;
            if self.raw_coder.is_some() {
                let mut raw_vector_cursor = self.raw_vector_cursor()?;
                raw_vector_cursor.remove(record_id)?;
            }
        }
        Ok(())
    }

    fn posting_cursor(&self) -> Result<TypedCursorGuard<'_, PostingKey, Vec<u8>>> {
        self.head_reader
            .session()
            .get_or_create_typed_cursor(&self.index.table_names.postings)
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

    fn read_centroid_ids(
        record_id: i64,
        cursor: &mut RecordCursorGuard<'_>,
    ) -> Result<Option<Vec<u32>>> {
        Ok(cursor.seek_exact(record_id).transpose()?.map(|r| {
            r.chunks(4)
                .map(|c| u32::from_be_bytes(c.try_into().expect("u32 centroid")))
                .collect()
        }))
    }

    fn remove_postings(
        centroid_ids: Vec<u32>,
        record_id: i64,
        cursor: &mut TypedCursorGuard<'_, PostingKey, Vec<u8>>,
    ) -> Result<()> {
        for centroid_id in centroid_ids {
            cursor
                .remove(PostingKey::new(centroid_id, record_id))
                .or_else(|e| {
                    if e == Error::not_found_error() {
                        Ok(())
                    } else {
                        Err(e)
                    }
                })?;
        }
        Ok(())
    }
}

fn select_centroids(
    head_reader: &impl GraphVectorIndexReader,
    candidates: Vec<Neighbor>,
    distance_fn: &dyn VectorDistance,
    replica_count: usize,
) -> Result<Vec<u32>> {
    assert!(!candidates.is_empty());
    let mut vectors = head_reader.high_fidelity_vectors()?;

    let mut centroid_ids: Vec<u32> = Vec::with_capacity(replica_count);
    let mut centroids =
        VecVectorStore::with_capacity(head_reader.config().dimensions.get() * 4, replica_count);
    for candidate in candidates {
        if centroid_ids.len() >= replica_count {
            break;
        }

        let v = vectors
            .get(candidate.vertex())
            .expect("returned vector should exist")?;
        if !centroids
            .iter()
            .any(|c| distance_fn.distance(c, v) < candidate.distance())
        {
            centroid_ids.push(
                candidate
                    .vertex()
                    .try_into()
                    .expect("centroid_ids <= u32::MAX"),
            );
            centroids.push(v);
        }
    }
    Ok(centroid_ids)
}

struct PostingIter<'a, 'b, 'c> {
    cursor: TypedCursorGuard<'a, PostingKey, Vec<u8>>,
    seen: &'b mut HashSet<i64>,
    tail_query: &'c dyn QueryVectorDistance,

    read: usize,
}

impl<'a, 'b, 'c> PostingIter<'a, 'b, 'c> {
    fn new(
        reader: &'a SessionIndexReader,
        centroid_id: u32,
        seen: &'b mut HashSet<i64>,
        tail_query: &'c dyn QueryVectorDistance,
    ) -> Result<Self> {
        // I _think_ wt copies bounds so we should be cool with temporaries here.
        let mut cursor = reader
            .session()
            .get_or_create_typed_cursor::<PostingKey, Vec<u8>>(
                &reader.index().table_names.postings,
            )?;
        cursor.set_bounds(
            PostingKey::for_centroid(centroid_id)..PostingKey::for_centroid(centroid_id + 1),
        )?;
        Ok(Self {
            cursor,
            seen,
            tail_query,
            read: 0,
        })
    }

    fn read(&self) -> usize {
        self.read
    }
}

impl Iterator for PostingIter<'_, '_, '_> {
    type Item = Result<Neighbor>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(record_result) = unsafe { self.cursor.next_unsafe() } {
            self.read += 1;
            let (record_id, vector) = match record_result {
                Ok((k, v)) => (k.record_id, v),
                Err(e) => return Some(Err(e)),
            };
            if self.seen.insert(record_id) {
                let dist = self.tail_query.distance(vector);
                return Some(Ok(Neighbor::new(record_id, dist)));
            }
        }
        None
    }
}

impl FusedIterator for PostingIter<'_, '_, '_> {}

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
        let tail_query = new_query_vector_distance_f32(
            query,
            reader.index().head_config().config().similarity,
            reader.index().config().posting_coder,
        );
        // TODO: replace the heap, try https://quickwit.io/blog/top-k-complexity
        let mut results = BinaryHeap::with_capacity(self.params.limit.get());
        for c in centroids {
            let centroid_id: u32 = c.vertex().try_into().expect("centroid_id is a u32");
            // TODO: consider structuring as a single iterator over centroids to avoid cursor freelisting.
            // TODO: if I can't read a posting list then skip and warn rather than exiting early.
            let mut it =
                PostingIter::new(reader, centroid_id, &mut self.seen, tail_query.as_ref())?;
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

        // XXX separate head rerank format from tail rerank format.
        let query = new_query_vector_distance_f32(
            query,
            reader.head_reader.config().similarity,
            reader
                .head_reader
                .config()
                .rerank_format
                .expect("using head format for no reason :)"),
        );
        let mut raw_cursor = reader
            .session()
            .open_record_cursor(&reader.index().table_names.raw_vectors)?;
        let mut reranked = results
            .into_sorted_vec()
            .into_iter()
            .take(self.params.num_rerank)
            .map(|n| {
                Ok(Neighbor::new(
                    n.vertex(),
                    query.distance(unsafe {
                        raw_cursor
                            .seek_exact_unsafe(n.vertex())
                            .expect("raw vector for candidate")?
                    }),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        reranked.sort_unstable();

        Ok(reranked)
    }
}
