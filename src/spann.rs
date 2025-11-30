//! Implement SPANN-like vector indexing on top of WiredTiger.
//!
//! This implemented by clustering the input dataset and building a graph-based index over the
//! select centroids. This index is used to build and navigate a posting index.

pub mod bulk;
pub mod centroid_stats;
pub mod search;

use std::{io, sync::Arc};

use serde::{Deserialize, Serialize};
use vectors::{soar::SoarQueryVectorDistance, F32VectorCoder, F32VectorCoding, VectorDistance};
use wt_mdb::{
    options::{CreateOptionsBuilder, DropOptions},
    session::{FormatString, Formatted},
    Connection, Error, RecordCursorGuard, Result, Session, TypedCursorGuard,
};

use crate::{
    input::{VecVectorStore, VectorStore},
    vamana::search::GraphSearcher,
    vamana::wt::{read_app_metadata, SessionGraphVectorIndex, TableGraphVectorIndex},
    vamana::{GraphConfig, GraphSearchParams, GraphVectorIndex, GraphVectorStore},
    Neighbor,
};

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct IndexConfig {
    pub replica_count: usize,
    pub replica_selection: ReplicaSelectionAlgorithm,
    pub head_search_params: GraphSearchParams,
    pub posting_coder: F32VectorCoding,
    pub rerank_format: Option<F32VectorCoding>,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReplicaSelectionAlgorithm {
    /// Select replicas using relative neighbor graph edge pruning.
    RNG,
    /// Select replicas using SOAR distance scoring.
    #[default]
    SOAR,
}

impl std::str::FromStr for ReplicaSelectionAlgorithm {
    type Err = io::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "rng" => Ok(Self::RNG),
            "soar" => Ok(Self::SOAR),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown replica selection algorithm {s}"),
            )),
        }
    }
}

impl std::fmt::Display for ReplicaSelectionAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RNG => write!(f, "rng"),
            Self::SOAR => write!(f, "soar"),
        }
    }
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
        let config: IndexConfig = serde_json::from_str(
            &read_app_metadata(&session, &table_names.postings)
                .ok_or(Error::not_found_error())??,
        )?;
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
    const FORMAT: FormatString = FormatString::new(c"u");

    type Ref<'a> = Self;

    #[inline(always)]
    fn to_formatted_ref(&self) -> Self::Ref<'_> {
        *self
    }

    #[inline(always)]
    fn pack(value: Self::Ref<'_>, packed: &mut Vec<u8>) -> Result<()> {
        packed.resize(12, 0);
        packed[..4].copy_from_slice(&value.centroid_id.to_be_bytes());
        packed[4..].copy_from_slice(&value.record_id.to_be_bytes());
        Ok(())
    }

    #[inline(always)]
    fn unpack<'b>(packed: &'b [u8]) -> Result<Self::Ref<'b>> {
        if packed.len() == 12 {
            let centroid_id = u32::from_be_bytes(packed[..4].try_into().unwrap());
            let record_id = i64::from_be_bytes(packed[4..].try_into().unwrap());
            Ok(Self::new(centroid_id, record_id))
        } else {
            Err(Error::WiredTiger(wt_mdb::WiredTigerError::Generic))
        }
    }
}

pub struct SessionIndexWriter {
    index: Arc<TableIndex>,
    distance_fn: Box<dyn VectorDistance>,
    posting_coder: Box<dyn F32VectorCoder>,
    raw_coder: Option<Box<dyn F32VectorCoder>>,

    head_reader: SessionGraphVectorIndex,
    head_searcher: GraphSearcher,
}

impl SessionIndexWriter {
    pub fn new(index: Arc<TableIndex>, session: Session) -> Self {
        let distance_fn = index.head.high_fidelity_table().new_distance_function();
        let posting_coder = index
            .config
            .posting_coder
            .new_coder(index.head.config().similarity);
        let raw_coder = index
            .config()
            .rerank_format
            .map(|t| t.new_coder(index.head_config().config().similarity));
        let head_reader = SessionGraphVectorIndex::new(index.head.clone(), session);
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
            self.index.config.replica_selection,
            self.index.config.replica_count,
            candidates,
            vector,
            &self.head_reader,
            self.distance_fn.as_ref(),
        )?;

        let mut centroid_cursor = self.centroid_cursor()?;
        let mut posting_cursor = self.posting_cursor()?;
        if let Some(centroid_ids) = Self::read_centroid_ids(record_id, &mut centroid_cursor)? {
            Self::remove_postings(centroid_ids, record_id, &mut posting_cursor)?;
        }

        if let Some((vectors, coder)) = self.raw_vector_cursor().zip(self.raw_coder.as_ref()) {
            vectors?.set(record_id, &coder.encode(vector))?;
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
            if let Some(cursor) = self.raw_vector_cursor() {
                cursor?.remove(record_id)?;
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

    fn raw_vector_cursor(&self) -> Option<Result<RecordCursorGuard<'_>>> {
        self.index.config().rerank_format.map(|_| {
            self.head_reader
                .session()
                .get_record_cursor(&self.index.table_names.raw_vectors)
        })
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
    algorithm: ReplicaSelectionAlgorithm,
    replica_count: usize,
    candidates: Vec<Neighbor>,
    vector: &[f32],
    head_reader: &impl GraphVectorIndex,
    distance_fn: &dyn VectorDistance,
) -> Result<Vec<u32>> {
    assert!(
        !candidates.is_empty(),
        "at least one candidate is required for replica selection"
    );
    if replica_count == 1 {
        // If we aren't selecting multiple replicas then just return the first candidate.
        return Ok(vec![candidates[0].vertex() as u32]);
    }

    match algorithm {
        ReplicaSelectionAlgorithm::RNG => {
            select_centroids_rng(replica_count, candidates, head_reader, distance_fn)
        }
        ReplicaSelectionAlgorithm::SOAR => {
            select_centroids_soar(replica_count, candidates, vector, head_reader)
        }
    }
}

fn select_centroids_rng(
    replica_count: usize,
    candidates: Vec<Neighbor>,
    head_reader: &impl GraphVectorIndex,
    distance_fn: &dyn VectorDistance,
) -> Result<Vec<u32>> {
    assert!(!candidates.is_empty());
    let mut vectors = head_reader.high_fidelity_vectors()?;
    let coder = vectors.new_coder();

    let mut centroid_ids: Vec<u32> = Vec::with_capacity(replica_count);
    let mut centroids = VecVectorStore::with_capacity(
        coder.byte_len(head_reader.config().dimensions.get()),
        replica_count,
    );
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

fn select_centroids_soar(
    replica_count: usize,
    candidates: Vec<Neighbor>,
    vector: &[f32],
    head_reader: &impl GraphVectorIndex,
) -> Result<Vec<u32>> {
    assert!(!candidates.is_empty());
    let mut vectors = head_reader.high_fidelity_vectors()?;
    let coder = vectors.new_coder();

    let primary = coder.decode(
        vectors
            .get(candidates[0].vertex())
            .unwrap_or(Err(Error::not_found_error()))?,
    );
    let soar_dist = if let Some(dist) = SoarQueryVectorDistance::new(vector, &primary) {
        dist
    } else {
        return Ok(vec![candidates[0].vertex() as u32]);
    };
    let mut secondary_centroid_ids = Vec::with_capacity(candidates.len() - 1);
    let mut candidate_vector = vec![0.0f32; primary.len()];
    for candidate in candidates.iter().skip(1) {
        coder.decode_to(
            vectors
                .get(candidate.vertex())
                .unwrap_or(Err(Error::not_found_error()))?,
            &mut candidate_vector,
        );
        secondary_centroid_ids.push(Neighbor::new(
            candidate.vertex(),
            soar_dist.distance(&candidate_vector),
        ));
    }

    secondary_centroid_ids.sort_unstable();
    Ok(std::iter::once(&candidates[0])
        .chain(secondary_centroid_ids.iter())
        .take(replica_count)
        .map(|n| n.vertex() as u32)
        .collect())
}

pub struct SessionIndexReader {
    index: Arc<TableIndex>,
    head_reader: SessionGraphVectorIndex,
}

impl SessionIndexReader {
    pub fn new(index: &Arc<TableIndex>, session: Session) -> Self {
        let head_reader = SessionGraphVectorIndex::new(index.head_config().clone(), session);
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
