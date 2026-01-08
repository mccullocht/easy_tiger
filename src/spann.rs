//! Implement SPANN-like vector indexing on top of WiredTiger.
//!
//! This implemented by clustering the input dataset and building a graph-based index over the
//! select centroids. This index is used to build and navigate a posting index.

pub mod bulk;
pub mod centroid_stats;
pub mod rebalance;
pub mod search;

use std::{
    io,
    ops::{Range, RangeInclusive},
    sync::Arc,
};

use rustix::io::Errno;
use serde::{Deserialize, Serialize};
use vectors::{soar::SoarQueryVectorDistance, F32VectorCoder, F32VectorCoding, VectorDistance};
use wt_mdb::{
    session::{CreateOptionsBuilder, DropOptions, FormatString, Formatted},
    Connection, Error, Result, Session,
};

use crate::{
    input::{VecVectorStore, VectorStore},
    spann::centroid_stats::CentroidCounts,
    vamana::{
        wt::{read_app_metadata, SessionGraphVectorIndex, TableGraphVectorIndex},
        GraphConfig, GraphSearchParams, GraphVectorIndex, GraphVectorStore,
    },
    Neighbor,
};

/// Configuration for the SPANN index.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct IndexConfig {
    /// Parameters used to search the head index when selecting centroids.
    pub head_search_params: GraphSearchParams,
    /// Vector coding used for each posting vector.
    pub posting_coder: F32VectorCoding,
    /// Minimum centroid length in vectors.
    ///
    /// Centroids with fewer vectors will be removed from the index and their vectors will be
    /// reassigned to other centroids.
    pub min_centroid_len: usize,
    /// Maximum centroid length in vectors.
    ///
    /// Centroids with more vectors than this will be split into 2 centroids and their vectors will
    /// be reassigned to other centroids. Vectors in nearby centroids may also be reassigned.
    pub max_centroid_len: usize,
    /// Number of posting replicas to write each vector to. Must be non-negative.
    pub replica_count: usize,
    /// Algorithm to select posting replicas from candidate centroids.
    ///
    /// If `replica_count` is one, only the closest centroid is selected.
    pub replica_selection: ReplicaSelectionAlgorithm,
    /// If set, build a vector id keyed vector table in this format for re-ranking results.
    pub rerank_format: Option<F32VectorCoding>,
}

impl IndexConfig {
    /// Range of minimum and maximum centroid lengths.
    pub fn centroid_len_range(&self) -> RangeInclusive<usize> {
        self.min_centroid_len..=self.max_centroid_len
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Default, Debug)]
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
    // Table that maps centroid_id -> (primary_count, secondary_count).
    // These pre-aggregated statistics are used to balance the index and influence search.
    centroid_stats: String,
    // Table that maps record_id -> raw vector.
    // This is used for re-scoring after a SPANN search.
    raw_vectors: String,
}

impl TableNames {
    fn from_index_name(index_name: &str) -> Self {
        Self {
            postings: format!("{index_name}.postings"),
            centroids: format!("{index_name}.centroids"),
            centroid_stats: format!("{index_name}.centroid_stats"),
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
            self.centroid_stats.as_str(),
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

    pub fn new_posting_coder(&self) -> Box<dyn F32VectorCoder> {
        self.config
            .posting_coder
            .new_coder(self.head_config().config().similarity)
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
            &table_names.centroid_stats,
            Some(
                CreateOptionsBuilder::default()
                    .key_format::<u32>()
                    .value_format::<CentroidCounts>()
                    .into(),
            ),
        )?;
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

    pub fn centroid_assignments_table_name(&self) -> &str {
        &self.table_names.centroids
    }

    pub fn postings_table_name(&self) -> &str {
        &self.table_names.postings
    }

    pub fn centroid_stats_table_name(&self) -> &str {
        &self.table_names.centroid_stats
    }

    pub fn raw_vectors_table_name(&self) -> &str {
        &self.table_names.raw_vectors
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
pub struct PostingKey {
    /// Centroid identifier.
    pub centroid_id: u32,
    /// Original record identifier assigned or provided on insertion.
    pub record_id: i64,
}

impl PostingKey {
    pub fn centroid_range(centroid_id: u32) -> Range<Self> {
        Self {
            centroid_id,
            record_id: 0,
        }..Self {
            centroid_id: centroid_id + 1,
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
            Ok(Self {
                centroid_id,
                record_id,
            })
        } else {
            Err(Error::WiredTiger(wt_mdb::WiredTigerError::Generic))
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CentroidAssignmentType {
    Primary,
    Secondary,
}

/// A value in the centroid assignment table.
///
/// This maps a record to its primary centroid and any secondary centroids.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CentroidAssignment {
    primary_id: u32,
    secondary_ids: Vec<[u8; 4]>,
}

impl CentroidAssignment {
    pub fn new(primary_id: u32, secondary_ids: &[u32]) -> Self {
        Self {
            primary_id,
            secondary_ids: secondary_ids.iter().map(|id| id.to_le_bytes()).collect(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (CentroidAssignmentType, u32)> + '_ {
        self.to_formatted_ref().iter()
    }
}

impl Formatted for CentroidAssignment {
    const FORMAT: FormatString = FormatString::new(c"u");

    type Ref<'a> = CentroidAssignmentRef<'a>;

    #[inline(always)]
    fn to_formatted_ref(&self) -> Self::Ref<'_> {
        CentroidAssignmentRef {
            primary_id: self.primary_id,
            secondary_ids: &self.secondary_ids,
        }
    }

    #[inline(always)]
    fn pack(value: Self::Ref<'_>, packed: &mut Vec<u8>) -> Result<()> {
        packed.resize(value.len() * std::mem::size_of::<u32>(), 0);
        packed[..4].copy_from_slice(&value.primary_id.to_le_bytes());
        for (i, o) in value.secondary_ids.iter().zip(
            packed[4..]
                .as_chunks_mut::<{ std::mem::size_of::<u32>() }>()
                .0
                .iter_mut(),
        ) {
            *o = *i;
        }
        Ok(())
    }

    #[inline(always)]
    fn unpack<'b>(packed: &'b [u8]) -> Result<Self::Ref<'b>> {
        if !packed.len().is_multiple_of(std::mem::size_of::<u32>()) || packed.is_empty() {
            return Err(Error::Errno(Errno::INVAL));
        }

        let ids = packed.as_chunks::<{ std::mem::size_of::<u32>() }>().0;
        let primary_id = u32::from_le_bytes(ids[0]);
        let secondary_ids: &[[u8; 4]] = &ids[1..];
        Ok(CentroidAssignmentRef {
            primary_id,
            secondary_ids,
        })
    }
}

impl From<CentroidAssignmentRef<'_>> for CentroidAssignment {
    fn from(value: CentroidAssignmentRef<'_>) -> Self {
        Self {
            primary_id: value.primary_id,
            secondary_ids: value.secondary_ids.to_vec(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct CentroidAssignmentRef<'a> {
    primary_id: u32,
    secondary_ids: &'a [[u8; 4]],
}

impl<'a> CentroidAssignmentRef<'a> {
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        1 + self.secondary_ids.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (CentroidAssignmentType, u32)> + 'a {
        std::iter::once((CentroidAssignmentType::Primary, self.primary_id)).chain(
            self.secondary_ids
                .iter()
                .map(|id| (CentroidAssignmentType::Secondary, u32::from_le_bytes(*id))),
        )
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

    pub fn head_reader(&self) -> &SessionGraphVectorIndex {
        &self.head_reader
    }
}
