//! Implement SPANN-like vector indexing on top of WiredTiger.
//!
//! This implemented by clustering the input dataset and building a graph-based index over the
//! select centroids. This index is used to build and navigate a posting index.

pub mod bulk;
pub mod centroid_stats;
pub mod postings;
pub mod rebalance;
pub mod search;

use std::{ops::RangeInclusive, sync::Arc};

use serde::{Deserialize, Serialize};
use vectors::{F32VectorCoder, F32VectorCoding};
use wt_mdb::{
    connection::{CreateOptionsBuilder, DropOptions},
    session::{CommitTransactionOptions, RollbackTransactionOptions},
    Connection, Error, Result, Transaction,
};

use crate::{
    spann::centroid_stats::CentroidCounts,
    vamana::{
        wt::{read_app_metadata, TableGraphVectorIndex, TransactionGraphVectorIndex},
        GraphConfig, GraphSearchParams,
    },
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
    /// If set, build a vector id keyed vector table in this format for re-ranking results.
    // XXX this can no longer be optional.
    pub rerank_format: Option<F32VectorCoding>,
}

impl IndexConfig {
    /// Range of minimum and maximum centroid lengths.
    pub fn centroid_len_range(&self) -> RangeInclusive<usize> {
        self.min_centroid_len..=self.max_centroid_len
    }
}

#[derive(Clone)]
struct TableNames {
    // Table that maps (centroid_id,record_id) -> quantized vector.
    // Ranges of this table are searched based on the outcome of searching the head.
    postings: String,
    // Table that maps centroid_id -> primary_count.
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
            centroid_stats: format!("{index_name}.centroid_stats"),
            raw_vectors: format!("{index_name}.raw_vectors"),
        }
    }

    fn record_table_names(&self) -> impl Iterator<Item = &str> {
        [self.raw_vectors.as_str()].into_iter()
    }

    fn all_names(&self) -> impl Iterator<Item = &str> {
        [
            self.postings.as_str(),
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
            .coder(self.head_config().config().similarity, None)
    }

    pub fn posting_vector_len(&self) -> usize {
        self.new_posting_coder()
            .byte_len(self.head_config().config().dimensions.get())
    }

    pub fn from_db(connection: &Arc<Connection>, index_name: &str) -> std::io::Result<Self> {
        let head = Arc::new(TableGraphVectorIndex::from_db(
            connection,
            &Self::head_name(index_name),
        )?);

        let table_names = TableNames::from_index_name(index_name);
        let txn = connection.begin_transaction(None)?;
        let config: IndexConfig = serde_json::from_str(
            &read_app_metadata(&txn, &table_names.postings).ok_or(Error::not_found_error())??,
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
    ) -> std::io::Result<Self> {
        let head_similarity = head_config.similarity;
        let head_dimensions = head_config.dimensions;
        let head = Arc::new(TableGraphVectorIndex::init_index(
            connection,
            head_config,
            &Self::head_name(index_name),
        )?);
        let table_names = TableNames::from_index_name(index_name);
        for table_name in table_names.record_table_names() {
            connection.create_table(
                table_name,
                Some(
                    CreateOptionsBuilder::default()
                        .key_format::<i64>()
                        .value_format::<Vec<u8>>()
                        .into(),
                ),
            )?;
        }
        connection.create_table(
            &table_names.centroid_stats,
            Some(
                CreateOptionsBuilder::default()
                    .key_format::<u32>()
                    .value_format::<CentroidCounts>()
                    .into(),
            ),
        )?;
        let posting_vector_len = spann_config
            .posting_coder
            .coder(head_similarity, None)
            .byte_len(head_dimensions.get());
        let leaf_page_size = crate::posting_block::leaf_page_max(
            spann_config.max_centroid_len,
            posting_vector_len,
            4096,
        ) as u32;
        connection.create_table(
            &table_names.postings,
            Some(
                CreateOptionsBuilder::default()
                    .key_format::<u32>()
                    .value_format::<Vec<u8>>()
                    .app_metadata(&serde_json::to_string(&spann_config)?)
                    .leaf_page_max(leaf_page_size)
                    .leaf_value_max(leaf_page_size)
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
        connection: &Arc<Connection>,
        index_name: &str,
        options: &Option<DropOptions>,
    ) -> Result<()> {
        TableGraphVectorIndex::drop_tables(connection, &Self::head_name(index_name), options)?;
        for table_name in TableNames::from_index_name(index_name).all_names() {
            connection.drop_table(table_name, options.clone())?;
        }
        Ok(())
    }

    pub fn postings_table_name(&self) -> &str {
        &self.table_names.postings
    }

    pub fn centroid_stats_table_name(&self) -> &str {
        &self.table_names.centroid_stats
    }

    // XXX this has to be renamed.
    pub fn raw_vectors_table_name(&self) -> &str {
        &self.table_names.raw_vectors
    }

    fn head_name(index_name: &str) -> String {
        format!("{index_name}.head")
    }
}

pub struct TransactionIndex {
    index: Arc<TableIndex>,
    head: TransactionGraphVectorIndex,
}

impl TransactionIndex {
    pub fn new(index: &Arc<TableIndex>, transaction: Transaction) -> Self {
        let head = TransactionGraphVectorIndex::new(Arc::clone(index.head_config()), transaction);
        Self {
            index: index.clone(),
            head,
        }
    }

    pub fn transaction(&self) -> &Transaction {
        self.head.transaction()
    }

    pub fn head(&self) -> &TransactionGraphVectorIndex {
        &self.head
    }

    pub fn index(&self) -> &Arc<TableIndex> {
        &self.index
    }

    /// Commit the underlying [`Transaction`] with the provided options.
    pub fn commit(self, options: Option<CommitTransactionOptions>) -> Result<()> {
        self.head.commit(options)
    }

    /// Rollback the underlying [`Transaction`] with the provided options.
    pub fn rollback(self, options: Option<RollbackTransactionOptions>) -> Result<()> {
        self.head.rollback(options)
    }
}
