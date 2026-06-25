//! A flat (exhaustive) vector index backed by WiredTiger.

pub mod search;

use std::{io, num::NonZero, sync::Arc};

use serde::{Deserialize, Serialize};
use vectors::{F32VectorCoding, VectorSimilarity};
use wt_mdb::{
    Connection,
    connection::{CreateOptionsBuilder, DropOptions},
};

use crate::vamana::wt::read_app_metadata;

/// Configuration for a flat vector index.
#[derive(Serialize, Deserialize, Clone)]
pub struct FlatIndexConfig {
    pub dimensions: NonZero<usize>,
    pub similarity: VectorSimilarity,
    pub format: F32VectorCoding,
}

/// Returns the WiredTiger table name for a flat index.
pub fn table_name(index_name: &str) -> String {
    index_name.to_string()
}

/// Reads the config for an existing flat index from the database.
pub fn open_config(connection: &Arc<Connection>, index_name: &str) -> io::Result<FlatIndexConfig> {
    let txn = connection.begin_transaction(None)?;
    let metadata = read_app_metadata(&txn, &table_name(index_name))
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "flat index table not found"))??;
    serde_json::from_str(&metadata).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Creates a new flat index table and stores `config` in its app metadata.
pub fn init_index(
    connection: &Arc<Connection>,
    index_name: &str,
    config: &FlatIndexConfig,
) -> io::Result<()> {
    connection
        .create_table(
            &table_name(index_name),
            Some(
                CreateOptionsBuilder::default()
                    .app_metadata(&serde_json::to_string(config)?)
                    .into(),
            ),
        )
        .map_err(io::Error::from)
}

/// Drops the table backing a flat index.
pub fn drop_index(
    connection: &Arc<Connection>,
    index_name: &str,
    options: Option<DropOptions>,
) -> io::Result<()> {
    connection
        .drop_table(&table_name(index_name), options)
        .map_err(io::Error::from)
}
