mod drop_index;
mod init_index;
mod insert_vectors;
mod search;

use std::{io, num::NonZero, sync::Arc};

use clap::{Args, Subcommand};
use easy_tiger::vamana::wt::read_app_metadata;
use serde::{Deserialize, Serialize};
use vectors::{F32VectorCoding, VectorSimilarity};
use wt_mdb::Connection;

use crate::wt_args::WiredTigerArgs;
use drop_index::drop_index;
use init_index::{InitIndexArgs, init_index};
use insert_vectors::{InsertVectorsArgs, insert_vectors};
use search::{SearchArgs, search};

#[derive(Serialize, Deserialize)]
pub(super) struct FlatIndexConfig {
    pub(super) dimensions: NonZero<usize>,
    pub(super) similarity: VectorSimilarity,
    pub(super) format: F32VectorCoding,
}

fn flat_table_name(index_name: &str) -> String {
    index_name.to_string()
}

fn open_config(connection: &Arc<Connection>, table_name: &str) -> io::Result<FlatIndexConfig> {
    let txn = connection.begin_transaction(None)?;
    let metadata = read_app_metadata(&txn, table_name)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "flat index table not found"))??;
    serde_json::from_str(&metadata).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

#[derive(Args)]
pub struct FlatArgs {
    #[command(flatten)]
    wt: WiredTigerArgs,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Initialize a new flat index.
    InitIndex(InitIndexArgs),
    /// Remove an existing flat index.
    DropIndex,
    /// Insert vectors into an existing flat index.
    InsertVectors(InsertVectorsArgs),
    /// Search a flat index exhaustively.
    Search(SearchArgs),
}

pub fn flat_command(args: FlatArgs) -> io::Result<()> {
    let cmd_connection = args.wt.open_connection()?;
    let connection = Arc::clone(&cmd_connection);
    let index_name = args.wt.index_name();
    match args.command {
        Command::InitIndex(args) => init_index(connection, index_name, args),
        Command::DropIndex => drop_index(connection, index_name),
        Command::InsertVectors(args) => insert_vectors(connection, index_name, args),
        Command::Search(args) => search(connection, index_name, args),
    }?;
    cmd_connection.checkpoint()?;
    Ok(())
}
