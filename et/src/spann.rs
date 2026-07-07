mod bulk_load;
mod centroid_stats;
mod drop_index;
mod export_head;
mod init_index;
mod insert_vectors;
mod rebalance;
mod search;

use std::{io, sync::Arc};

use clap::{Args, Subcommand};

use crate::wt_args::WiredTigerArgs;
use bulk_load::{bulk_load, BulkLoadArgs};
use centroid_stats::centroid_stats;
use drop_index::drop_index;
use export_head::{export_head, ExportHeadArgs};
use init_index::{init_index, InitIndexArgs};
use insert_vectors::{insert_vectors, InsertVectorsArgs};
use rebalance::{rebalance, RebalanceArgs};
use search::{search, SearchArgs};

#[derive(Args)]
pub struct SpannArgs {
    #[command(flatten)]
    wt: WiredTigerArgs,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Bulk load a set of vectors into an empty SPANN-ish index.
    BulkLoad(BulkLoadArgs),
    /// Initialize a SPANN-ish index with a single dummy centroid.
    InitIndex(InitIndexArgs),
    /// Insert vectors into an existing SPANN-ish index.
    InsertVectors(InsertVectorsArgs),
    /// Search a SPANN-ish index.
    Search(SearchArgs),
    /// Print centroid assignment statistics.
    CentroidStats,
    /// Export centroid vectors from the head index as little-endian f32 values.
    ExportHead(ExportHeadArgs),
    /// Rebalance the SPANN index.
    Rebalance(RebalanceArgs),
    /// Remove an existing index.
    DropIndex,
}

pub fn spann_command(args: SpannArgs) -> io::Result<()> {
    let cmd_connection = args.wt.open_connection()?;
    let connection = Arc::clone(&cmd_connection);
    let index_name = args.wt.index_name();
    match args.command {
        Command::BulkLoad(args) => bulk_load(connection, index_name, args),
        Command::InitIndex(args) => init_index(connection, index_name, args),
        Command::InsertVectors(args) => insert_vectors(connection, index_name, args),
        Command::Search(args) => search(connection, index_name, args),
        Command::CentroidStats => centroid_stats(connection, index_name),
        Command::ExportHead(args) => export_head(connection, index_name, args),
        Command::Rebalance(args) => rebalance(connection, index_name, args),
        Command::DropIndex => drop_index(connection, index_name),
    }?;
    cmd_connection.checkpoint()?;
    Ok(())
}
