mod bulk_load;
mod centroid_stats;
mod drop_index;
mod extract;
mod init_index;
mod insert_vectors;
mod rebalance;
mod search;
mod verify_primary_assignments;

use std::io;

use clap::{Args, Subcommand};

use crate::wt_args::WiredTigerArgs;
use bulk_load::{bulk_load, BulkLoadArgs};
use centroid_stats::centroid_stats;
use drop_index::drop_index;
use extract::{extract_index, ExtractIndexArgs};
use init_index::{init_index, InitIndexArgs};
use insert_vectors::{insert_vectors, InsertVectorsArgs};
use rebalance::{rebalance, RebalanceArgs};
use search::{search, SearchArgs};
use verify_primary_assignments::verify_primary_assignments;

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
    /// Extract the index to a set of flat files.
    ExtractIndex(ExtractIndexArgs),
    /// Insert vectors into an existing SPANN-ish index.
    InsertVectors(InsertVectorsArgs),
    /// Search a SPANN-ish index.
    Search(SearchArgs),
    /// Print centroid assignment statistics.
    CentroidStats,
    /// Rebalance the SPANN index.
    Rebalance(RebalanceArgs),
    /// Remove an existing index.
    DropIndex,
    /// Verify that the primary assignment of each vector is to its closest centroid.
    VerifyPrimaryAssignments,
}

pub fn spann_command(args: SpannArgs) -> io::Result<()> {
    let connection = args.wt.open_connection()?;
    let session = connection.open_session()?;
    let index_name = args.wt.index_name();
    match args.command {
        Command::BulkLoad(args) => bulk_load(connection, index_name, args),
        Command::InitIndex(args) => init_index(connection, index_name, args),
        Command::ExtractIndex(args) => extract_index(connection, index_name, args),
        Command::InsertVectors(args) => insert_vectors(connection, index_name, args),
        Command::Search(args) => search(connection, index_name, args),
        Command::CentroidStats => centroid_stats(connection, index_name),
        Command::Rebalance(args) => rebalance(connection, index_name, args),
        Command::DropIndex => drop_index(connection, index_name),
        Command::VerifyPrimaryAssignments => verify_primary_assignments(connection, index_name),
    }?;
    session.checkpoint()?;
    Ok(())
}
