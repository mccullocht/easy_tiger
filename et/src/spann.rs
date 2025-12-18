mod bulk_load;
mod centroid_stats;
mod closest_centroid;
mod drop_index;
mod search;

use std::io;

use clap::{Args, Subcommand};

use bulk_load::{bulk_load, BulkLoadArgs};
use centroid_stats::centroid_stats;
use closest_centroid::{closest_centroid, ClosestCentroidArgs};
use drop_index::drop_index;
use search::{search, SearchArgs};

use crate::wt_args::WiredTigerArgs;

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
    /// Search a SPANN-ish index.
    Search(SearchArgs),
    /// Print centroid assignment statistics.
    CentroidStats,
    /// Remove an existing index.
    DropIndex,
    /// Verify centroid assignments.
    ClosestCentroid(ClosestCentroidArgs),
}

pub fn spann_command(args: SpannArgs) -> io::Result<()> {
    let connection = args.wt.open_connection()?;
    let session = connection.open_session()?;
    let index_name = args.wt.index_name();
    match args.command {
        Command::BulkLoad(args) => bulk_load(connection, index_name, args),
        Command::Search(args) => search(connection, index_name, args),
        Command::CentroidStats => centroid_stats(connection, index_name),
        Command::DropIndex => drop_index(connection, index_name),
        Command::ClosestCentroid(args) => closest_centroid(connection, index_name, args),
    }?;
    session.checkpoint()?;
    Ok(())
}
