mod bulk_load;
mod search;

use std::{io, sync::Arc};

use bulk_load::{bulk_load, BulkLoadArgs};
use clap::{Args, Subcommand};
use search::{search, SearchArgs};
use wt_mdb::Connection;

#[derive(Args)]
pub struct SpannArgs {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Bulk load a set of vectors into an empty SPANN-ish index.
    BulkLoad(BulkLoadArgs),
    /// Search a SPANN-ish index.
    Search(SearchArgs),
}

pub fn spann_command(
    connection: Arc<Connection>,
    index_name: &str,
    args: SpannArgs,
) -> io::Result<()> {
    match args.command {
        Command::BulkLoad(args) => bulk_load(connection, index_name, args),
        Command::Search(args) => search(connection, index_name, args),
    }
}
