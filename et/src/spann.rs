mod bulk_load;
mod drop_index;
mod search;

use std::io;

use bulk_load::{bulk_load, BulkLoadArgs};
use clap::{Args, Subcommand};
use search::{search, SearchArgs};

use crate::{spann::drop_index::drop_index, wt_args::WiredTigerArgs};

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
    /// Remove an existing index.
    DropIndex,
}

pub fn spann_command(args: SpannArgs) -> io::Result<()> {
    let connection = args.wt.open_connection()?;
    let session = connection.open_session()?;
    let index_name = args.wt.index_name();
    match args.command {
        Command::BulkLoad(args) => bulk_load(connection, index_name, args),
        Command::Search(args) => search(connection, index_name, args),
        Command::DropIndex => drop_index(connection, index_name),
    }?;
    session.checkpoint()?;
    Ok(())
}
