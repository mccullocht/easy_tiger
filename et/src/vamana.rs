//mod bulk_load;
//mod search;
pub(crate) mod drop_index; // XXX FIXME
mod init_index;

use std::{io, sync::Arc};

//use bulk_load::{bulk_load, BulkLoadArgs};
use clap::{Args, Subcommand};
//use search::{search, SearchArgs};
use wt_mdb::Connection;

use drop_index::drop_index;
use init_index::{init_index, InitIndexArgs};

#[derive(Args)]
pub struct VamanaArgs {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Initialize a new (empty) index.
    InitIndex(InitIndexArgs),
    /// Drop an index.
    DropIndex,
    /*
    /// Bulk load a set of vectors into an empty SPANN-ish index.
    BulkLoad(BulkLoadArgs),
    /// Search a SPANN-ish index.
    Search(SearchArgs),
    */
}

pub fn vamana_command(
    connection: Arc<Connection>,
    index_name: &str,
    args: VamanaArgs,
) -> io::Result<()> {
    match args.command {
        Command::InitIndex(args) => init_index(connection, index_name, args),
        Command::DropIndex => drop_index(connection, index_name),
        //Command::BulkLoad(args) => bulk_load(connection, index_name, args),
        //Command::Search(args) => search(connection, index_name, args),
    }
}
