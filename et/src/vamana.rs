mod bulk_load;
mod delete;
mod drop_index;
mod init_index;
mod insert;
mod lookup;
mod search;

use std::{io, sync::Arc};

use clap::{Args, Subcommand};
use wt_mdb::Connection;

use bulk_load::{bulk_load, BulkLoadArgs};
use delete::{delete, DeleteArgs};
use drop_index::drop_index;
use init_index::{init_index, InitIndexArgs};
use insert::{insert, InsertArgs};
use lookup::{lookup, LookupArgs};
use search::{search, SearchArgs};

#[derive(Args)]
pub struct VamanaArgs {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Bulk load a set of vectors into an index.
    /// Requires that the index be uninitialized.
    BulkLoad(BulkLoadArgs),
    /// Search for a list of vectors and time the operation.
    Search(SearchArgs),
    /// Initialize a new (empty) index.
    InitIndex(InitIndexArgs),
    /// Drop an index.
    DropIndex,
    /// Lookup the contents of a single vertex.
    Lookup(LookupArgs),
    /// Insert a vectors from a file into the index.
    Insert(InsertArgs),
    /// Delete vectors by key range.
    Delete(DeleteArgs),
}

pub fn vamana_command(
    connection: Arc<Connection>,
    index_name: &str,
    args: VamanaArgs,
) -> io::Result<()> {
    match args.command {
        Command::BulkLoad(args) => bulk_load(connection, index_name, args),
        Command::Search(args) => search(connection, index_name, args),
        Command::InitIndex(args) => init_index(connection, index_name, args),
        Command::DropIndex => drop_index(connection, index_name),
        Command::Lookup(args) => lookup(connection, index_name, args),
        Command::Insert(args) => insert(connection, index_name, args),
        Command::Delete(args) => delete(connection, index_name, args),
    }
}
