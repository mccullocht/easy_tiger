mod bulk_load;
mod delete;
mod drop_index;
mod exhaustive_search;
mod init_index;
mod insert;
mod lookup;
mod search;
mod ui;

use std::{
    io::{self},
    num::NonZero,
};

use bulk_load::{bulk_load, BulkLoadArgs};
use clap::{command, Parser, Subcommand};
use delete::{delete, DeleteArgs};
use drop_index::drop_index;
use exhaustive_search::{exhaustive_search, ExhaustiveSearchArgs};
use init_index::{init_index, InitIndexArgs};
use insert::{insert, InsertArgs};
use lookup::{lookup, LookupArgs};
use search::{search, SearchArgs};
use wt_mdb::{
    options::{ConnectionOptionsBuilder, Statistics},
    Connection,
};

#[derive(Parser)]
#[command(version, about = "EasyTiger vector indexing tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Path to the WiredTiger database.
    #[arg(long)]
    wiredtiger_db_path: String,
    /// Size of the WiredTiger disk cache, in MB.
    #[arg(long, default_value = "1024")]
    wiredtiger_cache_size_mb: NonZero<usize>,
    /// If true, create the WiredTiger database if it does not exist.
    #[arg(long, default_value = "false")]
    wiredtiger_create_db: bool,

    /// Name of the index, used to derive table names in WiredTiger.
    #[arg(short, long)]
    index_name: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new (empty) index.
    InitIndex(InitIndexArgs),
    /// Bulk load a set of vectors into an index.
    /// Requires that the index be uninitialized.
    BulkLoad(BulkLoadArgs),
    /// Drop an index.
    DropIndex,
    /// Lookup the contents of a single vertex.
    Lookup(LookupArgs),
    /// Search for a list of vectors and time the operation.
    Search(SearchArgs),
    /// Exhaustively search an index and create new Neighbors files.
    ExhaustiveSearch(ExhaustiveSearchArgs),
    /// Insert a vectors from a file into the index.
    Insert(InsertArgs),
    /// Delete vectors by key range.
    Delete(DeleteArgs),
}

fn main() -> io::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    // TODO: Connection.filename should accept &Path. This will likely be very annoying to plumb to CString.
    let mut connection_options = ConnectionOptionsBuilder::default()
        .cache_size_mb(cli.wiredtiger_cache_size_mb)
        .statistics(Statistics::Fast);
    if cli.wiredtiger_create_db {
        connection_options = connection_options.create();
    }
    let connection = Connection::open(&cli.wiredtiger_db_path, Some(connection_options.into()))?;

    match cli.command {
        Commands::BulkLoad(args) => bulk_load(connection, args, &cli.index_name),
        Commands::Delete(args) => delete(connection, &cli.index_name, args),
        Commands::DropIndex => drop_index(connection, &cli.index_name),
        Commands::InitIndex(args) => init_index(connection, &cli.index_name, args),
        Commands::Insert(args) => insert(connection, &cli.index_name, args),
        Commands::Lookup(args) => lookup(connection, &cli.index_name, args),
        Commands::Search(args) => search(connection, &cli.index_name, args),
        Commands::ExhaustiveSearch(args) => {
            exhaustive_search(connection.clone(), &cli.index_name, args)
        }
    }
}
