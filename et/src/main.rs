mod bulk_load;
mod drop_index;
mod exhaustive_search;
mod lookup;
mod search;

use std::{
    io::{self, ErrorKind},
    num::NonZero,
};

use bulk_load::{bulk_load, BulkLoadArgs};
use clap::{command, Parser, Subcommand};
use drop_index::drop_index;
use easy_tiger::wt::WiredTigerGraphVectorIndex;
use exhaustive_search::{exhaustive_search, ExhaustiveSearchArgs};
use lookup::{lookup, LookupArgs};
use search::{search, SearchArgs};
use wt_mdb::{options::ConnectionOptionsBuilder, Connection};

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
    /// Add a list of vectors to the index.
    Add,
    /// Delete vectors by key range.
    Delete,
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();
    // TODO: Connection.filename should accept &Path. This will likely be very annoying to plumb to CString.
    let mut connection_options =
        ConnectionOptionsBuilder::default().cache_size_mb(cli.wiredtiger_cache_size_mb);
    if cli.wiredtiger_create_db {
        connection_options = connection_options.create();
    }
    let connection = Connection::open(&cli.wiredtiger_db_path, Some(connection_options.into()))?;

    match cli.command {
        Commands::BulkLoad(args) => bulk_load(connection, args, &cli.index_name),
        Commands::DropIndex => drop_index(connection, &cli.index_name),
        Commands::Lookup(args) => lookup(
            connection.clone(),
            WiredTigerGraphVectorIndex::from_db(&connection, &cli.index_name)?,
            args,
        ),
        Commands::Search(args) => search(
            connection.clone(),
            WiredTigerGraphVectorIndex::from_db(&connection, &cli.index_name)?,
            args,
        ),
        Commands::ExhaustiveSearch(args) => exhaustive_search(
            connection.clone(),
            WiredTigerGraphVectorIndex::from_db(&connection, &cli.index_name)?,
            args,
        ),
        _ => Err(io::Error::from(ErrorKind::Unsupported)),
    }
}
