mod lookup;
mod search;

use std::{
    io::{self, ErrorKind},
    num::NonZero,
};

use clap::{command, Parser, Subcommand};
use easy_tiger::wt::WiredTigerIndexParams;
use lookup::{lookup, LookupArgs};
use search::{search, SearchArgs};
use wt_mdb::{Connection, ConnectionOptionsBuilder};

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
    /// WiredTiger table basename use to locate the graph.
    #[arg(long)]
    wiredtiger_table_basename: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Lookup the contents of a single vertex.
    Lookup(LookupArgs),
    /// Search for a list of vectors and time the operation.
    Search(SearchArgs),
    /// Add a list of vectors to the index.
    Add,
    /// Delete vectors by key range.
    Delete,
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    let connection = Connection::open(
        &cli.wiredtiger_db_path,
        Some(
            ConnectionOptionsBuilder::default()
                .cache_size_mb(cli.wiredtiger_cache_size_mb)
                .into(),
        ),
    )
    .map_err(io::Error::from)?;
    let index_params =
        WiredTigerIndexParams::new(connection.clone(), &cli.wiredtiger_table_basename);

    match cli.command {
        Commands::Lookup(args) => lookup(connection, index_params, args),
        Commands::Search(args) => search(connection, index_params, args),
        Commands::Add => Err(std::io::Error::from(ErrorKind::Unsupported)),
        Commands::Delete => Err(std::io::Error::from(ErrorKind::Unsupported)),
    }
}
