mod exhaustive_search;
mod recall;
mod spann;
mod ui;
mod vamana;
mod wt_stats;

use std::{
    io::{self},
    num::NonZero,
};

use clap::{command, Parser, Subcommand};
use exhaustive_search::{exhaustive_search, ExhaustiveSearchArgs};
use spann::{spann_command, SpannArgs};
use vamana::{vamana_command, VamanaArgs};
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
    /// Perform SPANN index operations.
    Spann(SpannArgs),
    /// Perform Vamana/DiskANN index operations.
    Vamana(VamanaArgs),
    /// Exhaustively search an index and create new Neighbors files.
    ExhaustiveSearch(ExhaustiveSearchArgs),
}

fn main() -> io::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    // TODO: Connection.filename should accept &Path. This will likely be very annoying to plumb to CString.
    let mut connection_options = ConnectionOptionsBuilder::default()
        .cache_size_mb(cli.wiredtiger_cache_size_mb)
        .statistics(Statistics::Fast)
        .checkpoint_log_size(128 << 20);
    if cli.wiredtiger_create_db {
        connection_options = connection_options.create();
    }
    let connection = Connection::open(&cli.wiredtiger_db_path, Some(connection_options.into()))?;
    let session = connection.open_session()?;

    match cli.command {
        Commands::Spann(args) => spann_command(connection, &cli.index_name, args),
        Commands::Vamana(args) => vamana_command(connection, &cli.index_name, args),
        Commands::ExhaustiveSearch(args) => {
            exhaustive_search(connection.clone(), &cli.index_name, args)
        }
    }?;

    session.checkpoint()?;
    Ok(())
}
