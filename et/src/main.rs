use std::io::ErrorKind;

use clap::{command, Parser, Subcommand};

#[derive(Parser)]
#[command(version, about = "EasyTiger vector indexing tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Search for a list of vectors and time the operation.
    Search,
    /// Add a list of vectors to the index.
    Add,
    /// Delete vectors by key range.
    Delete,
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Search => Err(std::io::Error::from(ErrorKind::Unsupported)),
        Commands::Add => Err(std::io::Error::from(ErrorKind::Unsupported)),
        Commands::Delete => Err(std::io::Error::from(ErrorKind::Unsupported)),
    }
}
