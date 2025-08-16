mod recall;
mod spann;
mod ui;
mod vamana;
mod wt_args;
mod wt_stats;

use std::io::{self};

use clap::{command, Parser, Subcommand};
use spann::{spann_command, SpannArgs};
use vamana::{vamana_command, VamanaArgs};

#[derive(Parser)]
#[command(version, about = "EasyTiger vector indexing tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Perform SPANN index operations.
    Spann(SpannArgs),
    /// Perform Vamana/DiskANN index operations.
    Vamana(VamanaArgs),
}

fn main() -> io::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Spann(args) => spann_command(args),
        Commands::Vamana(args) => vamana_command(args),
    }
}
