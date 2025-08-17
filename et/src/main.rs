mod compute_neighbors;
mod quantization_recall;
mod recall;
mod spann;
mod ui;
mod vamana;
mod wt_args;
mod wt_stats;

use std::io::{self};

use clap::{command, Parser, Subcommand};
use compute_neighbors::{compute_neighbors, ComputeNeighborsArgs};
use spann::{spann_command, SpannArgs};
use vamana::{vamana_command, VamanaArgs};

use crate::quantization_recall::{quantization_recall, QuantizationRecallArgs};

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
    /// Compute the top k neighbors for a set of queries against a set of document vectors.
    ComputeNeighbors(ComputeNeighborsArgs),
    /// Quantize a vector set and compute exact recall against a ground truth.
    QuantizationRecall(QuantizationRecallArgs),
}

fn main() -> io::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Spann(args) => spann_command(args),
        Commands::Vamana(args) => vamana_command(args),
        Commands::ComputeNeighbors(args) => compute_neighbors(args),
        Commands::QuantizationRecall(args) => quantization_recall(args),
    }
}
