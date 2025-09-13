mod compute_neighbors;
mod hcrng;
mod neighbor_util;
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
use hcrng::{hcrng_command, HcrngArgs};
use quantization_recall::{quantization_recall, QuantizationRecallArgs};
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
    /// Perform Hierarchical Relative Neighbor graph index operations.
    Hcrng(HcrngArgs),
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
        Commands::Hcrng(args) => hcrng_command(args),
        Commands::ComputeNeighbors(args) => compute_neighbors(args),
        Commands::QuantizationRecall(args) => quantization_recall(args),
    }
}
