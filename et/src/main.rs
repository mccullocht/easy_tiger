mod compute_neighbors;
mod generate;
mod neighbor_util;
mod quantization;
mod recall;
mod spann;
mod ui;
mod vamana;
mod wt_args;
mod wt_stats;

use std::io::{self};

use clap::{Parser, Subcommand};
use compute_neighbors::{ComputeNeighborsArgs, compute_neighbors};
use generate::{GenerateArgs, generate};
use quantization::{QuantizationArgs, quantization};
use spann::{SpannArgs, spann_command};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;
use vamana::{VamanaArgs, vamana_command};

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
    /// Uses GPU acceleration via wgpu if a suitable adapter is available, otherwise CPU.
    ComputeNeighbors(ComputeNeighborsArgs),
    /// Quantization related utilities.
    Quantization(QuantizationArgs),
    /// Generate random vectors and write them to a file.
    Generate(GenerateArgs),
}

fn main() -> io::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy()
                .add_directive("wgpu_core=warn".parse().unwrap())
                .add_directive("wgpu_hal=warn".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Spann(args) => spann_command(args),
        Commands::Vamana(args) => vamana_command(args),
        Commands::ComputeNeighbors(args) => compute_neighbors(args),
        Commands::Quantization(args) => quantization(args),
        Commands::Generate(args) => generate(args),
    }
}
