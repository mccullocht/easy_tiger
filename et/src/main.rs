mod compute_neighbors;
mod neighbor_util;
mod quantization;
mod recall;
mod spann;
mod ui;
mod vamana;
mod wt_args;
mod wt_stats;

use std::{
    io::{self},
    path::PathBuf,
};

use clap::{Parser, Subcommand, command};
use compute_neighbors::{ComputeNeighborsArgs, compute_neighbors};
use quantization::{QuantizationArgs, quantization};
use spann::{SpannArgs, spann_command};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{
    EnvFilter, Layer, filter::filter_fn, fmt, fmt::format::FmtSpan, layer::SubscriberExt,
    util::SubscriberInitExt,
};
use vamana::{VamanaArgs, vamana_command};

#[derive(Parser)]
#[command(version, about = "EasyTiger vector indexing tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// If set, write OTEL trace spans as json to the output file.
    #[arg(long)]
    trace: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Perform SPANN index operations.
    Spann(SpannArgs),
    /// Perform Vamana/DiskANN index operations.
    Vamana(VamanaArgs),
    /// Compute the top k neighbors for a set of queries against a set of document vectors.
    ComputeNeighbors(ComputeNeighborsArgs),
    /// Quantization related utilities.
    Quantization(QuantizationArgs),
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();

    let (file_layer, _appender_guard) = if let Some(trace_file) = cli.trace {
        let file_appender = tracing_appender::rolling::never(".", trace_file);
        let (non_blocking_appender, guard) = tracing_appender::non_blocking(file_appender);
        (
            Some(
                fmt::layer()
                    .json()
                    .with_span_events(FmtSpan::CLOSE)
                    .with_writer(non_blocking_appender)
                    .with_filter(filter_fn(|metadata| metadata.is_span())),
            ),
            Some(guard),
        )
    } else {
        (None, None)
    };

    let stdout_layer = fmt::layer().compact().with_writer(io::stdout);

    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    tracing_subscriber::registry()
        .with(stdout_layer.with_filter(env_filter))
        .with(file_layer)
        .init();

    match cli.command {
        Commands::Spann(args) => spann_command(args),
        Commands::Vamana(args) => vamana_command(args),
        Commands::ComputeNeighbors(args) => compute_neighbors(args),
        Commands::Quantization(args) => quantization(args),
    }
}
