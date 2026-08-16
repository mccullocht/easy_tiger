mod check;
mod convert_f16;
mod convert_neighbors;

use std::io;

use clap::{Args, Subcommand};

use check::{CheckArgs, check};
use convert_f16::{ConvertF16Args, convert_f16};
use convert_neighbors::{ConvertNeighborsArgs, convert_neighbors};

#[derive(Args)]
pub struct DataArgs {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Convert f32 vector data to f16 and write in BigANN format.
    ConvertF16(ConvertF16Args),
    /// Add a BigANN header to a headerless neighbors (recall ground truth) file.
    ConvertNeighbors(ConvertNeighborsArgs),
    /// Read a BigANN header and print count, dimensionality, and entry width.
    Check(CheckArgs),
}

pub fn data_command(args: DataArgs) -> io::Result<()> {
    match args.command {
        Command::ConvertF16(args) => convert_f16(args),
        Command::ConvertNeighbors(args) => convert_neighbors(args),
        Command::Check(args) => check(args),
    }
}
