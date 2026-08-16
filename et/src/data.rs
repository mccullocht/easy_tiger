mod convert;

use std::io;

use clap::{Args, Subcommand};

use convert::{ConvertArgs, convert};

#[derive(Args)]
pub struct DataArgs {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Convert f32 vector data to f16 and write in BigANN format.
    Convert(ConvertArgs),
}

pub fn data_command(args: DataArgs) -> io::Result<()> {
    match args.command {
        Command::Convert(args) => convert(args),
    }
}
