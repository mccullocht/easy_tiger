mod convert_f16;

use std::io;

use clap::{Args, Subcommand};

use convert_f16::{ConvertF16Args, convert_f16};

#[derive(Args)]
pub struct DataArgs {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Convert f32 vector data to f16 and write in BigANN format.
    ConvertF16(ConvertF16Args),
}

pub fn data_command(args: DataArgs) -> io::Result<()> {
    match args.command {
        Command::ConvertF16(args) => convert_f16(args),
    }
}
