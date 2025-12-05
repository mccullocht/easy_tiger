mod read;

use std::io;

use clap::{Args, Subcommand};
use read::{read_archive, ReadArchiveArgs};

#[derive(Args)]
pub struct MongodumpArgs {
    #[command(subcommand)]
    command: MongodumpCommands,
}

#[derive(Subcommand)]
enum MongodumpCommands {
    /// Read and display documents from a mongodump archive.
    Read(ReadArchiveArgs),
}

pub fn mongodump_command(args: MongodumpArgs) -> io::Result<()> {
    match args.command {
        MongodumpCommands::Read(args) => read_archive(args),
    }
}

