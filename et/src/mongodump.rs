mod extract_archive_vectors;
mod parser;
mod read;

use std::io;

use clap::{Args, Subcommand};
use extract_archive_vectors::{extract_archive_vectors, ExtractArchiveVectorsArgs};
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
    ExtractArchiveVectors(ExtractArchiveVectorsArgs),
}

pub fn mongodump_command(args: MongodumpArgs) -> io::Result<()> {
    match args.command {
        MongodumpCommands::Read(args) => read_archive(args),
        MongodumpCommands::ExtractArchiveVectors(args) => extract_archive_vectors(args),
    }
}
