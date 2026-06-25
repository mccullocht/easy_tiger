mod drop_index;
mod init_index;
mod insert_vectors;
mod search;

use std::{io, sync::Arc};

use clap::{Args, Subcommand};

use crate::wt_args::WiredTigerArgs;
use drop_index::drop_index;
use init_index::{InitIndexArgs, init_index};
use insert_vectors::{InsertVectorsArgs, insert_vectors};
use search::{SearchArgs, search};

#[derive(Args)]
pub struct FlatArgs {
    #[command(flatten)]
    wt: WiredTigerArgs,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Initialize a new flat index.
    InitIndex(InitIndexArgs),
    /// Remove an existing flat index.
    DropIndex,
    /// Insert vectors into an existing flat index.
    InsertVectors(InsertVectorsArgs),
    /// Search a flat index exhaustively.
    Search(SearchArgs),
}

pub fn flat_command(args: FlatArgs) -> io::Result<()> {
    let cmd_connection = args.wt.open_connection()?;
    let connection = Arc::clone(&cmd_connection);
    let index_name = args.wt.index_name();
    match args.command {
        Command::InitIndex(args) => init_index(connection, index_name, args),
        Command::DropIndex => drop_index(connection, index_name),
        Command::InsertVectors(args) => insert_vectors(connection, index_name, args),
        Command::Search(args) => search(connection, index_name, args),
    }?;
    cmd_connection.checkpoint()?;
    Ok(())
}
