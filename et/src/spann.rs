use std::{io, sync::Arc};

use clap::{Args, Subcommand};
use wt_mdb::Connection;

use crate::spann_load::{spann_load, SpannLoadArgs};

#[derive(Args)]
pub struct SpannArgs {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Bulk load a set of vectors into an empty index.
    BulkLoad(SpannLoadArgs),
}

pub fn spann_command(
    connection: Arc<Connection>,
    index_name: &str,
    args: SpannArgs,
) -> io::Result<()> {
    match args.command {
        Command::BulkLoad(args) => spann_load(connection, index_name, args),
    }
}
