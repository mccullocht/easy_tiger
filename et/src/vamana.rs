mod bulk_load;
mod drop_index;
mod init_index;
mod insert;
mod lookup;
mod search;

use std::{io, num::NonZero, sync::Arc};

use clap::{Args, Subcommand};

use bulk_load::{BulkLoadArgs, bulk_load};
use drop_index::drop_index;
use easy_tiger::vamana::EdgePruningConfig;
use init_index::{InitIndexArgs, init_index};
use insert::{InsertArgs, insert};
use lookup::{LookupArgs, lookup};
use search::{SearchArgs, search};

use crate::wt_args::WiredTigerArgs;

#[derive(Args)]
pub struct VamanaArgs {
    #[command(flatten)]
    wt: WiredTigerArgs,

    #[command(subcommand)]
    command: Command,
}

#[derive(Args)]
pub struct EdgePruningArgs {
    /// Maximum number of edges for any vertex.
    #[arg(short, long, default_value = "32")]
    max_edges: NonZero<usize>,
    /// Maximum alpha value used to prune edges. Large values keep more edges.
    ///
    /// Must be >= 1.0.
    #[arg(long, default_value_t = 1.2)]
    max_alpha: f64,
    /// Alpha value scaling factor.
    ///
    /// This value is multiplied by the current alpha value (starting at 1.0) until max_alpha is
    /// exceeded. Lower values will trigger fewer iterations. Must be >= 1.0.
    #[arg(long, default_value_t = 1.2)]
    alpha_scale: f64,
}

impl From<EdgePruningArgs> for EdgePruningConfig {
    fn from(value: EdgePruningArgs) -> Self {
        Self {
            max_edges: value.max_edges,
            max_alpha: value.max_alpha,
            alpha_scale: value.alpha_scale,
        }
    }
}

#[derive(Subcommand)]
pub enum Command {
    /// Bulk load a set of vectors into an index.
    /// Requires that the index be uninitialized.
    BulkLoad(BulkLoadArgs),
    /// Search for a list of vectors and time the operation.
    Search(SearchArgs),
    /// Initialize a new (empty) index.
    InitIndex(InitIndexArgs),
    /// Drop an index.
    DropIndex,
    /// Lookup the contents of a single vertex.
    Lookup(LookupArgs),
    /// Insert a vectors from a file into the index.
    Insert(InsertArgs),
}

pub fn vamana_command(args: VamanaArgs) -> io::Result<()> {
    let cmd_connection = args.wt.open_connection()?;
    let connection = Arc::clone(&cmd_connection);
    let index_name = args.wt.index_name();
    match args.command {
        Command::BulkLoad(args) => bulk_load(connection, index_name, args),
        Command::Search(args) => search(connection, index_name, args),
        Command::InitIndex(args) => init_index(connection, index_name, args),
        Command::DropIndex => drop_index(connection, index_name),
        Command::Lookup(args) => lookup(connection, index_name, args),
        Command::Insert(args) => insert(connection, index_name, args),
    }?;
    cmd_connection.checkpoint()?;
    Ok(())
}
