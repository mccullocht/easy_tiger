mod bulk_load;
mod check_reachability;
mod drop_index;
mod init_index;
mod insert;
mod lookup;
mod search;

use std::{io, num::NonZero, sync::Arc};

use clap::{Args, Subcommand};

use bulk_load::{BulkLoadArgs, bulk_load};
use check_reachability::{CheckReachabilityArgs, check_reachability};
use drop_index::drop_index;
use easy_tiger::vamana::{EdgePruningConfig, EdgeType};
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
    /// Alpha value used to prune edges. Large values keep more edges.
    ///
    /// Must be >= 1.0.
    #[arg(long, default_value_t = 1.2)]
    alpha: f64,
    /// Saturate the edge set up to max_edges after pruning during insert.
    ///
    /// When set, the best pruned candidate edges are added back to fill the edge set up to
    /// max_edges, producing a denser graph that may yield higher recall.
    #[arg(long, default_value_t = false)]
    saturate_graph: bool,
}

#[derive(Copy, Clone, Debug, Default, clap::ValueEnum)]
pub enum EdgeTypeArg {
    #[default]
    Undirected,
    Directed,
}

impl From<EdgeTypeArg> for EdgeType {
    fn from(v: EdgeTypeArg) -> Self {
        match v {
            EdgeTypeArg::Undirected => EdgeType::Undirected,
            EdgeTypeArg::Directed => EdgeType::Directed,
        }
    }
}

impl From<EdgePruningArgs> for EdgePruningConfig {
    fn from(value: EdgePruningArgs) -> Self {
        Self {
            max_edges: value.max_edges,
            alpha: value.alpha,
            saturate_graph: value.saturate_graph,
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
    /// Check whether every vertex in the graph is reachable from the entry point.
    CheckReachability(CheckReachabilityArgs),
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
        Command::CheckReachability(args) => check_reachability(connection, index_name, args),
    }?;
    cmd_connection.checkpoint()?;
    Ok(())
}
