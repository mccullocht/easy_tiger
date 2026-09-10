mod centroid_stats;
mod delete_sim;
mod drop_index;
mod export_head;
mod init_index;
mod insert_vectors;
mod reassign;
mod rebalance;
mod search;

use std::{io, sync::Arc};

use clap::{Args, Subcommand};

use crate::wt_args::WiredTigerArgs;
use centroid_stats::{CentroidStatsArgs, centroid_stats};
use delete_sim::{DeleteSimArgs, delete_sim};
use drop_index::drop_index;
use export_head::{ExportHeadArgs, export_head};
use init_index::{InitIndexArgs, init_index};
use insert_vectors::{InsertVectorsArgs, insert_vectors};
use reassign::{ReassignArgs, reassign};
use rebalance::{RebalanceArgs, rebalance};
use search::{SearchArgs, search};

#[derive(Args)]
pub struct SpannArgs {
    #[command(flatten)]
    wt: WiredTigerArgs,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Initialize a SPANN-ish index with a single dummy centroid.
    InitIndex(InitIndexArgs),
    /// Insert vectors into an existing SPANN-ish index.
    InsertVectors(InsertVectorsArgs),
    /// Search a SPANN-ish index.
    Search(SearchArgs),
    /// Print centroid assignment statistics.
    CentroidStats(CentroidStatsArgs),
    /// Export centroid vectors from the head index as little-endian f32 values.
    ExportHead(ExportHeadArgs),
    /// Rebalance the SPANN index.
    Rebalance(RebalanceArgs),
    /// Recompute assignments for a single centroid's posting vectors by searching the head index
    /// with each vector's rerank vector; report where they would be assigned, or with --commit
    /// move them there.
    Reassign(ReassignArgs),
    /// Simulate deletes: for every rerank vector, search the head and read postings until the
    /// record is located, reporting the depth and how many records could not be found.
    DeleteSim(DeleteSimArgs),
    /// Remove an existing index.
    DropIndex,
}

pub fn spann_command(args: SpannArgs) -> io::Result<()> {
    let cmd_connection = args.wt.open_connection()?;
    let connection = Arc::clone(&cmd_connection);
    let index_name = args.wt.index_name();
    match args.command {
        Command::InitIndex(args) => init_index(connection, index_name, args),
        Command::InsertVectors(args) => insert_vectors(connection, index_name, args),
        Command::Search(args) => search(connection, index_name, args),
        Command::CentroidStats(args) => centroid_stats(connection, index_name, args),
        Command::ExportHead(args) => export_head(connection, index_name, args),
        Command::Rebalance(args) => rebalance(connection, index_name, args),
        Command::Reassign(args) => reassign(connection, index_name, args),
        Command::DeleteSim(args) => delete_sim(connection, index_name, args),
        Command::DropIndex => drop_index(connection, index_name),
    }?;
    cmd_connection.checkpoint()?;
    Ok(())
}
