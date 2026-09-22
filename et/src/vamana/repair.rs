use std::{
    collections::{BTreeSet, HashMap},
    io,
    sync::Arc,
};

use clap::Args;
use easy_tiger::vamana::{
    Graph, GraphVectorIndex,
    mutate::repair,
    wt::{TableGraphVectorIndex, TransactionGraphVectorIndex},
};
use wt_mdb::Connection;

#[derive(Args)]
pub struct RepairArgs {
    /// Comma-separated list of vertex ids to repair.
    ///
    /// Example: --ids 1,2,3
    #[arg(long, value_delimiter = ',', required = true)]
    ids: Vec<i64>,

    /// If true, commit the changes made by repair. Otherwise the transaction is rolled back
    /// and the index is left unmodified.
    #[arg(long, default_value_t = false)]
    commit: bool,
}

pub fn repair_command(
    connection: Arc<Connection>,
    index_name: &str,
    args: RepairArgs,
) -> io::Result<()> {
    let index = Arc::new(TableGraphVectorIndex::from_db(&connection, index_name)?);
    let txn_index = TransactionGraphVectorIndex::new(
        Arc::clone(&index),
        connection.begin_transaction(None)?,
    );

    let before = read_edges(&txn_index, &args.ids)?;

    for &id in &args.ids {
        repair(id, &txn_index)?;
    }

    let after = read_edges(&txn_index, &args.ids)?;

    for &id in &args.ids {
        print_diff(id, before.get(&id).unwrap(), after.get(&id).unwrap());
    }

    if args.commit {
        txn_index.commit(None)?;
        println!("Committed changes.");
    } else {
        txn_index.rollback(None)?;
        println!("Not committed (pass --commit to persist changes).");
    }

    Ok(())
}

/// Read the edge set for each of `ids`, returning `None` for any vertex that does not exist.
fn read_edges(
    index: &TransactionGraphVectorIndex,
    ids: &[i64],
) -> io::Result<HashMap<i64, Option<BTreeSet<i64>>>> {
    let mut graph = index.graph()?;
    let mut result = HashMap::with_capacity(ids.len());
    for &id in ids {
        let edges = match graph.edges(id) {
            None => None,
            Some(Ok(edges)) => Some(edges.collect::<BTreeSet<_>>()),
            Some(Err(e)) => return Err(e.into()),
        };
        result.insert(id, edges);
    }
    Ok(result)
}

fn print_diff(id: i64, before: &Option<BTreeSet<i64>>, after: &Option<BTreeSet<i64>>) {
    println!("vertex {id}:");
    match (before, after) {
        (None, None) => println!("  not found (before or after)"),
        (Some(_), None) => println!("  vertex no longer found after repair"),
        (None, Some(a)) => println!("  vertex now found after repair; edges: {a:?}"),
        (Some(b), Some(a)) => {
            let added = a.difference(b).copied().collect::<Vec<_>>();
            let removed = b.difference(a).copied().collect::<Vec<_>>();
            if added.is_empty() && removed.is_empty() {
                println!("  unchanged ({} edges)", b.len());
            } else {
                if !removed.is_empty() {
                    println!("  - {removed:?}");
                }
                if !added.is_empty() {
                    println!("  + {added:?}");
                }
            }
        }
    }
}
