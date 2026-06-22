use std::{collections::{HashSet, VecDeque}, io, sync::Arc};

use clap::Args;
use easy_tiger::vamana::{
    Graph,
    wt::{CursorGraph, TableGraphVectorIndex, ENTRY_POINT_KEY},
};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro128PlusPlus;
use wt_mdb::Connection;

#[derive(Args)]
pub struct CheckReachabilityArgs {
    /// Number of unreachable vertex IDs to include in the sample output.
    #[arg(long, default_value_t = 10)]
    sample_size: usize,
}

pub fn check_reachability(
    connection: Arc<Connection>,
    index_name: &str,
    args: CheckReachabilityArgs,
) -> io::Result<()> {
    let index = TableGraphVectorIndex::from_db(&connection, index_name)?;
    let txn = connection.begin_transaction(None)?;

    let mut graph = CursorGraph::new(txn.open_record_cursor(index.graph_table_name())?);

    let entry_point = match graph.entry_point() {
        None => {
            println!("Graph is empty (no entry point).");
            return Ok(());
        }
        Some(Err(e)) => return Err(e.into()),
        Some(Ok(ep)) => ep,
    };

    // BFS from the entry point to collect all reachable vertex IDs.
    let mut reachable: HashSet<i64> = HashSet::new();
    let mut queue: VecDeque<i64> = VecDeque::new();
    reachable.insert(entry_point);
    queue.push_back(entry_point);

    while let Some(vertex_id) = queue.pop_front() {
        let edges = match graph.edges(vertex_id) {
            None => continue,
            Some(Err(e)) => return Err(e.into()),
            Some(Ok(edges)) => edges,
        };
        for neighbor in edges {
            if reachable.insert(neighbor) {
                queue.push_back(neighbor);
            }
        }
    }

    // Scan all records to count total vertices and collect a reservoir sample of unreachable ones.
    // ENTRY_POINT_KEY (-1) holds entry point metadata, not a vertex; skip it.
    let scan = txn.open_record_cursor(index.graph_table_name())?;
    let mut total: usize = 0;
    let mut unreachable_count: usize = 0;
    let mut sample: Vec<i64> = Vec::with_capacity(args.sample_size);
    let mut rng = Xoshiro128PlusPlus::seed_from_u64(0);

    for result in scan {
        let (key, _) = result?;
        if key == ENTRY_POINT_KEY {
            continue;
        }
        total += 1;
        if !reachable.contains(&key) {
            unreachable_count += 1;
            if sample.len() < args.sample_size {
                sample.push(key);
            } else if args.sample_size > 0 {
                let j = rng.random_range(0..unreachable_count);
                if j < args.sample_size {
                    sample[j] = key;
                }
            }
        }
    }

    if unreachable_count == 0 {
        println!("All {total} vertices are reachable from the entry point.");
    } else {
        println!(
            "{unreachable_count} of {total} vertices ({:.2}%) are not reachable from the entry point.",
            unreachable_count as f64 * 100.0 / total as f64,
        );
        if !sample.is_empty() {
            sample.sort_unstable();
            println!(
                "Sample ({} of {unreachable_count}): {:?}",
                sample.len(),
                sample,
            );
        }
    }

    Ok(())
}
