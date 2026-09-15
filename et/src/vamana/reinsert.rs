use std::{io, sync::Arc};

use clap::Args;
use easy_tiger::vamana::{
    Graph, GraphVectorIndex, GraphVectorStore,
    mutate::upsert_vector,
    wt::{TableGraphVectorIndex, TransactionGraphVectorIndex},
};
use wt_mdb::{Connection, Error};

#[derive(Args)]
pub struct ReinsertArgs {
    /// Comma separated vertex ids to reinsert. Each vertex is deleted and re-inserted using its
    /// own stored (highest fidelity available) vector, re-running edge selection against the graph
    /// as it exists today. Later ids in the same invocation see the effects of earlier ones.
    #[arg(short, long = "id", required = true, value_delimiter = ',')]
    ids: Vec<i64>,
    /// Commit the new edges. Without this flag the whole batch is computed and printed, then
    /// rolled back, so nothing is written.
    #[arg(long)]
    commit: bool,
}

pub fn reinsert(
    connection: Arc<Connection>,
    index_name: &str,
    args: ReinsertArgs,
) -> io::Result<()> {
    let index = Arc::new(TableGraphVectorIndex::from_db(&connection, index_name)?);
    let txn =
        TransactionGraphVectorIndex::new(Arc::clone(&index), connection.begin_transaction(None)?);

    for id in &args.ids {
        let old_edges: Vec<i64> = match txn.graph()?.edges(*id).transpose()? {
            Some(edges) => edges.collect(),
            None => {
                println!("{id}: not found in graph, skipping");
                continue;
            }
        };

        let vector: Vec<f32> = {
            let mut store = txn.high_fidelity_vectors()?;
            let coder = store.new_coder();
            let encoded = store
                .get(*id)
                .unwrap_or_else(|| Err(Error::not_found_error()))?
                .to_vec();
            coder.decode(&encoded)
        };

        upsert_vector(*id, &vector, &txn)?;

        let new_edges: Vec<i64> = txn
            .graph()?
            .edges(*id)
            .unwrap_or_else(|| Err(Error::not_found_error()))?
            .collect();
        let gained: Vec<i64> = new_edges
            .iter()
            .copied()
            .filter(|e| !old_edges.contains(e))
            .collect();
        let lost: Vec<i64> = old_edges
            .iter()
            .copied()
            .filter(|e| !new_edges.contains(e))
            .collect();

        println!(
            "{id}: {} -> {} edges (+{} -{})",
            old_edges.len(),
            new_edges.len(),
            gained.len(),
            lost.len()
        );
        println!("  old:    {old_edges:?}");
        println!("  new:    {new_edges:?}");
        println!("  gained: {gained:?}");
        println!("  lost:   {lost:?}");
    }

    if args.commit {
        txn.commit(None)?;
        println!("committed.");
    } else {
        txn.rollback(None)?;
        println!("dry run: no changes committed.");
    }

    Ok(())
}
