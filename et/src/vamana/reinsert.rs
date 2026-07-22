use std::{io, sync::Arc};

use clap::Args;
use easy_tiger::vamana::{
    mutate::upsert_vector,
    wt::{TableGraphVectorIndex, TransactionGraphVectorIndex},
    GraphVectorIndex, GraphVectorStore,
};
use wt_mdb::{Connection, Error};

#[derive(Args)]
pub struct ReinsertArgs {
    /// Id of the vertex to reinsert.
    #[arg(short, long)]
    id: i64,
}

/// Re-perform insertion for `args.id`, rebuilding its edges.
///
/// The vertex's stored (high fidelity) vector is decoded and re-upserted at the same id, which
/// deletes the existing vertex -- repairing its neighbors -- and then re-inserts it, selecting fresh
/// edges as if it were a new vector.
pub fn reinsert(
    connection: Arc<Connection>,
    index_name: &str,
    args: ReinsertArgs,
) -> io::Result<()> {
    let index = Arc::new(TableGraphVectorIndex::from_db(&connection, index_name)?);
    let txn_index =
        TransactionGraphVectorIndex::new(Arc::clone(&index), connection.begin_transaction(None)?);

    // Recover the highest fidelity vector stored for the vertex and decode it to f32.
    let vector = {
        let mut vectors = txn_index.high_fidelity_vectors()?;
        let coder = vectors.new_coder();
        match vectors.get(args.id) {
            None => {
                println!("Vertex {} not found!", args.id);
                return Ok(());
            }
            Some(Err(e)) => return Err(e.into()),
            Some(Ok(encoded)) => coder.decode(encoded),
        }
    };

    upsert_vector(args.id, &vector, &txn_index).map_err(Error::from)?;
    txn_index.commit(None)?;

    println!("Reinserted vertex {}", args.id);
    Ok(())
}
