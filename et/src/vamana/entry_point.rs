use std::{io, num::NonZero, sync::Arc};

use clap::Args;
use easy_tiger::vamana::{
    Graph, GraphVectorIndex, GraphVectorStore,
    search::{GraphSearcher, Options as GraphSearchOptions},
    wt::{TableGraphVectorIndex, TransactionGraphVectorIndex},
};
use wt_mdb::{Connection, Error};

#[derive(Args)]
pub struct EntryPointArgs {
    /// Number of candidates in the search beam width used to look for a point closer to the mean
    /// vector than the current entry point.
    ///
    /// Defaults to the beam width configured for the index.
    #[arg(short, long)]
    beam_width: Option<NonZero<usize>>,

    /// If a point closer to the mean vector than the current entry point is found, set it as the
    /// new entry point and commit the change.
    #[arg(long, default_value_t = false)]
    commit: bool,
}

pub fn entry_point(
    connection: Arc<Connection>,
    index_name: &str,
    args: EntryPointArgs,
) -> io::Result<()> {
    let index = Arc::new(TableGraphVectorIndex::from_db(&connection, index_name)?);
    let reader = TransactionGraphVectorIndex::new(
        Arc::clone(&index),
        connection.begin_transaction(None)?,
    );

    let hi_table = index.high_fidelity_table();
    let coder = hi_table.new_coder();
    let dimensions = index.config().dimensions.get();

    // Decode every high fidelity vector and accumulate a mean in f64 for precision.
    let mut sum = vec![0.0f64; dimensions];
    let mut count: u64 = 0;
    for result in reader.transaction().open_record_cursor(hi_table.name())? {
        let (_, bytes) = result?;
        let vector = coder.decode(&bytes);
        for (s, v) in sum.iter_mut().zip(vector.iter()) {
            *s += *v as f64;
        }
        count += 1;
    }

    if count == 0 {
        println!("Index has no vectors; nothing to do.");
        return Ok(());
    }

    let mean: Vec<f32> = sum.iter().map(|s| (*s / count as f64) as f32).collect();

    let mut graph = reader.graph()?;
    let ep_id = match graph.entry_point().transpose()? {
        Some(ep) => ep,
        None => {
            println!("Graph is empty (no entry point).");
            return Ok(());
        }
    };

    let distance_fn = index.config().similarity.distance_f32();

    let mut vectors = reader.high_fidelity_vectors()?;
    let ep_bytes = vectors
        .get(ep_id)
        .unwrap_or_else(|| Err(Error::not_found_error()))?;
    let ep_vector = coder.decode(ep_bytes);
    let ep_distance = distance_fn.distance_f32(&mean, &ep_vector);
    drop(vectors);

    println!("Averaged {count} high fidelity vectors.");
    println!("Current entry point {ep_id}: distance to mean = {ep_distance:.6}");

    let mut search_params = index.config().index_search_params;
    if let Some(beam_width) = args.beam_width {
        search_params.beam_width = beam_width;
    }
    let mut searcher = GraphSearcher::new(search_params);
    let (results, _) =
        searcher.search_with_options(&mean, GraphSearchOptions::default(), &reader)?;

    let Some(top) = results.first() else {
        println!("Search found no candidates.");
        return Ok(());
    };
    let top_vertex = top.vertex();

    let mut vectors = reader.high_fidelity_vectors()?;
    let top_bytes = vectors
        .get(top_vertex)
        .unwrap_or_else(|| Err(Error::not_found_error()))?;
    let top_vector = coder.decode(top_bytes);
    let top_distance = distance_fn.distance_f32(&mean, &top_vector);
    drop(vectors);

    println!("Closest point found by search {top_vertex}: distance to mean = {top_distance:.6}");

    if top_vertex == ep_id {
        println!("Entry point is already the closest point found.");
        return Ok(());
    }

    if top_distance >= ep_distance {
        println!("No closer point found; current entry point is retained.");
        return Ok(());
    }

    println!(
        "Found a closer entry point candidate {top_vertex} ({top_distance:.6} < {ep_distance:.6})."
    );

    if args.commit {
        graph.set_entry_point(top_vertex)?;
        drop(graph);
        reader.commit(None)?;
        println!("Committed new entry point: {top_vertex}");
    } else {
        println!("Re-run with --commit to update the entry point.");
    }

    Ok(())
}
