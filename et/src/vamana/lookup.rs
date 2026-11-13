use std::{io, sync::Arc};

use clap::Args;
use easy_tiger::{
    Neighbor,
    vamana::{
        Graph, GraphVectorStore,
        wt::{CursorGraph, CursorVectorStore, TableGraphVectorIndex},
    },
};
use wt_mdb::Connection;

#[derive(Args)]
pub struct LookupArgs {
    /// Id of the vertex to lookup.
    #[arg(short, long)]
    id: i64,

    /// If true, print the undirected graph edges.
    #[arg(short, long, default_value = "true")]
    edges: bool,

    /// If true, score each edge against the source vertex using the highest fidelity vectors
    /// available and print edges sorted in ascending order by distance.
    #[arg(short, long, default_value_t = false)]
    score: bool,

    /// If true, print vector status and number of bytes.
    #[arg(short, long, default_value_t = false)]
    vector: bool,

    /// If true, print if the vector value.
    #[arg(short, long, default_value_t = false)]
    print_full_vector: bool,
}

pub fn lookup(connection: Arc<Connection>, index_name: &str, args: LookupArgs) -> io::Result<()> {
    let index = TableGraphVectorIndex::from_db(&connection, index_name)?;
    let txn = connection.begin_transaction(None)?;

    if args.edges {
        let mut graph = CursorGraph::new(txn.open_record_cursor(index.graph_table_name())?);
        match graph.edges(args.id) {
            None => {
                println!("Vertex not found!");
            }
            Some(result) => match result {
                Err(e) => println!("Vertex error {e}"),
                Ok(edges) => {
                    let edges = edges.collect::<Vec<_>>();
                    if args.score {
                        print_scored_edges(&txn, &index, args.id, &edges)?;
                    } else {
                        println!("edges: {edges:?}");
                    }
                }
            },
        };
    }

    if args.vector {
        let mut vectors = CursorVectorStore::new(
            txn.open_record_cursor(index.nav_table().name())?,
            index.config().similarity,
            index.nav_table().format(),
            index.nav_table().centroid().map(std::sync::Arc::from),
        );
        match vectors.get(args.id) {
            None => {
                println!("Vector not found!");
            }
            Some(Err(e)) => {
                println!("Vector error {e}");
            }
            Some(Ok(v)) => {
                if args.print_full_vector {
                    let coder = index.nav_table().format().coder();
                    let vector = coder.decode(v);
                    println!("vector: {vector:?}");
                } else {
                    println!("vector bytes: {}", v.len());
                }
            }
        }
    }

    Ok(())
}

/// Score `edges` against `src_id` using the highest fidelity vectors available for `index`
/// (the rerank vectors if present, otherwise the navigational vectors) and print them sorted
/// in ascending order by distance.
fn print_scored_edges(
    txn: &wt_mdb::Transaction,
    index: &TableGraphVectorIndex,
    src_id: i64,
    edges: &[i64],
) -> io::Result<()> {
    let table = index.high_fidelity_table();
    let mut vectors = CursorVectorStore::new(
        txn.open_record_cursor(table.name())?,
        index.config().similarity,
        table.format(),
        table.centroid().map(Arc::from),
    );

    let src = match vectors.get(src_id) {
        None => {
            println!("Source vector not found!");
            return Ok(());
        }
        Some(Err(e)) => {
            println!("Source vector error {e}");
            return Ok(());
        }
        Some(Ok(v)) => v.to_vec(),
    };

    let distance_fn = table.new_distance_function();
    let mut scored = Vec::with_capacity(edges.len());
    for &edge in edges {
        match vectors.get(edge) {
            None => println!("edge {edge}: vector not found"),
            Some(Err(e)) => println!("edge {edge}: vector error {e}"),
            Some(Ok(v)) => scored.push(Neighbor::new(edge, distance_fn.distance(&src, v))),
        }
    }
    scored.sort_unstable();

    println!("edges:");
    for neighbor in &scored {
        println!(
            "  {:>12}  distance={:.6}",
            neighbor.vertex(),
            neighbor.distance()
        );
    }

    Ok(())
}
