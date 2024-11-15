use std::{io, sync::Arc};

use clap::Args;
use easy_tiger::{
    graph::{Graph, GraphNode},
    wt::{WiredTigerGraph, WiredTigerGraphVectorIndex},
};
use wt_mdb::Connection;

#[derive(Args)]
pub struct LookupArgs {
    /// Id of the vertex to lookup.
    #[arg(short, long)]
    id: i64,

    /// If true, print the contents of the vector.
    #[arg(short, long, default_value = "false")]
    vector: bool,
    /// If true, print the undirected graph edges.
    #[arg(short, long, default_value = "true")]
    edges: bool,
}

pub fn lookup(
    connection: Arc<Connection>,
    index: WiredTigerGraphVectorIndex,
    args: LookupArgs,
) -> io::Result<()> {
    let mut session = connection.open_session()?;
    let mut graph = WiredTigerGraph::new(
        *index.metadata(),
        session.open_record_cursor(index.graph_table_name())?,
    );
    match graph.get(args.id) {
        None => {
            println!("Not found!");
        }
        Some(result) => match result {
            Err(e) => println!("Error: {}", e),
            Ok(v) => {
                if args.vector {
                    println!("{:?}", v.vector());
                }
                if args.edges {
                    println!("{:?}", v.edges().collect::<Vec<_>>());
                }
            }
        },
    };
    Ok(())
}
