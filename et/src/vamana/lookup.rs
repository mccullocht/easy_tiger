use std::{io, sync::Arc};

use clap::Args;
use easy_tiger::vamana::{
    wt::{CursorGraph, TableGraphVectorIndex},
    Graph,
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
}

pub fn lookup(connection: Arc<Connection>, index_name: &str, args: LookupArgs) -> io::Result<()> {
    let index = TableGraphVectorIndex::from_db(&connection, index_name)?;
    let session = connection.open_session()?;
    let mut graph = CursorGraph::new(session.get_record_cursor(index.graph_table_name())?);
    match graph.edges(args.id) {
        None => {
            println!("Not found!");
        }
        Some(result) => match result {
            Err(e) => println!("Error: {e}"),
            Ok(edges) => {
                if args.edges {
                    println!("{:?}", edges.collect::<Vec<_>>());
                }
            }
        },
    };
    Ok(())
}
