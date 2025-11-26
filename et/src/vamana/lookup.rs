use std::{io, sync::Arc};

use clap::Args;
use easy_tiger::vamana::{
    wt::{CursorGraph, CursorVectorStore, TableGraphVectorIndex},
    Graph, GraphVectorStore,
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

    /// If true, print vector status and number of bytes.
    #[arg(short, long, default_value_t = false)]
    vector: bool,

    /// If true, print if the vector value.
    #[arg(short, long, default_value_t = false)]
    print_full_vector: bool,
}

pub fn lookup(connection: Arc<Connection>, index_name: &str, args: LookupArgs) -> io::Result<()> {
    let index = TableGraphVectorIndex::from_db(&connection, index_name)?;
    let session = connection.open_session()?;

    if args.edges {
        let mut graph = CursorGraph::new(session.get_record_cursor(index.graph_table_name())?);
        match graph.edges(args.id) {
            None => {
                println!("Vertex not found!");
            }
            Some(result) => match result {
                Err(e) => println!("Vertex error {e}"),
                Ok(edges) => {
                    println!("edges: {:?}", edges.collect::<Vec<_>>());
                }
            },
        };
    }

    if args.vector {
        let mut vectors = CursorVectorStore::new(
            session.get_record_cursor(index.nav_table().name())?,
            index.config().similarity,
            index.nav_table().format(),
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
                    let coder = index
                        .nav_table()
                        .format()
                        .new_coder(index.config().similarity);
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
