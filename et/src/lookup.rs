use std::{io, num::NonZero, sync::Arc};

use clap::Args;
use easy_tiger::{
    graph::{Graph, GraphNode},
    search::GraphSearchParams,
    wt::{GraphMetadata, WiredTigerGraph, WiredTigerIndexParams},
};
use wt_mdb::Connection;

#[derive(Args)]
pub struct LookupArgs {
    /// Number of vector dimensions for index and query vectors.
    // TODO: this should appear in the table, derive it from that!
    #[arg(short, long)]
    dimensions: NonZero<usize>,

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
    index_params: WiredTigerIndexParams,
    args: LookupArgs,
) -> io::Result<()> {
    let session = connection.open_session().map_err(io::Error::from)?;
    // TODO: this should be read from the graph table.
    let metadata = GraphMetadata {
        dimensions: args.dimensions,
        max_edges: NonZero::new(1).unwrap(), // unused here.
        index_search_params: GraphSearchParams {
            //unused here.
            beam_width: NonZero::new(1).unwrap(),
            num_rerank: 0,
        },
    };
    let mut graph = WiredTigerGraph::new(
        metadata,
        session.open_record_cursor(&index_params.graph_table_name)?,
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
