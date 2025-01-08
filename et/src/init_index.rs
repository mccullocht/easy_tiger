use std::{io, num::NonZero, sync::Arc};

use clap::Args;
use easy_tiger::{
    graph::{GraphConfig, GraphSearchParams},
    quantization::VectorQuantizer,
    scoring::VectorSimilarity,
    wt::TableGraphVectorIndex,
};
use wt_mdb::Connection;

use crate::drop_index::drop_index;

#[derive(Args)]
pub struct InitIndexArgs {
    /// Number of dimensions in input vectors.
    #[arg(short, long)]
    dimensions: NonZero<usize>,
    /// Similarity function to use for vector scoring.
    #[arg(short, long, value_enum)]
    similarity: VectorSimilarity,

    /// Maximum number of edges for any vertex.
    #[arg(long, default_value = "64")]
    max_edges: NonZero<usize>,
    /// Number of edges to search for when indexing a vertex.
    ///
    /// Larger values make indexing more expensive but may also produce a larger, more
    /// saturated graph that has higher recall.
    #[arg(long, default_value = "256")]
    edge_candidates: NonZero<usize>,
    /// Number of edge candidates to rerank. Defaults to edge_candidates.
    ///
    /// When > 0 re-rank candidate edges using the highest fidelity vectors available.
    /// The candidate list is then truncated to this size, so this may effectively reduce
    /// the value of edge_candidates.
    #[arg(long)]
    rerank_edges: Option<usize>,

    /// If true, drop the named index if it exists and re-initialize.
    ///
    /// If false and the index already exists this command will fail.
    #[arg(long)]
    drop_if_exists: bool,
}

pub fn init_index(
    connection: Arc<Connection>,
    index_name: &str,
    args: InitIndexArgs,
) -> io::Result<()> {
    if args.drop_if_exists {
        drop_index(connection.clone(), index_name)?;
    } else if !TableGraphVectorIndex::from_db(&connection, index_name)
        .is_err_and(|e| e.kind() == io::ErrorKind::NotFound)
    {
        println!("Index {} already exists!", index_name);
        return Ok(());
    }

    TableGraphVectorIndex::init_index(
        &connection,
        None,
        GraphConfig {
            dimensions: args.dimensions,
            similarity: args.similarity,
            quantizer: VectorQuantizer::default(), // XXX
            max_edges: args.max_edges,
            index_search_params: GraphSearchParams {
                beam_width: args.edge_candidates,
                num_rerank: args.rerank_edges.unwrap_or(args.edge_candidates.get()),
            },
        },
        index_name,
    )
    .map(|_| ())
}
