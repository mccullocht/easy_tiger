use std::{io, num::NonZero, sync::Arc};

use clap::Args;
use easy_tiger::vamana::{
    wt::TableGraphVectorIndex, GraphConfig, GraphSearchParams, PatienceParams,
};
use vectors::{F32VectorCoding, VectorSimilarity};
use wt_mdb::Connection;

use crate::vamana::EdgePruningArgs;

use super::drop_index::drop_index;

#[derive(Args)]
pub struct InitIndexArgs {
    /// Number of dimensions in input vectors.
    #[arg(short, long)]
    dimensions: NonZero<usize>,
    /// Similarity function to use for vector scoring.
    #[arg(short, long, value_enum)]
    similarity: VectorSimilarity,
    /// Vector coding to use for navigational vectors.
    #[arg(long, value_enum)]
    nav_format: F32VectorCoding,
    /// Vector coding to use for rerank vectors.
    #[arg(long, value_enum, default_value = "f32")]
    rerank_format: Option<F32VectorCoding>,

    /// Number of edges to search for when indexing a vertex.
    ///
    /// Larger values make indexing more expensive but may also produce a larger, more
    /// saturated graph that has higher recall.
    #[arg(long, default_value = "256")]
    edge_candidates: NonZero<usize>,
    /// Number of edge candidates to rerank.
    ///
    /// Defaults to edge_candidates if --rerank-format is set.
    ///
    /// When > 0 re-rank candidate edges using the highest fidelity vectors available.
    /// The candidate list is then truncated to this size, so this may effectively reduce
    /// the value of edge_candidates.
    #[arg(long)]
    rerank_edges: Option<usize>,
    /// Patience threshold to use during edge candidate generation.
    #[arg(long, default_value_t = 0.995)]
    patience_saturation_threshold: f64,
    /// Patience count to use during edge candidate generation.
    ///
    /// If left unset, patience is not used to early terminate edge candidate generation search.
    #[arg(long)]
    patience_saturation_count: Option<usize>,

    #[command(flatten)]
    pruning: EdgePruningArgs,

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
        println!("Index {index_name} already exists!");
        return Ok(());
    }

    TableGraphVectorIndex::init_index(
        &connection,
        GraphConfig {
            dimensions: args.dimensions,
            similarity: args.similarity,
            nav_format: args.nav_format,
            rerank_format: args.rerank_format,
            pruning: args.pruning.into(),
            index_search_params: GraphSearchParams {
                beam_width: args.edge_candidates,
                num_rerank: args
                    .rerank_format
                    .map(|_| args.rerank_edges.unwrap_or(args.edge_candidates.get()))
                    .unwrap_or(0),
                patience: args.patience_saturation_count.map(|c| PatienceParams {
                    saturation_threshold: args.patience_saturation_threshold,
                    patience_count: c,
                }),
            },
        },
        index_name,
    )
    .map(|_| ())
}
