use std::{io, num::NonZero, sync::Arc};

use clap::Args;
use easy_tiger::{
    spann::{IndexConfig, ReplicaSelectionAlgorithm, TableIndex},
    vamana::{
        mutate::insert_vector, wt::SessionGraphVectorIndex, GraphConfig, GraphSearchParams,
        PatienceParams,
    },
};
use vectors::{F32VectorCoding, VectorSimilarity};
use wt_mdb::{session::DropOptionsBuilder, Connection};

use crate::vamana::EdgePruningArgs;

#[derive(Args)]
pub struct InitIndexArgs {
    /// Number of dimensions in input vectors.
    #[arg(short, long)]
    dimensions: NonZero<usize>,
    /// Similarity function to use for vector scoring.
    #[arg(short, long, value_enum)]
    similarity: VectorSimilarity,
    /// Encoding used for navigational vectors in the head index.
    #[arg(long, value_enum)]
    head_nav_format: F32VectorCoding,
    /// Encoding used for rerank vectors in the head index.
    #[arg(long)]
    head_rerank_format: Option<F32VectorCoding>,
    /// Number of edges to search for when indexing a vertex.
    ///
    /// Larger values make indexing more expensive but may also produce a larger, more
    /// saturated graph that has higher recall.
    #[arg(short, long, default_value_t = NonZero::new(128).unwrap())]
    edge_candidates: NonZero<usize>,
    /// Number of edge candidates to rerank. Defaults to edge_candidates.
    ///
    /// When > 0 re-rank candidate edges using the highest fidelity vectors available.
    /// The candidate list is then truncated to this size, so this may effectively reduce
    /// the value of edge_candidates.
    #[arg(short, long)]
    rerank_edges: Option<usize>,

    #[command(flatten)]
    pruning: EdgePruningArgs,

    /// Minimum number of vectors that should map to each head centroid.
    #[arg(long, default_value_t = 192)]
    head_min_centroid_len: usize,
    /// Maximum number of vectors that should map to each head centroid.
    /// This should be at least 2x --head-min-centroid-len.
    #[arg(long, default_value_t = 512)]
    head_max_centroid_len: usize,

    /// Number of edge candidates when searching head table for centroid ids during insertion.
    /// This should be at least as many as --replica-count
    #[arg(long)]
    head_edge_candidates: Option<usize>,
    /// Number of vectors to re-rank when searching head table for centroid ids during insertion.
    /// If unset, re-ranks all edge candidates.
    #[arg(long)]
    head_rerank_edges: Option<usize>,
    /// Patience saturation threshold.
    ///
    /// During each search round fewer than this fraction of candidates must change. If this
    /// threshold is exceeded --patience-saturation-count consecutive times then the search will be
    /// terminated.
    #[arg(long, default_value_t = 0.995)]
    head_patience_saturation_threshold: f64,
    /// Patience saturation count.
    ///
    /// If unset, patience early termination will not be used.
    #[arg(long)]
    head_patience_saturation_count: Option<usize>,

    /// Maximum number of replica centroids to assign each vector to.
    #[arg(long, default_value_t = NonZero::new(1).unwrap())]
    replica_count: NonZero<usize>,

    /// Replica selection algorithm to use.
    #[arg(long, default_value_t = ReplicaSelectionAlgorithm::SOAR)]
    replica_selection: ReplicaSelectionAlgorithm,

    /// Quantizer to use for vectors written to centroid posting lists.
    #[arg(long)]
    posting_coder: F32VectorCoding,

    /// Format to use for a rerank table. May be omitted.
    #[arg(long)]
    rerank_format: Option<F32VectorCoding>,

    /// If true, drop any WiredTiger tables with the same name before bulk upload.
    #[arg(long, default_value_t = false)]
    drop_tables: bool,
}

pub fn init_index(
    connection: Arc<Connection>,
    index_name: &str,
    args: InitIndexArgs,
) -> io::Result<()> {
    if args.drop_tables {
        TableIndex::drop_tables(
            &connection.open_session()?,
            index_name,
            &Some(DropOptionsBuilder::default().set_force().into()),
        )?;
    }

    let head_config = GraphConfig {
        dimensions: args.dimensions,
        similarity: args.similarity,
        nav_format: args.head_nav_format,
        rerank_format: args.head_rerank_format,
        pruning: args.pruning.into(),
        index_search_params: GraphSearchParams {
            beam_width: args.edge_candidates,
            num_rerank: args
                .head_rerank_format
                .map(|_| {
                    args.rerank_edges
                        .unwrap_or_else(|| args.edge_candidates.get())
                })
                .unwrap_or(0),
            patience: args.head_patience_saturation_count.map(|c| PatienceParams {
                saturation_threshold: args.head_patience_saturation_threshold,
                patience_count: c,
            }),
        },
    };
    let beam_width = args
        .head_edge_candidates
        .map(|v| NonZero::new(v).unwrap())
        .unwrap_or(args.edge_candidates);
    let spann_config = IndexConfig {
        replica_count: args.replica_count.get(),
        replica_selection: args.replica_selection,
        min_centroid_len: args.head_min_centroid_len,
        max_centroid_len: args.head_max_centroid_len,
        head_search_params: GraphSearchParams {
            beam_width,
            num_rerank: args
                .head_rerank_format
                .map(|_| args.head_rerank_edges.unwrap_or(beam_width.get()))
                .unwrap_or(0),
            patience: None,
        },
        posting_coder: args.posting_coder,
        rerank_format: args.rerank_format,
    };
    let index = Arc::new(TableIndex::init_index(
        &connection,
        index_name,
        head_config,
        spann_config,
    )?);

    let head_index =
        SessionGraphVectorIndex::new(Arc::clone(index.head_config()), connection.open_session()?);
    // Insert a dummy vector into the head index to provide somewhere for the first insert to go.
    // This vector is not a good vector (particularly for angular distance metrics) but it will
    // disappear as soon as we need to split the initial centroid.
    insert_vector(&vec![0.0f32; args.dimensions.get()], &head_index)?;

    Ok(())
}
