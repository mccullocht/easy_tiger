use std::{fs::File, io, num::NonZero, path::PathBuf, sync::Arc};

use clap::Args;
use easy_tiger::{
    bulk::BulkLoadBuilder,
    graph::{GraphConfig, GraphSearchParams},
    input::{DerefVectorStore, VectorStore},
    scoring::VectorSimilarity,
    wt::TableGraphVectorIndex,
};
use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};
use wt_mdb::{options::DropOptionsBuilder, Connection};

use crate::drop_index;

#[derive(Args)]
pub struct BulkLoadArgs {
    /// Path to the input vectors to bulk ingest.
    #[arg(short, long)]
    f32_vectors: PathBuf,
    /// Number of dimensions in input vectors.
    #[arg(short, long)]
    dimensions: NonZero<usize>,
    /// Similarity function to use for vector scoring.
    #[arg(short, long, value_enum)]
    similarity: VectorSimilarity,
    /// If true, load all quantized vectors into a trivial memory store for bulk loading.
    /// This can be significantly faster than reading these values from WiredTiger.
    #[arg(long)]
    memory_quantized_vectors: bool,

    /// Maximum number of edges for any vertex.
    #[arg(short, long, default_value = "64")]
    max_edges: NonZero<usize>,
    /// Number of edges to search for when indexing a vertex.
    ///
    /// Larger values make indexing more expensive but may also produce a larger, more
    /// saturated graph that has higher recall.
    #[arg(short, long, default_value = "256")]
    edge_candidates: NonZero<usize>,
    /// Number of edge candidates to rerank. Defaults to edge_candidates.
    ///
    /// When > 0 re-rank candidate edges using the highest fidelity vectors available.
    /// The candidate list is then truncated to this size, so this may effectively reduce
    /// the value of edge_candidates.
    #[arg(short, long)]
    rerank_edges: Option<usize>,

    /// If true, drop any WiredTiger tables with the same name before bulk upload.
    #[arg(long, default_value = "false")]
    drop_tables: bool,

    /// Limit the number of input vectors. Useful for testing.
    #[arg(short, long)]
    limit: Option<usize>,
}

pub fn bulk_load(
    connection: Arc<Connection>,
    args: BulkLoadArgs,
    index_name: &str,
) -> io::Result<()> {
    let f32_vectors = DerefVectorStore::new(
        unsafe { memmap2::Mmap::map(&File::open(args.f32_vectors)?)? },
        args.dimensions,
    )?;

    let config = GraphConfig {
        dimensions: args.dimensions,
        similarity: args.similarity,
        max_edges: args.max_edges,
        index_search_params: GraphSearchParams {
            beam_width: args.edge_candidates,
            num_rerank: args
                .rerank_edges
                .unwrap_or_else(|| args.edge_candidates.get()),
        },
    };
    if args.drop_tables {
        drop_index(connection.clone(), index_name)?;
    }
    let index = TableGraphVectorIndex::from_init(config, index_name)?;

    let num_vectors = f32_vectors.len();
    let limit = args.limit.unwrap_or(num_vectors);
    let mut builder = BulkLoadBuilder::new(
        connection,
        index,
        f32_vectors,
        args.memory_quantized_vectors,
        limit,
    );

    {
        let progress = progress_bar(limit, "load nav vectors");
        builder.load_nav_vectors(|| progress.inc(1))?;
    }
    {
        let progress = progress_bar(limit, "build graph");
        builder.insert_all(|| progress.inc(1))?;
    }
    {
        let progress = progress_bar(limit, "cleanup graph");
        builder.cleanup(|| progress.inc(1))?;
    }
    let stats = {
        let progress = progress_bar(limit, "load graph");
        builder.load_graph(|| progress.inc(1))?
    };
    println!("{:?}", stats);

    Ok(())
}

fn progress_bar(len: usize, message: &'static str) -> ProgressBar {
    ProgressBar::new(len as u64)
        .with_style(
            ProgressStyle::default_bar()
                .template(
                    "{msg} {wide_bar} {pos}/{len} ETA: {eta_precise} Elapsed: {elapsed_precise}",
                )
                .unwrap(),
        )
        .with_message(message)
        .with_finish(ProgressFinish::AndLeave)
}
