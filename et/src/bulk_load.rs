use std::{fs::File, io, num::NonZero, path::PathBuf, sync::Arc};

use clap::Args;
use easy_tiger::{
    bulk::{BulkLoadBuilder, Options},
    distance::VectorSimilarity,
    graph::{GraphConfig, GraphLayout, GraphSearchParams},
    input::{DerefVectorStore, VectorStore},
    quantization::VectorQuantizer,
    wt::TableGraphVectorIndex,
};
use wt_mdb::{Connection, Result, Session};
use wt_sys::{WT_STAT_CONN_CURSOR_SEARCH, WT_STAT_CONN_READ_IO};

use crate::{drop_index, ui::progress_bar};

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
    /// Quantizer to use for navigational vectors.
    ///
    /// This will also dictate the quantized scoring function used.
    #[arg(short, long, value_enum)]
    quantizer: VectorQuantizer,

    /// Physical layout used for graph.
    ///
    /// `split` puts raw vectors, nav vectors, and graph edges each in separate tables. If results
    /// are being re-ranked this will require additional reads to complete.
    ///
    /// `raw_vector_in_graph` places raw vectors and graph edges in the same table. When a vertex
    /// is visited the raw vector is read and saved for re-scoring. This minimizes the number of
    /// reads performed and is likely better for indices with less traffic.
    #[arg(long, value_enum, default_value = "raw_vector_in_graph")]
    layout: GraphLayout,
    /// If true, load all quantized vectors into a trivial memory store for bulk loading.
    /// This can be significantly faster than reading these values from WiredTiger.
    #[arg(long, default_value_t = false)]
    memory_quantized_vectors: bool,
    /// If true, load all the raw vectors into WiredTiger for bulk loading.
    /// This can be faster for dot similarity as the vectors are normalized just once.
    /// It also moves caching policy to WT rather than allowing the OS to manage it.
    #[arg(long, default_value_t = false)]
    wiredtiger_vector_store: bool,

    /// Maximum number of edges for any vertex.
    #[arg(short, long, default_value = "32")]
    max_edges: NonZero<usize>,
    /// Number of edges to search for when indexing a vertex.
    ///
    /// Larger values make indexing more expensive but may also produce a larger, more
    /// saturated graph that has higher recall.
    #[arg(short, long, default_value = "128")]
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
        quantizer: args.quantizer,
        layout: args.layout,
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
        connection.clone(),
        index,
        f32_vectors,
        Options {
            memory_quantized_vectors: args.memory_quantized_vectors,
            wt_vector_store: args.wiredtiger_vector_store,
        },
        limit,
    );

    for phase in builder.phases() {
        let progress = progress_bar(builder.len(), Some(phase.display_name()));
        builder.execute_phase(phase, || progress.inc(1))?;
    }
    println!("{:?}", builder.graph_stats().unwrap());

    if args.wiredtiger_vector_store {
        let (search_calls, read_io) = cache_hit_stats(&connection.open_session()?)?;
        println!(
            "cache hit rate {:.2}% ({} reads, {} lookups)",
            (search_calls - read_io) as f64 * 100.0 / search_calls as f64,
            read_io,
            search_calls,
        );
    }

    Ok(())
}

/// Count lookup calls and read IOs. This can be used to estimate cache hit rate.
fn cache_hit_stats(session: &Session) -> Result<(i64, i64)> {
    let mut stat_cursor = session.new_stats_cursor(wt_mdb::options::Statistics::Fast, None)?;
    let search_calls = stat_cursor
        .seek_exact(WT_STAT_CONN_CURSOR_SEARCH)
        .expect("WT_STAT_CONN_CURSOR_SEARCH")?;
    let read_ios = stat_cursor
        .seek_exact(WT_STAT_CONN_READ_IO)
        .expect("WT_STAT_CONN_READ_IO")?;
    Ok((search_calls, read_ios))
}
