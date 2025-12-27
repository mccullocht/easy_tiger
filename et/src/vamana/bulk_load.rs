use std::{fs::File, io, num::NonZero, path::PathBuf, sync::Arc};

use clap::Args;
use easy_tiger::{
    input::{DerefVectorStore, VectorStore},
    vamana::{
        bulk::{BulkLoadBuilder, Options},
        wt::TableGraphVectorIndex,
        GraphConfig, GraphSearchParams, PatienceParams,
    },
};
use vectors::{F32VectorCoding, VectorSimilarity};
use wt_mdb::{Connection, Result, Session};
use wt_sys::{WT_STAT_CONN_CACHE_BYTES_READ, WT_STAT_CONN_CURSOR_SEARCH, WT_STAT_CONN_READ_IO};

use crate::{
    ui::progress_bar,
    vamana::{drop_index::drop_index, EdgePruningArgs},
};

#[derive(Args)]
pub struct BulkLoadArgs {
    /// Path to the input vectors to bulk ingest.
    #[arg(short, long)]
    f32_vectors: PathBuf,
    /// Number of dimensions in input vectors.
    #[arg(short, long)]
    dimensions: NonZero<usize>,
    /// Similarity function to use for vector scoring.
    #[arg(long)]
    similarity: VectorSimilarity,
    /// Vector coding to use for navigational vectors.
    #[arg(long)]
    nav_format: F32VectorCoding,
    /// Vector coding to use for rerank vectors.
    #[arg(long)]
    rerank_format: Option<F32VectorCoding>,

    /// If true, load all quantized vectors into a trivial memory store for bulk loading.
    /// This can be significantly faster than reading these values from WiredTiger.
    #[arg(long, default_value_t = true)]
    memory_quantized_vectors: bool,

    /// Number of edges to search for when indexing a vertex.
    ///
    /// Larger values make indexing more expensive but may also produce a larger, more
    /// saturated graph that has higher recall.
    #[arg(short, long, default_value = "128")]
    edge_candidates: NonZero<usize>,
    /// Number of edge candidates to rerank.
    ///
    /// Defaults to edge_candidates if --rerank-format is set.
    ///
    /// When > 0 re-rank candidate edges using the highest fidelity vectors available.
    /// The candidate list is then truncated to this size, so this may effectively reduce
    /// the value of edge_candidates.
    #[arg(short, long)]
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

    /// If true, drop any WiredTiger tables with the same name before bulk upload.
    #[arg(long, default_value = "false")]
    drop_tables: bool,

    /// Limit the number of input vectors. Useful for testing.
    #[arg(short, long)]
    limit: Option<usize>,
}

pub fn bulk_load(
    connection: Arc<Connection>,
    index_name: &str,
    args: BulkLoadArgs,
) -> io::Result<()> {
    let f32_vectors = DerefVectorStore::new(
        unsafe { memmap2::Mmap::map(&File::open(args.f32_vectors)?)? },
        args.dimensions,
    )?;

    let config = GraphConfig {
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
        },
        limit,
    );

    for phase in builder.phases() {
        let progress = progress_bar(builder.len(), phase.display_name());
        builder.execute_phase(phase, |n| progress.inc(n))?;
    }
    println!("{:?}", builder.graph_stats().unwrap());

    let stats = wired_tiger_stats(&connection.open_session()?)?;
    println!(
        "cache hit rate {:.2}% ({} reads, {} lookups); {} bytes read into cache",
        (stats.search_calls - stats.read_ios) as f64 * 100.0 / stats.search_calls as f64,
        stats.read_ios,
        stats.search_calls,
        stats.read_bytes,
    );

    Ok(())
}

struct WiredTigerStats {
    search_calls: i64,
    read_ios: i64,
    read_bytes: i64,
}

/// Count lookup calls and read IOs. This can be used to estimate cache hit rate.
fn wired_tiger_stats(session: &Session) -> Result<WiredTigerStats> {
    let mut stat_cursor = session.new_stats_cursor(wt_mdb::Statistics::Fast, None)?;
    Ok(WiredTigerStats {
        search_calls: stat_cursor
            .seek_exact(WT_STAT_CONN_CURSOR_SEARCH as i32)
            .expect("WT_STAT_CONN_CURSOR_SEARCH")?
            .value,
        read_ios: stat_cursor
            .seek_exact(WT_STAT_CONN_READ_IO as i32)
            .expect("WT_STAT_CONN_READ_IO")?
            .value,
        read_bytes: stat_cursor
            .seek_exact(WT_STAT_CONN_CACHE_BYTES_READ as i32)
            .expect("WT_STATE_CONN_CACHE_BYTES_READ")?
            .value,
    })
}
