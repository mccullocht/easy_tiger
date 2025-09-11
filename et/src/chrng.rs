mod analyze_graph;
mod search;

use std::{fs::File, io, num::NonZero, path::PathBuf, sync::Arc};

use clap::{Args, Subcommand};
use easy_tiger::{
    chrng::{create_clusters, rewrite_graph_table, rewrite_table},
    input::{DerefVectorStore, VectorStore},
    vamana::{
        bulk::{BulkLoadBuilder, Options},
        wt::TableGraphVectorIndex,
        GraphConfig, GraphSearchParams,
    },
};
use memmap2::Mmap;
use vectors::{F32VectorCoding, VectorSimilarity};
use wt_mdb::Connection;

use crate::{
    chrng::{
        analyze_graph::{analyze_graph, AnalyzeGraphArgs},
        search::{search, SearchArgs},
    },
    ui::{progress_bar, progress_spinner},
    vamana::{drop_index::drop_index, EdgePruningArgs},
    wt_args::WiredTigerArgs,
    wt_stats::WiredTigerConnectionStats,
};

#[derive(Args)]
pub struct ChrngArgs {
    #[command(flatten)]
    wt: WiredTigerArgs,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Bulk load a set of vectors into an index.
    /// Requires that the index be uninitialized.
    BulkLoad(BulkLoadArgs),
    /// Analyze the output tail graph.
    AnalyzeGraph(AnalyzeGraphArgs),
    /// Search a CHRNG index.
    Search(SearchArgs),
}

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

    /// Maximum number of vectors in each cluster.
    #[arg(long)]
    max_cluster_len: NonZero<usize>,

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

    #[command(flatten)]
    pruning: EdgePruningArgs,

    /// If true, drop any WiredTiger tables with the same name before bulk upload.
    #[arg(long, default_value = "false")]
    drop_tables: bool,

    /// Limit the number of input vectors. Useful for testing.
    #[arg(short, long)]
    limit: Option<usize>,
}

pub fn chrng_command(args: ChrngArgs) -> io::Result<()> {
    let connection = args.wt.open_connection()?;
    let session = connection.open_session()?;
    let index_name = args.wt.index_name();
    match args.command {
        Command::BulkLoad(args) => bulk_load(connection, index_name, args),
        Command::AnalyzeGraph(args) => analyze_graph(connection, index_name, args),
        Command::Search(args) => search(connection, index_name, args),
    }?;
    session.checkpoint()?;
    Ok(())
}

fn bulk_load(connection: Arc<Connection>, index_name: &str, args: BulkLoadArgs) -> io::Result<()> {
    let (vectors, clusters, mapping) = {
        // XXX we need to be able to limit the clustering. nothing downstream needs to apply the
        // limit as of yet.
        let dataset = DerefVectorStore::new(
            unsafe { Mmap::map(&File::open(args.f32_vectors)?)? },
            args.dimensions,
        )?;
        let spinner = progress_spinner("clustering");
        create_clusters(
            &dataset,
            args.similarity,
            args.max_cluster_len.get(),
            &|i| spinner.inc(i),
        )?
    };

    let head_index_name = format!("{index_name}.head");
    let tail_index_name = format!("{index_name}.tail");

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
        },
    };
    if args.drop_tables {
        drop_index(connection.clone(), &head_index_name)?;
        drop_index(connection.clone(), &tail_index_name)?;
    }
    let head_index = TableGraphVectorIndex::from_init(config, &head_index_name)?;
    let tail_index = TableGraphVectorIndex::from_init(config, &tail_index_name)?;

    {
        let len = clusters.len();
        let mut builder = BulkLoadBuilder::new(
            connection.clone(),
            head_index,
            clusters,
            Options {
                memory_quantized_vectors: args.memory_quantized_vectors,
            },
            len,
        );

        for phase in builder.phases() {
            let progress = progress_bar(builder.len(), format!("head {}", phase.display_name()));
            builder.execute_phase(phase, |n| progress.inc(n))?;
        }
        println!("{:?}", builder.graph_stats().unwrap());
    }

    let len = args.limit.unwrap_or(vectors.len());
    {
        let mut builder = BulkLoadBuilder::new(
            connection.clone(),
            tail_index.clone(),
            vectors,
            Options {
                memory_quantized_vectors: args.memory_quantized_vectors,
            },
            len,
        );

        for phase in builder.phases() {
            let progress = progress_bar(builder.len(), format!("tail {}", phase.display_name()));
            builder.execute_phase(phase, |n| progress.inc(n))?;
        }
        println!("{:?}", builder.graph_stats().unwrap());
    }

    let stats = WiredTigerConnectionStats::try_from(&connection)?;
    println!(
        "cache hit rate {:.2}% ({} reads, {} lookups); {} bytes read into cache",
        (stats.search_calls - stats.read_ios) as f64 * 100.0 / stats.search_calls as f64,
        stats.read_ios,
        stats.search_calls,
        stats.read_bytes,
    );

    {
        let progress = progress_bar(len, "rewrite graph");
        rewrite_graph_table(&connection, &tail_index.graph_table_name(), &mapping, |i| {
            progress.inc(i)
        })?;
    }

    {
        let progress = progress_bar(len, "rewrite nav");
        rewrite_table(&connection, &tail_index.nav_table().name(), &mapping, |i| {
            progress.inc(i)
        })?;
    }

    Ok(())
}
