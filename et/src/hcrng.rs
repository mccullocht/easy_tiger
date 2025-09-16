use std::{collections::HashSet, fs::File, io, num::NonZero, path::PathBuf, sync::Arc};

use clap::{Args, Subcommand};
use easy_tiger::{
    bulk::{BulkLoadBuilder, Options},
    graph::{GraphConfig, GraphLayout, GraphSearchParams},
    hcrng::create_clusters,
    input::{DerefVectorStore, VectorStore},
    vectors::{F32VectorCoding, VectorSimilarity},
    wt::{Leb128EdgeIterator, TableGraphVectorIndex},
};
use memmap2::Mmap;
use wt_mdb::Connection;

use crate::{
    ui::{progress_bar, progress_spinner},
    vamana::drop_index::drop_index,
    wt_args::WiredTigerArgs,
    wt_stats::WiredTigerConnectionStats,
};

#[derive(Args)]
pub struct HcrngArgs {
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

    /// Maximum number of edges for any vertex.
    #[arg(short, long, default_value = "32")]
    max_edges: NonZero<usize>,
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

    /// If true, drop any WiredTiger tables with the same name before bulk upload.
    #[arg(long, default_value = "false")]
    drop_tables: bool,

    /// Limit the number of input vectors. Useful for testing.
    #[arg(short, long)]
    limit: Option<usize>,
}

pub fn hcrng_command(args: HcrngArgs) -> io::Result<()> {
    let connection = args.wt.open_connection()?;
    let session = connection.open_session()?;
    let index_name = args.wt.index_name();
    match args.command {
        Command::BulkLoad(args) => bulk_load(connection, index_name, args),
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
        layout: GraphLayout::Split,
        max_edges: args.max_edges,
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

    {
        let len = args.limit.unwrap_or(vectors.len());
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

    // XXX this is a bit silly.
    // XXX if i set the max cluster size to 1024 ~52% of links stay in cluster and there are links out to ~1.5 other clusters on avg.
    // XXX at 256 it is ~45% and ~1.3 other clusters.
    // XXX at 128 it is ~38% and ~1.3 other clusters.
    // XXX on search the number of nav vectors scored drops by 12% at 256
    // XXX 13% at 128
    // XXX 8% at 1024
    // XXX this begs some questions: am I saving due to locality or improved graph shape or both?
    // XXX it has to be both -- the drop is 13% on scoring but ~25% on latency and cpu time
    // XXX but if I just _insert_ in the bp order do i still get the 13%??? it's worth asking
    // because it's less invasive and potentially a good optimization step. you still need more to
    // maintain the advantage.
    // XXX it's also worth counting how many clusters we visit during a search, i wonder if we could
    // score more vectors but still do better due to access patterns.
    let session = connection.open_session()?;
    let cursor = session.open_record_cursor(tail_index.graph_table_name())?;
    let mut edges_in_cluster = 0usize;
    let mut edges_out_cluster = 0usize;
    let mut edges_out_clusters = 0;
    let mut cluster_id_delta = 0;
    for r in cursor {
        let (ord, raw_edges) = r?;
        let cluster_id = mapping.identify_cluster_id(ord as usize);

        let mut out_clusters = HashSet::new();
        for e in Leb128EdgeIterator::new(&raw_edges) {
            let e_cluster_id = mapping.identify_cluster_id(e as usize);
            cluster_id_delta += cluster_id.abs_diff(e_cluster_id);
            if e_cluster_id == cluster_id {
                edges_in_cluster += 1;
            } else {
                edges_out_cluster += 1;
                out_clusters.insert(e_cluster_id);
            }
        }
        edges_out_clusters += out_clusters.len();
    }

    println!("edges within cluster: {edges_in_cluster} edges outside cluster: {edges_out_cluster} count of out clusters {edges_out_clusters} cluster id delta {cluster_id_delta}");

    Ok(())
}
