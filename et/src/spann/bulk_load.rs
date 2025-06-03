use std::{cell::RefCell, fs::File, io, num::NonZero, path::PathBuf, sync::Arc};

use clap::Args;
use easy_tiger::{
    bulk::{self, BulkLoadBuilder},
    distance::VectorSimilarity,
    graph::{GraphConfig, GraphLayout, GraphSearchParams},
    input::{DerefVectorStore, SubsetViewVectorStore, VectorStore},
    kmeans::{iterative_balanced_kmeans, Params},
    quantization::VectorQuantizer,
    spann::{IndexConfig, SessionIndexWriter, TableIndex},
};
use rand::thread_rng;
use rayon::prelude::*;
use thread_local::ThreadLocal;
use wt_mdb::{options::DropOptionsBuilder, Connection};

use crate::ui::{progress_bar, progress_spinner};

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
    #[arg(long, value_enum, default_value = "split")]
    layout: GraphLayout,

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

    /// Minimum number of vectors that should map to each head centroid.
    #[arg(long, default_value_t = 64)]
    head_min_centroid_len: usize,
    /// Maximum number of vectors that should map to each head centroid.
    /// This should be at least 2x --head-min-centroid-len.
    #[arg(long, default_value_t = 192)]
    head_max_centroid_len: usize,

    /// Maximum number of replica centroids to assign each vector to.
    #[arg(long)]
    replica_count: NonZero<usize>,

    /// Quantizer to use for vectors written to centroid posting lists.
    #[arg(long)]
    posting_quantizer: VectorQuantizer,

    /// Number of edge candidates when searching head table for centroid ids during insertion.
    /// This should be at least as many as --replica-count
    #[arg(long)]
    head_edge_candidates: NonZero<usize>,

    /// Number of vectors to re-rank when searching head table for centroid ids during insertion.
    /// If unset, re-ranks all edge candidates.
    #[arg(long)]
    head_rerank_edges: Option<usize>,

    /// Limit the number of input vectors. Useful for testing.
    #[arg(short, long)]
    limit: Option<usize>,

    /// If true, drop any WiredTiger tables with the same name before bulk upload.
    #[arg(long, default_value_t = false)]
    drop_tables: bool,
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
    let spann_config = IndexConfig {
        replica_count: args.replica_count.get(),
        head_search_params: GraphSearchParams {
            beam_width: args.head_edge_candidates,
            num_rerank: args
                .head_rerank_edges
                .unwrap_or(args.head_edge_candidates.get()),
        },
        quantizer: args.posting_quantizer,
    };
    let index = Arc::new(TableIndex::init_index(
        &connection,
        index_name,
        head_config,
        spann_config,
    )?);

    let limit = args.limit.unwrap_or(f32_vectors.len());
    let index_vectors = SubsetViewVectorStore::new(&f32_vectors, (0..limit).into_iter().collect());
    let centroids = {
        let progress = progress_spinner("clustering head");
        let (centroids, _) = iterative_balanced_kmeans(
            &index_vectors,
            args.head_min_centroid_len..=args.head_max_centroid_len,
            32,
            1000, // batch size
            &Params {
                iters: 100,
                epsilon: 0.01,
                ..Params::default()
            },
            &mut thread_rng(),
            |x| progress.inc(x),
        );
        centroids
    };
    let centroids_len = centroids.len();

    {
        let mut head_loader = BulkLoadBuilder::new(
            connection.clone(),
            Arc::unwrap_or_clone(index.head_config().clone()),
            centroids,
            bulk::Options {
                memory_quantized_vectors: false,
                wt_vector_store: true,
                cluster_ordered_insert: false,
            },
            centroids_len,
        );
        for phase in head_loader.phases() {
            let progress = progress_bar(centroids_len, format!("head {}", phase.display_name()));
            head_loader.execute_phase(phase, |x| progress.inc(x))?;
        }
    }

    // TODO: consider writing a dedicated bulk loader. The bulk loader should at least do the raw
    // vectors to avoid writing to the checkpoint.
    let inserted = {
        let tl_writer: ThreadLocal<RefCell<SessionIndexWriter>> = ThreadLocal::new();
        let progress = progress_bar(limit, "postings index");
        (0..index_vectors.len())
            .into_par_iter()
            .map(|i| {
                let mut writer = tl_writer
                    .get_or_try(|| {
                        let session = connection.open_session()?;
                        Ok::<_, wt_mdb::Error>(RefCell::new(SessionIndexWriter::new(
                            index.clone(),
                            session,
                        )))
                    })?
                    .borrow_mut();
                writer.session().begin_transaction(None)?;
                let inserted = writer.upsert(i as i64, &f32_vectors[i])?;
                writer.session().commit_transaction(None)?;
                progress.inc(1);
                Ok::<_, wt_mdb::Error>(inserted)
            })
            .try_reduce(|| 0usize, |a, b| Ok(a + b))?
    };

    println!(
        "Head contains {} centroids ({:4.2}%)",
        centroids_len,
        (centroids_len as f64 / index_vectors.len() as f64) * 100.0
    );
    println!(
        "Inserted {} posting entries (avg {:4.2})",
        inserted,
        inserted as f64 / limit as f64
    );

    Ok(())
}
