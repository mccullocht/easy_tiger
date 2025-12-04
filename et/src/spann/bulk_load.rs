use std::{fs::File, io, num::NonZero, path::PathBuf, sync::Arc};

use clap::Args;
use easy_tiger::{
    input::{DerefVectorStore, SubsetViewVectorStore, VectorStore},
    kmeans::{iterative_balanced_kmeans, Params},
    spann::{
        bulk::{
            assign_to_centroids, bulk_load_centroids, bulk_load_postings, bulk_load_raw_vectors,
        },
        IndexConfig, ReplicaSelectionAlgorithm, TableIndex,
    },
    vamana::{
        bulk::{self, BulkLoadBuilder},
        GraphConfig, GraphSearchParams,
    },
};
use rand_xoshiro::{rand_core::SeedableRng, Xoshiro128PlusPlus};
use vectors::{F32VectorCoding, VectorSimilarity};
use wt_mdb::{options::DropOptionsBuilder, Connection};

use crate::{
    ui::{progress_bar, progress_spinner},
    vamana::EdgePruningArgs,
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
    head_edge_candidates: NonZero<usize>,

    /// Number of vectors to re-rank when searching head table for centroid ids during insertion.
    /// If unset, re-ranks all edge candidates.
    #[arg(long)]
    head_rerank_edges: Option<usize>,

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

    /// Limit the number of input vectors. Useful for testing.
    #[arg(short, long)]
    limit: Option<usize>,

    /// Random seed used for clustering computations.
    /// Use a fixed value for repeatability.
    #[arg(long, default_value_t = 0x7774_7370414E4E)]
    seed: u64,

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
        },
    };
    let spann_config = IndexConfig {
        replica_count: args.replica_count.get(),
        replica_selection: args.replica_selection,
        min_centroid_len: args.head_min_centroid_len,
        max_centroid_len: args.head_max_centroid_len,
        head_search_params: GraphSearchParams {
            beam_width: args.head_edge_candidates,
            num_rerank: args
                .head_rerank_format
                .map(|_| {
                    args.head_rerank_edges
                        .unwrap_or(args.head_edge_candidates.get())
                })
                .unwrap_or(0),
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
    let mut rng = Xoshiro128PlusPlus::seed_from_u64(args.seed);

    let limit = args.limit.unwrap_or(f32_vectors.len());
    let index_vectors = SubsetViewVectorStore::new(&f32_vectors, (0..limit).collect());
    let centroids = {
        let progress = progress_spinner("clustering head");
        let (centroids, _) = iterative_balanced_kmeans(
            &index_vectors,
            args.head_min_centroid_len..=args.head_max_centroid_len,
            32,
            8192, // batch size
            &Params {
                iters: 100,
                epsilon: 0.0001,
                ..Params::default()
            },
            &mut rng,
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
            },
            centroids_len,
        );
        for phase in head_loader.phases() {
            let progress = progress_bar(centroids_len, format!("head {}", phase.display_name()));
            head_loader.execute_phase(phase, |x| progress.inc(x))?;
        }
    }

    let session = connection.open_session()?;
    if index.config().rerank_format.is_some() {
        let progress = progress_bar(limit, "tail load raw vectors");
        bulk_load_raw_vectors(index.as_ref(), &session, &f32_vectors, limit, |i| {
            progress.inc(i)
        })?
    }

    let centroid_assignments = {
        let progress = progress_bar(limit, "tail assign centroids");
        assign_to_centroids(index.as_ref(), &connection, &f32_vectors, limit, |i| {
            progress.inc(i)
        })?
    };
    {
        let progress = progress_bar(limit, "tail load centroids");
        bulk_load_centroids(index.as_ref(), &session, &centroid_assignments, |i| {
            progress.inc(i)
        })?;
    }

    {
        let posting_count = centroid_assignments.iter().map(|c| c.len()).sum::<usize>();
        let progress = progress_bar(posting_count, "tail load postings");
        bulk_load_postings(
            index.as_ref(),
            &session,
            &centroid_assignments,
            &f32_vectors,
            |i| progress.inc(i),
        )?;
    }

    Ok(())
}
