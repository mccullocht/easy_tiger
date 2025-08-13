use std::{fs::File, io, num::NonZero, path::PathBuf, sync::Arc};

use clap::Args;
use easy_tiger::{
    bulk::{self, BulkLoadBuilder},
    graph::{GraphConfig, GraphLayout, GraphSearchParams},
    input::{DerefVectorStore, SubsetViewVectorStore, VectorStore},
    kmeans::{iterative_balanced_kmeans, Params},
    spann::{
        bulk::{
            assign_to_centroids, bulk_load_centroids, bulk_load_postings, bulk_load_raw_vectors,
        },
        IndexConfig, TableIndex,
    },
    vectors::{F32VectorCoding, VectorSimilarity},
};
use histogram::Histogram;
use rand_xoshiro::{rand_core::SeedableRng, Xoshiro128PlusPlus};
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
    /// Encoding used for navigational vectors in the head index.
    #[arg(long, value_enum)]
    head_nav_format: F32VectorCoding,
    /// Encoding used for rerank vectors in the head index.
    #[arg(long, value_enum, default_value = "raw")]
    head_rerank_format: F32VectorCoding,

    /// Physical layout used for the head graph.
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

    /// Number of edge candidates when searching head table for centroid ids during insertion.
    /// This should be at least as many as --replica-count
    #[arg(long)]
    head_edge_candidates: NonZero<usize>,

    /// Number of vectors to re-rank when searching head table for centroid ids during insertion.
    /// If unset, re-ranks all edge candidates.
    #[arg(long)]
    head_rerank_edges: Option<usize>,

    /// If set replace each head centroid with a representative mediod.
    #[arg(long, default_value_t = true)]
    head_use_mediods: bool,

    /// Maximum number of replica centroids to assign each vector to.
    #[arg(long)]
    replica_count: NonZero<usize>,

    /// Quantizer to use for vectors written to centroid posting lists.
    #[arg(long)]
    posting_coder: F32VectorCoding,

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

    /// If true, print additional information about tail assignments.
    #[arg(long, default_value_t = false)]
    print_tail_assignment_stats: bool,
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
        // XXX allow leaving this unset.
        rerank_format: Some(args.head_rerank_format),
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
        posting_coder: args.posting_coder,
    };
    let index = Arc::new(TableIndex::init_index(
        &connection,
        index_name,
        head_config,
        spann_config,
    )?);
    let mut rng = Xoshiro128PlusPlus::seed_from_u64(args.seed);

    let limit = args.limit.unwrap_or(f32_vectors.len());
    let index_vectors = SubsetViewVectorStore::new(&f32_vectors, (0..limit).into_iter().collect());
    let centroids = {
        let progress = progress_spinner("clustering head");
        let (mut centroids, assignments) = iterative_balanced_kmeans(
            &index_vectors,
            args.head_min_centroid_len..=args.head_max_centroid_len,
            32,
            1000, // batch size
            &Params {
                iters: 100,
                epsilon: 0.01,
                ..Params::default()
            },
            &mut rng,
            |x| progress.inc(x),
        );

        if args.head_use_mediods {
            let mediods = assignments.into_iter().enumerate().fold(
                vec![(usize::MAX, f64::MAX); centroids.len()],
                |mut m, (i, (c, d))| {
                    if d < m[c].1 {
                        m[c] = (i, d);
                    }
                    m
                },
            );
            for (i, c) in mediods.into_iter().map(|(c, _)| c).enumerate() {
                centroids[i].copy_from_slice(&index_vectors[c]);
            }
        }

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
                cluster_ordered_insert: false,
            },
            centroids_len,
        );
        for phase in head_loader.phases() {
            let progress = progress_bar(centroids_len, format!("head {}", phase.display_name()));
            head_loader.execute_phase(phase, |x| progress.inc(x))?;
        }
    }

    let session = connection.open_session()?;
    {
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

    let mut stats = CentroidAssignmentStats::new(centroids_len);
    for assignments in centroid_assignments.iter() {
        stats.add(&assignments);
    }

    println!(
        "Head contains {} centroids ({:4.2}%)",
        centroids_len,
        (centroids_len as f64 / index_vectors.len() as f64) * 100.0
    );
    println!(
        "Inserted {} tail posting entries (avg {:4.2})",
        stats.total_assigned(),
        stats.total_assigned() as f64 / limit as f64
    );
    if args.print_tail_assignment_stats {
        println!("Primary assignments per centroid:");
        CentroidAssignmentStats::print_histogram(stats.primary_assignment_histogram())?;
        println!("Secondary assignments per centroid:");
        CentroidAssignmentStats::print_histogram(stats.secondary_assignment_histogram())?;
        println!("Total assignments per centroid:");
        CentroidAssignmentStats::print_histogram(stats.total_assignment_histogram())?;
    }

    Ok(())
}

struct CentroidAssignmentStats {
    primary: Vec<usize>,
    secondary: Vec<usize>,
}

impl CentroidAssignmentStats {
    pub fn new(centroids_len: usize) -> Self {
        Self {
            primary: vec![0; centroids_len],
            secondary: vec![0; centroids_len],
        }
    }

    pub fn add(&mut self, centroids: &[u32]) {
        if let Some((primary, secondaries)) = centroids.split_first() {
            self.primary[*primary as usize] += 1;
            for s in secondaries {
                self.secondary[*s as usize] += 1;
            }
        }
    }

    pub fn total_assigned(&self) -> usize {
        self.primary.iter().copied().sum::<usize>() + self.secondary.iter().copied().sum::<usize>()
    }

    pub fn primary_assignment_histogram(&self) -> Histogram {
        Self::make_histogram(self.primary.iter().copied())
    }

    pub fn secondary_assignment_histogram(&self) -> Histogram {
        Self::make_histogram(self.secondary.iter().copied())
    }

    pub fn total_assignment_histogram(&self) -> Histogram {
        Self::make_histogram(
            self.primary
                .iter()
                .zip(self.secondary.iter())
                .map(|(p, s)| *p + *s),
        )
    }

    pub fn print_histogram(histogram: Histogram) -> io::Result<()> {
        use std::io::Write;
        let mut lock = std::io::stdout().lock();
        for b in histogram.into_iter().filter(|b| b.count() > 0) {
            writeln!(lock, "[{:5}..{:5}] {:7}", b.start(), b.end(), b.count())?;
        }
        Ok(())
    }

    fn make_histogram(centroid_sizes: impl Iterator<Item = usize> + Clone) -> Histogram {
        let max_value_power = centroid_sizes
            .clone()
            .max()
            .unwrap()
            .next_power_of_two()
            .ilog2() as u8;
        let mut histogram = Histogram::new(2, max_value_power.max(3)).unwrap();
        for c in centroid_sizes {
            histogram.add(c as u64, 1).unwrap();
        }
        histogram
    }
}
