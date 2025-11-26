use std::{io, num::NonZero, ops::RangeInclusive, sync::Arc};

use clap::Args;
use easy_tiger::{
    input::{VecVectorStore, VectorStore},
    kmeans::{self, iterative_balanced_kmeans},
    spann::{centroid_stats::CentroidStats, PostingKey, ReplicaSelectionAlgorithm, TableIndex},
    vamana::{
        crud::IndexMutator,
        graph::{GraphVectorIndexReader, GraphVectorStore},
        search::GraphSearcher,
    },
};
use rand::prelude::*;
use rand_xoshiro::Xoshiro128PlusPlus;
use wt_mdb::{Connection, Error, Result, Session};

// TODO: almost all of these arguments belong in the index config.
#[derive(Args)]
pub struct RebalanceArgs {
    /// Minimum number of vectors that should map to each centroid.
    #[arg(long)]
    min_centroid_len: NonZero<usize>,
    /// Maximum number of vectors that should map to each centroid.
    #[arg(long)]
    max_centroid_len: NonZero<usize>,
    /// Number of replica centroids to assign each vector to.
    #[arg(long)]
    replica_count: NonZero<usize>,
    /// Replica selection algorithm to use.
    #[arg(long, default_value_t = ReplicaSelectionAlgorithm::SOAR)]
    replica_selection: ReplicaSelectionAlgorithm,
    /// Number of rebalancing iterations to perform.
    #[arg(long)]
    iterations: NonZero<usize>,

    /// Random seed used for clustering computations.
    /// Use a fixed value for repeatability.
    #[arg(long, default_value_t = 0x7774_7370414E4E)]
    seed: u64,
}

#[derive(Debug, Default, Copy, Clone)]
struct BalanceSummary {
    in_bounds: usize,

    below_bounds: usize,
    below_exemplar: Option<(usize, usize)>,

    above_bounds: usize,
    above_exemplar: Option<(usize, usize)>,
}

impl BalanceSummary {
    fn new(stats: &CentroidStats, bounds: RangeInclusive<usize>) -> Self {
        let mut summary = Self::default();
        for (i, c) in stats.assignment_counts_iter().map(|(i, c)| (i, c as usize)) {
            if bounds.contains(&c) {
                summary.in_bounds += 1;
            } else if c < *bounds.start() {
                summary.below_bounds += 1;
                summary.below_exemplar = summary
                    .below_exemplar
                    .map(|(j, d)| if c < d { (i, c) } else { (j, d) })
                    .or(Some((i, c)));
            } else {
                summary.above_bounds += 1;
                summary.above_exemplar = summary
                    .above_exemplar
                    .map(|(j, d)| if c > d { (i, c) } else { (j, d) })
                    .or(Some((i, c)));
            }
        }
        summary
    }
}

// TODO: all of this belongs in the index config.
struct RebalancingPolicy {
    min_centroid_len: usize,
    max_centroid_len: usize,
    replica_count: usize,
    replica_selection: ReplicaSelectionAlgorithm,
}

impl RebalancingPolicy {
    fn centroid_len(&self) -> RangeInclusive<usize> {
        self.min_centroid_len..=self.max_centroid_len
    }
}

impl From<RebalanceArgs> for RebalancingPolicy {
    fn from(args: RebalanceArgs) -> Self {
        Self {
            min_centroid_len: args.min_centroid_len.get(),
            max_centroid_len: args.max_centroid_len.get(),
            replica_count: args.replica_count.get(),
            replica_selection: args.replica_selection,
        }
    }
}

pub struct Rebalancer {
    index: Arc<TableIndex>,
    policy: RebalancingPolicy,
    head_mutator: IndexMutator,
}

// XXX I hate the structure of this w.r.t. transactions. If I make any method mutable I can't use
// a transaction guard. Ideally the transaction guard would also take a mutable reference to the
// Session so like much much worse. The alternative is that the rebalancer can't take a session and
// I have to pass one to every single method which is awful.
impl Rebalancer {
    // Get MULT * max_centroid_len vectors for rebalancing.
    const MULT: usize = 8;

    fn new(index: Arc<TableIndex>, session: Session, policy: RebalancingPolicy) -> Self {
        let head_mutator = IndexMutator::new(Arc::clone(index.head_config()), session);
        Self {
            index,
            policy,
            head_mutator,
        }
    }

    fn centroid_stats(&self) -> Result<CentroidStats> {
        CentroidStats::from_index(&self.head_mutator.session(), &self.index)
    }

    fn summary(&self, stats: &CentroidStats) -> BalanceSummary {
        BalanceSummary::new(stats, self.policy.centroid_len())
    }

    fn select_centroids(&self, centroid_id: usize, stats: &CentroidStats) -> Result<Vec<usize>> {
        let reader = self.head_mutator.reader();
        // XXX I should be able to pass a QVD to the searcher to avoid this.
        let mut src_vectors = reader.high_fidelity_vectors()?;
        let coder = src_vectors.new_coder();
        let centroid_vector = coder.decode(
            src_vectors
                .get(centroid_id as i64)
                .unwrap_or(Err(Error::not_found_error()))?,
        );

        let mut searcher = GraphSearcher::new(self.index.config().head_search_params);
        let mut candidates = searcher.search(&centroid_vector, reader)?;
        let target_vectors = Self::MULT * self.policy.min_centroid_len;
        let mut selected_vectors = 0;
        for (i, c) in candidates.iter().enumerate() {
            if selected_vectors >= target_vectors {
                candidates.truncate(i);
                break;
            }
            selected_vectors += stats
                .assignment_counts(c.vertex() as usize)
                .map_or(0, |c| c.total() as usize);
        }
        Ok(candidates
            .into_iter()
            .map(|n| n.vertex() as usize)
            .collect())
    }

    fn get_centroid_vectors(
        &self,
        centroid_ids: &[usize],
    ) -> Result<(Vec<PostingKey>, VecVectorStore<f32>)> {
        let mut posting_keys = vec![];
        let mut vectors = VecVectorStore::new(self.index.head_config().config().dimensions.get());
        let mut scratch_vector = vec![0.0f32; vectors.elem_stride()];

        let coder = self
            .index
            .config()
            .posting_coder
            .new_coder(self.index.head_config().config().similarity);
        let mut posting_cursor = self
            .head_mutator
            .session()
            .get_or_create_typed_cursor::<PostingKey, Vec<u8>>(self.index.postings_table_name())?;
        for c in centroid_ids {
            posting_cursor.set_bounds(
                PostingKey::for_centroid(*c as u32)..PostingKey::for_centroid(*c as u32 + 1),
            )?;

            while let Some(r) = unsafe { posting_cursor.next_unsafe() } {
                let (key, vector) = r?;
                posting_keys.push(key);
                coder.decode_to(vector, &mut scratch_vector);
                vectors.push(&scratch_vector);
            }
        }
        Ok((posting_keys, vectors))
    }
}

pub fn rebalance(
    connection: Arc<Connection>,
    index_name: &str,
    args: RebalanceArgs,
) -> io::Result<()> {
    let index = Arc::new(TableIndex::from_db(&connection, index_name)?);
    let session = connection.open_session()?;
    let mut rng = Xoshiro128PlusPlus::seed_from_u64(args.seed);

    let rebalancer = Rebalancer::new(index, session, args.into());
    let stats = rebalancer.centroid_stats()?;
    let summary = rebalancer.summary(&stats);
    println!("SUMMARY");
    println!("  in bounds:    {:3}", summary.in_bounds);
    println!(
        "  below bounds: {:3} examplar {:?}",
        summary.below_bounds, summary.below_exemplar
    );
    println!(
        "  above bounds: {:3} exemplar {:?}",
        summary.above_bounds, summary.above_exemplar
    );

    let rebalance_centroid =
        if let Some(centroid) = summary.below_exemplar.or(summary.above_exemplar) {
            centroid.0
        } else {
            println!("No centroid to rebalance!");
            return Ok(());
        };

    println!("Rebalancing centroid {}", rebalance_centroid,);

    let centroids = rebalancer.select_centroids(rebalance_centroid, &stats)?;
    println!("Input rebalancing centroids {}", centroids.len());
    for c in centroids.iter() {
        println!(
            "  id {:6} len {:5}",
            *c,
            stats
                .assignment_counts(*c)
                .map_or(0, |c| c.total() as usize)
        );
    }

    let (keys, vectors) = rebalancer.get_centroid_vectors(&centroids)?;
    let k = vectors.len() / rebalancer.policy.max_centroid_len;
    println!(
        "Read {} vectors for rebalancing; target clusters {}",
        keys.len(),
        k
    );

    let (clusters, assignments) = iterative_balanced_kmeans(
        &vectors,
        rebalancer.policy.centroid_len(),
        k,
        vectors.len(),
        // XXX choose better params!
        &kmeans::Params {
            iters: 100,
            epsilon: 0.00001,
            initialization: kmeans::InitializationMethod::KMeansPlusPlus,
            ..kmeans::Params::default()
        },
        &mut rng,
        |_| {},
    );

    println!("Produced {} clusters", clusters.len());
    let cluster_sizes =
        assignments
            .iter()
            .fold(vec![0usize; clusters.len()], |mut counts, (cluster, _)| {
                counts[*cluster] += 1;
                counts
            });
    println!("  Cluster sizes: {cluster_sizes:?}");

    Ok(())
}
