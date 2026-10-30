use std::{ops::RangeInclusive, sync::Arc};

use rand::Rng;
use wt_mdb::{Connection, Result};

use crate::{
    spann::{TableIndex, TransactionIndex, centroid_stats::CentroidStats},
    vamana::search::GraphSearchStats,
};

use std::ops::{Add, AddAssign};

/// Statistics collected during a centroid merge operation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MergeStats {
    /// Number of vectors that were in the merged centroid.
    pub moved_vectors: usize,
    /// Number of unique centroids that vectors were reassigned to.
    pub unique_centroids: usize,
}

impl Add for MergeStats {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            moved_vectors: self.moved_vectors + rhs.moved_vectors,
            unique_centroids: self.unique_centroids + rhs.unique_centroids,
        }
    }
}

impl AddAssign for MergeStats {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

/// Statistics collected during a centroid split operation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SplitStats {
    /// Number of vectors in the centroid being split.
    pub moved_vectors: usize,
    /// Number of vectors where we had to search the head index again to find a new assigned centroid.
    pub searches: usize,
    /// The number of nearby vectors we examine for reassignment.
    pub nearby_seen: usize,
    /// The number of nearby vectors that were reassigned to a new centroid.
    pub nearby_moved: usize,
}

impl Add for SplitStats {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            moved_vectors: self.moved_vectors + rhs.moved_vectors,
            searches: self.searches + rhs.searches,
            nearby_seen: self.nearby_seen + rhs.nearby_seen,
            nearby_moved: self.nearby_moved + rhs.nearby_moved,
        }
    }
}

impl AddAssign for SplitStats {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

/// Time spent in each of the phases run during `parallel_rebalance`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RebalancePhaseDurations {
    pub split_update_head: std::time::Duration,
    pub insert_split_centroids: std::time::Duration,
    pub posting_reassignments: std::time::Duration,
    pub apply_posting_reassignments: std::time::Duration,
    pub remove_merge_sources: std::time::Duration,
    pub select_nearby_centroids: std::time::Duration,
    pub compute_nearby_reassignments: std::time::Duration,
    pub apply_nearby_reassignments: std::time::Duration,
}

impl Add for RebalancePhaseDurations {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            split_update_head: self.split_update_head + rhs.split_update_head,
            insert_split_centroids: self.insert_split_centroids + rhs.insert_split_centroids,
            posting_reassignments: self.posting_reassignments + rhs.posting_reassignments,
            apply_posting_reassignments: self.apply_posting_reassignments
                + rhs.apply_posting_reassignments,
            remove_merge_sources: self.remove_merge_sources + rhs.remove_merge_sources,
            select_nearby_centroids: self.select_nearby_centroids + rhs.select_nearby_centroids,
            compute_nearby_reassignments: self.compute_nearby_reassignments
                + rhs.compute_nearby_reassignments,
            apply_nearby_reassignments: self.apply_nearby_reassignments
                + rhs.apply_nearby_reassignments,
        }
    }
}

impl AddAssign for RebalancePhaseDurations {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

/// Statistics collected during a rebalance operation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RebalanceStats {
    pub merged: usize,
    pub merge_stats: MergeStats,
    pub split: usize,
    pub split_stats: SplitStats,
    /// Accumulated graph search stats for all searches performed during rebalancing.
    pub search_stats: GraphSearchStats,
    /// Time spent in each parallel rebalance phase.
    pub phase_durations: RebalancePhaseDurations,
}

impl Add for RebalanceStats {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            merged: self.merged + rhs.merged,
            merge_stats: self.merge_stats + rhs.merge_stats,
            split: self.split + rhs.split,
            split_stats: self.split_stats + rhs.split_stats,
            search_stats: self.search_stats + rhs.search_stats,
            phase_durations: self.phase_durations + rhs.phase_durations,
        }
    }
}

impl AddAssign for RebalanceStats {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Add<MergeStats> for RebalanceStats {
    type Output = Self;

    fn add(self, rhs: MergeStats) -> Self::Output {
        Self {
            merged: self.merged + 1,
            merge_stats: self.merge_stats + rhs,
            split: self.split,
            split_stats: self.split_stats,
            search_stats: self.search_stats,
            phase_durations: self.phase_durations,
        }
    }
}

impl AddAssign<MergeStats> for RebalanceStats {
    fn add_assign(&mut self, rhs: MergeStats) {
        *self = *self + rhs;
    }
}

impl Add<SplitStats> for RebalanceStats {
    type Output = Self;

    fn add(self, rhs: SplitStats) -> Self::Output {
        Self {
            merged: self.merged,
            merge_stats: self.merge_stats,
            split: self.split + 1,
            split_stats: self.split_stats + rhs,
            search_stats: self.search_stats,
            phase_durations: self.phase_durations,
        }
    }
}

impl AddAssign<SplitStats> for RebalanceStats {
    fn add_assign(&mut self, rhs: SplitStats) {
        *self = *self + rhs;
    }
}

/// A summary of centroid assignment balance.
///
/// This consumes centroid stats and a range of valid assignment counts and provides a summary of
/// the number of centroids that are in bounds, below bounds, and above bounds, as well as examplars
/// for above and below bounds centroids.
#[derive(Debug, Default, Copy, Clone)]
pub struct BalanceSummary {
    in_bounds: usize,

    below_bounds: usize,
    below_exemplar: Option<(usize, usize)>,

    above_bounds: usize,
    above_exemplar: Option<(usize, usize)>,
}

impl BalanceSummary {
    /// Create a balance summary from centroid stats and a range of valid assignment counts.
    pub fn new(stats: &CentroidStats, bounds: RangeInclusive<usize>) -> Self {
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

    /// Total number of centroids.
    pub fn total_clusters(&self) -> usize {
        self.in_bounds + self.below_bounds + self.above_bounds
    }

    /// Number of centroids that have assignment counts within bounds.
    pub fn in_bounds(&self) -> usize {
        self.in_bounds
    }

    /// Fraction of centroids that have assignment counts within bounds.
    pub fn in_policy_fraction(&self) -> f64 {
        self.in_bounds as f64 / self.total_clusters() as f64
    }

    /// Number of centroids that have assignment counts below bounds.
    pub fn below_bounds(&self) -> usize {
        self.below_bounds
    }

    /// Below bounds examplar centroid ID and assignment count.
    ///
    /// The returned centroid is the once with the fewest assignments in the index.
    pub fn below_exemplar(&self) -> Option<(usize, usize)> {
        self.below_exemplar
    }

    /// Number of centroids that have assignment counts above bounds.
    pub fn above_bounds(&self) -> usize {
        self.above_bounds
    }

    /// Above bounds examplar centroid ID and assignment count.
    ///
    /// The returned centroid is the once with the most assignments in the index.
    pub fn above_exemplar(&self) -> Option<(usize, usize)> {
        self.above_exemplar
    }
}

mod parallel {
    use rand::Rng;
    use rayon::prelude::*;
    use std::{
        collections::{HashMap, HashSet},
        num::NonZero,
        ops::RangeInclusive,
        sync::Arc,
    };
    use tracing::warn;
    use vectors::QueryVectorDistance;
    use wt_mdb::{Connection, Error, Result};

    use crate::{
        input::{VecVectorStore, VectorStore},
        posting_block::PostingBlock,
        spann::{
            CentroidAssignment, TableIndex, TransactionIndex,
            centroid_stats::{CentroidAssignmentUpdater, CentroidCounts, CentroidStats},
            postings::BlockPostingsMut,
            rebalance::{MergeStats, RebalanceStats, SplitStats},
        },
        vamana::{
            GraphVectorIndex, GraphVectorStore,
            mutate::{delete_vector, upsert_vector_with_options},
            search::{GraphSearchStats, GraphSearcher, Options as GraphSearchOptions},
        },
    };

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum RebalanceOp {
        /// Merge out the centroid, globally reassigning all vectors contain in its posting.
        Merge(u32),
        /// Split the centroid into two component centroids. The source centroid id is retained for
        /// one component and the second id is newly allocated for the other component.
        /// Posting vectors will remain in the source, move to the new centroid, or be globally
        /// reassigned.
        Split(u32, u32),
    }

    impl RebalanceOp {
        fn source(&self) -> u32 {
            *match self {
                Self::Merge(c) => c,
                Self::Split(c, _) => c,
            }
        }
    }

    /// Return the source ids of all merge ops. These centroids are removed during rebalancing.
    fn merge_sources(ops: &[RebalanceOp]) -> HashSet<u32> {
        ops.iter()
            .filter_map(|op| match op {
                RebalanceOp::Merge(c) => Some(*c),
                RebalanceOp::Split(_, _) => None,
            })
            .collect()
    }

    /// Centroids computed for a split op.
    pub struct SplitCentroids {
        /// Centroid being split. This id is retained and is assigned `centroids[0]`.
        pub source: u32,
        /// Newly allocated centroid id that is assigned `centroids[1]`.
        pub new: u32,
        /// High fidelity head vector for `source` before it is overwritten by the split.
        pub old_source_vector: Vec<u8>,
        /// The two component centroids.
        pub centroids: VecVectorStore<f32>,
    }

    /// Return a distance function for a centroid vector encoded in the head high fidelity format,
    /// such as one read before the centroid was overwritten.
    fn encoded_centroid_distance(
        txn_idx: &TransactionIndex,
        centroid: &[u8],
    ) -> Box<dyn QueryVectorDistance> {
        let head_config = txn_idx.index().head_config();
        let similarity = head_config.config().similarity;
        let head_format = head_config.high_fidelity_table().format();
        let posting_format = txn_idx.index().config().posting_coder;
        if head_format == posting_format {
            posting_format.query_distance_symmetric(similarity, centroid.to_vec())
        } else {
            posting_format
                .query_distance_asymmetric(similarity, head_format.coder().decode(centroid))
                .expect("centroid vector is finite")
        }
    }

    /// Return a distance function for the current head vector of `centroid_id`.
    fn centroid_distance(
        txn_idx: &TransactionIndex,
        centroid_id: u32,
    ) -> Result<Box<dyn QueryVectorDistance>> {
        let mut centroid_store = txn_idx.head().high_fidelity_vectors()?;
        let centroid = centroid_store
            .get(centroid_id as i64)
            .unwrap_or_else(|| Err(Error::not_found_error()))?;
        Ok(encoded_centroid_distance(txn_idx, centroid))
    }

    struct PostingUpdater<'a> {
        postings: BlockPostingsMut<'a>,
        assignments: CentroidAssignmentUpdater<'a>,
    }

    impl<'a> PostingUpdater<'a> {
        pub fn new(txn_idx: &'a TransactionIndex) -> Result<Self> {
            Ok(Self {
                postings: BlockPostingsMut::from_txn(txn_idx)?,
                assignments: CentroidAssignmentUpdater::new(txn_idx)?,
            })
        }

        pub fn read_centroid(&mut self, centroid_id: u32) -> Result<Vec<(i64, Vec<u8>)>> {
            self.postings.read_centroid(centroid_id)
        }

        pub fn move_posting(&mut self, record_id: i64, source: u32, target: u32) -> Result<()> {
            let old = self
                .assignments
                .update(record_id, CentroidAssignment::new(target))?;
            assert_eq!(old.primary_id, source);
            let v = self.postings.remove(source, record_id)?.unwrap().to_vec();
            self.postings.insert(target, record_id, &v)
        }

        pub fn flush(mut self) -> Result<()> {
            self.postings
                .flush()
                .and_then(|()| self.assignments.flush())
        }
    }

    /// Get a list of all rebalancing operations that need to be performed.
    pub fn get_rebalance_ops(
        stats: &CentroidStats,
        bounds: RangeInclusive<usize>,
    ) -> Vec<RebalanceOp> {
        let mut it = stats.available_centroid_ids();
        stats
            .assignment_counts_iter()
            .filter_map(|(centroid, count)| {
                if (count as usize) < *bounds.start() {
                    Some(RebalanceOp::Merge(centroid as u32))
                } else if count as usize > *bounds.end() {
                    Some(RebalanceOp::Split(
                        centroid as u32,
                        it.next().unwrap() as u32,
                    ))
                } else {
                    None
                }
            })
            .collect()
    }

    /// For all split operations, generate new centroid vectors in parallel.
    ///
    /// The returned centroids are ordered so that the one closest to the source centroid is first;
    /// this component retains the source id, which minimizes disruption to nearby postings.
    pub fn generate_split_centroids<R: Rng>(
        connection: &Arc<Connection>,
        index: &Arc<TableIndex>,
        ops: &[RebalanceOp],
        rng_supplier: &(impl Fn() -> R + Send + Sync),
    ) -> Result<Vec<SplitCentroids>> {
        ops.par_iter()
            .filter_map(|op| match op {
                RebalanceOp::Split(s, n) => Some((*s, *n)),
                RebalanceOp::Merge(_) => None,
            })
            .map_init(
                || {
                    TransactionIndex::new(
                        index,
                        connection
                            .begin_transaction(None)
                            .expect("open session and txn"),
                    )
                },
                |txn_idx, (s, n)| {
                    let mut posting_cursor = txn_idx
                        .transaction()
                        .open_cursor::<u32, Vec<u8>>(index.postings_table_name())?;
                    let Some(raw_posting) =
                        unsafe { posting_cursor.seek_exact_unsafe(s) }.transpose()?
                    else {
                        return Ok(None);
                    };
                    let block =
                        PostingBlock::new(raw_posting, txn_idx.index().posting_vector_len())
                            .expect("valid posting block");
                    let mut rng = rng_supplier();
                    let posting_vectors = extract_posting_vectors(txn_idx, &block)?;
                    let mut centroids = partition_postings(txn_idx, s, &posting_vectors, &mut rng);

                    let old_source_vector = txn_idx
                        .head()
                        .high_fidelity_vectors()?
                        .get(s as i64)
                        .unwrap_or_else(|| Err(Error::not_found_error()))?
                        .to_vec();
                    let source_distfn = encoded_centroid_distance(txn_idx, &old_source_vector);
                    let coder = index.new_posting_coder();
                    if source_distfn.distance(
                        &coder
                            .encode(&centroids[1])
                            .expect("split centroid is finite"),
                    ) < source_distfn.distance(
                        &coder
                            .encode(&centroids[0])
                            .expect("split centroid is finite"),
                    ) {
                        let c0 = centroids[0].to_vec();
                        centroids.swap_remove(0);
                        centroids.push(&c0);
                    }

                    Ok(Some(SplitCentroids {
                        source: s,
                        new: n,
                        old_source_vector,
                        centroids,
                    }))
                },
            )
            .filter_map(|r: Result<Option<_>>| r.transpose())
            .collect()
    }

    fn extract_posting_vectors(
        txn_idx: &TransactionIndex,
        block: &PostingBlock<'_>,
    ) -> Result<VecVectorStore<f32>> {
        let len = block.len();
        let mut scratch_vector =
            vec![0.0f32; txn_idx.index().head_config().config().dimensions.get()];
        let mut clustering_vectors = VecVectorStore::with_capacity(scratch_vector.len(), len);
        if txn_idx.index().config().split_rerank_vectors {
            let coder = txn_idx.index().config().rerank_format.coder();
            let mut cursor = txn_idx
                .transaction()
                .open_cursor::<i64, Vec<u8>>(txn_idx.index().raw_vectors_table_name())?;
            for vid in block.iter().map(|x| x.0) {
                let Some(v) = unsafe { cursor.seek_exact_unsafe(vid).transpose() }? else {
                    continue;
                };
                coder.decode_to(v, &mut scratch_vector);
                clustering_vectors.push(&scratch_vector);
            }
        } else {
            let coder = txn_idx.index().config().posting_coder.coder();
            for v in block.iter().map(|x| x.1) {
                coder.decode_to(v, &mut scratch_vector);
                clustering_vectors.push(&scratch_vector);
            }
        }
        Ok(clustering_vectors)
    }

    /// Insert pre-computed split centroids into the head index.
    ///
    /// Each split source is overwritten in place with its first component centroid, then each
    /// newly allocated centroid is inserted with the second component. Sources are updated first
    /// so that new centroids select edges against current rather than stale source vectors.
    ///
    /// Excludes every merge source in `ops` from selection as an edge. Merge sources are still
    /// present in the graph at this point and will be removed later by `remove_merge_sources`;
    /// linking to them would only create edges that need to be repaired on deletion.
    ///
    /// Also writes an empty stats row for each new centroid so that the id is reserved even if no
    /// vectors are assigned to it.
    // NB: deliberately sequential to avoid conflicts.
    pub fn insert_split_centroids(
        connection: &Arc<Connection>,
        index: &Arc<TableIndex>,
        ops: &[RebalanceOp],
        split_centroids: &[SplitCentroids],
    ) -> Result<()> {
        if split_centroids.is_empty() {
            return Ok(());
        }
        let merge_sources = merge_sources(ops);
        let txn_idx = TransactionIndex::new(index, connection.begin_transaction(None)?);
        for split in split_centroids {
            upsert_vector_with_options(
                split.source as i64,
                &split.centroids[0],
                txn_idx.head(),
                GraphSearchOptions::with_filter(|i: i64| !merge_sources.contains(&(i as u32))),
            )?;
        }
        let mut stats_cursor = txn_idx
            .transaction()
            .open_cursor::<u32, CentroidCounts>(index.centroid_stats_table_name())?;
        for split in split_centroids {
            upsert_vector_with_options(
                split.new as i64,
                &split.centroids[1],
                txn_idx.head(),
                GraphSearchOptions::with_filter(|i: i64| !merge_sources.contains(&(i as u32))),
            )?;
            stats_cursor.set(split.new, CentroidCounts::default())?;
        }
        drop(stats_cursor);
        txn_idx.commit(None)
    }

    fn partition_postings(
        txn_idx: &TransactionIndex,
        centroid_id: u32,
        vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
        rng: &mut impl Rng,
    ) -> VecVectorStore<f32> {
        let mut centroids = match crate::kmeans::balanced_binary_partition(
            vectors,
            100,
            txn_idx.index().config().min_centroid_len,
            rng,
        ) {
            Ok(r) => r,
            Err(r) => {
                warn!(
                    "split_centroid: binary partition of centroid {centroid_id} (count {}) failed to converge!",
                    vectors.len()
                );
                r
            }
        };

        // The partition centroids are arithmetic means of the (unit-length) posting vectors, and
        // the mean of unit vectors has norm < 1. Normalize so the split centroids live in the same
        // space the head index navigates when similarity is angular.
        if txn_idx.index().head_config().config().similarity.angular() {
            for centroid in centroids.iter_mut() {
                vectors::prepare_vector_in_place(centroid, None, true, None)
                    .expect("split centroid is finite");
            }
        }
        centroids
    }

    /// Generate a list of all posting moves out of ops assuming an updated head index.
    ///
    /// Vectors that remain in a split source centroid are not listed.
    pub fn posting_reassignments(
        connection: &Arc<Connection>,
        index: &Arc<TableIndex>,
        ops: &[RebalanceOp],
        split_centroids: &[SplitCentroids],
    ) -> Result<(Vec<PostingMove>, RebalanceStats)> {
        #[derive(Debug, Clone)]
        enum Target {
            Centroid(u32),
            Query(Vec<u8>),
        }

        let splits = split_centroids
            .iter()
            .map(|s| (s.source, s))
            .collect::<HashMap<_, _>>();

        // TODO: this is two separate statements to maximize parallelism between the searches, but
        // the searches only take a few ms at most to finish so it would probably be better to
        // collapse the work down to per-centroid.

        // For each op compute the set of vectors that need to be moved to a target.
        let centroid_reassignments = ops
            .par_iter()
            .map_init(
                || {
                    TransactionIndex::new(
                        index,
                        connection
                            .begin_transaction(None)
                            .expect("open session and txn"),
                    )
                },
                |txn, op| {
                    let mut posting_cursor = txn
                        .transaction()
                        .open_cursor::<u32, Vec<u8>>(index.postings_table_name())?;
                    let Some(raw_posting) =
                        unsafe { posting_cursor.seek_exact_unsafe(op.source()) }.transpose()?
                    else {
                        return Ok::<(u32, Vec<(i64, Target)>, RebalanceStats), Error>((
                            op.source(),
                            vec![],
                            RebalanceStats::default(),
                        ));
                    };
                    let block = PostingBlock::new(raw_posting, txn.index().posting_vector_len())
                        .expect("valid posting block");
                    match op {
                        RebalanceOp::Split(s, _) => {
                            // If the split was skipped the source is unchanged.
                            let Some(split) = splits.get(s) else {
                                return Ok((*s, vec![], RebalanceStats::default()));
                            };
                            // The source has been overwritten in the head index, use the vector
                            // captured before the split.
                            let s_distfn = encoded_centroid_distance(txn, &split.old_source_vector);
                            let t0_distfn = centroid_distance(txn, split.source)?;
                            let t1_distfn = centroid_distance(txn, split.new)?;
                            let postings = block
                                .iter()
                                .map(|(id, v)| {
                                    let s_dist = s_distfn.distance(v);
                                    let t0_dist = t0_distfn.distance(v);
                                    let t1_dist = t1_distfn.distance(v);
                                    let target = if s_dist < t0_dist && s_dist < t1_dist {
                                        Target::Query(v.to_vec())
                                    } else if t0_dist < t1_dist {
                                        Target::Centroid(split.source)
                                    } else {
                                        Target::Centroid(split.new)
                                    };
                                    (id, target)
                                })
                                .collect::<Vec<_>>();
                            let stats = RebalanceStats::default()
                                + SplitStats {
                                    moved_vectors: postings.len(),
                                    searches: postings
                                        .iter()
                                        .filter(|x| matches!(x.1, Target::Query(_)))
                                        .count(),
                                    nearby_seen: 0,
                                    nearby_moved: 0,
                                };
                            Ok((*s, postings, stats))
                        }
                        RebalanceOp::Merge(s) => {
                            let postings = block
                                .iter()
                                .map(|(i, v)| (i, Target::Query(v.to_vec())))
                                .collect::<Vec<_>>();
                            let stats = RebalanceStats::default()
                                + MergeStats {
                                    moved_vectors: postings.len(),
                                    unique_centroids: 0,
                                };
                            Ok((*s, postings, stats))
                        }
                    }
                },
            )
            .collect::<Result<Vec<_>>>()?;
        let mut stats = centroid_reassignments
            .iter()
            .map(|(_, _, stats)| *stats)
            .fold(RebalanceStats::default(), |acc, s| acc + s);

        // Assign targets to any vector that does not yet have one.
        // Only merge sources are excluded; split sources remain valid targets.
        let coder = index.new_posting_coder();
        let filter = merge_sources(ops);
        let (moves, search_stats) = centroid_reassignments
            .into_par_iter()
            .map_init(
                || {
                    let txn = TransactionIndex::new(
                        index,
                        connection
                            .begin_transaction(None)
                            .expect("open session and txn"),
                    );
                    // Use a lower budget for during reassignment. We're going to seed the search
                    // with the source node which makes convergence faster and more accurate.
                    let mut params = txn.index().config().head_search_params;
                    params.beam_width = NonZero::new(16.min(params.beam_width.get())).unwrap();
                    let searcher = GraphSearcher::new(params);
                    let result_scratch = Vec::with_capacity(params.beam_width.get());
                    (txn, searcher, result_scratch)
                },
                |(txn, searcher, result_scratch), (source, postings, _)| {
                    let mut m = vec![];
                    let mut s = GraphSearchStats::default();
                    for (vector_id, target) in postings {
                        let centroid = match target {
                            Target::Centroid(c) => c,
                            Target::Query(q) => {
                                let (mut candidates, _) = searcher.search_with_options(
                                    &coder.decode(&q),
                                    GraphSearchOptions::with_filter(|i| {
                                        !filter.contains(&(i as u32))
                                    })
                                    .with_seeds([source as i64])
                                    .with_result_scratch(std::mem::take(result_scratch)),
                                    txn.head(),
                                )?;
                                let centroid = candidates[0].vertex() as u32;
                                std::mem::swap(result_scratch, &mut candidates); // return scratch.
                                s += searcher.stats();
                                centroid
                            }
                        };
                        // The vector remains in the (split) source centroid.
                        if centroid == source {
                            continue;
                        }
                        m.push(PostingMove {
                            record_id: vector_id,
                            source,
                            target: centroid,
                        });
                    }
                    Ok::<(Vec<PostingMove>, GraphSearchStats), Error>((m, s))
                },
            )
            .try_reduce(
                || (vec![], GraphSearchStats::default()),
                |(mut acc, mut acc_stats), (partial, partial_stats)| {
                    acc.extend_from_slice(&partial);
                    acc_stats += partial_stats;
                    Ok((acc, acc_stats))
                },
            )?;
        stats.search_stats = search_stats;
        Ok((moves, stats))
    }

    /// Remove merge source centroid ids from stats, postings, and the head graph.
    ///
    /// All postings must have been moved out of the merge sources before calling this.
    // NB: deliberately sequential to avoid high conflict rate.
    pub fn remove_merge_sources(
        connection: &Arc<Connection>,
        index: &Arc<TableIndex>,
        ops: &[RebalanceOp],
    ) -> Result<()> {
        let txn_idx = TransactionIndex::new(index, connection.begin_transaction(None)?);
        let mut stats_cursor = txn_idx
            .transaction()
            .open_cursor::<u32, CentroidCounts>(index.centroid_stats_table_name())?;
        let mut postings = BlockPostingsMut::from_txn(&txn_idx)?;

        for s in merge_sources(ops) {
            stats_cursor.remove(s).or_else(|e| {
                if e == Error::not_found_error() {
                    Ok(())
                } else {
                    Err(e)
                }
            })?;
            postings.remove_centroid(s)?;
            delete_vector(s as i64, txn_idx.head())?;
        }

        postings.flush()?;
        drop(postings);
        drop(stats_cursor);
        txn_idx.commit(None)
    }

    /// For all split ops produce a map from each nearby centroid that should be considered for
    /// reassignment to a list of potential targets that should be considered.
    pub fn select_nearby_centroids(
        connection: &Arc<Connection>,
        index: &Arc<TableIndex>,
        ops: &[RebalanceOp],
        split_centroids: &[SplitCentroids],
    ) -> Result<(HashMap<u32, Vec<u32>>, GraphSearchStats)> {
        // Split sources are retained so only merge sources need to be filtered.
        let filter_centroids = merge_sources(ops);
        let target_to_nearby = split_centroids
            .par_iter()
            .flat_map(|split| [split.source, split.new])
            .map_init(
                || {
                    let txn = TransactionIndex::new(
                        index,
                        connection
                            .begin_transaction(None)
                            .expect("open session and txn"),
                    );
                    // Search for 128 vectors regardless of settings, we will truncate to 64 for
                    // nearby check.
                    let mut params = txn.index().config().head_search_params;
                    params.beam_width = NonZero::new(128).unwrap();
                    let searcher = GraphSearcher::new(params);
                    (txn, searcher)
                },
                |(txn_idx, searcher), centroid| {
                    let mut store = txn_idx.head().high_fidelity_vectors()?;
                    let coder = store.new_coder();
                    let (candidates, _) = searcher.search_with_options(
                        &coder.decode(
                            store
                                .get(centroid as i64)
                                .unwrap_or_else(|| Err(Error::not_found_error()))?,
                        ),
                        GraphSearchOptions::with_filter(|i| {
                            i != centroid as i64 && !filter_centroids.contains(&(i as u32))
                        })
                        .with_seeds([centroid as i64]),
                        txn_idx.head(),
                    )?;
                    Ok::<_, Error>((
                        centroid,
                        candidates
                            .iter()
                            .take(64)
                            .map(|n| n.vertex() as u32)
                            .collect::<Vec<_>>(),
                        searcher.stats(),
                    ))
                },
            )
            .collect::<Result<Vec<_>>>()?;

        let mut nearby_to_targets: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut search_stats = GraphSearchStats::default();
        for (target, nearby, stats) in target_to_nearby {
            search_stats += stats;
            for n in nearby {
                nearby_to_targets.entry(n).or_default().push(target);
            }
        }
        Ok((nearby_to_targets, search_stats))
    }

    /// Move of a single posting from `source` to `target` centroid.
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct PostingMove {
        pub record_id: i64,
        pub source: u32,
        pub target: u32,
    }

    /// For each nearby centroid examine all postings and compare them against each of the target
    /// centroid vectors. Yield any vector that is closer to one of the targets.
    pub fn compute_nearby_reassignments(
        connection: &Arc<Connection>,
        index: &Arc<TableIndex>,
        nearby_to_targets: &HashMap<u32, Vec<u32>>,
    ) -> Result<(Vec<PostingMove>, SplitStats)> {
        let per_nearby = nearby_to_targets
            .par_iter()
            .map_init(
                || {
                    TransactionIndex::new(
                        index,
                        connection
                            .begin_transaction(None)
                            .expect("open session and txn"),
                    )
                },
                |txn, (nearby, targets)| {
                    let mut updater = PostingUpdater::new(txn)?;
                    let postings = updater.read_centroid(*nearby)?;
                    let nearby_distfn = centroid_distance(txn, *nearby)?;
                    let targets_distfn = targets
                        .iter()
                        .map(|c| centroid_distance(txn, *c))
                        .collect::<Result<Vec<_>>>()?;
                    let moves = postings
                        .iter()
                        .filter_map(|(id, v)| {
                            let closest_target = targets
                                .iter()
                                .zip(targets_distfn.iter())
                                .map(|(c, d)| (*c, d.distance(v)))
                                .min_by(|a, b| a.1.total_cmp(&b.1))
                                .unwrap();
                            if closest_target.1 < nearby_distfn.distance(v) {
                                Some(PostingMove {
                                    record_id: *id,
                                    source: *nearby,
                                    target: closest_target.0,
                                })
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();
                    let stats = SplitStats {
                        nearby_seen: (postings.len() * targets.len()),
                        nearby_moved: moves.len(),
                        ..Default::default()
                    };
                    Ok::<_, Error>((moves, stats))
                },
            )
            .collect::<Result<Vec<_>>>()?;

        let stats = per_nearby
            .iter()
            .map(|x| x.1)
            .fold(SplitStats::default(), |acc, s| acc + s);
        let moves = per_nearby.into_iter().flat_map(|x| x.0).collect();
        Ok((moves, stats))
    }

    /// Apply posting moves to the index in a single transaction.
    // NB: deliberately sequential as moves out of a single source would conflict.
    pub fn apply_posting_moves(
        connection: &Arc<Connection>,
        index: &Arc<TableIndex>,
        moves: &[PostingMove],
    ) -> Result<()> {
        if moves.is_empty() {
            return Ok(());
        }

        let txn_idx = TransactionIndex::new(index, connection.begin_transaction(None)?);
        let mut updater = PostingUpdater::new(&txn_idx)?;
        for &m in moves {
            updater.move_posting(m.record_id, m.source, m.target)?;
        }
        updater.flush()?;
        txn_idx.commit(None)
    }
}

/// Rebalance all centroids that do not match size policy in parallel.
pub fn parallel_rebalance<R: Rng>(
    connection: &Arc<Connection>,
    index: &Arc<TableIndex>,
    rng_supplier: &(impl Fn() -> R + Send + Sync),
) -> Result<RebalanceStats> {
    let centroid_stats = {
        let txn = TransactionIndex::new(index, connection.begin_transaction(None)?);
        CentroidStats::from_index_stats(&txn)?
    };
    let ops = parallel::get_rebalance_ops(&centroid_stats, index.config().centroid_len_range());

    let start = std::time::Instant::now();
    let split_centroids =
        parallel::generate_split_centroids(connection, index, &ops, rng_supplier)?;
    let split_update_head = start.elapsed();

    let start = std::time::Instant::now();
    parallel::insert_split_centroids(connection, index, &ops, &split_centroids)?;
    let insert_split_centroids = start.elapsed();

    let start = std::time::Instant::now();
    let (reassignments, mut stats) =
        parallel::posting_reassignments(connection, index, &ops, &split_centroids)?;
    let posting_reassignments = start.elapsed();

    let start = std::time::Instant::now();
    parallel::apply_posting_moves(connection, index, &reassignments)?;
    let apply_posting_reassignments = start.elapsed();

    let start = std::time::Instant::now();
    parallel::remove_merge_sources(connection, index, &ops)?;
    let remove_merge_sources = start.elapsed();

    let start = std::time::Instant::now();
    let (nearby_to_targets, _) =
        parallel::select_nearby_centroids(connection, index, &ops, &split_centroids)?;
    let select_nearby_centroids = start.elapsed();

    let start = std::time::Instant::now();
    let (nearby_reassignments, nearby_stats) =
        parallel::compute_nearby_reassignments(connection, index, &nearby_to_targets)?;
    let compute_nearby_reassignments = start.elapsed();
    stats.split_stats += nearby_stats;

    let start = std::time::Instant::now();
    parallel::apply_posting_moves(connection, index, &nearby_reassignments)?;
    let apply_nearby_reassignments = start.elapsed();

    stats.phase_durations += RebalancePhaseDurations {
        split_update_head,
        insert_split_centroids,
        posting_reassignments,
        apply_posting_reassignments,
        remove_merge_sources,
        select_nearby_centroids,
        compute_nearby_reassignments,
        apply_nearby_reassignments,
    };
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        num::NonZero,
        sync::Arc,
    };

    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;
    use vectors::{F32VectorCoding, VectorSimilarity};
    use wt_mdb::Connection;

    use super::{
        BalanceSummary, RebalanceStats,
        parallel::{self, RebalanceOp},
        parallel_rebalance,
    };
    use crate::{
        posting_block::PostingBlock,
        spann::{
            CentroidAssignment, IndexConfig, TableIndex, TransactionIndex,
            centroid_stats::{CentroidAssignmentUpdater, CentroidCounts, CentroidStats},
            postings::BlockPostingsMut,
        },
        vamana::{
            EdgePruningConfig, EdgeType, Graph, GraphConfig, GraphSearchParams, GraphVectorIndex,
            GraphVectorStore, mutate::insert_vector, search::GraphSearcher,
        },
    };

    const MIN_CENTROID_LEN: usize = 4;
    const MAX_CENTROID_LEN: usize = 16;

    struct Fixture {
        conn: Arc<Connection>,
        index: Arc<TableIndex>,
        /// All vectors inserted into the index keyed by record id.
        vectors: HashMap<i64, Vec<f32>>,
        _dir: tempfile::TempDir,
    }

    /// Intermediate state from each rebalance phase.
    struct Phases {
        ops: Vec<RebalanceOp>,
        reassignments: Vec<parallel::PostingMove>,
        nearby: HashMap<u32, Vec<u32>>,
    }

    /// State of the index read after verifying consistency.
    struct Snapshot {
        /// Record id -> assigned centroid id.
        assignments: HashMap<i64, u32>,
        /// Centroid id -> number of assigned vectors.
        counts: HashMap<u32, u32>,
        /// Centroid id -> head vector.
        centroids: HashMap<u32, Vec<f32>>,
    }

    impl Fixture {
        fn new(similarity: VectorSimilarity, edge_type: EdgeType) -> Self {
            let dir = tempfile::TempDir::new().unwrap();
            let conn = Connection::open(
                dir.path().to_str().unwrap(),
                Some(
                    wt_mdb::connection::OptionsBuilder::default()
                        .create()
                        .into(),
                ),
            )
            .unwrap();
            let search_params = GraphSearchParams {
                beam_width: NonZero::new(32).unwrap(),
                num_rerank: 0,
                patience: None,
            };
            let index = Arc::new(
                TableIndex::init_index(
                    &conn,
                    "test",
                    GraphConfig {
                        dimensions: NonZero::new(2).unwrap(),
                        similarity,
                        nav_format: F32VectorCoding::F32,
                        rerank_format: None,
                        pruning: EdgePruningConfig::new(NonZero::new(8).unwrap()),
                        index_search_params: search_params,
                        centroid: None,
                        edge_type,
                    },
                    IndexConfig {
                        head_search_params: search_params,
                        posting_coder: F32VectorCoding::F32,
                        min_centroid_len: MIN_CENTROID_LEN,
                        max_centroid_len: MAX_CENTROID_LEN,
                        rerank_format: F32VectorCoding::F32,
                        split_rerank_vectors: false,
                        rotation_seed: None,
                    },
                )
                .unwrap(),
            );
            Self {
                conn,
                index,
                vectors: HashMap::new(),
                _dir: dir,
            }
        }

        fn txn_index(&self) -> TransactionIndex {
            TransactionIndex::new(&self.index, self.conn.begin_transaction(None).unwrap())
        }

        /// Add a centroid to the head index and return its id.
        fn add_centroid(&self, vector: &[f32]) -> u32 {
            let txn = self.txn_index();
            let id = insert_vector(vector, txn.head()).unwrap();
            txn.commit(None).unwrap();
            id as u32
        }

        /// Add `vectors` assigned to `centroid` and return their record ids.
        fn add_vectors(&mut self, centroid: u32, vectors: &[Vec<f32>]) -> Vec<i64> {
            let ids = (self.vectors.len()..(self.vectors.len() + vectors.len()))
                .map(|i| i as i64)
                .collect::<Vec<_>>();
            let txn = self.txn_index();
            {
                let mut assignments = CentroidAssignmentUpdater::new(&txn).unwrap();
                let mut postings = BlockPostingsMut::from_txn(&txn).unwrap();
                let mut raw_cursor = txn
                    .transaction()
                    .open_cursor::<i64, Vec<u8>>(self.index.raw_vectors_table_name())
                    .unwrap();
                let posting_coder = self.index.new_posting_coder();
                let raw_coder = self.index.config().rerank_format.coder();
                for (id, v) in ids.iter().zip(vectors) {
                    assignments
                        .insert(*id, CentroidAssignment::new(centroid))
                        .unwrap();
                    postings
                        .insert(
                            centroid,
                            *id,
                            &posting_coder.encode(v).expect("finite vector"),
                        )
                        .unwrap();
                    raw_cursor
                        .set(*id, &raw_coder.encode(v).expect("finite vector"))
                        .unwrap();
                }
                postings.flush().unwrap();
                assignments.flush().unwrap();
            }
            txn.commit(None).unwrap();
            self.vectors
                .extend(ids.iter().copied().zip(vectors.iter().cloned()));
            ids
        }

        /// Add `vectors` assigned to the nearest centroid in the head index.
        fn insert_vectors(&mut self, vectors: &[Vec<f32>]) {
            let mut by_centroid: HashMap<u32, Vec<Vec<f32>>> = HashMap::new();
            {
                let txn = self.txn_index();
                let mut searcher = GraphSearcher::new(self.index.config().head_search_params);
                for v in vectors {
                    let centroid = searcher.search(v, txn.head()).unwrap()[0].vertex() as u32;
                    by_centroid.entry(centroid).or_default().push(v.clone());
                }
            }
            for (centroid, vectors) in by_centroid {
                self.add_vectors(centroid, &vectors);
            }
        }

        fn centroid_stats(&self) -> CentroidStats {
            CentroidStats::from_index_stats(&self.txn_index()).unwrap()
        }

        fn rebalance(&self) -> RebalanceStats {
            parallel_rebalance(&self.conn, &self.index, &|| {
                Xoshiro256PlusPlus::seed_from_u64(0x5eed)
            })
            .unwrap()
        }

        /// Run each phase of `parallel_rebalance` and return intermediate state for inspection.
        fn rebalance_phases(&self) -> Phases {
            let (conn, index) = (&self.conn, &self.index);
            let ops = parallel::get_rebalance_ops(
                &self.centroid_stats(),
                index.config().centroid_len_range(),
            );
            let splits = parallel::generate_split_centroids(conn, index, &ops, &|| {
                Xoshiro256PlusPlus::seed_from_u64(0x5eed)
            })
            .unwrap();
            parallel::insert_split_centroids(conn, index, &ops, &splits).unwrap();
            let (reassignments, _) =
                parallel::posting_reassignments(conn, index, &ops, &splits).unwrap();
            parallel::apply_posting_moves(conn, index, &reassignments).unwrap();
            parallel::remove_merge_sources(conn, index, &ops).unwrap();
            let (nearby, _) =
                parallel::select_nearby_centroids(conn, index, &ops, &splits).unwrap();
            let (moves, _) = parallel::compute_nearby_reassignments(conn, index, &nearby).unwrap();
            parallel::apply_posting_moves(conn, index, &moves).unwrap();
            Phases {
                ops,
                reassignments,
                nearby,
            }
        }

        /// Verify that assignments, postings, stats, and the head graph are mutually consistent
        /// and that no vectors were lost.
        fn check(&self) -> Snapshot {
            let txn = self.txn_index();
            let index = &self.index;

            let mut assignments = HashMap::new();
            for r in txn
                .transaction()
                .open_cursor::<i64, CentroidAssignment>(index.centroid_assignments_table_name())
                .unwrap()
            {
                let (id, a) = r.unwrap();
                assignments.insert(id, a.primary_id);
            }
            assert_eq!(
                assignments.keys().copied().collect::<HashSet<_>>(),
                self.vectors.keys().copied().collect::<HashSet<_>>(),
                "assigned records"
            );

            // Each record appears exactly once, in the posting of its assigned centroid.
            let posting_coder = index.new_posting_coder();
            let mut seen = HashSet::new();
            let mut posting_lens = HashMap::new();
            for r in txn
                .transaction()
                .open_cursor::<u32, Vec<u8>>(index.postings_table_name())
                .unwrap()
            {
                let (centroid, data) = r.unwrap();
                let block = PostingBlock::new(&data, index.posting_vector_len())
                    .expect("valid posting block");
                assert!(
                    !block.is_empty(),
                    "empty posting row for centroid {centroid}"
                );
                for (id, v) in block.iter() {
                    assert!(seen.insert(id), "record {id} in multiple postings");
                    assert_eq!(
                        assignments.get(&id),
                        Some(&centroid),
                        "record {id} posting does not match assignment"
                    );
                    assert_eq!(
                        v,
                        posting_coder
                            .encode(&self.vectors[&id])
                            .expect("finite vector")
                            .as_slice(),
                        "record {id} posting vector"
                    );
                }
                posting_lens.insert(centroid, block.len() as u32);
            }
            assert_eq!(
                seen.len(),
                assignments.len(),
                "records missing from postings"
            );

            // Stats match the postings for every centroid.
            let mut counts = HashMap::new();
            for r in txn
                .transaction()
                .open_cursor::<u32, CentroidCounts>(index.centroid_stats_table_name())
                .unwrap()
            {
                let (centroid, c) = r.unwrap();
                counts.insert(centroid, c.primary);
            }
            for (centroid, count) in counts.iter() {
                assert_eq!(
                    *count,
                    posting_lens.get(centroid).copied().unwrap_or(0),
                    "centroid {centroid} stats"
                );
            }
            for centroid in posting_lens.keys() {
                assert!(
                    counts.contains_key(centroid),
                    "centroid {centroid} has postings but no stats"
                );
            }

            // Every head vertex is reachable from the entry point and is a centroid with stats.
            let head = txn.head();
            let mut graph = head.graph().unwrap();
            let entry_point = graph.entry_point().expect("entry point").unwrap();
            let mut visited = HashSet::from([entry_point]);
            let mut queue = vec![entry_point];
            while let Some(v) = queue.pop() {
                let edges = graph
                    .edges(v)
                    .unwrap_or_else(|| panic!("edge to missing vertex {v}"))
                    .unwrap()
                    .collect::<Vec<_>>();
                for e in edges {
                    if visited.insert(e) {
                        queue.push(e);
                    }
                }
            }
            assert_eq!(
                visited.iter().map(|v| *v as u32).collect::<HashSet<_>>(),
                counts.keys().copied().collect::<HashSet<_>>(),
                "head vertices vs centroid stats"
            );

            // Every centroid is the top result when searching for its own vector.
            let mut store = head.high_fidelity_vectors().unwrap();
            let coder = store.new_coder();
            let centroids = counts
                .keys()
                .map(|c| (*c, coder.decode(store.get(*c as i64).unwrap().unwrap())))
                .collect::<HashMap<_, _>>();
            let mut searcher = GraphSearcher::new(index.config().head_search_params);
            for (centroid, v) in centroids.iter() {
                let results = searcher.search(v, head).unwrap();
                assert_eq!(
                    results.first().map(|n| n.vertex()),
                    Some(*centroid as i64),
                    "search for centroid {centroid}"
                );
            }

            Snapshot {
                assignments,
                counts,
                centroids,
            }
        }
    }

    fn cluster(rng: &mut impl RngExt, center: [f32; 2], radius: f32, n: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|_| {
                vec![
                    center[0] + rng.random_range(-radius..=radius),
                    center[1] + rng.random_range(-radius..=radius),
                ]
            })
            .collect()
    }

    fn l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(a, b)| (a - b) * (a - b)).sum()
    }

    fn nearest(centroids: &HashMap<u32, Vec<f32>>, v: &[f32]) -> u32 {
        *centroids
            .iter()
            .min_by(|a, b| l2(a.1, v).total_cmp(&l2(b.1, v)))
            .unwrap()
            .0
    }

    #[test]
    fn get_rebalance_ops_allocates_one_id() {
        let mut f = Fixture::new(VectorSimilarity::Euclidean, EdgeType::Undirected);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        for (center, n) in [
            ([0.0, 0.0], 20),
            ([4.0, 0.0], 8),
            ([0.0, 4.0], 1),
            ([4.0, 4.0], 20),
            ([8.0, 0.0], 8),
        ] {
            let c = f.add_centroid(&center);
            f.add_vectors(c, &cluster(&mut rng, center, 0.5, n));
        }
        let ops =
            parallel::get_rebalance_ops(&f.centroid_stats(), f.index.config().centroid_len_range());
        assert_eq!(
            ops,
            vec![
                RebalanceOp::Split(0, 5),
                RebalanceOp::Merge(2),
                RebalanceOp::Split(3, 6)
            ]
        );
    }

    // Splitting the only centroid also splits the head entry point.
    #[test]
    fn split_only_centroid_reuses_source_id() {
        let mut f = Fixture::new(VectorSimilarity::Euclidean, EdgeType::Undirected);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2);
        let s = f.add_centroid(&[0.0, 0.0]);
        let left = f.add_vectors(s, &cluster(&mut rng, [-2.0, 0.0], 0.5, 12));
        let right = f.add_vectors(s, &cluster(&mut rng, [2.5, 0.0], 0.5, 12));

        let stats = f.rebalance();
        assert_eq!((stats.split, stats.merged), (1, 0));

        let snap = f.check();
        assert_eq!(
            snap.counts.keys().copied().collect::<HashSet<_>>(),
            HashSet::from([s, s + 1])
        );
        // The source retains the component closest to its original vector.
        assert!(left.iter().all(|id| snap.assignments[id] == s));
        assert!(right.iter().all(|id| snap.assignments[id] == s + 1));
        assert!(l2(&snap.centroids[&s], &[-2.0, 0.0]) < 0.25);
    }

    // The component closest to the source's pre-split vector retains the source id, regardless of
    // the order produced by partitioning.
    #[test]
    fn split_closest_component_retains_source_id() {
        for seed in 0..8 {
            let mut f = Fixture::new(VectorSimilarity::Euclidean, EdgeType::Undirected);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
            let sign = if seed % 2 == 0 { 1.0 } else { -1.0 };
            let s = f.add_centroid(&[0.0, 0.0]);
            let near = f.add_vectors(s, &cluster(&mut rng, [sign * 1.0, 0.0], 0.3, 12));
            let far = f.add_vectors(s, &cluster(&mut rng, [sign * -2.0, 0.0], 0.3, 12));

            f.rebalance();
            let snap = f.check();
            assert!(
                near.iter().all(|id| snap.assignments[id] == s),
                "seed {seed}"
            );
            assert!(
                far.iter().all(|id| snap.assignments[id] == s + 1),
                "seed {seed}"
            );
        }
    }

    // Vectors closer to the source's pre-split vector than to either component must be searched.
    #[test]
    fn split_requery_uses_old_centroid() {
        let mut f = Fixture::new(VectorSimilarity::Euclidean, EdgeType::Undirected);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(3);
        let old_source = [0.0, 0.0];
        let s = f.add_centroid(&old_source);
        let n = f.add_centroid(&[0.0, 1.0]);
        f.add_vectors(n, &cluster(&mut rng, [0.0, 1.2], 0.2, 8));
        f.add_vectors(s, &cluster(&mut rng, [-2.0, 0.0], 0.3, 10));
        f.add_vectors(s, &cluster(&mut rng, [2.0, 0.0], 0.3, 10));
        let center = f.add_vectors(s, &cluster(&mut rng, [0.0, 0.0], 0.1, 4));
        let source_ids = f
            .vectors
            .keys()
            .copied()
            .filter(|id| *id >= 8)
            .collect::<Vec<_>>();

        let stats = f.rebalance();
        assert_eq!((stats.split, stats.merged), (1, 0));

        let snap = f.check();
        let new = n + 1;
        let expected_searches = source_ids
            .iter()
            .filter(|id| {
                let v = &f.vectors[id];
                let s_dist = l2(&old_source, v);
                s_dist < l2(&snap.centroids[&s], v) && s_dist < l2(&snap.centroids[&new], v)
            })
            .count();
        assert_eq!(expected_searches, center.len());
        assert_eq!(stats.split_stats.searches, expected_searches);
        for id in center {
            assert_eq!(snap.assignments[&id], n);
        }
    }

    #[test]
    fn merge_and_split_same_round() {
        let mut f = Fixture::new(VectorSimilarity::Euclidean, EdgeType::Undirected);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(4);
        let s = f.add_centroid(&[0.0, 0.0]);
        let m = f.add_centroid(&[1.5, 0.0]);
        f.add_vectors(s, &cluster(&mut rng, [-1.5, 0.0], 0.3, 10));
        let right = f.add_vectors(s, &cluster(&mut rng, [1.0, 0.0], 0.2, 10));
        let merged = f.add_vectors(m, &cluster(&mut rng, [1.3, 0.0], 0.05, 2));

        let phases = f.rebalance_phases();
        assert_eq!(
            phases.ops,
            vec![RebalanceOp::Split(s, m + 1), RebalanceOp::Merge(m)]
        );
        // The retained source is a valid target when reassigning merged vectors. Check this
        // directly as nearby reassignment may otherwise correct the assignment.
        let to_source = phases.reassignments.iter().collect::<HashSet<_>>();
        assert!(
            merged
                .iter()
                .all(|id| to_source.contains(&parallel::PostingMove {
                    record_id: *id,
                    source: m,
                    target: s,
                }))
        );

        let snap = f.check();
        assert_eq!(
            snap.counts.keys().copied().collect::<HashSet<_>>(),
            HashSet::from([s, m + 1])
        );
        assert!(right.iter().all(|id| snap.assignments[id] == s));
        assert!(merged.iter().all(|id| snap.assignments[id] == s));
        assert_eq!(snap.counts[&s], 12);
    }

    #[test]
    fn multiple_splits_same_round() {
        let mut f = Fixture::new(VectorSimilarity::Euclidean, EdgeType::Undirected);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(5);
        let a = f.add_centroid(&[0.0, 0.0]);
        let b = f.add_centroid(&[3.0, 0.0]);
        let c = f.add_centroid(&[1.5, 2.0]);
        f.add_vectors(a, &cluster(&mut rng, [-0.6, 0.0], 0.2, 10));
        f.add_vectors(a, &cluster(&mut rng, [0.6, 0.0], 0.2, 10));
        f.add_vectors(b, &cluster(&mut rng, [2.4, 0.0], 0.2, 10));
        f.add_vectors(b, &cluster(&mut rng, [3.6, 0.0], 0.2, 10));
        f.add_vectors(c, &cluster(&mut rng, [1.5, 2.0], 0.2, 8));

        let phases = f.rebalance_phases();
        assert_eq!(
            phases.ops,
            vec![RebalanceOp::Split(a, 3), RebalanceOp::Split(b, 4)]
        );
        // Retained split sources are candidates for nearby reassignment from other splits.
        assert!(phases.nearby[&a].contains(&b) && phases.nearby[&a].contains(&4));
        assert!(phases.nearby[&b].contains(&a) && phases.nearby[&b].contains(&3));

        let snap = f.check();
        assert_eq!(
            snap.counts.keys().copied().collect::<HashSet<_>>(),
            HashSet::from([a, b, c, 3, 4])
        );
        for (id, centroid) in snap.assignments.iter() {
            assert_eq!(*centroid, nearest(&snap.centroids, &f.vectors[id]));
        }
    }

    #[test]
    fn nearby_reassignment_into_reused_source() {
        let mut f = Fixture::new(VectorSimilarity::Euclidean, EdgeType::Undirected);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(6);
        let s = f.add_centroid(&[0.0, 0.0]);
        let n = f.add_centroid(&[2.5, 0.0]);
        f.add_vectors(s, &cluster(&mut rng, [-1.5, 0.0], 0.2, 10));
        f.add_vectors(s, &cluster(&mut rng, [1.0, 0.0], 0.2, 10));
        let near = f.add_vectors(n, &cluster(&mut rng, [1.6, 0.0], 0.05, 4));
        f.add_vectors(n, &cluster(&mut rng, [2.6, 0.0], 0.2, 6));

        let stats = f.rebalance();
        assert_eq!((stats.split, stats.merged), (1, 0));
        assert!(stats.split_stats.nearby_moved >= near.len());

        let snap = f.check();
        assert!(near.iter().all(|id| snap.assignments[id] == s));
        assert_eq!(snap.counts[&n], 6);
    }

    fn rebalance_until_balanced(similarity: VectorSimilarity, edge_type: EdgeType) {
        let mut f = Fixture::new(similarity, edge_type);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        // Mirror `et spann init-index`, which seeds the head with a single zero vector.
        f.add_centroid(&[0.0, 0.0]);
        let centers = (0..8)
            .map(|_| {
                [
                    rng.random_range(-4.0f32..4.0),
                    rng.random_range(-4.0f32..4.0),
                ]
            })
            .collect::<Vec<_>>();
        let range = f.index.config().centroid_len_range();
        for _ in 0..4 {
            let vectors = (0..100)
                .map(|_| {
                    let center = centers[rng.random_range(0..centers.len())];
                    let mut v = cluster(&mut rng, center, 1.0, 1).pop().unwrap();
                    if similarity.angular() {
                        let norm = l2(&v, &[0.0, 0.0]).sqrt();
                        v.iter_mut().for_each(|d| *d /= norm);
                    }
                    v
                })
                .collect::<Vec<_>>();
            f.insert_vectors(&vectors);

            for round in 0.. {
                let before = f.centroid_stats();
                let summary = BalanceSummary::new(&before, range.clone());
                if summary.below_bounds() == 0 && summary.above_bounds() == 0 {
                    break;
                }
                assert!(round < 100, "rebalance did not converge");
                let next_id = before.available_centroid_ids().next().unwrap();

                let stats = f.rebalance();
                assert_eq!(stats.split, summary.above_bounds());
                assert_eq!(stats.merged, summary.below_bounds());
                f.check();
                if stats.split > 0 {
                    // Exactly one new id is allocated per split.
                    assert_eq!(
                        f.centroid_stats().available_centroid_ids().next().unwrap(),
                        next_id + stats.split
                    );
                }
            }
        }
    }

    #[test]
    fn rebalance_until_balanced_euclidean_undirected() {
        rebalance_until_balanced(VectorSimilarity::Euclidean, EdgeType::Undirected);
    }

    #[test]
    fn rebalance_until_balanced_euclidean_directed() {
        rebalance_until_balanced(VectorSimilarity::Euclidean, EdgeType::Directed);
    }

    #[test]
    fn rebalance_until_balanced_dot_undirected() {
        rebalance_until_balanced(VectorSimilarity::Dot, EdgeType::Undirected);
    }
}
