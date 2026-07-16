use std::{ops::RangeInclusive, sync::Arc};

use rand::Rng;
use wt_mdb::{Connection, Result};

use crate::{
    spann::{centroid_stats::CentroidStats, TableIndex, TransactionIndex},
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

/// Statistics collected during a rebalance operation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RebalanceStats {
    pub merged: usize,
    pub merge_stats: MergeStats,
    pub split: usize,
    pub split_stats: SplitStats,
    /// Accumulated graph search stats for all searches performed during rebalancing.
    pub search_stats: GraphSearchStats,
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
    use vectors::{F32VectorCoder, F32VectorCoding, QueryVectorDistance};
    use wt_mdb::{Connection, Error, Result};

    use crate::{
        input::{VecVectorStore, VectorStore},
        posting_block::PostingBlock,
        spann::{
            centroid_stats::{CentroidAssignmentUpdater, CentroidCounts, CentroidStats},
            postings::BlockPostingsMut,
            rebalance::{MergeStats, RebalanceStats, SplitStats},
            CentroidAssignment, TableIndex, TransactionIndex,
        },
        vamana::{
            mutate::{delete_vector, upsert_vector},
            search::{GraphSearchStats, GraphSearcher, Options as GraphSearchOptions},
            wt::CursorVectorStore,
            GraphVectorIndex, GraphVectorStore,
        },
    };

    #[derive(Debug, Clone)]
    pub enum RebalanceOp {
        /// Merge out the centroid, globally reassigning all vectors contain in its posting.
        Merge(u32),
        /// Split the centroid into two new component centroids.
        /// Posting vectors will be assigned to new targets or globally reassigned.
        Split(u32, u32, u32),
    }

    impl RebalanceOp {
        fn source(&self) -> u32 {
            *match self {
                Self::Merge(c) => c,
                Self::Split(c, _, _) => c,
            }
        }
    }

    pub struct CentroidDistanceFactory<'a> {
        centroid_store: CursorVectorStore<'a>,
        posting_format: F32VectorCoding,
        head_coder: Option<Box<dyn F32VectorCoder>>,
    }

    impl<'a> CentroidDistanceFactory<'a> {
        pub fn new(txn_idx: &'a TransactionIndex) -> Result<Self> {
            let centroid_store = txn_idx.head().high_fidelity_vectors()?;
            let posting_format = txn_idx.index().config().posting_coder;
            let head_format = centroid_store.format();
            let head_coder = if posting_format == head_format {
                None
            } else {
                Some(head_format.coder(centroid_store.similarity(), None))
            };
            Ok(Self {
                centroid_store,
                posting_format,
                head_coder,
            })
        }

        pub fn distance_to_centroid(
            &mut self,
            centroid_id: u32,
        ) -> Result<Box<dyn QueryVectorDistance>> {
            let similarity = self.centroid_store.similarity();
            let query = self
                .centroid_store
                .get(centroid_id as i64)
                .unwrap_or_else(|| Err(Error::not_found_error()))?;
            if let Some(head_coder) = self.head_coder.as_ref() {
                let query = head_coder.decode(query);
                Ok(self
                    .posting_format
                    .query_distance_asymmetric(similarity, query, None))
            } else {
                Ok(self
                    .posting_format
                    .query_distance_symmetric(similarity, query.to_vec(), None))
            }
        }
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

        pub fn copy_posting(&mut self, record_id: i64, source: u32, target: u32) -> Result<()> {
            self.assignments
                .overwrite(record_id, CentroidAssignment::new(target))?;
            let v = self.postings.get(source, record_id)?;
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
                        it.next().unwrap() as u32,
                    ))
                } else {
                    None
                }
            })
            .collect()
    }

    /// For all split operations, generates new target centroids and inserts them in head index.
    pub fn split_update_head<R: Rng>(
        connection: &Arc<Connection>,
        index: &Arc<TableIndex>,
        ops: &[RebalanceOp],
        rng_supplier: &(impl Fn() -> R + Send + Sync),
    ) -> Result<()> {
        let target_centroids = ops.par_iter().filter(|op| matches!(op, RebalanceOp::Split(_, _, _))).map(|op| {
            if let RebalanceOp::Split(s, t0, t1) = op {
                let txn_idx = TransactionIndex::new(index, connection.begin_transaction(None)?);
                let vectors = extract_vectors(&txn_idx, *s)?;
                let mut rng = rng_supplier();
                let centroids = match crate::kmeans::balanced_binary_partition(
                    &vectors,
                    100,
                    txn_idx.index().config().min_centroid_len,
                    &mut rng,
                ) {
                    Ok(r) => r,
                    Err(r) => {
                        warn!(
                            "split_centroid: binary partition of centroid {s} (count {}) failed to converge!",
                            vectors.len()
                        );
                        r
                    }
                };
                Ok::<_, Error>((*t0, *t1, centroids))
            } else {
                unreachable!("filtered to splits");
            }
        }).collect::<Result<Vec<_>>>()?;

        // NB: deliberately done sequentially to avoid conflicts.
        let txn_idx = TransactionIndex::new(index, connection.begin_transaction(None)?);
        for (t0, t1, centroids) in target_centroids {
            upsert_vector(t0 as i64, &centroids[0], txn_idx.head())?;
            upsert_vector(t1 as i64, &centroids[1], txn_idx.head())?;
        }
        txn_idx.commit(None)
    }

    fn extract_vectors(
        txn_idx: &TransactionIndex,
        centroid_id: u32,
    ) -> Result<VecVectorStore<f32>> {
        let mut posting_cursor = txn_idx
            .transaction()
            .open_cursor::<u32, Vec<u8>>(txn_idx.index().postings_table_name())?;
        let Some(raw_posting) =
            unsafe { posting_cursor.seek_exact_unsafe(centroid_id) }.transpose()?
        else {
            return Ok(VecVectorStore::new(txn_idx.index().dimensions()));
        };
        let block = PostingBlock::new(raw_posting, txn_idx.index().posting_vector_len())
            .expect("valid posting block");

        let mut scratch = vec![0.0f32; txn_idx.index().dimensions()];
        let mut vectors = VecVectorStore::with_capacity(scratch.len(), block.len());
        if let Some(coder) = txn_idx.index().rerank_coder() {
            let mut rerank_cursor = txn_idx
                .transaction()
                .open_cursor::<i64, Vec<u8>>(txn_idx.index().raw_vectors_table_name())?;
            for (r, _) in block.iter() {
                let v = unsafe { rerank_cursor.seek_exact_unsafe(r) }
                    .unwrap_or_else(|| Err(Error::not_found_error()))?;
                coder.decode_to(v, &mut scratch);
                vectors.push(&scratch);
            }
        } else {
            let coder = txn_idx.index().new_posting_coder();
            for (_, v) in block.iter() {
                coder.decode_to(&v, &mut scratch);
                vectors.push(&scratch);
            }
        }
        Ok(vectors)
    }

    pub type TargetCentroidSourceMap = HashMap<u32, Vec<(u32, i64)>>;

    /// Generate a list of all reassignments out of ops assuming an updated head index.
    /// Assignments are grouped by _target_ centroid, listing the source centroid and vector.
    pub fn posting_reassignments(
        connection: &Arc<Connection>,
        index: &Arc<TableIndex>,
        ops: &[RebalanceOp],
    ) -> Result<(TargetCentroidSourceMap, RebalanceStats)> {
        #[derive(Debug, Clone)]
        enum Target {
            Centroid(u32),
            Query(Vec<u8>),
        }

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
                        RebalanceOp::Split(s, t0, t1) => {
                            let mut f = CentroidDistanceFactory::new(txn)?;
                            let s_distfn = f.distance_to_centroid(*s)?;
                            let t0_distfn = f.distance_to_centroid(*t0)?;
                            let t1_distfn = f.distance_to_centroid(*t1)?;
                            let postings = block
                                .iter()
                                .map(|(id, v)| {
                                    let s_dist = s_distfn.distance(v);
                                    let t0_dist = t0_distfn.distance(v);
                                    let t1_dist = t1_distfn.distance(v);
                                    let target = if s_dist < t0_dist && s_dist < t1_dist {
                                        Target::Query(v.to_vec())
                                    } else if t0_dist < t1_dist {
                                        Target::Centroid(*t0)
                                    } else {
                                        Target::Centroid(*t1)
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

        // Assign targets to any vector that does not yet have one and group by target centroid.
        let coder = index.new_posting_coder();
        let filter = ops.iter().map(|o| o.source()).collect::<HashSet<_>>();
        let (by_target, search_stats) = centroid_reassignments
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
                    let mut m = HashMap::<u32, Vec<(u32, i64)>>::new();
                    let mut s = GraphSearchStats::default();
                    for (vector_id, target) in postings {
                        let centroid = match target {
                            Target::Centroid(c) => c,
                            Target::Query(q) => {
                                let mut candidates = searcher.search_with_options(
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
                        m.entry(centroid).or_default().push((source, vector_id));
                    }
                    Ok::<(HashMap<u32, Vec<(u32, i64)>>, GraphSearchStats), Error>((m, s))
                },
            )
            .try_reduce(
                || (HashMap::new(), GraphSearchStats::default()),
                |(mut acc, mut acc_stats), (partial, partial_stats)| {
                    for (centroid, list) in partial {
                        acc.entry(centroid).or_default().extend_from_slice(&list);
                    }
                    acc_stats += partial_stats;
                    Ok((acc, acc_stats))
                },
            )?;
        stats.search_stats = search_stats;
        Ok((by_target, stats))
    }

    /// Apply all posting reassignments then remove the source centroid ids.
    pub fn apply_posting_reassignments(
        connection: &Arc<Connection>,
        index: &Arc<TableIndex>,
        reassignments: &HashMap<u32, Vec<(u32, i64)>>,
        ops: &[RebalanceOp],
    ) -> Result<()> {
        reassignments
            .par_iter()
            .try_for_each(|(target, reassignments)| {
                let txn = TransactionIndex::new(index, connection.begin_transaction(None)?);
                let mut updater = PostingUpdater::new(&txn)?;
                for (source, vector_id) in reassignments {
                    updater.copy_posting(*vector_id, *source, *target)?
                }
                updater.flush()?;
                txn.commit(None)?;
                Ok::<_, Error>(())
            })?;

        // NB: deliberately sequential to avoid high conflict rate.
        let txn_idx = TransactionIndex::new(index, connection.begin_transaction(None)?);
        for op in ops {
            let mut stats_cursor = txn_idx
                .transaction()
                .open_cursor::<u32, CentroidCounts>(index.centroid_stats_table_name())?;
            stats_cursor.remove(op.source()).or_else(|e| {
                if e == Error::not_found_error() {
                    Ok(())
                } else {
                    Err(e)
                }
            })?;
            let mut posting_cursor = txn_idx
                .transaction()
                .open_cursor::<u32, Vec<u8>>(index.postings_table_name())?;
            posting_cursor.remove(op.source()).or_else(|e| {
                if e == Error::not_found_error() {
                    Ok(())
                } else {
                    Err(e)
                }
            })?;
            delete_vector(op.source() as i64, txn_idx.head())?;
        }
        txn_idx.commit(None)
    }

    /// For all split ops produce a map from each nearby centroid that should be considered for
    /// reassignment to a list of potential targets that should be considered.
    pub fn select_nearby_centroids(
        connection: &Arc<Connection>,
        index: &Arc<TableIndex>,
        ops: &[RebalanceOp],
    ) -> Result<(HashMap<u32, Vec<u32>>, GraphSearchStats)> {
        let filter_centroids = ops.iter().map(RebalanceOp::source).collect::<HashSet<_>>();
        let target_to_nearby = ops
            .par_iter()
            .filter(|op| matches!(op, RebalanceOp::Split(_, _, _)))
            .flat_map(|op| {
                let RebalanceOp::Split(_, t0, t1) = op else {
                    unreachable!("filtered to splits")
                };
                [*t0, *t1]
            })
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
                    let candidates = searcher.search_with_options(
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

    #[derive(Debug, Copy, Clone)]
    pub struct PostingMove {
        record_id: i64,
        source: u32,
        target: u32,
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
                    let mut f = CentroidDistanceFactory::new(txn)?;
                    let postings = updater.read_centroid(*nearby)?;
                    let nearby_distfn = f.distance_to_centroid(*nearby)?;
                    let targets_distfn = targets
                        .iter()
                        .map(|c| f.distance_to_centroid(*c))
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

    /// Apply all computed nearby reassignments to the index.
    pub fn apply_nearby_reassignments(
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
    parallel::split_update_head(connection, index, &ops, rng_supplier)?;
    let (reassignments, mut stats) = parallel::posting_reassignments(connection, index, &ops)?;
    parallel::apply_posting_reassignments(connection, index, &reassignments, &ops)?;
    let (nearby_to_targets, _) = parallel::select_nearby_centroids(connection, index, &ops)?;
    let (nearby_reassignments, nearby_stats) =
        parallel::compute_nearby_reassignments(connection, index, &nearby_to_targets)?;
    stats += nearby_stats;
    parallel::apply_nearby_reassignments(connection, index, &nearby_reassignments)?;
    Ok(stats)
}
