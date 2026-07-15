use std::{collections::HashSet, ops::RangeInclusive, sync::Arc};

use rand::Rng;
use wt_mdb::{Connection, Error, Kind, Result, WiredTigerError};

use crate::{
    spann::{
        centroid_stats::{CentroidAssignmentUpdater, CentroidCounts, CentroidStats},
        postings::BlockPostingsMut,
        CentroidAssignment, TableIndex, TransactionIndex,
    },
    vamana::{mutate::delete_vector, search::GraphSearcher, search::Options as GraphSearchOptions},
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
}

impl Add for RebalanceStats {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            merged: self.merged + rhs.merged,
            merge_stats: self.merge_stats + rhs.merge_stats,
            split: self.split + rhs.split,
            split_stats: self.split_stats + rhs.split_stats,
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
        }
    }
}

impl AddAssign<SplitStats> for RebalanceStats {
    fn add_assign(&mut self, rhs: SplitStats) {
        *self = *self + rhs;
    }
}

/// Remove `centroid_id` and merge each of its vectors into the next closest centroid.
pub fn merge_centroid(
    txn_idx: &TransactionIndex,
    centroid_id: usize,
    len: usize,
) -> Result<MergeStats> {
    // Collect all of the vectors for the centroid to merge.
    let index = Arc::clone(txn_idx.index());
    let posting_cursor = txn_idx
        .transaction()
        .open_cursor::<u32, Vec<u8>>(index.postings_table_name())?;
    let mut postings = BlockPostingsMut::new(posting_cursor, index.posting_vector_len());
    let vectors = postings.read_centroid(centroid_id as u32)?;
    postings.remove_centroid(centroid_id as u32)?;
    assert_eq!(
        vectors.len(),
        len,
        "merge_centroid of {centroid_id} expected {len} vectors; actual {}",
        vectors.len()
    );

    // Remove the centroid from the graph.
    delete_vector(centroid_id as i64, txn_idx.head())?;

    // If the centroid is already empty then there is nothing to do.
    if vectors.is_empty() {
        txn_idx
            .transaction()
            .open_cursor::<u32, CentroidCounts>(index.centroid_stats_table_name())?
            .remove(centroid_id as u32)?;
        return Ok(MergeStats::default());
    }

    // Query the head index for each vector and assign a new centroid.
    let coder = index.new_posting_coder();
    let removed_vectors = vectors.len();
    let mut searcher = GraphSearcher::new(index.config().head_search_params);
    let mut float_vector = vec![0.0f32; index.head_config().config().dimensions.get()];
    let mut unique_centroids = HashSet::new();
    let mut assignment_updater = CentroidAssignmentUpdater::new(txn_idx)?;
    for (record_id, vector) in vectors {
        coder.decode_to(&vector, &mut float_vector);
        // TODO: seed the search with the existing assignments for this record; reduce budget.
        let candidates = searcher.search_with_options(
            &float_vector,
            GraphSearchOptions::with_filter(|i| i != centroid_id as i64),
            txn_idx.head(),
        )?;
        let new_assignments = CentroidAssignment::new(candidates[0].vertex() as u32);
        let old_assignments = assignment_updater.update(record_id, new_assignments)?;
        move_postings(
            record_id,
            &vector,
            old_assignments,
            new_assignments,
            &mut postings,
        )?;
        unique_centroids.insert(new_assignments.primary_id);
    }

    postings.flush()?;
    assignment_updater.flush()?;
    txn_idx
        .transaction()
        .open_cursor::<u32, CentroidCounts>(index.centroid_stats_table_name())?
        .remove(centroid_id as u32)?;

    Ok(MergeStats {
        moved_vectors: removed_vectors,
        unique_centroids: unique_centroids.len(),
    })
}

/// Split `centroid_id` in two, creating two new centroids.
///
/// `rng` is used to partition the input centroid into two clusters.
pub fn split_centroid(
    txn_idx: &TransactionIndex,
    centroid_id: usize,
    rng: &mut impl Rng,
) -> Result<SplitStats> {
    let mut updater = split::PostingUpdater::new(txn_idx)?;
    let centroid_split = split::top_half(&mut updater, centroid_id as u32, rng)?;

    let mut searches = 0;
    let posting_coder = txn_idx.index().new_posting_coder();
    let mut searcher = GraphSearcher::new(txn_idx.index().config().head_search_params);
    for target in &centroid_split.targets {
        for &record_id in &target.to_reassign {
            searches += split::bottom_half_reassign_one_with_search(
                target.centroid_id,
                record_id,
                posting_coder.as_ref(),
                &mut searcher,
                &mut updater,
            )?;
        }
    }

    let mut nearby_stats = SplitStats::default();
    for target in &centroid_split.targets {
        let nearby = split::bottom_half_find_nearby_centroids(txn_idx, target.centroid_id)?;
        for nearby_centroid_id in nearby {
            nearby_stats += split::bottom_half_nearby_reassign_centroid(
                &mut updater,
                nearby_centroid_id,
                target.centroid_id,
            )?;
        }
    }

    updater.flush()?;
    txn_idx
        .transaction()
        .open_cursor::<u32, CentroidCounts>(txn_idx.index().centroid_stats_table_name())?
        .remove(centroid_id as u32)?;

    Ok(SplitStats {
        moved_vectors: centroid_split.stats.moved_vectors,
        searches,
        nearby_seen: nearby_stats.nearby_seen,
        nearby_moved: nearby_stats.nearby_moved,
    })
}

/// Target centroid produced by split of a centroid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CentroidSplitTarget {
    /// The centroid_id the split was written to.
    pub centroid_id: u32,
    /// Vectors within the block that should be reassigned by search. This means that the vector
    /// is closer to the previous centroid than it is to this target centroid.
    pub to_reassign: Vec<i64>,
}

impl CentroidSplitTarget {
    fn new(centroid_id: u32) -> Self {
        Self {
            centroid_id,
            to_reassign: vec![],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CentroidSplit {
    /// Stats generated for the split.
    pub stats: SplitStats,
    /// The two output split targets. This contains additional information for bottom half use.
    pub targets: [CentroidSplitTarget; 2],
}

impl CentroidSplit {
    fn new(len: usize, targets: [u32; 2]) -> Self {
        Self {
            stats: SplitStats {
                moved_vectors: len,
                ..Default::default()
            },
            targets: [
                CentroidSplitTarget::new(targets[0]),
                CentroidSplitTarget::new(targets[1]),
            ],
        }
    }
}

/// Run the top half of the centroid split operation.
///
/// The top half clusters the contents of the centroid and produces two new centroids. Every
/// vector assigned to the centroid is reassigned to one of the new centroids and the head index
/// is updated.
///
/// This method will only mutate the head index and the centroid to split as well as assignments
/// for all vectors in the centroid. This mutation is not sufficient to ensure that vectors are
/// assigned to their closest centroid; callers should invoke the bottom half to finish rebalancing.
///
/// Returns NOT_FOUND if `centroid_id` cannot be found in `index` -- this may mean that another
/// thread has split this centroid already.
pub fn split_centroid_top_half(
    connection: &Arc<Connection>,
    index: &Arc<TableIndex>,
    centroid_id: u32,
    rng: &mut impl Rng,
) -> Result<CentroidSplit> {
    retry_on_rollback(connection, index, |txn_idx| {
        let centroid_split = {
            let mut updater = split::PostingUpdater::new(&txn_idx)?;
            let centroid_split = split::top_half(&mut updater, centroid_id, rng)?;
            updater.flush().map(|()| centroid_split)?
        };
        txn_idx
            .transaction()
            .open_cursor::<u32, CentroidCounts>(txn_idx.index().centroid_stats_table_name())?
            .remove(centroid_id)
            .or_else(|e| {
                if e == Error::not_found_error() {
                    Ok(())
                } else {
                    Err(e)
                }
            })?;
        txn_idx.commit(None).map(|_| centroid_split)
    })
}

/// Run the bottom half of the centroid split operation.
///
/// This should be invoked once for each centroid produced by the split. It will reassign any
/// vectors in the centroid that may have a closer centroid in the broader pool and also check if
/// any vectors from nearby centroids ought to be reassigned. This may be broken into many smaller
/// transactions to minimize the likelihood of conflicts.
///
/// Returns NOT_FOUND if `split_target.centroid_id` cannot be found in `index` -- this may mean that
/// another thread has elected to merge or split this centroid.
pub fn split_centroid_bottom_half(
    connection: &Arc<Connection>,
    index: &Arc<TableIndex>,
    split_target: CentroidSplitTarget,
) -> Result<SplitStats> {
    let mut split_stats = SplitStats::default();

    let mut searcher = GraphSearcher::new(index.config().head_search_params);
    let posting_coder = index.new_posting_coder();
    for record_id in split_target.to_reassign {
        split_stats.searches += retry_on_rollback(connection, index, |txn_idx| {
            let mut updater = split::PostingUpdater::new(&txn_idx)?;
            let r = split::bottom_half_reassign_one_with_search(
                split_target.centroid_id,
                record_id,
                posting_coder.as_ref(),
                &mut searcher,
                &mut updater,
            )?;
            updater.flush()?;
            txn_idx.commit(None).map(|()| r)
        })?;
    }

    let nearby_clusters: Vec<u32> = {
        let txn_idx = TransactionIndex::new(index, connection.begin_transaction(None)?);
        split::bottom_half_find_nearby_centroids(&txn_idx, split_target.centroid_id)?
    };

    for nearby_centroid_id in nearby_clusters {
        split_stats += retry_on_rollback(connection, index, |txn_idx| {
            let stats = {
                let mut updater = split::PostingUpdater::new(&txn_idx)?;
                let stats = split::bottom_half_nearby_reassign_centroid(
                    &mut updater,
                    nearby_centroid_id,
                    split_target.centroid_id,
                )?;
                updater.flush().map(|()| stats)?
            };
            txn_idx.commit(None).map(|()| stats)
        })?
    }

    Ok(split_stats)
}

mod split {
    use rand::Rng;
    use tracing::warn;
    use vectors::{F32VectorCoder, F32VectorCoding, QueryVectorDistance};
    use wt_mdb::{Error, Result};

    use crate::input::VecVectorStore;
    use crate::spann::{
        centroid_stats::CentroidAssignmentUpdater, postings::BlockPostingsMut, CentroidAssignment,
        TransactionIndex,
    };
    use crate::vamana::wt::CursorVectorStore;
    use crate::vamana::{
        mutate::{delete_vector, upsert_vector},
        search::GraphSearcher,
        search::Options as GraphSearchOptions,
        GraphVectorIndex, GraphVectorStore,
    };

    use super::{CentroidSplit, SplitStats};

    pub fn top_half(
        updater: &mut PostingUpdater<'_>,
        centroid_id: u32,
        rng: &mut impl Rng,
    ) -> Result<CentroidSplit> {
        let vectors = updater.read_centroid(centroid_id)?;
        if vectors.is_empty() {
            return Err(Error::not_found_error());
        }

        let c0_dist_fn = updater.distance_to_centroid(centroid_id)?;

        // Remove postings and vector quickly to trigger OCC rollbacks.
        let target_centroid_ids = updater.postings.next_centroid_id().map(|x| [x, x + 1])?;
        updater.postings.remove_centroid(centroid_id)?;
        delete_vector(centroid_id as i64, updater.txn_idx.head())?;

        // Partition the postings for `centroid_id` into two new postings and insert into index.
        let centroids = partition_postings(
            updater.txn_idx,
            centroid_id,
            vectors.iter().map(|x| x.1.as_slice()),
            rng,
        );
        upsert_vector(
            target_centroid_ids[0] as i64,
            &centroids[0],
            updater.txn_idx.head(),
        )?;
        upsert_vector(
            target_centroid_ids[1] as i64,
            &centroids[1],
            updater.txn_idx.head(),
        )?;

        let c1_dist_fn = updater.distance_to_centroid(target_centroid_ids[0])?;
        let c2_dist_fn = updater.distance_to_centroid(target_centroid_ids[1])?;

        let mut centroid_split = CentroidSplit::new(vectors.len(), target_centroid_ids);
        for (record_id, vector) in vectors {
            let c0_dist = c0_dist_fn.distance(&vector);
            let c1_dist = c1_dist_fn.distance(&vector);
            let c2_dist = c2_dist_fn.distance(&vector);
            let split_centroid_id = if c1_dist < c2_dist {
                if c0_dist < c1_dist {
                    centroid_split.targets[0].to_reassign.push(record_id)
                }
                centroid_split.targets[0].centroid_id
            } else {
                if c0_dist < c2_dist {
                    centroid_split.targets[1].to_reassign.push(record_id)
                }
                centroid_split.targets[1].centroid_id
            };

            updater.move_posting(record_id, &vector, centroid_id, split_centroid_id)?
        }

        Ok(centroid_split)
    }

    fn partition_postings<'a>(
        txn_idx: &TransactionIndex,
        centroid_id: u32,
        vectors: impl ExactSizeIterator<Item = &'a [u8]>,
        rng: &mut impl Rng,
    ) -> VecVectorStore<f32> {
        let len = vectors.len();
        let posting_coder = txn_idx
            .index()
            .config()
            .posting_coder
            .coder(txn_idx.index().head_config().config().similarity, None);
        let mut scratch_vector =
            vec![0.0f32; txn_idx.index().head_config().config().dimensions.get()];
        let mut clustering_vectors = VecVectorStore::with_capacity(scratch_vector.len(), len);
        for v in vectors {
            posting_coder.decode_to(v, &mut scratch_vector);
            clustering_vectors.push(&scratch_vector);
        }

        match crate::kmeans::balanced_binary_partition(
            &clustering_vectors,
            100,
            txn_idx.index().config().min_centroid_len,
            rng,
        ) {
            Ok(r) => r,
            Err(r) => {
                warn!(
                    "split_centroid: binary partition of centroid {centroid_id} (count {}) failed to converge!",
                    len
                );
                r
            }
        }
    }

    pub fn bottom_half_reassign_one_with_search(
        centroid_id: u32,
        record_id: i64,
        posting_coder: &dyn F32VectorCoder,
        searcher: &mut GraphSearcher,
        updater: &mut PostingUpdater<'_>,
    ) -> Result<usize> {
        // If the target posting can't be found, then skip it.
        // There's a broader case where the centroid has been split again and in this case we
        // could skip the rest of the function.
        let posting = match updater.postings.get(centroid_id, record_id) {
            Err(e) if e == Error::not_found_error() => return Ok(0usize),
            result => result,
        }?;
        let query = posting_coder.decode(&posting);
        let candidates = searcher.search(&query, updater.txn_idx.head())?;
        updater
            .move_posting(
                record_id,
                &posting,
                centroid_id,
                candidates[0].vertex() as u32,
            )
            .map(|()| 1usize)
    }

    pub fn bottom_half_find_nearby_centroids(
        txn_idx: &TransactionIndex,
        centroid_id: u32,
    ) -> Result<Vec<u32>> {
        let mut store = txn_idx.head().high_fidelity_vectors()?;
        let query = store.new_coder().decode(
            store
                .get(centroid_id as i64)
                .unwrap_or_else(|| Err(Error::not_found_error()))?,
        );
        let mut searcher = GraphSearcher::new(txn_idx.index().config().head_search_params);
        let mut candidates = searcher.search_with_options(
            &query,
            GraphSearchOptions::with_filter(|i| i != centroid_id as i64),
            txn_idx.head(),
        )?;
        candidates.truncate(64);
        Ok(candidates.into_iter().map(|n| n.vertex() as u32).collect())
    }

    /// Moves vectors from `nearby_centroid_id` to `target_centroid_id` if they are closer to the
    /// target.
    pub fn bottom_half_nearby_reassign_centroid(
        updater: &mut PostingUpdater<'_>,
        nearby_centroid_id: u32,
        target_centroid_id: u32,
    ) -> Result<SplitStats> {
        let mut stats = SplitStats::default();
        if updater.centroid_empty(nearby_centroid_id)?
            || updater.centroid_empty(target_centroid_id)?
        {
            return Ok(stats);
        }

        let nearby_dist_fn = updater.distance_to_centroid(nearby_centroid_id)?;
        let target_dist_fn = updater.distance_to_centroid(target_centroid_id)?;
        for (record_id, vector) in updater.read_centroid(nearby_centroid_id)? {
            stats.nearby_seen += 1;

            if target_dist_fn.distance(&vector) < nearby_dist_fn.distance(&vector) {
                updater.move_posting(record_id, &vector, nearby_centroid_id, target_centroid_id)?;
            }
        }

        Ok(stats)
    }

    pub struct PostingUpdater<'a> {
        txn_idx: &'a TransactionIndex,
        postings: BlockPostingsMut<'a>,
        assignments: CentroidAssignmentUpdater<'a>,
        centroid_store: CursorVectorStore<'a>,
        posting_format: F32VectorCoding,
        head_coder: Option<Box<dyn F32VectorCoder>>,
    }

    impl<'a> PostingUpdater<'a> {
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
                txn_idx,
                postings: BlockPostingsMut::from_txn(txn_idx)?,
                assignments: CentroidAssignmentUpdater::new(txn_idx)?,
                centroid_store,
                posting_format,
                head_coder,
            })
        }

        fn distance_to_centroid(
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

        pub fn centroid_empty(&mut self, centroid_id: u32) -> Result<bool> {
            self.assignments.centroid_len(centroid_id).map(|c| c == 0)
        }

        pub fn read_centroid(&mut self, centroid_id: u32) -> Result<Vec<(i64, Vec<u8>)>> {
            self.postings.read_centroid(centroid_id)
        }

        pub fn move_posting(
            &mut self,
            record_id: i64,
            vector: &[u8],
            from: u32,
            to: u32,
        ) -> Result<()> {
            let old = self
                .assignments
                .update(record_id, CentroidAssignment::new(to))?;
            assert_eq!(old.primary_id, from);
            self.postings.remove(from, record_id)?;
            self.postings.insert(to, record_id, vector)
        }

        pub fn flush(mut self) -> Result<()> {
            self.postings
                .flush()
                .and_then(|()| self.assignments.flush())
        }
    }
}

fn retry_on_rollback<R>(
    connection: &Arc<Connection>,
    index: &Arc<TableIndex>,
    mut op: impl FnMut(TransactionIndex) -> Result<R>,
) -> Result<R> {
    loop {
        match op(TransactionIndex::new(
            index,
            connection.begin_transaction(None)?,
        )) {
            Err(e) if e.kind() == Kind::WiredTiger(WiredTigerError::Rollback) => continue,
            result => break result,
        }
    }
}

fn move_postings(
    record_id: i64,
    vector: &[u8],
    old_assignment: CentroidAssignment,
    new_assignment: CentroidAssignment,
    postings: &mut BlockPostingsMut<'_>,
) -> Result<usize> {
    if old_assignment.primary_id != new_assignment.primary_id {
        postings.remove(old_assignment.primary_id, record_id)?;
        postings.insert(new_assignment.primary_id, record_id, vector)?;
        Ok(1)
    } else {
        Ok(0)
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
