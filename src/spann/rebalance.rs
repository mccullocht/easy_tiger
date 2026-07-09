use std::{collections::HashSet, num::NonZero, ops::RangeInclusive, sync::Arc};

use rand::Rng;
use tracing::warn;
use wt_mdb::{Connection, Error, Result, WiredTigerError};

use crate::{
    input::VecVectorStore,
    kmeans,
    spann::{
        CentroidAssignment, TableIndex, TransactionIndex,
        centroid_stats::{CentroidAssignmentUpdater, CentroidCounts, CentroidStats},
        postings::BlockPostingsMut,
    },
    vamana::{
        GraphVectorIndex, GraphVectorStore,
        mutate::{delete_vector, upsert_vector},
        search::GraphSearcher,
    },
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
    /// Number of unique centroids that split vectors were reassigned to.
    pub unique_centroids: usize,
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
            unique_centroids: self.unique_centroids + rhs.unique_centroids,
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
        let candidates = searcher.search_with_filter(
            &float_vector,
            |i| i != centroid_id as i64,
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

/// Split `centroid_id` in two, creating a `next_centroid_id`.
///
/// `rng` is used to partition the input centroid into two clusters.
pub fn split_centroid(
    txn_idx: &TransactionIndex,
    centroid_id: usize,
    target_centroid_ids: (usize, usize),
    len: usize,
    rng: &mut impl Rng,
) -> Result<SplitStats> {
    let posting_cursor = txn_idx
        .transaction()
        .open_cursor::<u32, Vec<u8>>(txn_idx.index().postings_table_name())?;
    let mut postings = BlockPostingsMut::new(posting_cursor, txn_idx.index().posting_vector_len());
    let vectors = postings.read_centroid(centroid_id as u32)?;
    postings.remove_centroid(centroid_id as u32)?;
    assert_eq!(
        vectors.len(),
        len,
        "split_centroid of {centroid_id} expected {len} vectors; actual {}",
        vectors.len()
    );

    // Unpack all of the vectors as floats and split into two clusters.
    let posting_format = txn_idx.index().config().posting_coder;
    let similarity = txn_idx.index().head_config().config().similarity;
    let posting_coder = posting_format.coder(similarity, None);
    let mut scratch_vector = vec![0.0f32; txn_idx.index().head_config().config().dimensions.get()];
    let mut clustering_vectors = VecVectorStore::with_capacity(scratch_vector.len(), vectors.len());
    for (_, vector) in vectors.iter() {
        posting_coder.decode_to(vector, &mut scratch_vector);
        clustering_vectors.push(&scratch_vector);
    }

    let centroids = match kmeans::balanced_binary_partition(
        &clustering_vectors,
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

    // Extract the original centroid vector from the head index and delete original centroid.
    let mut head_vectors = txn_idx.head().high_fidelity_vectors()?;
    let head_format = head_vectors.format();
    let head_coder = head_vectors.new_coder();
    let original_centroidq = head_vectors
        .get(centroid_id as i64)
        .unwrap_or(Err(Error::not_found_error()))?;
    let original_centroid = head_coder.decode(original_centroidq);
    delete_vector(centroid_id as i64, txn_idx.head())?;

    // For each vector if it is closer to the original centroid than either of the new centroids
    // then query the whole index to select a new assignment. Otherwise assign it to the closest
    // of the two new centroids.
    let mut params = txn_idx.index().config().head_search_params;
    // TODO: figure out if nearby update beam width needs to be configurable.
    params.beam_width = NonZero::new(64).unwrap();
    let mut searcher = GraphSearcher::new(params);
    let nearby_clusters = searcher.search(&original_centroid, txn_idx.head())?;

    // Write the new centroids back into the index.
    upsert_vector(target_centroid_ids.0 as i64, &centroids[0], txn_idx.head())?;
    upsert_vector(target_centroid_ids.1 as i64, &centroids[1], txn_idx.head())?;

    let mut assignment_updater = CentroidAssignmentUpdater::new(txn_idx)?;
    let c0_dist_fn = if head_format == posting_format {
        posting_format.query_distance_symmetric(similarity, original_centroidq, None)
    } else {
        posting_format.query_distance_asymmetric(similarity, original_centroid, None)
    };
    let c1_dist_fn = posting_format.query_distance_symmetric(
        similarity,
        posting_coder.encode(&centroids[0]),
        None,
    );
    let c2_dist_fn = posting_format.query_distance_symmetric(
        similarity,
        posting_coder.encode(&centroids[1]),
        None,
    );
    let mut searches = 0;
    let moved_vectors = vectors.len();
    // Re-use searcher with the unmodified head search params for vector reassignment.
    searcher = GraphSearcher::new(txn_idx.index().config().head_search_params);
    let mut float_vector = vec![0.0f32; txn_idx.index().head_config().config().dimensions.get()];
    let mut unique_centroids = HashSet::new();
    for (record_id, vector) in vectors {
        let c0_dist = c0_dist_fn.distance(&vector);
        let c1_dist = c1_dist_fn.distance(&vector);
        let c2_dist = c2_dist_fn.distance(&vector);
        let new_assignments = if c0_dist <= c1_dist && c0_dist <= c2_dist {
            searches += 1;
            posting_coder.decode_to(&vector, &mut float_vector);
            let candidates = searcher.search_with_filter(
                &float_vector,
                |i| i != centroid_id as i64,
                txn_idx.head(),
            )?;
            CentroidAssignment::new(candidates[0].vertex() as u32)
        } else {
            let updated_centroid_id = if c1_dist < c2_dist {
                target_centroid_ids.0
            } else {
                target_centroid_ids.1
            };
            CentroidAssignment::new(updated_centroid_id as u32)
        };

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

    let mut nearby_seen = 0;
    let mut nearby_moved = 0;
    // For a list of nearby centroids, examine all vectors and reassign them if they are closer
    // to one of the new centroids than they are to the current centroid.
    for n in nearby_clusters {
        let nearby_centroid_id = n.vertex() as u32;
        let mut head_vectors = txn_idx.head().high_fidelity_vectors()?;
        let head_coder = head_vectors.new_coder();
        let nearby_centroid = head_vectors
            .get(nearby_centroid_id as i64)
            .unwrap_or(Err(Error::not_found_error()))?;
        let c0_dist_fn = if head_format == posting_format {
            posting_format.query_distance_symmetric(similarity, nearby_centroid, None)
        } else {
            posting_format.query_distance_asymmetric(
                similarity,
                head_coder.decode(nearby_centroid),
                None,
            )
        };

        for (record_id, vector) in postings.read_centroid(nearby_centroid_id)? {
            let c0_dist = c0_dist_fn.distance(&vector);
            let c1_dist = c1_dist_fn.distance(&vector);
            let c2_dist = c2_dist_fn.distance(&vector);
            nearby_seen += 1;

            let assigned_centroid_id = if c1_dist < c0_dist {
                target_centroid_ids.0 as u32
            } else if c2_dist < c0_dist {
                target_centroid_ids.1 as u32
            } else {
                continue;
            };

            let new_assignments = CentroidAssignment::new(assigned_centroid_id);
            let old_assignments = assignment_updater.update(record_id, new_assignments)?;
            nearby_moved += move_postings(
                record_id,
                &vector,
                old_assignments,
                new_assignments,
                &mut postings,
            )?;
        }
    }

    postings.flush()?;
    assignment_updater.flush()?;
    txn_idx
        .transaction()
        .open_cursor::<u32, CentroidCounts>(txn_idx.index().centroid_stats_table_name())?
        .remove(centroid_id as u32)?;

    Ok(SplitStats {
        moved_vectors,
        searches,
        unique_centroids: unique_centroids.len(),
        nearby_seen,
        nearby_moved,
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
    // XXX should TableIndex contain a centroid_id allocator? If it doesn't we will frequently hit
    // conflicts when two splits happen concurrently.
    // XXX an alt is to lock when performing the top half. might be necessary any way since it's
    // hard to atomize head updates.
    retry_on_rollback(connection, index, |txn_idx| {
        let centroid_split = split::top_half(&txn_idx, centroid_id, rng)?;
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
            let mut postings = BlockPostingsMut::from_txn(&txn_idx)?;

            // If the target posting can't be found, then skip it.
            // There's a broader case where the centroid has been split again and in this case we
            // could skip the rest of the function.
            let posting = match postings.get(split_target.centroid_id, record_id) {
                Err(e) if e == Error::not_found_error() => return Ok(0usize),
                result => result,
            }?;
            let query = posting_coder.decode(&posting);
            let candidates = searcher.search(&query, txn_idx.head())?;
            let new_assignments = CentroidAssignment::new(candidates[0].vertex() as u32);

            let mut assignment_updater = CentroidAssignmentUpdater::new(&txn_idx)?;
            let old_assignments = assignment_updater.update(record_id, new_assignments)?;
            move_postings(
                record_id,
                &posting,
                old_assignments,
                new_assignments,
                &mut postings,
            )?;

            postings.flush()?;
            drop(postings);
            assignment_updater.flush()?;
            drop(assignment_updater);

            txn_idx.commit(None).map(|()| 1usize)
        })?;
    }

    let nearby_clusters: Vec<u32> = {
        let txn_idx = TransactionIndex::new(index, connection.begin_transaction(None)?);
        let mut store = txn_idx.head().high_fidelity_vectors()?;
        let query = store.new_coder().decode(
            store
                .get(split_target.centroid_id as i64)
                .unwrap_or(Err(Error::not_found_error()))?,
        );
        let mut searcher = GraphSearcher::new(txn_idx.index().config().head_search_params);
        let mut candidates = searcher.search_with_filter(
            &query,
            |i| i != split_target.centroid_id as i64,
            txn_idx.head(),
        )?;
        candidates.truncate(64);
        candidates.into_iter().map(|n| n.vertex() as u32).collect()
    };

    for nearby_centroid_id in nearby_clusters {
        split_stats += retry_on_rollback(connection, index, |txn_idx| {
            split::bottom_half_nearby_reassign(
                &txn_idx,
                split_target.centroid_id,
                nearby_centroid_id,
            )
            .and_then(|s| txn_idx.commit(None).map(|()| s))
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
        CentroidAssignment, TransactionIndex,
        centroid_stats::{CentroidAssignmentUpdater, CentroidCounts},
        postings::BlockPostingsMut,
    };
    use crate::vamana::wt::CursorVectorStore;
    use crate::vamana::{
        GraphVectorIndex, GraphVectorStore,
        mutate::{delete_vector, upsert_vector},
    };

    use super::{CentroidSplit, SplitStats, move_postings};

    pub fn top_half(
        txn_idx: &TransactionIndex,
        centroid_id: u32,
        rng: &mut impl Rng,
    ) -> Result<CentroidSplit> {
        let mut postings = BlockPostingsMut::new(
            txn_idx
                .transaction()
                .open_cursor::<u32, Vec<u8>>(txn_idx.index().postings_table_name())?,
            txn_idx.index().posting_vector_len(),
        );
        let vectors = postings.read_centroid(centroid_id)?;
        if vectors.is_empty() {
            return Err(Error::WiredTiger(wt_mdb::WiredTigerError::NotFound));
        }

        let mut distance_factory = PostingDistanceFactory::new(txn_idx)?;
        let c0_dist_fn = distance_factory.query_distance(centroid_id)?;

        // Remove postings and vector quickly to trigger OCC rollbacks.
        let target_centroid_ids = postings.next_centroid_id().map(|x| [x, x + 1])?;
        postings.remove_centroid(centroid_id)?;
        delete_vector(centroid_id as i64, txn_idx.head())?;

        // Partition the postings for `centroid_id` into two new postings and insert into index.
        let centroids = partition_postings(
            txn_idx,
            centroid_id,
            vectors.iter().map(|x| x.1.as_slice()),
            rng,
        );
        upsert_vector(target_centroid_ids[0] as i64, &centroids[0], txn_idx.head())?;
        upsert_vector(target_centroid_ids[1] as i64, &centroids[1], txn_idx.head())?;

        let c1_dist_fn = distance_factory.query_distance(target_centroid_ids[0])?;
        let c2_dist_fn = distance_factory.query_distance(target_centroid_ids[1])?;

        let mut centroid_split = CentroidSplit::new(vectors.len(), target_centroid_ids);
        let mut assignment_updater = CentroidAssignmentUpdater::new(txn_idx)?;
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

            let new_assignments = CentroidAssignment::new(split_centroid_id);

            let old_assignments = assignment_updater.update(record_id, new_assignments)?;
            move_postings(
                record_id,
                &vector,
                old_assignments,
                new_assignments,
                &mut postings,
            )?;
        }

        postings.flush()?;
        assignment_updater.flush()?;
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

    pub fn bottom_half_nearby_reassign(
        txn_idx: &TransactionIndex,
        target_centroid_id: u32,
        nearby_centroid_id: u32,
    ) -> Result<SplitStats> {
        let mut stats = SplitStats::default();
        let mut postings = BlockPostingsMut::from_txn(txn_idx)?;
        if postings.centroid_len(nearby_centroid_id)? == 0
            || postings.centroid_len(target_centroid_id)? == 0
        {
            return Ok(stats);
        }

        let mut distance_factory = PostingDistanceFactory::new(txn_idx)?;

        let c0_dist_fn = match distance_factory.query_distance(nearby_centroid_id) {
            Err(e) if e == Error::not_found_error() => return Ok(stats),
            result => result,
        }?;
        let c1_dist_fn = match distance_factory.query_distance(target_centroid_id) {
            Err(e) if e == Error::not_found_error() => return Ok(stats),
            result => result,
        }?;

        let mut assignment_updater = CentroidAssignmentUpdater::new(txn_idx)?;
        for (record_id, vector) in postings.read_centroid(nearby_centroid_id)? {
            stats.nearby_seen += 1;

            if c0_dist_fn.distance(&vector) <= c1_dist_fn.distance(&vector) {
                continue; // closer to nearby than target.
            }

            let new_assignments = CentroidAssignment::new(target_centroid_id);

            let old_assignments = assignment_updater.update(record_id, new_assignments)?;
            stats.nearby_moved += move_postings(
                record_id,
                &vector,
                old_assignments,
                new_assignments,
                &mut postings,
            )?;
        }

        postings.flush()?;
        assignment_updater.flush()?;

        Ok(stats)
    }

    struct PostingDistanceFactory<'a> {
        store: CursorVectorStore<'a>,
        posting_format: F32VectorCoding,
        head_coder: Option<Box<dyn F32VectorCoder>>,
    }

    impl<'a> PostingDistanceFactory<'a> {
        fn new(index: &'a TransactionIndex) -> Result<Self> {
            let store = index.head().high_fidelity_vectors()?;
            let posting_format = index.index().config().posting_coder;
            let head_format = store.format();
            let head_coder = if posting_format == head_format {
                None
            } else {
                Some(head_format.coder(store.similarity(), None))
            };
            Ok(Self {
                store,
                posting_format,
                head_coder,
            })
        }

        fn query_distance(&mut self, centroid_id: u32) -> Result<Box<dyn QueryVectorDistance>> {
            let similarity = self.store.similarity();
            let query = self
                .store
                .get(centroid_id as i64)
                .unwrap_or(Err(Error::not_found_error()))?;
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
            Err(Error::WiredTiger(WiredTigerError::Rollback)) => continue,
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
