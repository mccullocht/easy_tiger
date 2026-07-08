use std::{
    collections::HashSet,
    num::NonZero,
    ops::RangeInclusive,
    sync::Arc,
};

use rand::Rng;
use tracing::warn;
use wt_mdb::{Error, Result};

use crate::{
    input::VecVectorStore,
    kmeans,
    spann::{
        centroid_stats::{CentroidAssignmentUpdater, CentroidCounts, CentroidStats},
        postings::BlockPostingsMut,
        CentroidAssignment, TransactionIndex,
    },
    vamana::{
        mutate::{delete_vector, upsert_vector},
        search::GraphSearcher,
        GraphVectorIndex, GraphVectorStore,
    },
};

use std::ops::{Add, AddAssign};

/// Statistics collected during a centroid merge operation.
#[derive(Debug, Default, Clone, Copy)]
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
#[derive(Debug, Default, Clone, Copy)]
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
#[derive(Debug, Default, Clone, Copy)]
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
        let old_assignments =
            assignment_updater.update(record_id, new_assignments)?;
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
            warn!("split_centroid: binary partition of centroid {centroid_id} (count {}) failed to converge!", vectors.len());
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

        let old_assignments =
            assignment_updater.update(record_id, new_assignments)?;
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
            let old_assignments =
                assignment_updater.update(record_id, new_assignments)?;
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
