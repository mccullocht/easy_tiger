use std::{collections::BTreeSet, num::NonZero, ops::RangeInclusive, sync::Arc};

use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tracing::warn;
use wt_mdb::{session::Formatted, Error, Result, TypedCursor};

use crate::{
    input::VecVectorStore,
    kmeans,
    spann::{
        centroid_stats::{CentroidAssignmentUpdater, CentroidCounts, CentroidStats},
        select_centroids, CentroidAssignment, PostingKey, TableIndex,
    },
    vamana::{
        mutate::{delete_vector, upsert_vector},
        search::GraphSearcher,
        wt::SessionGraphVectorIndex,
        GraphVectorIndex, GraphVectorStore,
    },
};

use std::ops::{Add, AddAssign};

/// Statistics collected during a centroid merge operation.
#[derive(Debug, Default, Clone, Copy)]
pub struct MergeStats {
    /// Number of vectors that were in the merged centroid.
    pub moved_vectors: usize,
}

impl Add for MergeStats {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            moved_vectors: self.moved_vectors + rhs.moved_vectors,
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
    index: &TableIndex,
    head_index: &SessionGraphVectorIndex,
    centroid_id: usize,
    len: usize,
) -> Result<MergeStats> {
    // Collect all of the vectors for the centroid to merge.
    let mut posting_cursor = head_index
        .session()
        .get_or_create_typed_cursor::<PostingKey, Vec<u8>>(index.postings_table_name())?;
    let vectors = drain_centroid(centroid_id, &mut posting_cursor)?;
    assert_eq!(
        vectors.len(),
        len,
        "merge_centroid of {centroid_id} expected {len} vectors; actual {}",
        vectors.len()
    );

    // Remove the centroid from the graph.
    delete_vector(centroid_id as i64, head_index)?;

    // If the centroid is already empty then there is nothing to do.
    if vectors.is_empty() {
        head_index
            .session()
            .get_or_create_typed_cursor::<u32, CentroidCounts>(index.centroid_stats_table_name())?
            .remove(centroid_id as u32)?;
        return Ok(MergeStats::default());
    }

    // Query the head index for each vector and assign a new centroid.
    let coder = index.new_posting_coder();
    let mut float_vector = vec![0.0f32; index.head_config().config().dimensions.get()];
    let mut searcher = GraphSearcher::new(index.config().head_search_params);
    let mut assignment_updater = CentroidAssignmentUpdater::new(index, head_index.session())?;

    posting_cursor.reset()?;
    let removed_vectors = vectors.len();
    for (record_id, vector) in vectors {
        coder.decode_to(&vector, &mut float_vector);
        // TODO: seed the search with the existing assignments for this record; reduce budget.
        let candidates = searcher.search(&float_vector, head_index)?;
        let new_assignment = select_centroids(
            index.config().replica_selection,
            index.config().replica_count,
            candidates,
            &float_vector,
            head_index,
        )?;
        let old_assignment =
            assignment_updater.update(record_id, new_assignment.to_formatted_ref())?;
        move_postings(
            PostingKey {
                centroid_id: centroid_id as u32,
                record_id,
            },
            &vector,
            &old_assignment,
            &new_assignment,
            &mut posting_cursor,
        )?;
    }

    assignment_updater.flush()?;
    head_index
        .session()
        .get_or_create_typed_cursor::<u32, CentroidCounts>(index.centroid_stats_table_name())?
        .remove(centroid_id as u32)?;

    Ok(MergeStats {
        moved_vectors: removed_vectors,
    })
}

/// Split `centroid_id` in two, creating a `next_centroid_id`.
///
/// `rng` is used to partition the input centroid into two clusters.
pub fn split_centroid(
    index: &TableIndex,
    head_index: &SessionGraphVectorIndex,
    centroid_id: usize,
    target_centroid_ids: (usize, usize),
    len: usize,
    rng: &mut impl Rng,
) -> Result<SplitStats> {
    let mut posting_cursor = head_index
        .session()
        .get_or_create_typed_cursor::<PostingKey, Vec<u8>>(index.postings_table_name())?;
    let vectors = drain_centroid(centroid_id, &mut posting_cursor)?;
    assert_eq!(
        vectors.len(),
        len,
        "split_centroid of {centroid_id} expected {len} vectors; actual {}",
        vectors.len()
    );

    // Unpack all of the vectors as floats and split into two clusters.
    let posting_format = index.config().posting_coder;
    let similarity = index.head_config().config().similarity;
    let posting_coder = posting_format.new_coder(similarity);
    let mut scratch_vector = vec![0.0f32; index.head_config().config().dimensions.get()];
    let mut clustering_vectors = VecVectorStore::with_capacity(scratch_vector.len(), vectors.len());
    for (_, vector) in vectors.iter() {
        posting_coder.decode_to(vector, &mut scratch_vector);
        clustering_vectors.push(&scratch_vector);
    }
    posting_cursor.reset()?;

    let centroids = match kmeans::balanced_binary_partition(
        &clustering_vectors,
        100,
        index.config().min_centroid_len,
        rng,
    ) {
        Ok(r) => r,
        Err(r) => {
            warn!("split_centroid: binary partition of centroid {centroid_id} (count {}) failed to converge!", vectors.len());
            r
        }
    };

    // Extract the original centroid vector from the head index and delete original centroid.
    let mut head_vectors = head_index.high_fidelity_vectors()?;
    let head_coder = head_vectors.new_coder();
    let original_centroid = head_coder.decode(
        head_vectors
            .get(centroid_id as i64)
            .unwrap_or(Err(Error::not_found_error()))?,
    );
    delete_vector(centroid_id as i64, head_index)?;

    // For each vector if it is closer to the original centroid than either of the new centroids
    // then query the whole index to select a new assignment. Otherwise assign it to the closest
    // of the two new centroids.
    let mut params = index.config().head_search_params;
    // TODO: figure out if nearby update beam width needs to be configurable.
    params.beam_width = NonZero::new(64).unwrap();
    let mut searcher = GraphSearcher::new(params);
    let nearby_clusters = searcher.search(&original_centroid, head_index)?;

    // Write the new centroids back into the index.
    upsert_vector(target_centroid_ids.0 as i64, &centroids[0], head_index)?;
    upsert_vector(target_centroid_ids.1 as i64, &centroids[1], head_index)?;

    searcher = GraphSearcher::new(index.config().head_search_params);
    let mut assignment_updater = CentroidAssignmentUpdater::new(index, head_index.session())?;
    // TODO: skip decoding if head index and posting index format are the same.
    let c0_dist_fn = posting_format
        .query_vector_distance_indexing(posting_coder.encode(&original_centroid), similarity);
    let c1_dist_fn = posting_format
        .query_vector_distance_indexing(posting_coder.encode(&centroids[0]), similarity);
    let c2_dist_fn = posting_format
        .query_vector_distance_indexing(posting_coder.encode(&centroids[1]), similarity);
    let mut searches = 0;
    let moved_vectors = vectors.len();
    for (record_id, vector) in vectors {
        let c0_dist = c0_dist_fn.distance(&vector);
        let c1_dist = c1_dist_fn.distance(&vector);
        let c2_dist = c2_dist_fn.distance(&vector);
        let new_assignment = if c0_dist <= c1_dist && c0_dist <= c2_dist {
            searches += 1;
            posting_coder.decode_to(&vector, &mut scratch_vector);
            let mut candidates = searcher.search(&scratch_vector, head_index)?;
            candidates.truncate(index.config().replica_count * 4);
            select_centroids(
                index.config().replica_selection,
                index.config().replica_count,
                candidates,
                &scratch_vector,
                head_index,
            )?
        } else {
            let updated_centroid_id = if c1_dist < c2_dist {
                target_centroid_ids.0
            } else {
                target_centroid_ids.1
            };
            if index.config().replica_count <= 1 {
                CentroidAssignment::new(updated_centroid_id as u32, &[])
            } else {
                let mut current_assignment = assignment_updater
                    .read(record_id)
                    .unwrap_or(Err(Error::not_found_error()))?;
                current_assignment.replace(centroid_id as u32, updated_centroid_id as u32);
                current_assignment
            }
        };
        let old_assignment =
            assignment_updater.update(record_id, new_assignment.to_formatted_ref())?;

        move_postings(
            PostingKey {
                centroid_id: centroid_id as u32,
                record_id,
            },
            &vector,
            &old_assignment,
            &new_assignment,
            &mut posting_cursor,
        )?;
    }

    let mut nearby_seen = 0;
    let mut nearby_moved = 0;
    // For a list of nearby centroids, examine all vectors and reassign them if they are closer
    // to one of the new centroids than they are to the current centroid.
    let connection = Arc::clone(head_index.session().connection());
    let reassignments = nearby_clusters
        .into_par_iter()
        .map_init(
            || {
                (
                    SessionGraphVectorIndex::new(
                        Arc::clone(index.head_config()),
                        connection.open_session().expect("open session"),
                    ),
                    GraphSearcher::new(index.config().head_search_params),
                    scratch_vector.clone(),
                )
            },
            |(head_index, searcher, ref mut scratch_vector), n| {
                let nearby_centroid_id = n.vertex() as u32;
                let mut head_vectors = head_index.high_fidelity_vectors()?;
                let nearby_centroid = head_coder.decode(
                    head_vectors
                        .get(nearby_centroid_id as i64)
                        .unwrap_or(Err(Error::not_found_error()))?,
                );
                let c0_dist_fn = posting_format.query_vector_distance_indexing(
                    posting_coder.encode(&nearby_centroid),
                    similarity,
                );
                let mut posting_cursor = head_index
                    .session()
                    .get_or_create_typed_cursor::<PostingKey, Vec<u8>>(
                        index.postings_table_name(),
                    )?;
                let mut assignment_cursor = head_index
                    .session()
                    .get_or_create_typed_cursor::<i64, CentroidAssignment>(
                        index.centroid_assignments_table_name(),
                    )?;

                posting_cursor.set_bounds(PostingKey::centroid_range(nearby_centroid_id))?;
                let mut to_reassign = vec![];
                let mut nearby_seen = 0;
                let mut searches = 0;
                while let Some(r) = unsafe { posting_cursor.next_unsafe() } {
                    let (key, vector) = r?;
                    let c0_dist = c0_dist_fn.distance(vector);
                    let c1_dist = c1_dist_fn.distance(vector);
                    let c2_dist = c2_dist_fn.distance(vector);

                    // Do not move postings that are not assigned to the original centroid; we're not
                    // willing to do this work for secondaries.
                    if index.config().replica_count > 1
                        && assignment_cursor
                            .seek_exact(key.record_id)
                            .map(|r| r.map(|a| a.primary_id != key.centroid_id))
                            .unwrap_or(Ok(true))?
                    {
                        continue;
                    }
                    nearby_seen += 1;

                    let assigned_centroid_id = if c1_dist < c0_dist {
                        target_centroid_ids.0 as u32
                    } else if c2_dist < c0_dist {
                        target_centroid_ids.1 as u32
                    } else {
                        continue;
                    };

                    let new_assignment = if index.config().replica_count > 1 {
                        searches += 1;
                        posting_coder.decode_to(vector, scratch_vector);
                        let candidates = searcher.search(scratch_vector, head_index)?;
                        select_centroids(
                            index.config().replica_selection,
                            index.config().replica_count,
                            candidates,
                            scratch_vector,
                            head_index,
                        )?
                    } else {
                        CentroidAssignment::new(assigned_centroid_id, &[])
                    };

                    to_reassign.push((key, new_assignment, vector.to_vec()));
                }
                Ok((to_reassign, nearby_seen, searches))
            },
        )
        .collect::<Result<Vec<_>>>()?;

    for (_, seen, searched) in reassignments.iter() {
        nearby_seen += *seen;
        searches += *searched;
    }

    // We do not need to deduplicate reassignments -- only primary assignments are evaluated for
    // reassignment so no record_id will appear twice in the list.
    posting_cursor.reset()?;
    for (key, new_assignments, vector) in reassignments.into_iter().flat_map(|(r, _, _)| r) {
        let old_assignments =
            assignment_updater.update(key.record_id, new_assignments.to_formatted_ref())?;
        nearby_moved += move_postings(
            key,
            &vector,
            &old_assignments,
            &new_assignments,
            &mut posting_cursor,
        )?;
    }

    assignment_updater.flush()?;
    head_index
        .session()
        .get_or_create_typed_cursor::<u32, CentroidCounts>(index.centroid_stats_table_name())?
        .remove(centroid_id as u32)?;

    Ok(SplitStats {
        moved_vectors,
        searches,
        nearby_seen,
        nearby_moved,
    })
}

/// Remove all the vectors from `centroid_id` using `cursor` and return them.
fn drain_centroid(
    centroid_id: usize,
    cursor: &mut TypedCursor<'_, PostingKey, Vec<u8>>,
) -> Result<Vec<(i64, Vec<u8>)>> {
    let mut vectors = vec![];
    cursor.set_bounds(PostingKey::centroid_range(centroid_id as u32))?;
    while let Some(r) = cursor.next() {
        let (key, vector) = r?;
        vectors.push((key.record_id, vector));
        cursor.remove(key)?;
    }
    Ok(vectors)
}

fn move_postings(
    original_key: PostingKey,
    vector: &[u8],
    old_assignment: &CentroidAssignment,
    new_assignment: &CentroidAssignment,
    posting_cursor: &mut TypedCursor<'_, PostingKey, Vec<u8>>,
) -> Result<usize> {
    let old_assignment = old_assignment
        .iter()
        .map(|(_, id)| id)
        .collect::<BTreeSet<_>>();
    let new_assignment = new_assignment
        .iter()
        .map(|(_, id)| id)
        .collect::<BTreeSet<_>>();
    for to_remove in old_assignment
        .difference(&new_assignment)
        .map(|&c| original_key.with_centroid_id(c))
    {
        posting_cursor
            .remove(to_remove)
            .or_else(|e| {
                if e == Error::not_found_error() {
                    Ok(())
                } else {
                    Err(e)
                }
            })
            .expect("failed to remove posting");
    }
    let mut added = 0;
    for to_add in new_assignment
        .difference(&old_assignment)
        .map(|&c| original_key.with_centroid_id(c))
    {
        posting_cursor.set(to_add, vector)?;
        added += 1;
    }
    Ok(added)
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
