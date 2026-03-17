use std::{collections::BTreeSet, num::NonZero, ops::RangeInclusive, sync::Arc};

use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tracing::warn;
use wt_mdb::{
    session::{Formatted, TransactionGuard},
    Error, Result, TypedCursor,
};

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
    Neighbor,
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
    let mut posting_cursor = head_index
        .session()
        .get_or_create_typed_cursor::<PostingKey, Vec<u8>>(index.postings_table_name())?;
    let reassignments = merge_reassign(index, head_index, centroid_id, &mut posting_cursor)?;
    assert_eq!(
        reassignments.len(),
        len,
        "merge_centroid of {centroid_id} expected {len} vectors; actual {}",
        reassignments.len()
    );

    // Remove the centroid from the graph.
    delete_vector(centroid_id as i64, head_index)?;

    // If the centroid is already empty then there is nothing to do.
    if reassignments.is_empty() {
        head_index
            .session()
            .get_or_create_typed_cursor::<u32, CentroidCounts>(index.centroid_stats_table_name())?
            .remove(centroid_id as u32)?;
        return Ok(MergeStats::default());
    }

    let removed_vectors = reassignments.len();
    let mut assignment_updater = CentroidAssignmentUpdater::new(index, head_index.session())?;
    posting_cursor.reset()?;
    for (record_id, new_assignments, vector) in reassignments {
        let old_assignments =
            assignment_updater.update(record_id, new_assignments.to_formatted_ref())?;
        move_postings(
            PostingKey {
                centroid_id: centroid_id as u32,
                record_id,
            },
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

    Ok(MergeStats {
        moved_vectors: removed_vectors,
    })
}

/// Read all vectors for `centroid_id` and compute new centroid assignments for each.
fn merge_reassign(
    index: &TableIndex,
    head_index: &SessionGraphVectorIndex,
    centroid_id: usize,
    posting_cursor: &mut TypedCursor<'_, PostingKey, Vec<u8>>,
) -> Result<Vec<(i64, CentroidAssignment, Vec<u8>)>> {
    let vectors = read_centroid(centroid_id, posting_cursor)?;
    if vectors.is_empty() {
        return Ok(vec![]);
    }

    // Query the head index for each vector and assign a new centroid.
    let coder = index.new_posting_coder();
    let connection = Arc::clone(head_index.session().connection());
    vectors
        .into_par_iter()
        .map_init(
            || {
                (
                    SessionGraphVectorIndex::new(
                        Arc::clone(index.head_config()),
                        connection.open_session().expect("open session"),
                    ),
                    GraphSearcher::new(index.config().head_search_params),
                    vec![0.0f32; index.head_config().config().dimensions.get()],
                )
            },
            |(head_index, searcher, float_vector), (record_id, vector)| {
                coder.decode_to(&vector, float_vector);
                // TODO: seed the search with the existing assignments for this record; reduce budget.
                let candidates = searcher.search_with_filter(
                    float_vector,
                    |i| i != centroid_id as i64,
                    head_index,
                )?;
                let new_assignments = select_centroids(
                    index.config().replica_selection,
                    index.config().replica_count,
                    candidates,
                    float_vector,
                    head_index,
                )?;
                Ok((record_id, new_assignments, vector))
            },
        )
        .collect::<Result<Vec<_>>>()
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
    let (vectors, centroids) = partition_centroid(index, head_index, centroid_id, rng)?;
    assert_eq!(
        vectors.len(),
        len,
        "split_centroid of {centroid_id} expected {len} vectors; actual {}",
        vectors.len()
    );

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

    let moved_vectors = vectors.len();
    let (split_reassignments, mut searches) = split_reassign(
        index,
        head_index,
        centroid_id,
        vectors,
        &original_centroid,
        (target_centroid_ids.0, &centroids[0]),
        (target_centroid_ids.1, &centroids[1]),
    )?;
    let mut posting_cursor = head_index
        .session()
        .get_or_create_typed_cursor::<PostingKey, Vec<u8>>(index.postings_table_name())?;
    let mut assignment_updater = CentroidAssignmentUpdater::new(index, head_index.session())?;
    for (record_id, new_assignments, vector) in split_reassignments {
        let old_assignments =
            assignment_updater.update(record_id, new_assignments.to_formatted_ref())?;
        move_postings(
            PostingKey {
                centroid_id: centroid_id as u32,
                record_id,
            },
            &vector,
            &old_assignments,
            &new_assignments,
            &mut posting_cursor,
        )?;
    }

    let mut nearby_seen = 0;
    let mut nearby_moved = 0;
    // For a list of nearby centroids, examine all vectors and reassign them if they are closer
    // to one of the new centroids than they are to the current centroid.
    let (nearby_reassignments, nearby_seen_delta, searches_delta) = nearby_reassign(
        index,
        head_index,
        centroid_id,
        nearby_clusters,
        (target_centroid_ids.0, &centroids[0]),
        (target_centroid_ids.1, &centroids[1]),
    )?;
    nearby_seen += nearby_seen_delta;
    searches += searches_delta;

    // We do not need to deduplicate reassignments -- only primary assignments are evaluated for
    // reassignment so no record_id will appear twice in the list.
    posting_cursor.reset()?;
    for (key, new_assignments, vector) in nearby_reassignments {
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

/// Read all vectors for `centroid_id`, decode them to floats, and run balanced binary k-means
/// to produce two new centroid vectors.
///
/// Returns the original encoded vectors (for subsequent reassignment) together with a
/// two-entry `VecVectorStore` whose entries are the two new centroid vectors.
fn partition_centroid(
    index: &TableIndex,
    head_index: &SessionGraphVectorIndex,
    centroid_id: usize,
    rng: &mut impl Rng,
) -> Result<(Vec<(i64, Vec<u8>)>, VecVectorStore<f32>)> {
    let mut posting_cursor = head_index
        .session()
        .get_or_create_typed_cursor::<PostingKey, Vec<u8>>(index.postings_table_name())?;
    let vectors = read_centroid(centroid_id, &mut posting_cursor)?;

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
            warn!(
                "split_centroid: binary partition of centroid {centroid_id} (count {}) failed to converge!",
                vectors.len()
            );
            r
        }
    };

    Ok((vectors, centroids))
}

/// Compute new centroid assignments for each vector in a centroid being split.
///
/// Vectors closer to `original_centroid` than to either new centroid trigger a full graph search;
/// others are assigned to the closer of `centroid_a` or `centroid_b`.
///
/// Returns the reassignment list together with the total number of graph searches performed.
fn split_reassign(
    index: &TableIndex,
    head_index: &SessionGraphVectorIndex,
    centroid_id: usize,
    vectors: Vec<(i64, Vec<u8>)>,
    original_centroid: &[f32],
    centroid_a: (usize, &[f32]),
    centroid_b: (usize, &[f32]),
) -> Result<(Vec<(i64, CentroidAssignment, Vec<u8>)>, usize)> {
    let posting_format = index.config().posting_coder;
    let similarity = index.head_config().config().similarity;
    let posting_coder = posting_format.new_coder(similarity);
    let scratch_vector = vec![0.0f32; index.head_config().config().dimensions.get()];
    // TODO: skip decoding if head index and posting index format are the same.
    let c0_dist_fn = posting_format
        .query_vector_distance_indexing(posting_coder.encode(original_centroid), similarity);
    let c1_dist_fn = posting_format
        .query_vector_distance_indexing(posting_coder.encode(centroid_a.1), similarity);
    let c2_dist_fn = posting_format
        .query_vector_distance_indexing(posting_coder.encode(centroid_b.1), similarity);
    let connection = Arc::clone(head_index.session().connection());
    let raw = vectors
        .into_par_iter()
        .map_init(
            || {
                (
                    SessionGraphVectorIndex::new(
                        Arc::clone(index.head_config()),
                        connection.open_session().expect("open session"),
                    ),
                    GraphSearcher::new(index.config().head_search_params),
                    vec![0.0f32; index.head_config().config().dimensions.get()],
                )
            },
            |(head_index, searcher, float_vector), (record_id, vector)| {
                let mut searched = 0;
                let c0_dist = c0_dist_fn.distance(&vector);
                let c1_dist = c1_dist_fn.distance(&vector);
                let c2_dist = c2_dist_fn.distance(&vector);
                let new_assignments = if c0_dist <= c1_dist && c0_dist <= c2_dist {
                    searched += 1;
                    posting_coder.decode_to(&vector, float_vector);
                    let mut candidates = searcher.search_with_filter(
                        float_vector,
                        |i| i != centroid_id as i64,
                        head_index,
                    )?;
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
                        centroid_a.0
                    } else {
                        centroid_b.0
                    };
                    if index.config().replica_count <= 1 {
                        CentroidAssignment::new(updated_centroid_id as u32, &[])
                    } else {
                        let mut cursor = head_index
                            .session()
                            .get_or_create_typed_cursor::<i64, CentroidAssignment>(
                                &index.table_names.centroids,
                            )?;
                        let mut current_assignments = cursor
                            .seek_exact(record_id)
                            .unwrap_or(Err(Error::not_found_error()))?;
                        current_assignments.replace(centroid_id as u32, updated_centroid_id as u32);
                        current_assignments
                    }
                };
                Ok((record_id, new_assignments, vector, searched))
            },
        )
        .collect::<Result<Vec<_>>>()?;

    let mut searches = 0;
    let reassignments = raw
        .into_iter()
        .map(|(record_id, new_assignments, vector, searched)| {
            searches += searched;
            (record_id, new_assignments, vector)
        })
        .collect();
    Ok((reassignments, searches))
}

/// For each centroid in `nearby_clusters`, read its vectors and compute reassignments for any
/// that are closer to `centroid_a` or `centroid_b` than to their current centroid.
///
/// Returns the list of reassignments together with aggregate `nearby_seen` and `searches` counts.
fn nearby_reassign(
    index: &TableIndex,
    head_index: &SessionGraphVectorIndex,
    centroid_id: usize,
    nearby_clusters: Vec<Neighbor>,
    centroid_a: (usize, &[f32]),
    centroid_b: (usize, &[f32]),
) -> Result<(Vec<(PostingKey, CentroidAssignment, Vec<u8>)>, usize, usize)> {
    let posting_format = index.config().posting_coder;
    let similarity = index.head_config().config().similarity;
    let posting_coder = posting_format.new_coder(similarity);
    let head_vectors = head_index.high_fidelity_vectors()?;
    let head_coder = head_vectors.new_coder();
    let c1_dist_fn = posting_format
        .query_vector_distance_indexing(posting_coder.encode(centroid_a.1), similarity);
    let c2_dist_fn = posting_format
        .query_vector_distance_indexing(posting_coder.encode(centroid_b.1), similarity);
    let scratch_vector = vec![0.0f32; index.head_config().config().dimensions.get()];
    let connection = Arc::clone(head_index.session().connection());
    let results = nearby_clusters
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
                        centroid_a.0 as u32
                    } else if c2_dist < c0_dist {
                        centroid_b.0 as u32
                    } else {
                        continue;
                    };

                    let new_assignment = if index.config().replica_count > 1 {
                        searches += 1;
                        posting_coder.decode_to(vector, scratch_vector);
                        let candidates = searcher.search_with_filter(
                            scratch_vector,
                            |i| i != centroid_id as i64,
                            head_index,
                        )?;
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

    let mut nearby_seen = 0;
    let mut searches = 0;
    let mut reassignments = vec![];
    for (to_reassign, seen, searched) in results {
        nearby_seen += seen;
        searches += searched;
        reassignments.extend(to_reassign);
    }
    Ok((reassignments, nearby_seen, searches))
}

/// Read all the vectors from `centroid_id` using `cursor` and return them.
fn read_centroid(
    centroid_id: usize,
    cursor: &mut TypedCursor<'_, PostingKey, Vec<u8>>,
) -> Result<Vec<(i64, Vec<u8>)>> {
    let mut vectors = vec![];
    cursor.set_bounds(PostingKey::centroid_range(centroid_id as u32))?;
    while let Some(r) = cursor.next() {
        let (key, vector) = r?;
        vectors.push((key.record_id, vector));
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
        posting_cursor.remove(to_remove)?;
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

/// Rebalance all centroids by repeatedly merging and splitting until all are within the
/// configured assignment count bounds.
pub fn rebalance_all(
    index: &TableIndex,
    head_index: &SessionGraphVectorIndex,
    rng: &mut impl Rng,
) -> Result<RebalanceStats> {
    // XXX rebalancing has to be separate from insertion because the parallel reads in other
    // sessions would otherwise be unable to observe the batch being inserted.
    let bounds = index.config().centroid_len_range();
    let mut stats = RebalanceStats::default();
    loop {
        let txn = TransactionGuard::new(head_index.session(), None)?;
        let centroid_stats = CentroidStats::from_index_stats(head_index.session(), index)?;
        let summary = BalanceSummary::new(&centroid_stats, bounds.clone());
        if let Some((centroid_id, len)) = summary
            .below_exemplar()
            .filter(|_| summary.total_clusters() > 1)
        {
            stats += merge_centroid(index, head_index, centroid_id, len)?;
        } else if let Some((centroid_id, len)) = summary.above_exemplar() {
            let mut avail = centroid_stats.available_centroid_ids();
            let id_a = avail.next().expect("centroid IDs are unbounded");
            let id_b = avail.next().expect("centroid IDs are unbounded");
            stats += split_centroid(index, head_index, centroid_id, (id_a, id_b), len, rng)?;
        } else {
            break;
        }
        txn.commit(None)?;
    }
    Ok(stats)
}
