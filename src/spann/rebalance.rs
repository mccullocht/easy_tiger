use std::{
    collections::{BTreeSet, HashSet},
    num::NonZero,
    ops::RangeInclusive,
    sync::Arc,
};

use rand::Rng;
use rayon::prelude::*;
use tracing::warn;
use wt_mdb::{
    session::{Formatted, TransactionGuard},
    Error, Result, TypedCursor,
};

use crate::{
    input::{VecVectorStore, VectorStore},
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
    let vectors = read_centroid(centroid_id, &mut posting_cursor)?;
    let reassignments = merge_reassign(
        index,
        head_index,
        vectors,
        &HashSet::from([centroid_id]),
        &[],
        &VecVectorStore::new(index.head_config().config().dimensions.get()),
    )?;
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
///
/// `deleted_centroids` are excluded from consideration as reassignment targets. `new_centroid_ids`
/// and `new_centroids` (a parallel slice and store of float vectors) are injected as additional
/// candidates for every reassignment regardless of graph connectivity.
fn merge_reassign(
    index: &TableIndex,
    head_index: &SessionGraphVectorIndex,
    vectors: Vec<(i64, Vec<u8>)>,
    deleted_centroids: &HashSet<usize>,
    new_centroid_ids: &[usize],
    new_centroids: &VecVectorStore<f32>,
) -> Result<Vec<(i64, CentroidAssignment, Vec<u8>)>> {
    if vectors.is_empty() {
        return Ok(vec![]);
    }

    // Query the head index for each vector and assign a new centroid.
    let posting_format = index.config().posting_coder;
    let similarity = index.head_config().config().similarity;
    let coder = posting_format.new_coder(similarity);
    let new_centroid_dist_fns: Vec<_> = new_centroids
        .iter()
        .map(|c| posting_format.query_vector_distance_indexing(coder.encode(c), similarity))
        .collect();
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
                let mut candidates = searcher.search_with_filter(
                    float_vector,
                    |i| !deleted_centroids.contains(&(i as usize)),
                    head_index,
                )?;
                candidates.extend(
                    new_centroid_ids
                        .iter()
                        .zip(new_centroid_dist_fns.iter())
                        .map(|(&id, dist_fn)| Neighbor::new(id as i64, dist_fn.distance(&vector))),
                );
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
        &HashSet::from([centroid_id]),
        &[],
        &VecVectorStore::new(index.head_config().config().dimensions.get()),
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
    let mut nearby_new_vecs =
        VecVectorStore::with_capacity(index.head_config().config().dimensions.get(), 2);
    nearby_new_vecs.push(&centroids[0]);
    nearby_new_vecs.push(&centroids[1]);
    let (nearby_reassignments, nearby_seen_delta, searches_delta) = nearby_reassign(
        index,
        head_index,
        nearby_clusters,
        &HashSet::from([centroid_id]),
        &[target_centroid_ids.0, target_centroid_ids.1],
        &nearby_new_vecs,
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
/// `deleted_centroids` are excluded from consideration as reassignment targets. `new_centroid_ids`
/// and `new_centroids` (a parallel slice and store of float vectors) are injected as additional
/// candidates for every reassignment regardless of graph connectivity.
fn split_reassign(
    index: &TableIndex,
    head_index: &SessionGraphVectorIndex,
    centroid_id: usize,
    vectors: Vec<(i64, Vec<u8>)>,
    original_centroid: &[f32],
    centroid_a: (usize, &[f32]),
    centroid_b: (usize, &[f32]),
    deleted_centroids: &HashSet<usize>,
    new_centroid_ids: &[usize],
    new_centroids: &VecVectorStore<f32>,
) -> Result<(Vec<(i64, CentroidAssignment, Vec<u8>)>, usize)> {
    let posting_format = index.config().posting_coder;
    let similarity = index.head_config().config().similarity;
    let posting_coder = posting_format.new_coder(similarity);
    let scratch_vector = vec![0.0f32; index.head_config().config().dimensions.get()];
    let new_centroid_dist_fns: Vec<_> = new_centroids
        .iter()
        .map(|c| posting_format.query_vector_distance_indexing(posting_coder.encode(c), similarity))
        .collect();
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
                        |i| !deleted_centroids.contains(&(i as usize)),
                        head_index,
                    )?;
                    // NB: this may result in evaluting c1/c2 distance twice, but shouldn't change
                    // the outcome of reassignment.
                    candidates.extend(
                        new_centroid_ids
                            .iter()
                            .zip(new_centroid_dist_fns.iter())
                            .map(|(&id, dist_fn)| {
                                Neighbor::new(id as i64, dist_fn.distance(&vector))
                            }),
                    );
                    candidates.truncate(index.config().replica_count * 4);
                    select_centroids(
                        index.config().replica_selection,
                        index.config().replica_count,
                        candidates,
                        &scratch_vector,
                        head_index,
                    )?
                } else {
                    let updated_centroid_id = {
                        let (mut best_id, mut best_dist) = if c1_dist < c2_dist {
                            (centroid_a.0, c1_dist)
                        } else {
                            (centroid_b.0, c2_dist)
                        };
                        for (&id, dist_fn) in
                            new_centroid_ids.iter().zip(new_centroid_dist_fns.iter())
                        {
                            let d = dist_fn.distance(&vector);
                            if d < best_dist {
                                best_dist = d;
                                best_id = id;
                            }
                        }
                        best_id
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
/// that are closer to one of the `new_centroids` than to their current centroid.
///
/// `deleted_centroids` are excluded from replica search targets. `new_centroid_ids` and
/// `new_centroids` define the candidate reassignment targets and are considered for every vector.
///
/// Returns the list of reassignments together with aggregate `nearby_seen` and `searches` counts.
fn nearby_reassign(
    index: &TableIndex,
    head_index: &SessionGraphVectorIndex,
    nearby_clusters: Vec<Neighbor>,
    deleted_centroids: &HashSet<usize>,
    new_centroid_ids: &[usize],
    new_centroids: &VecVectorStore<f32>,
) -> Result<(Vec<(PostingKey, CentroidAssignment, Vec<u8>)>, usize, usize)> {
    let posting_format = index.config().posting_coder;
    let similarity = index.head_config().config().similarity;
    let posting_coder = posting_format.new_coder(similarity);
    let head_vectors = head_index.high_fidelity_vectors()?;
    let head_coder = head_vectors.new_coder();
    let new_centroid_dist_fns: Vec<_> = new_centroids
        .iter()
        .map(|c| posting_format.query_vector_distance_indexing(posting_coder.encode(c), similarity))
        .collect();
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

                    // Pick the closest new centroid; skip this vector if none beats the current.
                    let best = new_centroid_ids
                        .iter()
                        .zip(new_centroid_dist_fns.iter())
                        .map(|(&id, dist_fn)| (id as u32, dist_fn.distance(vector)))
                        .min_by(|a, b| a.1.total_cmp(&b.1));
                    let assigned_centroid_id = match best {
                        Some((id, d)) if d < c0_dist => id,
                        _ => continue,
                    };

                    let new_assignment = if index.config().replica_count > 1 {
                        searches += 1;
                        posting_coder.decode_to(vector, scratch_vector);
                        let mut candidates = searcher.search_with_filter(
                            scratch_vector,
                            |i| !deleted_centroids.contains(&(i as usize)),
                            head_index,
                        )?;
                        candidates.extend(
                            new_centroid_ids
                                .iter()
                                .zip(new_centroid_dist_fns.iter())
                                .map(|(&id, dist_fn)| {
                                    Neighbor::new(id as i64, dist_fn.distance(vector))
                                }),
                        );
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
///
/// All read/compute work (reassignment searches) runs in parallel via rayon. Writes are applied
/// serially on the calling thread.
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
        let (merge_centroids, split_centroids) = centroid_stats.assignment_counts_iter().fold(
            (vec![], vec![]),
            |(mut merge_centroids, mut split_centroids), (centroid_id, count)| {
                if count < *bounds.start() as u32 {
                    merge_centroids.push((centroid_id as usize, count as usize));
                } else if count > *bounds.end() as u32 {
                    split_centroids.push((centroid_id as usize, count as usize));
                }
                (merge_centroids, split_centroids)
            },
        );

        // If there is one under-sized centroid then we shouldn't do anything. Bootstrap indices
        // with zero vectors and other smaller indices may trip this condition.
        if (centroid_stats.centroid_count() == 1 && !merge_centroids.is_empty())
            || (merge_centroids.is_empty() && split_centroids.is_empty())
        {
            break;
        }

        let deleted_centroids: HashSet<usize> = merge_centroids
            .iter()
            .chain(split_centroids.iter())
            .map(|(id, _)| *id)
            .collect();

        // Phase 1 (serial): partition each split centroid into two clusters, minting new IDs.
        let dim = head_index.config().dimensions.get();
        let mut next_centroid_id = centroid_stats.available_centroid_ids();
        let mut split_centroid_ids: Vec<usize> = Vec::with_capacity(split_centroids.len() * 2);
        let mut split_centroid_vecs =
            VecVectorStore::<f32>::with_capacity(dim, split_centroids.len() * 2);
        // Work items: (centroid_id, len, vectors, a_idx, b_idx into split_centroid_ids/vecs)
        let split_work: Vec<(usize, usize, Vec<(i64, Vec<u8>)>, usize, usize)> = split_centroids
            .iter()
            .map(|(centroid_id, len)| {
                let (vectors, centroids) =
                    partition_centroid(index, head_index, *centroid_id, rng)?;
                let a_idx = split_centroid_ids.len();
                for c in centroids.iter() {
                    split_centroid_ids
                        .push(next_centroid_id.next().expect("centroid IDs are unbounded"));
                    split_centroid_vecs.push(c);
                }
                Ok((*centroid_id, *len, vectors, a_idx, a_idx + 1))
            })
            .collect::<Result<_>>()?;

        let connection = Arc::clone(head_index.session().connection());

        // Phase 2 (parallel): for merges — read posting vectors then merge_reassign; for splits —
        // read the original centroid vector, search for nearby clusters, then split_reassign.
        // Each task opens its own session so reads don't contend on the calling thread's session.

        // Merge results: (centroid_id, reassignments)
        let merge_results: Vec<(usize, Vec<(i64, CentroidAssignment, Vec<u8>)>)> = merge_centroids
            .par_iter()
            .map(|(centroid_id, _)| {
                let h = SessionGraphVectorIndex::new(
                    Arc::clone(index.head_config()),
                    connection.open_session()?,
                );
                let mut cursor = h
                    .session()
                    .get_or_create_typed_cursor::<PostingKey, Vec<u8>>(
                        index.postings_table_name(),
                    )?;
                let vectors = read_centroid(*centroid_id, &mut cursor)?;
                let reassignments = merge_reassign(
                    index,
                    &h,
                    vectors,
                    &deleted_centroids,
                    &split_centroid_ids,
                    &split_centroid_vecs,
                )?;
                Ok((*centroid_id, reassignments))
            })
            .collect::<Result<_>>()?;

        // Split results: (centroid_id, a_idx, b_idx, reassignments, searches, nearby_clusters)
        let split_results: Vec<(
            usize,
            usize,
            usize,
            Vec<(i64, CentroidAssignment, Vec<u8>)>,
            usize,
            Vec<Neighbor>,
        )> = split_work
            .into_par_iter()
            .map(|(centroid_id, _, vectors, a_idx, b_idx)| {
                let h = SessionGraphVectorIndex::new(
                    Arc::clone(index.head_config()),
                    connection.open_session()?,
                );
                // Read the original centroid vector before the graph is mutated.
                let mut head_vecs = h.high_fidelity_vectors()?;
                let head_coder = head_vecs.new_coder();
                let original_centroid = head_coder.decode(
                    head_vecs
                        .get(centroid_id as i64)
                        .unwrap_or(Err(Error::not_found_error()))?,
                );
                // Search for nearby clusters, excluding deleted centroids (not yet removed).
                let mut params = index.config().head_search_params;
                params.beam_width = NonZero::new(64).unwrap();
                let nearby_clusters = GraphSearcher::new(params).search_with_filter(
                    &original_centroid,
                    |i| !deleted_centroids.contains(&(i as usize)),
                    &h,
                )?;
                let (reassignments, searches) = split_reassign(
                    index,
                    &h,
                    centroid_id,
                    vectors,
                    &original_centroid,
                    (split_centroid_ids[a_idx], &split_centroid_vecs[a_idx]),
                    (split_centroid_ids[b_idx], &split_centroid_vecs[b_idx]),
                    &deleted_centroids,
                    &split_centroid_ids,
                    &split_centroid_vecs,
                )?;
                Ok((
                    centroid_id,
                    a_idx,
                    b_idx,
                    reassignments,
                    searches,
                    nearby_clusters,
                ))
            })
            .collect::<Result<_>>()?;

        // Phase 3 (parallel): nearby_reassign for each split.
        // Nearby results: (centroid_id, a_idx, b_idx, reassignments, nearby_seen, searches)
        let nearby_results: Vec<(
            usize,
            usize,
            usize,
            Vec<(PostingKey, CentroidAssignment, Vec<u8>)>,
            usize,
            usize,
        )> = split_results
            .par_iter()
            .map(|(centroid_id, a_idx, b_idx, _, _, nearby_clusters)| {
                let h = SessionGraphVectorIndex::new(
                    Arc::clone(index.head_config()),
                    connection.open_session()?,
                );
                let (reassignments, nearby_seen, searches) = nearby_reassign(
                    index,
                    &h,
                    nearby_clusters.clone(),
                    &deleted_centroids,
                    &split_centroid_ids,
                    &split_centroid_vecs,
                )?;
                Ok((
                    *centroid_id,
                    *a_idx,
                    *b_idx,
                    reassignments,
                    nearby_seen,
                    searches,
                ))
            })
            .collect::<Result<_>>()?;

        // Phase 4 (serial): apply all computed changes.
        let mut assignment_updater = CentroidAssignmentUpdater::new(index, head_index.session())?;
        let mut posting_cursor = head_index
            .session()
            .get_or_create_typed_cursor::<PostingKey, Vec<u8>>(index.postings_table_name())?;

        // Apply merges: delete from graph, write reassignments, remove stats.
        for (centroid_id, reassignments) in merge_results {
            delete_vector(centroid_id as i64, head_index)?;
            let moved = reassignments.len();
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
            head_index
                .session()
                .get_or_create_typed_cursor::<u32, CentroidCounts>(
                    index.centroid_stats_table_name(),
                )?
                .remove(centroid_id as u32)?;
            stats += MergeStats {
                moved_vectors: moved,
            };
        }

        // Apply splits: delete old centroid, upsert two new centroids, write reassignments.
        for (centroid_id, a_idx, b_idx, reassignments, _, _) in &split_results {
            delete_vector(*centroid_id as i64, head_index)?;
            upsert_vector(
                split_centroid_ids[*a_idx] as i64,
                &split_centroid_vecs[*a_idx],
                head_index,
            )?;
            upsert_vector(
                split_centroid_ids[*b_idx] as i64,
                &split_centroid_vecs[*b_idx],
                head_index,
            )?;
            let moved = reassignments.len();
            for (record_id, new_assignments, vector) in reassignments.iter() {
                let old_assignments =
                    assignment_updater.update(*record_id, new_assignments.to_formatted_ref())?;
                move_postings(
                    PostingKey {
                        centroid_id: *centroid_id as u32,
                        record_id: *record_id,
                    },
                    vector,
                    &old_assignments,
                    new_assignments,
                    &mut posting_cursor,
                )?;
            }
            let _ = moved; // used below via split_reassignments.len() in stats
        }

        // Apply nearby reassignments, remove old centroid stats, and accumulate per-split stats.
        posting_cursor.reset()?;
        for (
            (centroid_id, _, _, split_reassignments, searches, _),
            (_, _, _, nearby_reassignments, nearby_seen, nearby_searches),
        ) in split_results.iter().zip(nearby_results.into_iter())
        {
            let mut nearby_moved = 0;
            for (key, new_assignments, vector) in nearby_reassignments.iter() {
                let old_assignments =
                    assignment_updater.update(key.record_id, new_assignments.to_formatted_ref())?;
                nearby_moved += move_postings(
                    *key,
                    &vector,
                    &old_assignments,
                    &new_assignments,
                    &mut posting_cursor,
                )?;
            }
            head_index
                .session()
                .get_or_create_typed_cursor::<u32, CentroidCounts>(
                    index.centroid_stats_table_name(),
                )?
                .remove(*centroid_id as u32)?;
            stats += SplitStats {
                moved_vectors: split_reassignments.len(),
                searches: searches + nearby_searches,
                nearby_seen,
                nearby_moved,
            };
        }

        assignment_updater.flush()?;

        // XXX remove all the attempts to fix this upstream, it's copypasta.
        let mut stats_cursor = head_index
            .session()
            .get_or_create_typed_cursor::<u32, CentroidCounts>(index.centroid_stats_table_name())?;
        for centroid_id in deleted_centroids {
            stats_cursor.remove(centroid_id as u32)?;
        }
        txn.commit(None)?;
    }
    Ok(stats)
}
