use std::{num::NonZero, ops::RangeInclusive};

use rand::Rng;
use tracing::warn;
use wt_mdb::{session::Formatted, Error, Result, TypedCursor};

use crate::{
    input::{VecVectorStore, VectorStore},
    kmeans,
    spann::{
        centroid_stats::{CentroidAssignmentUpdater, CentroidCounts, CentroidStats},
        CentroidAssignment, PostingKey, TableIndex,
    },
    vamana::{
        mutate::{delete_vector, upsert_vector},
        search::GraphSearcher,
        wt::SessionGraphVectorIndex,
        GraphVectorIndex, GraphVectorStore,
    },
};

/// Remove `centroid_id` and merge each of its vectors into the next closest centroid.
pub fn merge_centroid(
    index: &TableIndex,
    head_index: &SessionGraphVectorIndex,
    centroid_id: usize,
) -> Result<()> {
    assert_eq!(
        index.config().replica_count,
        1,
        "rebalance only implemented for replica count 1"
    );

    // Collect all of the vectors for the centroid to merge.
    let mut posting_cursor = head_index
        .session()
        .get_or_create_typed_cursor::<PostingKey, Vec<u8>>(index.postings_table_name())?;
    let vectors = drain_centroid(centroid_id, &mut posting_cursor)?;

    // Remove the centroid from the graph.
    delete_vector(centroid_id as i64, head_index)?;

    // If the centroid is already empty then there is nothing to do.
    if vectors.is_empty() {
        return Ok(());
    }

    // TODO: run the required searches in parallel. WiredTiger sessions will make it challenging
    // to abstract this away from the storage engine.

    // Query the head index for each vector and assign a new centroid.
    let coder = index.new_posting_coder();
    let mut float_vector =
        vec![0.0f32; coder.dimensions(index.head_config().config().dimensions.get())];
    let mut searcher = GraphSearcher::new(index.config().head_search_params);
    let mut assignment_updater = CentroidAssignmentUpdater::new(index, head_index.session())?;
    posting_cursor.reset()?;
    for (record_id, vector) in vectors {
        coder.decode_to(&vector, &mut float_vector);
        // TODO: seed the search with the existing assignments for this record; reduce budget.
        let candidates = searcher.search(&float_vector, head_index)?;
        assert!(!candidates.is_empty());
        // TODO: handle replica_count > 1. This is still a search but we may have to move
        // postings that are not in centroid_id.
        let new_centroid_id = candidates[0].vertex() as u32;
        assignment_updater.update(
            record_id,
            CentroidAssignment::new(new_centroid_id, &[]).to_formatted_ref(),
        )?;
        let key = PostingKey {
            centroid_id: new_centroid_id,
            record_id,
        };
        posting_cursor.set(key, &vector)?;
    }

    assignment_updater.flush()?;
    head_index
        .session()
        .get_or_create_typed_cursor::<u32, CentroidCounts>(index.centroid_stats_table_name())?
        .remove(centroid_id as u32)?;

    Ok(())
}

/// Split `centroid_id` in two, creating a `next_centroid_id`.
///
/// `rng` is used to partition the input centroid into two clusters.
pub fn split_centroid(
    index: &TableIndex,
    head_index: &SessionGraphVectorIndex,
    centroid_id: usize,
    next_centroid_id: usize,
    rng: &mut impl Rng,
) -> Result<()> {
    assert_eq!(
        index.config().replica_count,
        1,
        "rebalance only implemented for replica count 1"
    );

    let mut posting_cursor = head_index
        .session()
        .get_or_create_typed_cursor::<PostingKey, Vec<u8>>(index.postings_table_name())?;
    let vectors = drain_centroid(centroid_id, &mut posting_cursor)?;
    assert!(!vectors.is_empty());

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
            if clustering_vectors
                .len()
                .div_ceil(index.config.max_centroid_len)
                <= 2
            {
                warn!("split_centroid: binary partition of centroid {centroid_id} (count {}) failed to converge!", vectors.len());
            }
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

    // TODO: perform searches in parallel.
    searcher = GraphSearcher::new(index.config().head_search_params);
    let mut assignment_updater = CentroidAssignmentUpdater::new(index, head_index.session())?;
    let c0_dist_fn = posting_format.query_vector_distance_f32(original_centroid, similarity);
    let c1_dist_fn = posting_format.query_vector_distance_f32(&centroids[0], similarity);
    let c2_dist_fn = posting_format.query_vector_distance_f32(&centroids[1], similarity);
    for (record_id, vector) in vectors {
        // TODO: handle replica_count > 1. If this centroid is _not_ the primary for record_id
        // then we always have to search and generate new candidates. We may also need to move
        // postings that are not in centroid_id.
        let c0_dist = c0_dist_fn.distance(&vector);
        let c1_dist = c1_dist_fn.distance(&vector);
        let c2_dist = c2_dist_fn.distance(&vector);
        let new_centroid_id = if c0_dist <= c1_dist && c0_dist <= c2_dist {
            posting_coder.decode_to(&vector, &mut scratch_vector);
            searcher.search(&scratch_vector, head_index)?[0].vertex() as u32
        } else if c1_dist < c2_dist {
            centroid_id as u32
        } else {
            next_centroid_id as u32
        };

        let key = PostingKey {
            centroid_id: new_centroid_id as u32,
            record_id,
        };
        posting_cursor.set(key, &vector)?;
        assignment_updater.update(
            record_id,
            CentroidAssignment::new(new_centroid_id, &[]).to_formatted_ref(),
        )?;
    }

    // For a list of nearby centroids, examine all vectors and reassign them if they are closer
    // to one of the new centroids than they are to the current centroid.
    // TODO: process nearby centroids in parallel.
    let mut update_posting_cursor = head_index
        .session()
        .get_or_create_typed_cursor::<PostingKey, Vec<u8>>(index.postings_table_name())?;
    for nearby_centroid_id in nearby_clusters.into_iter().map(|n| n.vertex() as u32) {
        let nearby_centroid = head_coder.decode(
            head_vectors
                .get(nearby_centroid_id as i64)
                .unwrap_or(Err(Error::not_found_error()))?,
        );
        let c0_dist_fn = posting_format.query_vector_distance_f32(nearby_centroid, similarity);

        posting_cursor.set_bounds(PostingKey::centroid_range(nearby_centroid_id))?;
        // SAFETY: memory remains valid because this path does not commit or rollback txns.
        while let Some(r) = unsafe { posting_cursor.next_unsafe() } {
            // TODO: handle replica_count > 1. If this is the primary for record_id and the
            // assignment remains unchanged then we can continue to do nothing. If the primary
            // changes or this is a secondary assignment then we need to search and generate new
            // candidates. In any case we may need to move some unexpected postings around.
            let (key, vector) = r?;
            let c0_dist = c0_dist_fn.distance(vector);
            let c1_dist = c1_dist_fn.distance(vector);
            let c2_dist = c2_dist_fn.distance(vector);

            let assigned_centroid_id = if c1_dist < c0_dist {
                centroid_id as u32
            } else if c2_dist < c0_dist {
                next_centroid_id as u32
            } else {
                continue;
            };

            let new_key = PostingKey {
                centroid_id: assigned_centroid_id,
                record_id: key.record_id,
            };
            update_posting_cursor.remove(key)?;
            update_posting_cursor.set(new_key, vector)?;
            assignment_updater.update(
                key.record_id,
                CentroidAssignment::new(assigned_centroid_id, &[]).to_formatted_ref(),
            )?;
        }
    }

    // Write the new centroids back into the index.
    upsert_vector(centroid_id as i64, &centroids[0], head_index)?;
    upsert_vector(next_centroid_id as i64, &centroids[1], head_index)?;

    assignment_updater.flush()
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
