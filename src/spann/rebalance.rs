use std::ops::RangeInclusive;

use wt_mdb::{session::Formatted, Result, TypedCursor};

use crate::{
    spann::{
        centroid_stats::{CentroidAssignmentUpdater, CentroidCounts, CentroidStats},
        CentroidAssignment, PostingKey, TableIndex,
    },
    vamana::{mutate::delete_vector, search::GraphSearcher, wt::SessionGraphVectorIndex},
};

/// Remove `centroid_id` and merge each of its vectors into the next closest centroid.
pub fn merge_centroid(
    index: &TableIndex,
    head_index: &SessionGraphVectorIndex,
    centroid_id: usize,
) -> Result<()> {
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
