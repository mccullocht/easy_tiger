use std::ops::RangeInclusive;

use crate::spann::centroid_stats::CentroidStats;

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
