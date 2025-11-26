use std::{io, num::NonZero, ops::RangeInclusive, sync::Arc};

use clap::Args;
use easy_tiger::spann::{centroid_stats::CentroidStats, ReplicaSelectionAlgorithm, TableIndex};
use wt_mdb::Connection;

// TODO: almost all of these arguments belong in the index config.
#[derive(Args)]
pub struct RebalanceArgs {
    /// Minimum number of vectors that should map to each centroid.
    #[arg(long)]
    min_centroid_len: NonZero<usize>,
    /// Maximum number of vectors that should map to each centroid.
    #[arg(long)]
    max_centroid_len: NonZero<usize>,
    /// Number of replica centroids to assign each vector to.
    #[arg(long)]
    replica_count: NonZero<usize>,
    /// Replica selection algorithm to use.
    #[arg(long, default_value_t = ReplicaSelectionAlgorithm::SOAR)]
    replica_selection: ReplicaSelectionAlgorithm,
    /// Number of rebalancing iterations to perform.
    #[arg(long)]
    iterations: NonZero<usize>,
}

#[derive(Debug, Default, Copy, Clone)]
struct BalanceSummary {
    in_bounds: usize,

    below_bounds: usize,
    below_exemplar: Option<(usize, usize)>,

    above_bounds: usize,
    above_exemplar: Option<(usize, usize)>,
}

impl BalanceSummary {
    fn new(stats: &CentroidStats, bounds: RangeInclusive<usize>) -> Self {
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
}

pub fn rebalance(
    connection: Arc<Connection>,
    index_name: &str,
    args: RebalanceArgs,
) -> io::Result<()> {
    let index = Arc::new(TableIndex::from_db(&connection, index_name)?);
    let session = connection.open_session()?;

    let stats = CentroidStats::from_index(&session, &index)?;
    let summary = BalanceSummary::new(
        &stats,
        args.min_centroid_len.get()..=args.max_centroid_len.get(),
    );
    println!("SUMMARY");
    println!("  in bounds:    {:3}", summary.in_bounds);
    println!(
        "  below bounds: {:3} examplar {:?}",
        summary.below_bounds, summary.below_exemplar
    );
    println!(
        "  above bounds: {:3} exemplar {:?}",
        summary.above_bounds, summary.above_exemplar
    );

    Ok(())
}
