use std::{io, sync::Arc};

use easy_tiger::spann::{centroid_stats::CentroidStats, TableIndex};
use histogram::Histogram;
use wt_mdb::Connection;

pub fn centroid_stats(connection: Arc<Connection>, index_name: &str) -> io::Result<()> {
    let index = TableIndex::from_db(&connection, index_name)?;
    let session = connection.open_session()?;
    let stats = CentroidStats::from_index_stats(&session, &index)?;

    println!("Head contains {} centroids", stats.centroid_count());
    println!("{} tail posting entries", stats.vector_count());
    let max_value_power = max_value_power(&stats);
    println!("Primary assignments per centroid:");
    print_histogram(
        max_value_power,
        stats.primary_assignment_counts_iter().map(|(_, c)| c),
    )?;
    println!("Secondary assignments per centroid:");
    print_histogram(
        max_value_power,
        stats.secondary_assignment_counts_iter().map(|(_, c)| c),
    )?;
    println!("Total assignments per centroid:");
    print_histogram(
        max_value_power,
        stats.assignment_counts_iter().map(|(_, c)| c),
    )
}

fn max_value_power(stats: &CentroidStats) -> u8 {
    (stats
        .assignment_counts_iter()
        .map(|(_, c)| c)
        .max()
        .unwrap()
        .next_power_of_two()
        .ilog2() as u8)
        .max(3)
}

fn print_histogram(max_value_power: u8, input: impl Iterator<Item = u32>) -> io::Result<()> {
    let mut histogram = Histogram::new(2, max_value_power).unwrap();
    for c in input {
        histogram.add(c.into(), 1).unwrap();
    }
    use std::io::Write;
    let mut lock = std::io::stdout().lock();
    for b in histogram.into_iter().filter(|b| b.count() > 0) {
        writeln!(lock, "[{:5}..{:5}] {:7}", b.start(), b.end(), b.count())?;
    }
    Ok(())
}
