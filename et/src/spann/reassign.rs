//! Reassignment audit for a single centroid's posting list.
//!
//! Reads the posting block for one centroid, looks up the rerank (raw) vector for every record in
//! it, and searches the head (centroid) index with that vector to recompute the record's
//! assignment the same way `insert-vectors` assigns it. The result is printed as a histogram of
//! how many vectors would be assigned to each centroid. With `--commit`, vectors whose recomputed
//! assignment differs from the source centroid are actually moved: the posting vector is moved to
//! the target centroid's posting, and the assignment table and centroid stats are updated.

use std::{collections::HashMap, io, sync::Arc};

use clap::Args;
use easy_tiger::{
    Neighbor,
    posting_block::PostingBlock,
    spann::{
        CentroidAssignment, TableIndex, TransactionIndex,
        centroid_stats::CentroidAssignmentUpdater, postings::BlockPostingsMut,
    },
    vamana::{GraphSearchParams, search::GraphSearcher},
};
use rayon::prelude::*;
use wt_mdb::Connection;

use crate::ui::progress_bar;

#[derive(Args)]
pub struct ReassignArgs {
    /// Id of the centroid whose posting list should be audited.
    #[arg(long)]
    centroid: u32,

    /// Actually move vectors whose recomputed assignment differs to their new centroid and commit
    /// the change. Without this flag the reassignment is only simulated and printed.
    #[arg(long, default_value_t = false)]
    commit: bool,
}

/// The recomputed assignment for a single record.
#[derive(Debug, Clone, Copy)]
enum Target {
    Centroid(u32),
    /// The record has no vector in the rerank table.
    MissingRerank,
    /// The head search returned no candidates.
    NoCandidates,
}

pub fn reassign(
    connection: Arc<Connection>,
    index_name: &str,
    args: ReassignArgs,
) -> io::Result<()> {
    let index = Arc::new(TableIndex::from_db(&connection, index_name)?);

    // Read the source posting and collect its record ids.
    let records: Vec<i64> = {
        let txn_idx = TransactionIndex::new(&index, connection.begin_transaction(None)?);
        let vector_len = index.posting_vector_len();
        let mut cursor = txn_idx
            .transaction()
            .open_cursor::<u32, Vec<u8>>(index.postings_table_name())?;
        let Some(data) = cursor.seek_exact(args.centroid) else {
            println!("centroid {} has no posting list", args.centroid);
            return Ok(());
        };
        let data = data?;
        let Some(block) = PostingBlock::new(&data, vector_len) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("malformed posting block for centroid {}", args.centroid),
            ));
        };
        block.iter().map(|(record_id, _)| record_id).collect()
    };

    println!(
        "centroid {}: {} vectors in posting",
        args.centroid,
        records.len()
    );

    // Recompute each record's assignment by searching the head with its rerank vector.
    let progress = progress_bar(records.len(), "reassign");
    let head_params = index.config().head_search_params;

    struct Worker {
        connection: Arc<Connection>,
        index: Arc<TableIndex>,
        searcher: GraphSearcher,
    }

    impl Worker {
        fn new(
            index: &Arc<TableIndex>,
            connection: &Arc<Connection>,
            head_params: GraphSearchParams,
        ) -> Self {
            Self {
                connection: Arc::clone(connection),
                index: Arc::clone(index),
                searcher: GraphSearcher::new(head_params),
            }
        }

        /// Look up `record_id`'s rerank vector and search the head index with it.
        fn recompute(&mut self, record_id: i64) -> io::Result<Target> {
            let reader =
                TransactionIndex::new(&self.index, self.connection.begin_transaction(None)?);

            // The rerank table stores the ingress-prepared vector (rotation and any normalization
            // already applied), so decoding it yields the same query used at insert time.
            let rerank_coder = self.index.config().rerank_format.coder();
            let query: Vec<f32> = {
                let mut raw_cursor = reader
                    .transaction()
                    .open_cursor::<i64, Vec<u8>>(self.index.raw_vectors_table_name())?;
                // SAFETY: no other WT operations occur before the returned slice is decoded.
                match unsafe { raw_cursor.seek_exact_unsafe(record_id) } {
                    Some(Ok(encoded)) => rerank_coder.decode(encoded),
                    Some(Err(e)) => return Err(e.into()),
                    None => return Ok(Target::MissingRerank),
                }
            };

            let centroids: Vec<Neighbor> = self.searcher.search(&query, reader.head())?;
            match centroids.first() {
                Some(c) => Ok(Target::Centroid(c.vertex() as u32)),
                None => Ok(Target::NoCandidates),
            }
        }
    }

    let assignments: Vec<(i64, Target)> = records
        .into_par_iter()
        .map_init(
            || Worker::new(&index, &connection, head_params),
            |worker, record_id| {
                let target = worker.recompute(record_id);
                progress.inc(1);
                target.map(|target| (record_id, target))
            },
        )
        .collect::<io::Result<Vec<_>>>()?;

    progress.finish_using_style();

    // Histogram of recomputed assignments.
    let mut histogram: HashMap<u32, usize> = HashMap::new();
    let mut missing_rerank = 0usize;
    let mut no_candidates = 0usize;
    for (_, target) in &assignments {
        match target {
            Target::Centroid(c) => *histogram.entry(*c).or_default() += 1,
            Target::MissingRerank => missing_rerank += 1,
            Target::NoCandidates => no_candidates += 1,
        }
    }

    let total = assignments.len();
    let mut ranked: Vec<(u32, usize)> = histogram.into_iter().collect();
    // Sort by descending count, then ascending centroid id for stable output.
    ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    println!(
        "\nrecomputed assignments ({} distinct centroids):",
        ranked.len()
    );
    for (centroid_id, count) in &ranked {
        let marker = if *centroid_id == args.centroid {
            "  (stays)"
        } else {
            ""
        };
        println!(
            "  centroid {:>10}: {:>8} ({:5.1}%){}",
            centroid_id,
            count,
            100.0 * *count as f64 / total.max(1) as f64,
            marker,
        );
    }
    if missing_rerank > 0 {
        println!("  missing rerank vector:    {missing_rerank:>8}");
    }
    if no_candidates > 0 {
        println!("  head search empty:        {no_candidates:>8}");
    }

    let staying = ranked
        .iter()
        .find(|(c, _)| *c == args.centroid)
        .map(|&(_, count)| count)
        .unwrap_or(0);
    println!(
        "\n{} of {} vectors stay in centroid {}",
        staying, total, args.centroid
    );
    if args.commit && staying == 0 && total > 0 {
        println!(
            "note: the posting list for centroid {} will become empty; \
             the centroid itself remains in the head index",
            args.centroid
        );
    }

    // Moves to apply: every record whose recomputed target is a different centroid.
    let moves: Vec<(i64, u32)> = assignments
        .into_iter()
        .filter_map(|(record_id, target)| match target {
            Target::Centroid(c) if c != args.centroid => Some((record_id, c)),
            _ => None,
        })
        .collect();

    if !args.commit {
        println!(
            "{} vectors would move out of centroid {} (dry run; pass --commit to apply)",
            moves.len(),
            args.centroid
        );
        return Ok(());
    }

    if moves.is_empty() {
        println!("nothing to move; centroid {} is unchanged", args.centroid);
        return Ok(());
    }

    let txn_idx = TransactionIndex::new(&index, connection.begin_transaction(None)?);
    let mut postings = BlockPostingsMut::from_txn(&txn_idx)?;
    let mut assignment_updater = CentroidAssignmentUpdater::new(&txn_idx)?;

    let mut moved: HashMap<u32, usize> = HashMap::new();
    let mut mismatched = 0usize;
    for (record_id, target) in &moves {
        // Read the posting vector before touching anything so a missing posting fails cleanly.
        let vector = postings.get(args.centroid, *record_id)?;
        let old = assignment_updater.update(*record_id, CentroidAssignment::new(*target))?;
        if old.primary_id != args.centroid {
            mismatched += 1;
            eprintln!(
                "warning: record {record_id} assignment says centroid {} but was found in posting {}",
                old.primary_id, args.centroid,
            );
        }
        postings.remove(args.centroid, *record_id)?;
        postings.insert(*target, *record_id, &vector)?;
        *moved.entry(*target).or_default() += 1;
    }

    postings.flush()?;
    assignment_updater.flush()?;
    drop(postings);
    drop(assignment_updater);
    txn_idx.commit(None)?;

    println!(
        "\ncommitted: moved {} vectors out of centroid {}{}",
        moves.len(),
        args.centroid,
        if mismatched > 0 {
            format!(" ({mismatched} assignment-table mismatches)")
        } else {
            String::new()
        },
    );
    let mut ranked: Vec<(u32, usize)> = moved.into_iter().collect();
    ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (centroid_id, count) in ranked {
        println!("  -> centroid {centroid_id:>10}: {count}");
    }

    Ok(())
}
