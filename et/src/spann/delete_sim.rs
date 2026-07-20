//! Delete simulation for a SPANN index.
//!
//! For every rerank (raw) vector in the corpus this reads the vector, searches the head (centroid)
//! index with it, then reads posting lists in centroid order until it finds the posting that
//! contains the vector's own record id. The depth (number of postings that had to be read) is
//! recorded. This models the cost of locating a record for deletion, and in particular surfaces
//! records that cannot be found in any of the centroids returned by the head search.

use std::{
    collections::{HashMap, HashSet},
    io::{self},
    num::NonZero,
    sync::Arc,
};

use clap::Args;
use easy_tiger::posting_block::PostingBlock;
use easy_tiger::{
    Neighbor,
    spann::{TableIndex, TransactionIndex},
    vamana::{GraphSearchParams, PatienceParams, search::GraphSearcher},
};
use wt_mdb::Connection;

use crate::ui::progress_bar;

#[derive(Args)]
pub struct DeleteSimArgs {
    /// Number of head (centroid) candidates in the search list.
    #[arg(long)]
    head_candidates: NonZero<usize>,
    /// Number of head (centroid) results to re-rank at the end of each search.
    /// If unset, use the same figure as --head-candidates.
    #[arg(long)]
    head_rerank_budget: Option<usize>,
    /// Patience saturation threshold.
    #[arg(long, default_value_t = 0.995)]
    head_patience_saturation_threshold: f64,
    /// Patience saturation count. If unset, patience early termination will not be used.
    #[arg(long)]
    head_patience_saturation_count: Option<usize>,
    /// Maximum number of postings (centroids) to read while searching for each record.
    /// If unset, read every centroid returned by the head search.
    #[arg(long)]
    max_postings: Option<NonZero<usize>>,
    /// Maximum number of records to simulate. If unset, process every record in the corpus.
    #[arg(short, long)]
    limit: Option<usize>,
    /// After the simulation, run a second phase that scans every posting to locate the records
    /// that could not be found, then compares the true distance to the centroid they actually
    /// live in against the distances returned by the head search. This is expensive (a full
    /// posting scan) but explains whether misses are due to head-search approximation.
    #[arg(long, default_value_t = false)]
    diagnose_missing: bool,
    /// Maximum number of per-record diagnostic lines to print in the diagnosis phase.
    #[arg(long, default_value_t = 50)]
    diagnose_sample: usize,
}

pub fn delete_sim(
    connection: Arc<Connection>,
    index_name: &str,
    args: DeleteSimArgs,
) -> io::Result<()> {
    let index = Arc::new(TableIndex::from_db(&connection, index_name)?);
    if index.config().rerank_format.is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "index has no rerank format; there are no rerank vectors to simulate deletes over",
        ));
    }

    let head_params = GraphSearchParams {
        beam_width: args.head_candidates,
        num_rerank: args
            .head_rerank_budget
            .unwrap_or(args.head_candidates.get()),
        patience: args.head_patience_saturation_count.map(|c| PatienceParams {
            saturation_threshold: args.head_patience_saturation_threshold,
            patience_count: c,
        }),
    };
    let max_postings = args.max_postings.map(NonZero::get).unwrap_or(usize::MAX);

    // Rather than scan the whole rerank table up front, find the largest record id and derive the
    // inclusive range [0, max_key]. Record ids are assumed to be dense-ish and non-negative;
    // missing keys in the range are simply skipped (not counted as processed).
    let max_key = {
        let txn_idx = TransactionIndex::new(&index, connection.begin_transaction(None)?);
        let mut cursor = txn_idx
            .transaction()
            .open_cursor::<i64, Vec<u8>>(index.raw_vectors_table_name())?;
        match cursor.largest_key() {
            Some(Ok(k)) => k,
            Some(Err(e)) => return Err(e.into()),
            None => {
                println!("rerank table is empty; nothing to simulate");
                return Ok(());
            }
        }
    };

    // Cap the number of candidate keys by --limit if provided.
    let last_key = match args.limit {
        Some(limit) if limit > 0 => std::cmp::min(max_key, limit as i64 - 1),
        Some(_) => return Ok(()),
        None => max_key,
    };
    let record_ids = 0..=last_key;

    let progress = progress_bar((last_key + 1) as usize, "delete-sim");

    let reduce = |a: SimStats, b: SimStats| a + b;

    let stats: SimStats = {
        use rayon::prelude::*;
        record_ids
            .into_par_iter()
            .map_init(
                || Worker::new(&index, &connection, head_params, args.diagnose_missing),
                |worker, record_id| {
                    let stats = worker.simulate(record_id, max_postings);
                    progress.inc(1);
                    stats
                },
            )
            .try_reduce(SimStats::default, |a, b| Ok(reduce(a, b)))?
    };

    progress.finish_using_style();
    stats.report();

    if args.diagnose_missing {
        diagnose_missing(&connection, &index, &stats.missing, args.diagnose_sample)?;
    }

    Ok(())
}

struct Worker {
    connection: Arc<Connection>,
    index: Arc<TableIndex>,
    searcher: GraphSearcher,
    diagnose: bool,
}

impl Worker {
    fn new(
        index: &Arc<TableIndex>,
        connection: &Arc<Connection>,
        head_params: GraphSearchParams,
        diagnose: bool,
    ) -> Self {
        Self {
            connection: Arc::clone(connection),
            index: Arc::clone(index),
            searcher: GraphSearcher::new(head_params),
            diagnose,
        }
    }

    /// Search the head index for the vector belonging to `record_id`, then read postings in
    /// centroid order until the posting containing `record_id` is found. Returns the outcome as a
    /// [`SimStats`] contribution.
    fn simulate(&mut self, record_id: i64, max_postings: usize) -> io::Result<SimStats> {
        let reader = TransactionIndex::new(&self.index, self.connection.begin_transaction(None)?);

        // Read and decode this record's rerank vector to use as the query.
        let rerank_coder = self
            .index
            .config()
            .rerank_format
            .expect("rerank format is set")
            .coder(self.index.head_config().config().similarity, None);
        let query: Vec<f32> = {
            let mut raw_cursor = reader
                .transaction()
                .open_record_cursor(self.index.raw_vectors_table_name())?;
            // SAFETY: no other WT operations occur before the returned slice is decoded.
            match unsafe { raw_cursor.seek_exact_unsafe(record_id) } {
                Some(Ok(encoded)) => rerank_coder.decode(encoded),
                Some(Err(e)) => return Err(e.into()),
                // No record at this key in the range; skip it without counting it as processed.
                None => return Ok(SimStats::skipped()),
            }
        };

        let centroids: Vec<Neighbor> = self.searcher.search(&query, reader.head())?;
        if centroids.is_empty() {
            return Ok(SimStats::not_found(self.make_missing(record_id, query, &[])));
        }

        let vector_len = self.index.posting_vector_len();
        let mut posting_cursor = reader
            .transaction()
            .open_cursor::<u32, Vec<u8>>(self.index.postings_table_name())?;

        for (rank, c) in centroids.iter().take(max_postings).enumerate() {
            let centroid_id: u32 = c.vertex().try_into().expect("centroid_id is a u32");
            // SAFETY: we are not performing any WT operations in between seeks.
            let data = match unsafe { posting_cursor.seek_exact_unsafe(centroid_id) } {
                Some(Ok(data)) => data,
                Some(Err(e)) => return Err(e.into()),
                None => continue,
            };
            let Some(block) = PostingBlock::new(data, vector_len) else {
                continue;
            };
            if block.lookup(record_id).is_some() {
                // depth is 1-based: number of postings read to find the record.
                return Ok(SimStats::found(rank + 1));
            }
        }

        Ok(SimStats::not_found(
            self.make_missing(record_id, query, &centroids),
        ))
    }

    /// Build a [`MissingRecord`] describing this not-found record and the centroids the head
    /// search returned, but only when the diagnosis phase is enabled (otherwise `None` to avoid
    /// retaining query vectors for the whole corpus).
    fn make_missing(
        &self,
        record_id: i64,
        query: Vec<f32>,
        head_centroids: &[Neighbor],
    ) -> Option<MissingRecord> {
        if !self.diagnose {
            return None;
        }
        Some(MissingRecord {
            record_id,
            query,
            head: head_centroids
                .iter()
                .map(|n| (n.vertex(), n.distance()))
                .collect(),
        })
    }
}

/// Details about a record that the head search failed to locate, retained for the optional
/// diagnosis phase.
#[derive(Clone)]
struct MissingRecord {
    record_id: i64,
    /// The record's rerank vector, used as the query.
    query: Vec<f32>,
    /// Centroids returned by the head search, as (centroid_id, search distance), ascending.
    head: Vec<(i64, f64)>,
}

#[derive(Default, Clone)]
struct SimStats {
    /// Keys in the range that had no record and were skipped.
    skipped: usize,
    processed: usize,
    found: usize,
    not_found: usize,
    depth_sum: u64,
    max_depth: usize,
    /// Number of records found at each depth (index 0 == depth 1).
    depth_histogram: Vec<usize>,
    /// Not-found records retained for diagnosis (empty unless --diagnose-missing).
    missing: Vec<MissingRecord>,
}

impl SimStats {
    fn found(depth: usize) -> Self {
        let mut depth_histogram = vec![0; depth];
        depth_histogram[depth - 1] = 1;
        Self {
            processed: 1,
            found: 1,
            depth_sum: depth as u64,
            max_depth: depth,
            depth_histogram,
            ..Default::default()
        }
    }

    fn not_found(missing: Option<MissingRecord>) -> Self {
        Self {
            processed: 1,
            not_found: 1,
            missing: missing.into_iter().collect(),
            ..Default::default()
        }
    }

    fn skipped() -> Self {
        Self {
            skipped: 1,
            ..Default::default()
        }
    }

    fn percentile(&self, p: f64) -> Option<usize> {
        if self.found == 0 {
            return None;
        }
        let target = (self.found as f64 * p).ceil() as usize;
        let target = target.max(1);
        let mut cumulative = 0;
        for (i, &count) in self.depth_histogram.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return Some(i + 1);
            }
        }
        Some(self.max_depth)
    }

    fn report(&self) {
        println!(
            "keys in range:     {} ({} skipped as missing)",
            self.processed + self.skipped,
            self.skipped
        );
        println!("records processed: {}", self.processed);
        println!(
            "found:             {} ({:.4}%)",
            self.found,
            100.0 * self.found as f64 / self.processed.max(1) as f64
        );
        println!(
            "NOT FOUND:         {} ({:.4}%)",
            self.not_found,
            100.0 * self.not_found as f64 / self.processed.max(1) as f64
        );
        if self.found > 0 {
            println!(
                "depth (postings read to locate record): mean {:.2} max {}",
                self.depth_sum as f64 / self.found as f64,
                self.max_depth,
            );
            for (label, p) in [("p50", 0.50), ("p90", 0.90), ("p99", 0.99)] {
                if let Some(d) = self.percentile(p) {
                    println!("  {label}: {d}");
                }
            }
        }
    }
}

impl std::ops::Add for SimStats {
    type Output = SimStats;

    fn add(mut self, mut rhs: SimStats) -> SimStats {
        self.skipped += rhs.skipped;
        self.processed += rhs.processed;
        self.found += rhs.found;
        self.not_found += rhs.not_found;
        self.depth_sum += rhs.depth_sum;
        self.max_depth = self.max_depth.max(rhs.max_depth);
        if self.depth_histogram.len() < rhs.depth_histogram.len() {
            self.depth_histogram.resize(rhs.depth_histogram.len(), 0);
        }
        for (slot, count) in self
            .depth_histogram
            .iter_mut()
            .zip(rhs.depth_histogram.iter())
        {
            *slot += count;
        }
        self.missing.append(&mut rhs.missing);
        self
    }
}

/// Second, expensive phase: scan every posting to find where each not-found record actually lives,
/// then compare the true distance to that centroid against the distances the head search returned.
///
/// This distinguishes two failure modes:
///   * `search-miss`: the record's true centroid is at least as close as the farthest centroid the
///     head search *did* return, so the graph search should have reached it — an approximation
///     failure in the head index search.
///   * `beyond-candidates`: the true centroid is farther than every returned centroid, so it simply
///     falls outside the candidate list at this `--head-candidates` setting.
///   * `orphaned`: the record was not found in any posting at all.
fn diagnose_missing(
    connection: &Arc<Connection>,
    index: &Arc<TableIndex>,
    missing: &[MissingRecord],
    sample: usize,
) -> io::Result<()> {
    if missing.is_empty() {
        println!("\nno missing records to diagnose");
        return Ok(());
    }

    println!(
        "\n=== diagnosis: scanning all postings to locate {} missing records ===",
        missing.len()
    );

    // record_id -> centroids whose posting contains it.
    let missing_ids: HashSet<i64> = missing.iter().map(|m| m.record_id).collect();
    let mut located: HashMap<i64, Vec<u32>> = HashMap::new();
    // Total number of vectors in each centroid, captured during the scan.
    let mut centroid_sizes: HashMap<u32, usize> = HashMap::new();

    let reader = TransactionIndex::new(index, connection.begin_transaction(None)?);
    let vector_len = index.posting_vector_len();
    {
        let cursor = reader
            .transaction()
            .open_cursor::<u32, Vec<u8>>(index.postings_table_name())?;
        let progress = progress_bar(0, "scan-postings");
        for item in cursor {
            let (centroid_id, data) = item?;
            progress.inc(1);
            let Some(block) = PostingBlock::new(&data, vector_len) else {
                continue;
            };
            centroid_sizes.insert(centroid_id, block.len());
            for (record_id, _) in block.iter() {
                if missing_ids.contains(&record_id) {
                    located.entry(record_id).or_default().push(centroid_id);
                }
            }
        }
        progress.finish_using_style();
    }

    // Distance between the (raw f32) query and the centroid's quantized vector, computed with an
    // asymmetric query distance in the head's coding rather than decoding the centroid.
    let similarity = index.head_config().config().similarity;
    let hf_table = reader.head().index().high_fidelity_table();
    let hf_format = hf_table.format();
    let hf_center = hf_table.centroid().map(|c| c.to_vec());
    let hf_name = hf_table.name().to_owned();
    let mut hf_cursor = reader.transaction().open_record_cursor(&hf_name)?;

    let mut search_miss = 0usize;
    let mut beyond_candidates = 0usize;
    let mut orphaned = 0usize;
    let mut printed = 0usize;
    // How many missing records actually live in each centroid.
    let mut per_centroid: HashMap<u32, usize> = HashMap::new();

    for m in missing {
        let centroids = located.get(&m.record_id);
        let (worst_returned, closest_returned) = (
            m.head.last().map(|(_, d)| *d),
            m.head.first().map(|(_, d)| *d),
        );

        // Distance from this record's query to each candidate centroid, using an asymmetric
        // quantized distance against the centroid's stored (encoded) vector.
        let query_distance =
            hf_format.query_distance_asymmetric(similarity, &m.query, hf_center.as_deref());
        let mut best: Option<(u32, f64)> = None;
        if let Some(centroids) = centroids {
            for &centroid_id in centroids {
                // SAFETY: the encoded vector is scored before the next cursor operation.
                let encoded = match unsafe { hf_cursor.seek_exact_unsafe(centroid_id as i64) } {
                    Some(Ok(encoded)) => encoded,
                    Some(Err(e)) => return Err(e.into()),
                    None => continue,
                };
                let dist = query_distance.distance(encoded);
                if best.is_none_or(|(_, bd)| dist < bd) {
                    best = Some((centroid_id, dist));
                }
            }
        }

        let Some((actual_centroid, actual_dist)) = best else {
            orphaned += 1;
            if printed < sample {
                println!(
                    "  record {:>10}: ORPHANED (not present in any posting)",
                    m.record_id
                );
                printed += 1;
            }
            continue;
        };

        *per_centroid.entry(actual_centroid).or_default() += 1;

        // How many returned centroids were strictly closer than the true centroid?
        let closer_returned = m.head.iter().filter(|(_, d)| *d < actual_dist).count();
        let classification = match worst_returned {
            Some(worst) if actual_dist <= worst => {
                search_miss += 1;
                "search-miss"
            }
            _ => {
                beyond_candidates += 1;
                "beyond-candidates"
            }
        };

        if printed < sample {
            println!(
                "  record {:>10}: {:<17} actual centroid {} dist {:.5} | head returned {} centroids, \
                 closest {:.5} worst {:.5} | {} returned closer than actual",
                m.record_id,
                classification,
                actual_centroid,
                actual_dist,
                m.head.len(),
                closest_returned.unwrap_or(f64::NAN),
                worst_returned.unwrap_or(f64::NAN),
                closer_returned,
            );
            printed += 1;
        }
    }

    if missing.len() > sample {
        println!("  ... {} more not shown", missing.len() - sample);
    }

    println!("\ndiagnosis summary ({} missing records):", missing.len());
    println!(
        "  search-miss:       {} (true centroid within the returned distance range; head search approximation)",
        search_miss
    );
    println!(
        "  beyond-candidates: {} (true centroid farther than every returned centroid)",
        beyond_candidates
    );
    println!(
        "  orphaned:          {} (record absent from every posting)",
        orphaned
    );

    if !per_centroid.is_empty() {
        let mut ranked: Vec<(u32, usize)> = per_centroid.into_iter().collect();
        // Sort by descending count, then ascending centroid id for stable output.
        ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        println!(
            "\ncentroids containing missing vectors: {} distinct centroids",
            ranked.len()
        );
        for (centroid_id, count) in ranked.iter().take(sample) {
            let total = centroid_sizes.get(centroid_id).copied().unwrap_or(0);
            println!("  centroid {centroid_id:>10}: {count} missing of {total} total");
        }
        if ranked.len() > sample {
            println!("  ... {} more centroids not shown", ranked.len() - sample);
        }
    }

    Ok(())
}
