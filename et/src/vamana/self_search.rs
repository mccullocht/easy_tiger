//! Self-search probe for a vamana index.
//!
//! For every vertex in the graph (or a caller-provided list of ids) this searches the index using
//! the vertex's own stored high-fidelity vector as the query, at configurable search parameters,
//! and records the vertex's own fate via the search trace. The vertex is its own exact nearest
//! neighbor at ~distance zero, so this is the most favorable query that can be issued for it:
//!
//! * `Found` (rank 0): the graph is navigable toward the vertex. If query-time searches still miss
//!   it, the failure is in the trajectory of those queries (beam width, candidate admission,
//!   patience), not the graph.
//! * `Unseen`: traversal never scored the vertex -- no expanded vertex has an edge to it. With
//!   `--trace`, the reported closest-approach vertex (the scored vertex nearest the target; because
//!   the query *is* the target's vector, scored nav distances are distances to the target) and
//!   whether it links to the target distinguish a missing edge (close approach, no edge) from a
//!   detached region (far approach).
//! * `Seen`/`RerankDropped`: the vertex was reached but outranked; a search-parameter problem.

use std::{
    io,
    num::NonZero,
    path::PathBuf,
    sync::Arc,
};

use clap::Args;
use easy_tiger::vamana::{
    Graph, GraphSearchParams, GraphVectorIndex, GraphVectorStore, PatienceParams,
    search::{GraphSearcher, GraphSearchStats, ScoredVertex, VertexTrace},
    wt::{ENTRY_POINT_KEY, TableGraphVectorIndex, TransactionGraphVectorIndex},
};
use serde::Serialize;
use wt_mdb::{Connection, Error};

use crate::ui::progress_bar;

#[derive(Args)]
pub struct SelfSearchArgs {
    /// Maximum number of vertices to probe. If unset, probe every vertex in the graph.
    #[arg(short, long)]
    limit: Option<usize>,
    /// Path to a file of vertex ids (one per line, blank lines and #-comments ignored) to probe
    /// instead of scanning the graph table. Useful for re-probing a specific list of offenders.
    #[arg(long)]
    ids: Option<PathBuf>,
    /// Beam width (candidate list capacity) for the probe search. If unset, use the index's
    /// build-time search parameters. NB: query-time parameters may differ from build-time ones
    /// (e.g. a SPANN head is searched with --head-candidates at query time); pass them explicitly
    /// to probe under production conditions.
    #[arg(long)]
    beam_width: Option<NonZero<usize>>,
    /// Number of results to rerank at the end of each probe search. If unset, use the index's
    /// build-time value.
    #[arg(long)]
    rerank_budget: Option<usize>,
    /// Patience saturation threshold (only used with --patience-count).
    #[arg(long, default_value_t = 0.995)]
    patience_threshold: f64,
    /// Patience saturation count. If set, overrides the index's build-time patience; zero
    /// disables patience early termination. If unset, the build-time patience (if any) is used.
    #[arg(long)]
    patience_count: Option<usize>,
    /// Emit one JSON line per probed vertex with its trace outcome and, for vertices that were
    /// reached but not found, the closest-approach diagnostics.
    #[arg(long)]
    trace: bool,
}

/// Per-vertex trace line emitted with --trace.
#[derive(Serialize)]
struct ProbeTrace {
    id: i64,
    outcome: VertexTrace,
    stats: GraphSearchStats,
    /// Present for non-`Found` outcomes: the scored vertex that came closest to the target and
    /// whether it has an edge to it.
    closest: Option<ClosestApproach>,
}

#[derive(Serialize)]
struct ClosestApproach {
    id: i64,
    distance: f64,
    has_edge: bool,
}

pub fn self_search(
    connection: Arc<Connection>,
    index_name: &str,
    args: SelfSearchArgs,
) -> io::Result<()> {
    let index = Arc::new(TableGraphVectorIndex::from_db(&connection, index_name)?);

    // Start from the index's own build-time search parameters and apply overrides.
    let mut params = index.config().index_search_params;
    if let Some(beam_width) = args.beam_width {
        params.beam_width = beam_width;
    }
    if let Some(rerank_budget) = args.rerank_budget {
        params.num_rerank = rerank_budget;
    }
    if let Some(patience_count) = args.patience_count {
        params.patience = (patience_count > 0).then_some(PatienceParams {
            saturation_threshold: args.patience_threshold,
            patience_count,
        });
    }

    let ids: Vec<i64> = match &args.ids {
        Some(path) => parse_ids(std::fs::read_to_string(path)?)?,
        None => {
            let txn = connection.begin_transaction(None)?;
            let scan = txn.open_record_cursor(index.graph_table_name())?;
            let mut ids = Vec::new();
            for result in scan {
                let (key, _) = result?;
                if key != ENTRY_POINT_KEY {
                    ids.push(key);
                }
            }
            ids
        }
    };
    let ids = match args.limit {
        Some(limit) => ids.into_iter().take(limit).collect::<Vec<_>>(),
        None => ids,
    };
    if ids.is_empty() {
        println!("no vertices to probe");
        return Ok(());
    }

    println!(
        "probing {} vertices at beam_width {} num_rerank {} patience {}",
        ids.len(),
        params.beam_width.get(),
        params.num_rerank,
        params
            .patience
            .map_or("off".to_owned(), |p| format!(
                "on (threshold {}, count {})",
                p.saturation_threshold, p.patience_count
            )),
    );

    let progress = progress_bar(ids.len(), "self-search");
    let stats: ProbeStats = {
        use rayon::prelude::*;
        ids.into_par_iter()
            .map_init(
                || Worker::new(&index, &connection, params, args.trace),
                |worker, id| {
                    let stats = worker.probe(id);
                    progress.inc(1);
                    stats
                },
            )
            .try_reduce(ProbeStats::default, |a, b| Ok(a + b))?
    };
    progress.finish_using_style();
    stats.report();

    Ok(())
}

fn parse_ids(contents: String) -> io::Result<Vec<i64>> {
    contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            line.parse::<i64>().map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("{line:?}: {e}"))
            })
        })
        .collect()
}

struct Worker {
    connection: Arc<Connection>,
    index: Arc<TableGraphVectorIndex>,
    searcher: GraphSearcher,
    trace: bool,
}

impl Worker {
    fn new(
        index: &Arc<TableGraphVectorIndex>,
        connection: &Arc<Connection>,
        params: GraphSearchParams,
        trace: bool,
    ) -> Self {
        Self {
            connection: Arc::clone(connection),
            index: Arc::clone(index),
            searcher: GraphSearcher::new(params),
            trace,
        }
    }

    /// Search for `id` using its own stored high-fidelity vector, returning the outcome as a
    /// [`ProbeStats`] contribution.
    fn probe(&mut self, id: i64) -> io::Result<ProbeStats> {
        let txn = TransactionGraphVectorIndex::new(
            Arc::clone(&self.index),
            self.connection.begin_transaction(None)?,
        );

        // Read and decode the vertex's high-fidelity vector to use as the query. Stored vectors
        // are centered against the index centroid (if any); un-center so the search's own query
        // preparation re-centers it exactly like a raw query vector.
        let mut query: Vec<f32> = {
            let mut store = txn.high_fidelity_vectors()?;
            let coder = store.new_coder();
            let encoded = store
                .get(id)
                .unwrap_or_else(|| Err(Error::not_found_error()))?
                .to_vec();
            coder.decode(&encoded)
        };
        if let Some(centroid) = &self.index.config().centroid {
            for (q, c) in query.iter_mut().zip(centroid.iter()) {
                *q += c;
            }
        }

        let (_results, trace) = self.searcher.search_with_trace(&query, &txn, &[id])?;
        let stats = self.searcher.stats();
        let outcome = trace.vectors[0].trace;

        let closest = if matches!(outcome, VertexTrace::Found { .. }) {
            None
        } else {
            closest_approach(&txn, id, &trace.scored)?
        };
        if self.trace {
            let probe_trace = ProbeTrace {
                id,
                outcome,
                stats,
                closest,
            };
            println!(
                "{}",
                serde_json::to_string(&probe_trace).expect("ProbeTrace is serializable")
            );
        }

        Ok(ProbeStats::outcome(outcome, stats))
    }
}

/// The scored vertex that came closest to the query. Because the probe's query is the target's own
/// vector, scored nav distances are distances *to the target*, so the minimum over `scored` is the
/// search's closest approach to it. `has_edge` reports whether that vertex links to the target --
/// it cannot when the outcome was `Unseen`, which is itself the missing-edge evidence.
fn closest_approach(
    txn: &TransactionGraphVectorIndex,
    id: i64,
    scored: &[ScoredVertex],
) -> io::Result<Option<ClosestApproach>> {
    let Some(closest) = scored
        .iter()
        .filter(|s| s.id != id)
        .min_by(|a, b| a.distance.total_cmp(&b.distance))
    else {
        return Ok(None);
    };
    let has_edge = txn
        .graph()?
        .edges(closest.id)
        .transpose()?
        .is_some_and(|mut edges| edges.any(|e| e == id));
    Ok(Some(ClosestApproach {
        id: closest.id,
        distance: closest.distance,
        has_edge,
    }))
}

#[derive(Default, Clone)]
struct ProbeStats {
    probed: usize,
    not_found: usize,
    unseen: usize,
    seen: usize,
    rerank_dropped: usize,
    found: usize,
    rank_sum: u64,
    max_rank: usize,
    /// Number of vertices found at each rank (index 0 == rank 0).
    rank_histogram: Vec<usize>,
    visited_sum: u64,
    max_visited: usize,
    /// Number of vertices whose search visited each number of vertices (index 0 == 1 visited).
    visited_histogram: Vec<usize>,
    candidates_sum: u64,
}

impl ProbeStats {
    fn outcome(outcome: VertexTrace, stats: GraphSearchStats) -> Self {
        let visited = stats.visited;
        let mut s = Self {
            probed: 1,
            visited_sum: visited as u64,
            max_visited: visited,
            visited_histogram: vec![0; visited],
            candidates_sum: stats.candidates as u64,
            ..Default::default()
        };
        if visited > 0 {
            s.visited_histogram[visited - 1] = 1;
        }
        match outcome {
            VertexTrace::Found { rank, .. } => {
                s.found = 1;
                s.rank_sum = rank as u64;
                s.max_rank = rank;
                s.rank_histogram = vec![0; rank + 1];
                s.rank_histogram[rank] = 1;
            }
            VertexTrace::Seen { .. } => s.seen = 1,
            VertexTrace::RerankDropped { .. } => s.rerank_dropped = 1,
            VertexTrace::Unseen => s.unseen = 1,
            VertexTrace::NotFound => s.not_found = 1,
        }
        s
    }

    /// Percentile over a histogram where index `i` counts value `i`, returned as `i` itself.
    fn percentile(histogram: &[usize], total: usize, p: f64) -> Option<usize> {
        if total == 0 {
            return None;
        }
        let target = ((total as f64) * p).ceil() as usize;
        let mut cumulative = 0;
        for (i, &count) in histogram.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return Some(i);
            }
        }
        None
    }

    fn report(&self) {
        let probed = self.probed.max(1);
        println!("vertices probed: {}", self.probed);
        for (label, count) in [
            ("found", self.found),
            ("seen (reached, outranked)", self.seen),
            ("rerank dropped", self.rerank_dropped),
            ("UNSEEN (never scored)", self.unseen),
            ("not found in nav table", self.not_found),
        ] {
            println!("{label:<24} {} ({:.4}%)", count, 100.0 * count as f64 / probed as f64);
        }
        if self.found > 0 {
            println!(
                "found rank: mean {:.2} max {}",
                self.rank_sum as f64 / self.found as f64,
                self.max_rank,
            );
            for (label, p) in [("p50", 0.50), ("p90", 0.90), ("p99", 0.99)] {
                if let Some(d) = Self::percentile(&self.rank_histogram, self.found, p) {
                    println!("  {label}: {d}");
                }
            }
        }
        if self.unseen > 0 {
            println!(
                "unseen vertices: re-run with --trace for closest-approach diagnostics, or with a \
                 larger --beam-width to test whether they are reachable at all"
            );
        }
        println!(
            "visited per search: mean {:.2} max {}",
            self.visited_sum as f64 / probed as f64,
            self.max_visited,
        );
        for (label, p) in [("p50", 0.50), ("p90", 0.90), ("p99", 0.99)] {
            if let Some(d) = Self::percentile(&self.visited_histogram, self.probed, p) {
                println!("  {label}: {}", d + 1);
            }
        }
        println!(
            "candidates per search: mean {:.2}",
            self.candidates_sum as f64 / probed as f64
        );
    }
}

impl std::ops::Add for ProbeStats {
    type Output = ProbeStats;

    fn add(mut self, rhs: ProbeStats) -> ProbeStats {
        self.probed += rhs.probed;
        self.not_found += rhs.not_found;
        self.unseen += rhs.unseen;
        self.seen += rhs.seen;
        self.rerank_dropped += rhs.rerank_dropped;
        self.found += rhs.found;
        self.rank_sum += rhs.rank_sum;
        self.max_rank = self.max_rank.max(rhs.max_rank);
        merge_histogram(&mut self.rank_histogram, &rhs.rank_histogram);
        self.visited_sum += rhs.visited_sum;
        self.max_visited = self.max_visited.max(rhs.max_visited);
        merge_histogram(&mut self.visited_histogram, &rhs.visited_histogram);
        self.candidates_sum += rhs.candidates_sum;
        self
    }
}

fn merge_histogram(target: &mut Vec<usize>, rhs: &[usize]) {
    if target.len() < rhs.len() {
        target.resize(rhs.len(), 0);
    }
    for (slot, count) in target.iter_mut().zip(rhs.iter()) {
        *slot += count;
    }
}
