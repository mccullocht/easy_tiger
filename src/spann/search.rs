//! SPANN search implementation.

use std::{
    cmp::Ordering,
    collections::HashSet,
    io,
    num::NonZero,
    ops::{Add, AddAssign},
    str::FromStr,
};

use ahash::{HashMap, HashMapExt};
use min_max_heap::MinMaxHeap;
use serde::{Deserialize, Serialize};
use tracing::warn;
use vectors::EstimatedDistance;
use wt_mdb::{Result, TypedCursorGuard};

use crate::{
    Neighbor,
    posting_block::PostingBlock,
    spann::{CentroidAssignment, TransactionIndex, centroid_stats::CentroidStats},
    vamana::{
        GraphSearchParams, GraphVectorIndex,
        search::{GraphSearchStats, GraphSearcher},
    },
};

/// The algorithm used to select centroids to search in the tail index.
#[derive(Debug, Copy, Clone)]
pub enum CentroidSelectorAlgorithm {
    /// Select the top N centroids based on the head graph search.
    TopN(usize),
    /// Select centroids from closest to farthest until we will score the request number of vectors.
    VectorCount(usize),
}

impl FromStr for CentroidSelectorAlgorithm {
    type Err = io::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            topn if topn.starts_with("top_n:") => {
                let n = topn
                    .strip_prefix("top_n:")
                    .expect("starts with prefix")
                    .parse::<usize>()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid number"))?;
                Ok(Self::TopN(n))
            }
            vc if vc.starts_with("vector_count:") => {
                let n = vc
                    .strip_prefix("vector_count:")
                    .expect("starts with prefix")
                    .parse::<usize>()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid number"))?;
                Ok(Self::VectorCount(n))
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unknown centroid selection algorithm",
            )),
        }
    }
}

impl std::fmt::Display for CentroidSelectorAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TopN(n) => write!(f, "top_n:{}", n),
            Self::VectorCount(n) => write!(f, "vector_count:{}", n),
        }
    }
}

/// Selects a set of centroids to search the tail postings of.
#[derive(Debug, Clone)]
pub enum CentroidSelector {
    /// Select the top N centroids by distance.
    TopN(usize),
    /// Select centroids until we will score the requested number of vectors, using statistics about
    /// the distribution of vectors across centroids.
    VectorCount { count: usize, stats: CentroidStats },
}

impl CentroidSelector {
    /// Create a new centroid selector based on the algorithm and data that can be derived from the
    /// index.
    pub fn new(algorithm: CentroidSelectorAlgorithm, txn_idx: &TransactionIndex) -> Result<Self> {
        match algorithm {
            CentroidSelectorAlgorithm::TopN(n) => Ok(Self::TopN(n)),
            CentroidSelectorAlgorithm::VectorCount(n) => {
                let stats = CentroidStats::from_index_stats(txn_idx)?;
                Ok(Self::VectorCount { count: n, stats })
            }
        }
    }

    /// Select the set of centroids to search from 'candidates'.
    pub fn select(&self, mut candidates: Vec<Neighbor>) -> Vec<Neighbor> {
        match self {
            Self::TopN(n) => {
                candidates.truncate(*n);
            }
            Self::VectorCount { count, stats } => {
                let mut selected = 0;
                for (i, c) in candidates.iter().enumerate() {
                    if selected >= *count {
                        candidates.truncate(i);
                        break;
                    }
                    selected += stats
                        .assignment_counts(c.vertex() as usize)
                        .map_or(0usize, |counts| counts.total() as usize);
                }
            }
        };
        candidates
    }
}

/// Tuning parameters for searching a SPANN index.
#[derive(Debug, Clone)]
pub struct SearchParams {
    /// Parameters for searching the head graph.
    /// NB: `head_params.beam_width` should be at least as large as `num_centroids`
    pub head_params: GraphSearchParams,
    /// Selects the centroids from candidates produced by searching the head graph.
    pub centroid_selector: CentroidSelector,
    /// The number of vectors to rerank using raw vectors.
    pub num_rerank: usize,
    /// The number of results to return.
    pub limit: NonZero<usize>,
    /// Z-score to use for estimated distance bound fed to rerank queue.
    pub z_score: f64,
}

/// Statistics for SPANN searches.
#[derive(Debug, Copy, Clone, Default, Serialize)]
pub struct SearchStats {
    /// Stats from the search of the head graph.
    pub head: GraphSearchStats,
    /// Number of posting lists read.
    pub postings_read: usize,
    /// Number of posting vectors read.
    pub posting_vectors_read: usize,
    /// Number of posting entries scored.
    pub posting_vectors_scored: usize,
    /// Number of results reranked using raw vectors.
    pub posting_vectors_reranked: usize,
}

impl Add for SearchStats {
    type Output = SearchStats;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            head: self.head + rhs.head,
            postings_read: self.postings_read + rhs.postings_read,
            posting_vectors_read: self.posting_vectors_read + rhs.posting_vectors_read,
            posting_vectors_scored: self.posting_vectors_scored + rhs.posting_vectors_scored,
            posting_vectors_reranked: self.posting_vectors_reranked + rhs.posting_vectors_reranked,
        }
    }
}

impl AddAssign for SearchStats {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

/// Maintains information about the status of a single traced vector during search.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum VectorTrace {
    /// The named vector does not exist in the index.
    NotFound,
    /// The centroid that this vector is assigned to was not returned by the head query.
    CentroidMissed { centroid_id: u32 },
    /// The centroid was returned from the head index but pruned before searching postings.
    CentroidPruned { centroid_id: u32, rank: usize },
    /// The centroid was searched but the vector was not found.
    PostingNotFound { centroid_id: u32 },
    /// The posting was found and distance was computed, but the posting was not competitive enough
    /// to be reranked. Note that error is not Z-adjusted.
    PostingQueuePruned { distance: f64, error: f64 },
    /// The vector was reranked but ranked beyond the limit at the end.
    RerankPruned { rank: usize, distance: f64 },
    /// The vector was found and returned.
    Found,
}

struct SearchTraceState {
    ids: HashMap<i64, (usize, VectorTrace)>,
    centroids: HashMap<u32, Vec<i64>>,
}

impl SearchTraceState {
    fn new(vectors: &[i64], index: &TransactionIndex) -> Result<Self> {
        let mut cursor = index.transaction().open_cursor::<i64, CentroidAssignment>(
            index.index().centroid_assignments_table_name(),
        )?;
        let mut ids = HashMap::<i64, (usize, VectorTrace)>::with_capacity(vectors.len());
        let mut centroids = HashMap::<u32, Vec<i64>>::with_capacity(vectors.len());
        for (i, &v) in vectors.iter().enumerate() {
            if let Some(a) = cursor.seek_exact(v) {
                let centroid_id = a?.primary_id;
                ids.insert(v, (i, VectorTrace::CentroidMissed { centroid_id }));
                centroids.entry(centroid_id).or_default().push(v);
            } else {
                ids.insert(v, (i, VectorTrace::NotFound));
            }
        }
        Ok(Self { ids, centroids })
    }

    fn observe_seen_centroids(&mut self, centroids: impl Iterator<Item = u32>) {
        for (i, cid) in centroids.enumerate() {
            let Some(ids) = self.centroids.get(&cid) else {
                continue;
            };
            for id in ids {
                self.ids.get_mut(id).unwrap().1 = VectorTrace::CentroidPruned {
                    centroid_id: cid,
                    rank: i,
                };
            }
        }
    }

    fn observe_selected_centroids(&mut self, centroids: impl Iterator<Item = u32>) {
        for cid in centroids {
            let Some(ids) = self.centroids.get(&cid) else {
                continue;
            };
            for id in ids {
                self.ids.get_mut(id).unwrap().1 = VectorTrace::PostingNotFound { centroid_id: cid };
            }
        }
    }

    #[inline]
    fn observe_posting(&mut self, vector: i64, d: EstimatedDistance) {
        if let Some((_, trace)) = self.ids.get_mut(&vector) {
            *trace = VectorTrace::PostingQueuePruned {
                distance: d.distance,
                error: d.error,
            }
        }
    }

    fn observe_rerank(&mut self, results: &[Neighbor], limit: usize) {
        for (i, r) in results.iter().enumerate() {
            let Some((_, trace)) = self.ids.get_mut(&r.vertex()) else {
                continue;
            };
            *trace = if i < limit {
                VectorTrace::Found
            } else {
                VectorTrace::RerankPruned {
                    rank: i,
                    distance: r.distance(),
                }
            }
        }
    }

    fn into_trace(self) -> Vec<VectorTrace> {
        let mut traces = vec![];
        for (rank, trace) in self.ids.into_iter().map(|(_, s)| s) {
            if rank >= traces.len() {
                traces.resize(rank + 1, VectorTrace::NotFound);
            }
            traces[rank] = trace;
        }
        traces
    }
}

pub struct Searcher {
    params: SearchParams,
    head_searcher: GraphSearcher,
    seen: HashSet<i64>,
    stats: SearchStats,
}

impl Searcher {
    pub fn new(params: SearchParams) -> Self {
        Self {
            head_searcher: GraphSearcher::new(params.head_params),
            params,
            seen: HashSet::new(),
            stats: SearchStats::default(),
        }
    }

    pub fn stats(&self) -> SearchStats {
        self.stats
    }

    pub fn search(
        &mut self,
        query: &[f32],
        reader: &TransactionIndex,
        posting_cursor: &mut TypedCursorGuard<'_, u32, Vec<u8>>,
    ) -> Result<Vec<Neighbor>> {
        self.search_with_trace(query, reader, posting_cursor, None)
            .map(|x| x.0)
    }

    pub fn search_with_trace(
        &mut self,
        query: &[f32],
        reader: &TransactionIndex,
        posting_cursor: &mut TypedCursorGuard<'_, u32, Vec<u8>>,
        traced: Option<&[i64]>,
    ) -> Result<(Vec<Neighbor>, Option<Vec<VectorTrace>>)> {
        self.stats = SearchStats::default();

        // Apply the ingress rotation (if configured) once, up front: every downstream distance
        // computation (head search, posting scoring, rerank) then operates in rotated space,
        // matching the rotated vectors written during ingestion.
        let prepared_query = vectors::prepare_vector(
            query,
            reader.index().rotator(),
            reader.head().config().similarity.angular(),
            None,
        );
        let query: &[f32] = &prepared_query;

        let mut trace_state = if let Some(vectors) = traced {
            Some(SearchTraceState::new(vectors, reader)?)
        } else {
            None
        };

        let mut centroids = self.head_searcher.search(query, reader.head())?;
        self.stats.head = self.head_searcher.stats();
        if centroids.is_empty() {
            return Ok((vec![], trace_state.map(|s| s.into_trace())));
        }
        if let Some(t) = trace_state.as_mut() {
            t.observe_seen_centroids(centroids.iter().map(|n| n.vertex() as u32));
        }

        centroids = self.params.centroid_selector.select(centroids);
        if let Some(t) = trace_state.as_mut() {
            t.observe_selected_centroids(centroids.iter().map(|n| n.vertex() as u32));
        }
        self.stats.postings_read = centroids.len();

        self.seen.clear();
        let dist_fn = reader
            .index
            .config()
            .posting_coder
            .query_distance_asymmetric(reader.index().head_config().config().similarity, query);
        let mut result_queue = ResultQueue::new(self.params.limit.get(), self.params.z_score);
        let vector_len = reader.index().posting_vector_len();
        for c in centroids {
            let centroid_id: u32 = c.vertex().try_into().expect("centroid_id is a u32");
            // SAFETY: we are not performing any WT operations in between seeks.
            let data = match unsafe { posting_cursor.seek_exact_unsafe(centroid_id) } {
                Some(Ok(data)) => data,
                Some(Err(e)) => {
                    warn!("failed to read posting for centroid {centroid_id}: {e}");
                    continue;
                }
                None => continue,
            };
            let Some(block) = PostingBlock::new(data, vector_len) else {
                warn!("malformed posting block for centroid {centroid_id}");
                continue;
            };
            for (record_id, vector) in block.iter() {
                self.stats.posting_vectors_read += 1;
                if !self.seen.insert(record_id) {
                    continue; // already seen
                }
                let d = dist_fn.estimated_distance(vector);
                if let Some(t) = trace_state.as_mut() {
                    t.observe_posting(record_id, d);
                }
                result_queue.push(record_id, d);
            }
        }

        self.stats.posting_vectors_scored = result_queue.scored;

        self.maybe_rerank_results(query, result_queue, trace_state, reader)
    }

    fn maybe_rerank_results(
        &mut self,
        query: &[f32],
        result_queue: ResultQueue,
        mut trace_state: Option<SearchTraceState>,
        reader: &TransactionIndex,
    ) -> Result<(Vec<Neighbor>, Option<Vec<VectorTrace>>)> {
        if self.params.num_rerank == 0 {
            return Ok((
                result_queue.into_results(),
                trace_state.map(|s| s.into_trace()),
            ));
        }

        let format = reader.index().config().rerank_format;
        let query = format.query_distance_asymmetric(reader.head.config().similarity, query);
        let mut raw_cursor = reader
            .transaction()
            .open_record_cursor(&reader.index().table_names.raw_vectors)?;
        let results = result_queue.into_results();
        self.stats.posting_vectors_reranked = results.len().min(self.params.num_rerank);
        let mut reranked = results
            .into_iter()
            .take(self.params.num_rerank)
            .map(|n| {
                Ok(Neighbor::new(
                    n.vertex(),
                    query.distance(unsafe {
                        raw_cursor
                            .seek_exact_unsafe(n.vertex())
                            .expect("raw vector for candidate")?
                    }),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        reranked.sort_unstable();
        if let Some(t) = trace_state.as_mut() {
            t.observe_rerank(&reranked, self.params.limit.get());
        }
        reranked.truncate(self.params.limit.get());

        Ok((reranked, trace_state.map(|s| s.into_trace())))
    }
}

#[derive(Debug, Copy, Clone)]
struct ErrorBoundNeighbor {
    n: Neighbor,
    e: EstimatedDistance,
}

impl ErrorBoundNeighbor {
    fn from_lower(vector_id: i64, e: EstimatedDistance) -> Self {
        Self {
            n: Neighbor::new(vector_id, e.distance - e.error),
            e,
        }
    }

    fn from_upper(vector_id: i64, e: EstimatedDistance) -> Self {
        Self {
            n: Neighbor::new(vector_id, e.distance + e.error),
            e,
        }
    }
}

impl PartialEq for ErrorBoundNeighbor {
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n
    }
}

impl Eq for ErrorBoundNeighbor {}

impl PartialOrd for ErrorBoundNeighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ErrorBoundNeighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.n.cmp(&other.n)
    }
}

struct ResultQueue {
    max_len: usize,
    z_score: f64,

    results: MinMaxHeap<ErrorBoundNeighbor>,
    overflow: MinMaxHeap<ErrorBoundNeighbor>,
    scored: usize,
}

impl ResultQueue {
    fn new(max_len: usize, z_score: f64) -> Self {
        Self {
            max_len,
            z_score,
            results: MinMaxHeap::with_capacity(max_len),
            overflow: MinMaxHeap::new(),
            scored: 0,
        }
    }

    fn push(&mut self, vector_id: i64, mut e: EstimatedDistance) {
        e.error *= self.z_score;
        self.scored += 1;
        let n = ErrorBoundNeighbor::from_upper(vector_id, e);
        if self.results.len() < self.max_len {
            self.results.push(n);
            return;
        }

        let ub = *self.results.peek_max().unwrap();
        if n < ub {
            let evicted = self.results.push_pop_max(n);
            // Put the evicted result in overflow if it is still competitive.
            let ub = *self.results.peek_max().unwrap();
            if ErrorBoundNeighbor::from_lower(evicted.n.vertex(), evicted.e) < ub {
                self.overflow
                    .push(ErrorBoundNeighbor::from_lower(evicted.n.vertex, evicted.e));
            }
            while self.overflow.peek_max().is_some_and(|x| ub < *x) {
                self.overflow.pop_max();
            }
        } else if ErrorBoundNeighbor::from_lower(vector_id, e) < ub {
            self.overflow
                .push(ErrorBoundNeighbor::from_lower(vector_id, e));
        }
    }

    fn into_results(self) -> Vec<Neighbor> {
        let mut results = std::iter::chain(self.results, self.overflow)
            .map(|en| Neighbor::new(en.n.vertex(), en.e.distance))
            .collect::<Vec<_>>();
        results.sort_unstable();
        results
    }
}
