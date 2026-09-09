//! SPANN search implementation.

use std::{
    cmp::Ordering,
    collections::HashSet,
    io,
    num::NonZero,
    ops::{Add, AddAssign},
    str::FromStr,
    sync::OnceLock,
};

use min_max_heap::MinMaxHeap;
use tracing::warn;
use vectors::{EstimatedDistance, QueryVectorDistance};
use wt_mdb::{Result, TypedCursorGuard};

use crate::{
    Neighbor,
    posting_block::PostingBlock,
    spann::{TransactionIndex, centroid_stats::CentroidStats},
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
}

/// Statistics for SPANN searches.
#[derive(Debug, Copy, Clone, Default)]
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
    /// 1-based rank, in distance-from-query order, of the last centroid posting that contributed a
    /// vector to the set of results to rerank. 0 when no results were produced.
    pub last_contribution_posting_rank: usize,
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
            last_contribution_posting_rank: self.last_contribution_posting_rank
                + rhs.last_contribution_posting_rank,
        }
    }
}

impl AddAssign for SearchStats {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
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
        self.stats = SearchStats::default();

        let mut centroids = self.head_searcher.search(query, reader.head())?;
        self.stats.head = self.head_searcher.stats();
        if centroids.is_empty() {
            return Ok(vec![]);
        }

        centroids = self.params.centroid_selector.select(centroids);
        self.stats.postings_read = centroids.len();

        self.seen.clear();
        let mut result_queue = ResultQueue::new(
            self.params.limit.get(),
            reader
                .index
                .config()
                .posting_coder
                .query_distance_asymmetric(reader.index().head_config().config().similarity, query),
        );
        let vector_len = reader.index().posting_vector_len();
        // Trace every selected centroid, even when its posting is missing or malformed, so that
        // trace entries stay aligned with centroid distance order.
        let mut postings = Vec::with_capacity(centroids.len());
        for (rank, c) in centroids.into_iter().enumerate() {
            let centroid_id: u32 = c.vertex().try_into().expect("centroid_id is a u32");
            // SAFETY: we are not performing any WT operations in between seeks.
            let data = match unsafe { posting_cursor.seek_exact_unsafe(centroid_id) } {
                Some(Ok(data)) => data,
                Some(Err(e)) => {
                    warn!("failed to read posting for centroid {centroid_id}: {e}");
                    postings.push(PostingTrace::new(c.distance()));
                    continue;
                }
                None => {
                    postings.push(PostingTrace::new(c.distance()));
                    continue;
                }
            };
            let Some(block) = PostingBlock::new(data, vector_len) else {
                warn!("malformed posting block for centroid {centroid_id}");
                postings.push(PostingTrace::new(c.distance()));
                continue;
            };
            postings.push(PostingTrace {
                distance: c.distance(),
                vectors: block.len(),
                contributing: 0,
            });
            for (record_id, vector) in block.iter() {
                self.stats.posting_vectors_read += 1;
                if !self.seen.insert(record_id) {
                    continue; // already seen
                }
                result_queue.push(record_id, vector, rank + 1);
            }
        }

        self.stats.posting_vectors_scored = result_queue.scored;

        let (r, contributions) = self.maybe_rerank_results(query, result_queue, reader)?;
        for (p, c) in postings.iter_mut().zip(contributions) {
            p.contributing = c;
        }
        self.trace_search(&postings);
        Ok(r)
    }

    /// Emit one JSON document describing this query's posting search when tracing is enabled via
    /// the `ET_SEARCH_TRACE` environment variable. Each `centroids` entry is `[centroid distance,
    /// posting vector count, contributing vector count]` in distance-from-query order.
    fn trace_search(&self, postings: &[PostingTrace]) {
        if !trace_enabled() {
            return;
        }
        let centroids = postings
            .iter()
            .map(|p| serde_json::json!([p.distance, p.vectors, p.contributing]))
            .collect::<Vec<_>>();
        let doc = serde_json::json!({
            "limit": self.params.limit.get() as u64,
            "num_rerank": self.params.num_rerank as u64,
            "postings_read": self.stats.postings_read as u64,
            "posting_vectors_read": self.stats.posting_vectors_read as u64,
            "posting_vectors_scored": self.stats.posting_vectors_scored as u64,
            "posting_vectors_reranked": self.stats.posting_vectors_reranked as u64,
            "last_contribution_posting_rank": self.stats.last_contribution_posting_rank as u64,
            "centroids": centroids,
        });
        println!("{doc}");
    }

    fn maybe_rerank_results(
        &mut self,
        query: &[f32],
        result_queue: ResultQueue<'_>,
        reader: &TransactionIndex,
    ) -> Result<(Vec<Neighbor>, Vec<usize>)> {
        if self.params.num_rerank == 0 {
            let (results, contributions) = result_queue.into_results();
            self.stats.last_contribution_posting_rank = last_contributing_rank(&contributions);
            return Ok((results, contributions));
        }

        let format = reader.index().config().rerank_format;
        let query = format.query_distance_asymmetric(reader.head.config().similarity, query);
        let mut raw_cursor = reader
            .transaction()
            .open_record_cursor(&reader.index().table_names.raw_vectors)?;
        let (results, contributions) = result_queue.into_results();
        self.stats.last_contribution_posting_rank = last_contributing_rank(&contributions);
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

        Ok((reranked, contributions))
    }
}

#[derive(Debug, Copy, Clone)]
struct ErrorBoundNeighbor {
    n: Neighbor,
    e: EstimatedDistance,
    /// 1-based rank, in distance-from-query order, of the centroid whose posting this neighbor
    /// was read from.
    centroid_rank: usize,
}

impl ErrorBoundNeighbor {
    fn from_lower(vector_id: i64, e: EstimatedDistance, centroid_rank: usize) -> Self {
        Self {
            n: Neighbor::new(vector_id, e.distance - e.error),
            e,
            centroid_rank,
        }
    }

    fn from_upper(vector_id: i64, e: EstimatedDistance, centroid_rank: usize) -> Self {
        Self {
            n: Neighbor::new(vector_id, e.distance + e.error),
            e,
            centroid_rank,
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

struct ResultQueue<'a> {
    dist_fn: Box<dyn QueryVectorDistance + 'a>,
    results: MinMaxHeap<ErrorBoundNeighbor>,
    overflow: MinMaxHeap<ErrorBoundNeighbor>,
    max_len: usize,
    scored: usize,
}

impl<'a> ResultQueue<'a> {
    fn new(max_len: usize, dist_fn: Box<dyn QueryVectorDistance + 'a>) -> Self {
        Self {
            dist_fn,
            results: MinMaxHeap::with_capacity(max_len),
            overflow: MinMaxHeap::new(),
            max_len,
            scored: 0,
        }
    }

    fn push(&mut self, vector_id: i64, vector: &[u8], centroid_rank: usize) {
        self.scored += 1;
        let e = self.dist_fn.estimated_distance(vector);
        let n = ErrorBoundNeighbor::from_upper(vector_id, e, centroid_rank);
        if self.results.len() < self.max_len {
            self.results.push(n);
            return;
        }

        let ub = *self.results.peek_max().unwrap();
        if n < ub {
            let evicted = self.results.push_pop_max(n);
            // Put the evicted result in overflow if it is still competitive.
            let ub = *self.results.peek_max().unwrap();
            let lower = ErrorBoundNeighbor::from_lower(
                evicted.n.vertex(),
                evicted.e,
                evicted.centroid_rank,
            );
            if lower < ub {
                self.overflow.push(lower);
            }
            while self.overflow.peek_max().is_some_and(|x| ub < *x) {
                self.overflow.pop_max();
            }
        } else if ErrorBoundNeighbor::from_lower(vector_id, e, centroid_rank) < ub {
            self.overflow
                .push(ErrorBoundNeighbor::from_lower(vector_id, e, centroid_rank));
        }
    }

    /// Returns the final results, along with the number of vectors contributed to the rerank
    /// candidate set from each centroid's posting, indexed by 1-based centroid rank.
    fn into_results(self) -> (Vec<Neighbor>, Vec<usize>) {
        let mut results = Vec::with_capacity(self.results.len() + self.overflow.len());
        let mut contributions = Vec::new();
        for en in std::iter::chain(self.results, self.overflow) {
            if contributions.len() < en.centroid_rank {
                contributions.resize(en.centroid_rank, 0);
            }
            contributions[en.centroid_rank - 1] += 1;
            results.push(Neighbor::new(en.n.vertex(), en.e.distance));
        }
        results.sort_unstable();
        (results, contributions)
    }
}

/// Per-centroid record for query tracing: distance from the query, the total number of vectors in
/// the centroid's posting, and the number of those that contributed to the rerank candidate set.
#[derive(Debug, Copy, Clone)]
struct PostingTrace {
    distance: f64,
    vectors: usize,
    contributing: usize,
}

impl PostingTrace {
    /// Trace a centroid whose posting is missing, malformed, or not yet read.
    fn new(distance: f64) -> Self {
        Self {
            distance,
            vectors: 0,
            contributing: 0,
        }
    }
}

/// Whether search tracing is enabled via the `ET_SEARCH_TRACE` environment variable.
fn trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("ET_SEARCH_TRACE").is_some())
}

/// 1-based rank, in distance-from-query order, of the last centroid posting that contributed a
/// vector to the rerank candidate set. 0 when no results were produced.
fn last_contributing_rank(contributions: &[usize]) -> usize {
    contributions.iter().rposition(|&c| c > 0).map_or(0, |i| i + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distance function for tests: the input vector is a little-endian (distance, error) f64
    /// pair.
    struct FixedEstimatedDistance;

    fn vector(distance: f64, error: f64) -> Vec<u8> {
        let mut v = distance.to_le_bytes().to_vec();
        v.extend_from_slice(&error.to_le_bytes());
        v
    }

    impl QueryVectorDistance for FixedEstimatedDistance {
        fn distance(&self, vector: &[u8]) -> f64 {
            f64::from_le_bytes(vector[..8].try_into().expect("distance bytes"))
        }

        fn estimated_distance(&self, vector: &[u8]) -> EstimatedDistance {
            EstimatedDistance {
                distance: self.distance(vector),
                error: f64::from_le_bytes(vector[8..].try_into().expect("error bytes")),
            }
        }
    }

    fn queue(max_len: usize) -> ResultQueue<'static> {
        ResultQueue::new(max_len, Box::new(FixedEstimatedDistance))
    }

    fn vertices(results: &[Neighbor]) -> Vec<i64> {
        results.iter().map(Neighbor::vertex).collect()
    }

    #[test]
    fn contributions_tallied_per_centroid_rank() {
        let mut q = queue(2);
        q.push(1, &vector(2.0, 5.0), 1);
        q.push(2, &vector(1.0, 0.0), 1);
        // The wide error bound keeps the evicted vector alive in overflow.
        q.push(3, &vector(0.5, 0.0), 2);
        let (results, contributions) = q.into_results();
        assert_eq!(contributions, vec![2, 1]);
        assert_eq!(last_contributing_rank(&contributions), 2);
        assert_eq!(vertices(&results), vec![3, 2, 1]);
    }

    #[test]
    fn non_competitive_evictions_do_not_contribute() {
        let mut q = queue(2);
        q.push(1, &vector(2.0, 0.0), 1);
        q.push(2, &vector(1.0, 0.0), 1);
        // The evicted vector's lower bound is not competitive so it is dropped entirely.
        q.push(3, &vector(0.5, 0.0), 2);
        let (results, contributions) = q.into_results();
        assert_eq!(contributions, vec![1, 1]);
        assert_eq!(last_contributing_rank(&contributions), 2);
        assert_eq!(vertices(&results), vec![3, 2]);
    }

    #[test]
    fn last_contributing_rank_handles_empty_and_trailing_zeros() {
        assert_eq!(last_contributing_rank(&[]), 0);
        assert_eq!(last_contributing_rank(&[0, 0]), 0);
        assert_eq!(last_contributing_rank(&[1, 0, 2]), 3);
    }
}
