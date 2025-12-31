//! SPANN search implementation.

use std::{
    collections::HashSet,
    io,
    num::NonZero,
    ops::{Add, AddAssign},
    str::FromStr,
};

use min_max_heap::MinMaxHeap;
use tracing::warn;
use vectors::{F32VectorCoding, QueryVectorDistance, VectorSimilarity};
use wt_mdb::{Result, Session};

use crate::{
    spann::{centroid_stats::CentroidStats, PostingKey, SessionIndexReader, TableIndex},
    vamana::{
        search::{GraphSearchStats, GraphSearcher},
        {GraphSearchParams, GraphVectorIndex},
    },
    Neighbor,
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
    pub fn new(
        algorithm: CentroidSelectorAlgorithm,
        index: &TableIndex,
        session: &Session,
    ) -> Result<Self> {
        match algorithm {
            CentroidSelectorAlgorithm::TopN(n) => Ok(Self::TopN(n)),
            CentroidSelectorAlgorithm::VectorCount(n) => {
                let stats = CentroidStats::from_index_stats(session, index)?;
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
    ///
    /// This may be greater than the number scored if vectors are replicated across centroids.
    pub posting_vectors_read: usize,
    /// Number of posting entries "fast" scored.
    ///
    /// Certain vector encodings support a "fast" distance function that is less accurate but faster
    /// than the "slow" distance function. Depending on the posting coding this will either be 0
    /// or some number less than `posting_vectors_slow_scored`.
    pub posting_vectors_fast_scored: usize,
    /// Number of posting entries "slow" scored.
    pub posting_vectors_slow_scored: usize,
}

impl Add for SearchStats {
    type Output = SearchStats;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            head: self.head + rhs.head,
            postings_read: self.postings_read + rhs.postings_read,
            posting_vectors_read: self.posting_vectors_read + rhs.posting_vectors_read,
            posting_vectors_fast_scored: self.posting_vectors_fast_scored
                + rhs.posting_vectors_fast_scored,
            posting_vectors_slow_scored: self.posting_vectors_slow_scored
                + rhs.posting_vectors_slow_scored,
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
        reader: &mut SessionIndexReader,
    ) -> Result<Vec<Neighbor>> {
        self.stats = SearchStats::default();

        let mut centroids = self.head_searcher.search(query, &reader.head_reader)?;
        self.stats.head = self.head_searcher.stats();
        if centroids.is_empty() {
            return Ok(vec![]);
        }

        centroids = self.params.centroid_selector.select(centroids);
        self.stats.postings_read = centroids.len();

        self.seen.clear();
        let mut result_queue = MultiResultQueue::new(
            query,
            reader.index.config().posting_coder,
            reader.index().head_config().config().similarity,
            self.params.limit.get(),
        );
        for c in centroids {
            let centroid_id: u32 = c.vertex().try_into().expect("centroid_id is a u32");
            // TODO: if I can't read a posting list then skip and warn rather than exiting early.
            let mut cursor = reader
                .session()
                .get_or_create_typed_cursor::<PostingKey, Vec<u8>>(
                    &reader.index().table_names.postings,
                )?;
            cursor.set_bounds(PostingKey::centroid_range(centroid_id))?;
            while let Some(r) = unsafe { cursor.next_unsafe() } {
                self.stats.posting_vectors_read += 1;
                let (record_id, vector) = match r {
                    Ok((k, v)) => (k.record_id, v),
                    Err(e) => {
                        warn!("failed to read posting in centroid {centroid_id}: {e}");
                        continue;
                    }
                };

                if !self.seen.insert(record_id) {
                    continue; // already seen
                }

                result_queue.push(record_id, vector);
            }
        }

        self.stats.posting_vectors_fast_scored = result_queue.fast_count;
        self.stats.posting_vectors_slow_scored = result_queue.slow_count;

        self.maybe_rerank_results(query, result_queue, reader)
    }

    fn maybe_rerank_results(
        &mut self,
        query: &[f32],
        result_queue: MultiResultQueue<'_>,
        reader: &mut SessionIndexReader,
    ) -> Result<Vec<Neighbor>> {
        if self.params.num_rerank == 0 || reader.index().config().rerank_format.is_none() {
            return Ok(result_queue.into_results());
        }

        let format = reader
            .index()
            .config()
            .rerank_format
            .expect("rerank format is set");
        let query = format.query_vector_distance_f32(query, reader.head_reader.config().similarity);
        let mut raw_cursor = reader
            .session()
            .open_record_cursor(&reader.index().table_names.raw_vectors)?;
        let mut reranked = result_queue
            .into_results()
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

        Ok(reranked)
    }
}

struct ResultQueue<'a> {
    dist_fn: Box<dyn QueryVectorDistance + 'a>,
    results: MinMaxHeap<Neighbor>,
    max_len: usize,
}

impl<'a> ResultQueue<'a> {
    fn new(max_len: usize, dist_fn: Box<dyn QueryVectorDistance + 'a>) -> Self {
        Self {
            dist_fn,
            results: MinMaxHeap::with_capacity(max_len),
            max_len,
        }
    }

    /// Returns `true` if `v` is kept in the queue rather than discarded.
    fn push(&mut self, vertex: i64, vector: &[u8]) -> bool {
        let n = Neighbor::new(vertex, self.dist_fn.distance(vector));
        if self.results.len() < self.max_len {
            self.results.push(n);
            true
        } else {
            self.results.push_pop_max(n).vertex() != vertex
        }
    }
}

struct MultiResultQueue<'a> {
    fast: Option<ResultQueue<'a>>,
    fast_count: usize,

    slow: ResultQueue<'a>,
    slow_count: usize,
}

impl<'a> MultiResultQueue<'a> {
    fn new(
        query: &'a [f32],
        coding: F32VectorCoding,
        similarity: VectorSimilarity,
        limit: usize,
    ) -> Self {
        let fast = coding
            .query_vector_distance_f32_fast(query, similarity)
            .map(|d| ResultQueue::new(limit, d));
        let slow = ResultQueue::new(limit, coding.query_vector_distance_f32(query, similarity));
        Self {
            fast,
            fast_count: 0,
            slow,
            slow_count: 0,
        }
    }

    fn push(&mut self, vertex: i64, vector: &[u8]) {
        if let Some(fast) = self.fast.as_mut() {
            // If we have a lo queue and the result doesn't rank, don't bother with the hi score.
            self.fast_count += 1;
            if !fast.push(vertex, vector) {
                return;
            }
        }
        self.slow_count += 1;
        self.slow.push(vertex, vector);
    }

    fn into_results(self) -> Vec<Neighbor> {
        self.slow.results.into_vec_asc()
    }
}
