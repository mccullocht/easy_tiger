//! SPANN search implementation.

use std::{
    cmp::Ordering,
    collections::HashSet,
    io,
    num::NonZero,
    ops::{Add, AddAssign},
    str::FromStr,
};

use min_max_heap::MinMaxHeap;
use tracing::warn;
use vectors::{EstimatedDistance, QueryVectorDistance};
use wt_mdb::{Result, TypedCursorGuard};

use crate::{
    Neighbor,
    posting_block::PostingBlock,
    spann::{TransactionIndex, centroid_stats::CentroidStats, centroids::CentroidVectorSource},
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
        let mut result_queue = ResultQueue::new(self.params.limit.get());
        let config = reader.index().config();
        let similarity = reader.index().head_config().config().similarity;
        let vector_len = reader.index().posting_vector_len();
        // When postings are not centered a single distance function scores every posting vector
        // against the query. When postings are centered each centroid's postings are stored as
        // residuals (v - c), so the query must be adjusted per centroid (q - c) to score
        // |(q - c) - r| = |q - v|.
        let shared_dist_fn = (!config.center_postings)
            .then(|| config.posting_coder.query_distance_asymmetric(similarity, query));
        let mut centroid_source = config
            .center_postings
            .then(|| CentroidVectorSource::new(reader.head()))
            .transpose()?;
        let mut centroid_dist_fn: Option<Box<dyn QueryVectorDistance>>;
        for c in centroids {
            let centroid_id: u32 = c.vertex().try_into().expect("centroid_id is a u32");
            let dist_fn: &dyn QueryVectorDistance = match shared_dist_fn.as_ref() {
                Some(dist_fn) => dist_fn.as_ref(),
                None => {
                    let Some(source) = centroid_source.as_mut() else {
                        unreachable!("centroid source present when postings are centered");
                    };
                    // Read the centroid vector before seeking the posting cursor: reading it
                    // performs WT operations that would invalidate data borrowed from the
                    // posting cursor below.
                    let centroid_vector = match source.centroid_vector(centroid_id) {
                        Ok(v) => v,
                        Err(e) => {
                            warn!("failed to read centroid {centroid_id}: {e}");
                            continue;
                        }
                    };
                    let adjusted_query =
                        vectors::prepare_vector(query, None, false, Some(&centroid_vector));
                    centroid_dist_fn =
                        Some(config.posting_coder.query_distance_asymmetric(similarity, adjusted_query));
                    centroid_dist_fn.as_ref().unwrap().as_ref()
                }
            };
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
                result_queue.push(dist_fn, record_id, vector);
            }
        }

        self.stats.posting_vectors_scored = result_queue.scored;

        self.maybe_rerank_results(query, result_queue, reader)
    }

    fn maybe_rerank_results(
        &mut self,
        query: &[f32],
        result_queue: ResultQueue,
        reader: &TransactionIndex,
    ) -> Result<Vec<Neighbor>> {
        if self.params.num_rerank == 0 {
            return Ok(result_queue.into_results());
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

        Ok(reranked)
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
    results: MinMaxHeap<ErrorBoundNeighbor>,
    overflow: MinMaxHeap<ErrorBoundNeighbor>,
    max_len: usize,
    scored: usize,
}

impl ResultQueue {
    fn new(max_len: usize) -> Self {
        Self {
            results: MinMaxHeap::with_capacity(max_len),
            overflow: MinMaxHeap::new(),
            max_len,
            scored: 0,
        }
    }

    fn push(&mut self, dist_fn: &dyn QueryVectorDistance, vector_id: i64, vector: &[u8]) {
        self.scored += 1;
        let e = dist_fn.estimated_distance(vector);
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        input::{VecVectorStore, VectorStore},
        spann::{
            CentroidAssignment, IndexConfig, TableIndex,
            bulk::{
                assign_to_centroids, load_centroid_stats, load_centroids, load_postings,
                load_raw_vectors,
            },
            centroids::CentroidVectorSource,
            postings::BlockPostingsMut,
            rebalance::parallel_rebalance,
        },
        vamana::{EdgePruningConfig, EdgeType, GraphConfig, mutate::insert_vector},
    };
    use vectors::{F32VectorCoding, VectorSimilarity};
    use wt_mdb::{Connection, connection::OptionsBuilder};

    use rand_xoshiro::rand_core::SeedableRng;

    const DIMENSIONS: usize = 4;

    /// Push vectors from two "centroids" scored with per-centroid adjusted distance functions
    /// and verify the merged top-k matches brute-force `|(q - c_k) - r|` scoring.
    #[test]
    fn result_queue_per_centroid_distance_functions() {
        let c0 = [0.0f32, 0.0];
        let c1 = [10.0f32, 10.0];
        let query = [1.0f32, 1.0];
        // Residuals of 4 vectors, the first two assigned to c0 and the rest to c1.
        let residuals = [[0.5, 0.5], [3.0, -3.0], [0.25, 0.25], [-2.0, 2.0]];
        let ids = [0i64, 1, 2, 3];

        let coder = F32VectorCoding::F32.coder();
        let adjusted = |c: &[f32; 2]| -> Vec<f32> {
            query.iter().zip(c.iter()).map(|(q, c)| q - c).collect()
        };
        let dfn0 = F32VectorCoding::F32
            .query_distance_asymmetric(VectorSimilarity::Euclidean, adjusted(&c0));
        let dfn1 = F32VectorCoding::F32
            .query_distance_asymmetric(VectorSimilarity::Euclidean, adjusted(&c1));

        let mut queue = ResultQueue::new(2);
        queue.push(dfn0.as_ref(), 0, &coder.encode(&residuals[0]));
        queue.push(dfn0.as_ref(), 1, &coder.encode(&residuals[1]));
        queue.push(dfn1.as_ref(), 2, &coder.encode(&residuals[2]));
        queue.push(dfn1.as_ref(), 3, &coder.encode(&residuals[3]));
        let results = queue.into_results();

        let mut expected = ids
            .iter()
            .zip(residuals.iter().zip([c0, c0, c1, c1]))
            .map(|(id, (r, c))| {
                let distance: f64 = query
                    .iter()
                    .zip(r.iter().zip(c.iter()))
                    .map(|(q, (r, c))| {
                        let d = *q as f64 - (*c + *r) as f64;
                        d * d
                    })
                    .sum();
                Neighbor::new(*id, distance)
            })
            .collect::<Vec<_>>();
        expected.sort_unstable_by(|a, b| {
            a.distance()
                .partial_cmp(&b.distance())
                .unwrap()
                .then(a.vertex().cmp(&b.vertex()))
        });

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].vertex(), expected[0].vertex());
        assert!((results[0].distance() - expected[0].distance()).abs() < 1e-9);
        assert_eq!(results[1].vertex(), expected[1].vertex());
        assert!((results[1].distance() - expected[1].distance()).abs() < 1e-9);
    }

    /// A SPANN index built in a temporary WiredTiger home.
    ///
    /// NB: `_dir` must drop after `conn` so the directory outlives WiredTiger's close-time
    /// checkpoint.
    struct Fixture {
        conn: Arc<Connection>,
        index: Arc<TableIndex>,
        vectors: VecVectorStore<f32>,
        _dir: tempfile::TempDir,
    }

    impl Fixture {
        /// Build an index with 4 head centroids: a dummy zero vector (id 0, as `et spann
        /// init-index` inserts) plus one centroid per data cluster below.
        fn new(center_postings: bool) -> std::io::Result<Self> {
            let dir = tempfile::TempDir::new().unwrap();
            let conn = Connection::open(
                dir.path().to_str().unwrap(),
                Some(OptionsBuilder::default().create().into()),
            )?;
            let search_params = GraphSearchParams {
                beam_width: NonZero::new(8).unwrap(),
                num_rerank: 0,
                patience: None,
            };
            let head_config = GraphConfig {
                dimensions: NonZero::new(DIMENSIONS).unwrap(),
                similarity: VectorSimilarity::Euclidean,
                nav_format: F32VectorCoding::F32,
                rerank_format: None,
                pruning: EdgePruningConfig::new(NonZero::new(4).unwrap()),
                index_search_params: search_params,
                centroid: None,
                edge_type: EdgeType::Undirected,
            };
            let spann_config = IndexConfig {
                head_search_params: search_params,
                posting_coder: F32VectorCoding::F32,
                // Bounds chosen so that loading the test data triggers both splits (A, B, O
                // exceed 12) and a merge (C has a single posting, below 4).
                min_centroid_len: 4,
                max_centroid_len: 12,
                rerank_format: F32VectorCoding::F32,
                center_postings,
            };
            let index = Arc::new(TableIndex::init_index(
                &conn,
                "test",
                head_config,
                spann_config,
            )?);

            let vectors = test_vectors();
            {
                // Insert centroid vectors: id 0 is the dummy zero vector, then one per cluster.
                let txn_idx = TransactionIndex::new(&index, conn.begin_transaction(None)?);
                insert_vector(&[0.0f32; DIMENSIONS], txn_idx.head())?;
                for c in [
                    [10.0f32, 10.0, 10.0, 10.0],
                    [-10.0f32, -10.0, -10.0, -10.0],
                    [10.0f32, -10.0, 10.0, -10.0],
                ] {
                    insert_vector(&c, txn_idx.head())?;
                }
                txn_idx.commit(None)?;
            }
            {
                let limit = vectors.len();
                let assignments =
                    assign_to_centroids(index.as_ref(), &conn, &vectors, limit, |_| {})?;
                load_centroids(index.as_ref(), &conn, &assignments, |_| {})?;
                load_centroid_stats(index.as_ref(), &conn, &assignments, |_| {})?;
                load_raw_vectors(index.as_ref(), &conn, &vectors, limit, |_| {})?;
                let txn = conn.begin_transaction(None)?;
                {
                    let cursor = txn.open_cursor::<u32, Vec<u8>>(index.postings_table_name())?;
                    let mut postings = BlockPostingsMut::new(cursor, index.posting_vector_len());
                    load_postings(
                        index.as_ref(),
                        &conn,
                        &mut postings,
                        &assignments,
                        &vectors,
                        |_| {},
                    )?;
                }
                txn.commit(None)?;
            }
            Ok(Self {
                _dir: dir,
                conn,
                index,
                vectors,
            })
        }

        /// Search for `query`, returning the ids of the top `limit` results.
        fn search(&self, query: &[f32]) -> Result<Vec<i64>> {
            let txn_idx = TransactionIndex::new(&self.index, self.conn.begin_transaction(None)?);
            let mut searcher = Searcher::new(SearchParams {
                head_params: GraphSearchParams {
                    beam_width: NonZero::new(8).unwrap(),
                    num_rerank: 0,
                    patience: None,
                },
                centroid_selector: CentroidSelector::TopN(3),
                num_rerank: 10,
                limit: NonZero::new(10).unwrap(),
            });
            let mut posting_cursor =
                txn_idx.transaction().open_cursor::<u32, Vec<u8>>(self.index.postings_table_name())?;
            Ok(searcher
                .search(query, &txn_idx, &mut posting_cursor)?
                .into_iter()
                .map(|n| n.vertex())
                .collect())
        }

        /// Verify the index invariants: every record has exactly one posting, in the block of
        /// its assigned centroid, and that posting decodes to a residual which, combined with
        /// the centroid's vector, reproduces the original vector.
        fn assert_postings_are_residuals(&self) -> Result<()> {
            let txn_idx = TransactionIndex::new(&self.index, self.conn.begin_transaction(None)?);
            let mut centroid_source = CentroidVectorSource::new(txn_idx.head())?;
            let mut assignment_cursor = txn_idx.transaction().open_cursor::<i64, CentroidAssignment>(
                self.index.centroid_assignments_table_name(),
            )?;
            let posting_coder = self.index.new_posting_coder();
            // Collect the posting blocks before the per-record checks: those seek other
            // cursors, which we avoid mixing with iteration over the postings cursor.
            let blocks: Vec<(u32, Vec<u8>)> = txn_idx
                .transaction()
                .open_cursor::<u32, Vec<u8>>(self.index.postings_table_name())?
                .collect::<Result<Vec<_>>>()?;
            let mut seen = HashSet::new();
            let mut scratch = vec![0.0f32; DIMENSIONS];
            for (centroid_id, data) in blocks {
                let block =
                    PostingBlock::new(&data, self.index.posting_vector_len()).expect("valid block");
                let centroid = centroid_source.centroid_vector(centroid_id)?;
                for (record_id, encoded) in block.iter() {
                    assert!(
                        seen.insert(record_id),
                        "record {record_id} appears in more than one posting block"
                    );
                    let assignment = assignment_cursor
                        .seek_exact(record_id)
                        .unwrap_or_else(|| Err(wt_mdb::Error::not_found_error()))?;
                    assert_eq!(
                        assignment.primary_id, centroid_id,
                        "record {record_id} is posted in centroid {centroid_id} but assigned to {}",
                        assignment.primary_id
                    );
                    posting_coder.decode_to(encoded, &mut scratch);
                    let original = &self.vectors[record_id as usize];
                    for ((r, o), c) in scratch.iter().zip(original.iter()).zip(centroid.iter()) {
                        assert!(
                            (r + c - o).abs() < 1e-4,
                            "posting {record_id} of centroid {centroid_id} does not reconstruct \
                             the original vector: r={r}, c={c}, o={o}"
                        );
                    }
                }
            }
            assert_eq!(
                seen.len(),
                self.vectors.len(),
                "every record must have exactly one posting"
            );
            Ok(())
        }
    }

    fn test_vectors() -> VecVectorStore<f32> {
        let mut store = VecVectorStore::with_capacity(DIMENSIONS, 50);
        // Cluster A around (10, 10, 10, 10), bi-modal around (12, ...) and (8, ...) so a
        // split partitions it into sub-clusters near those modes: 25 vectors (> max len).
        for i in 0..12 {
            let d = i as f32 * 0.01;
            store.push(&[12.0 + d, 12.0 - d, 12.0 + 2.0 * d, 12.0 - 2.0 * d]);
        }
        for i in 0..13 {
            let d = i as f32 * 0.01;
            store.push(&[8.0 + d, 8.0 - d, 8.0 + 2.0 * d, 8.0 - 2.0 * d]);
        }
        // Cluster B around (-10, -10, -10, -10): 12 vectors (within policy, no split).
        for i in 0..12 {
            let d = i as f32 * 0.01;
            store.push(&[-10.0 - d, -10.0 + d, -10.0 - 2.0 * d, -10.0 + 2.0 * d]);
        }
        // Cluster C around (10, -10, 10, -10): a single vector, below min_centroid_len.
        store.push(&[10.0, -10.0, 10.0, -10.0]);
        // Cluster O around the dummy zero centroid: 11 tight vectors (within policy)...
        for i in 0..11 {
            let d = i as f32 * 0.001;
            store.push(&[d, -d, 2.0 * d, -2.0 * d]);
        }
        // ...plus one boundary vector that is assigned to O (the origin is its nearest
        // centroid at load time) but is closer to A's (8, ...) split target, forcing a
        // nearby reassignment move during rebalance.
        store.push(&[4.5, 4.5, 4.5, 4.5]);
        store
    }

    /// Brute-force top-k ids by squared Euclidean distance.
    fn brute_force_top(vectors: &VecVectorStore<f32>, query: &[f32], k: usize) -> Vec<i64> {
        let mut scored: Vec<(i64, f64)> = (0..vectors.len())
            .map(|i| {
                let distance: f64 = query
                    .iter()
                    .zip(vectors[i].iter())
                    .map(|(q, v)| (*q as f64 - *v as f64).powi(2))
                    .sum();
                (i as i64, distance)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.into_iter().take(k).map(|(i, _)| i).collect()
    }

    #[test]
    fn centered_postings_search_recall() -> std::io::Result<()> {
        let fixture = Fixture::new(true)?;
        fixture.assert_postings_are_residuals()?;

        let query = [12.05f32, 11.95, 12.1, 11.9];
        let expected = brute_force_top(&fixture.vectors, &query, 10);
        let results = fixture.search(&query)?;
        // The query sits at the heart of cluster A so the top 10 are all A vectors; the
        // per-centroid adjusted scoring must find exactly the brute-force set.
        assert_eq!(results, expected);
        Ok(())
    }

    #[test]
    fn centered_postings_match_uncentered_results() -> std::io::Result<()> {
        let centered = Fixture::new(true)?;
        let uncentered = Fixture::new(false)?;
        for query in [
            [12.05f32, 11.95, 12.1, 11.9],
            [-9.9f32, -10.1, -9.95, -10.05],
            [0.01f32, -0.01, 0.02, -0.02],
        ] {
            assert_eq!(centered.search(&query)?, uncentered.search(&query)?);
        }
        Ok(())
    }

    #[test]
    fn rebalance_preserves_centered_postings() -> std::io::Result<()> {
        let fixture = Fixture::new(true)?;
        // Loading the fixture data leaves centroids out of policy; rebalancing splits the
        // oversized clusters and merges the singleton, re-centering every moved posting from
        // its rerank vector.
        let stats = parallel_rebalance(
            &fixture.conn,
            &fixture.index,
            &|| rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0x5EED),
        )?;
        // The fixture data must exercise both rebalance op kinds.
        assert!(stats.split >= 1, "fixture should trigger at least one split");
        assert!(stats.merged >= 1, "fixture should trigger at least one merge");
        assert!(
            stats.split_stats.nearby_moved >= 1,
            "fixture should trigger at least one nearby move"
        );
        fixture.assert_postings_are_residuals()?;

        let query = [12.05f32, 11.95, 12.1, 11.9];
        let expected = brute_force_top(&fixture.vectors, &query, 10);
        assert_eq!(fixture.search(&query)?, expected);
        Ok(())
    }
}
