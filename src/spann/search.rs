//! SPANN search implementation.

use std::{
    collections::HashSet,
    num::NonZero,
    ops::{Add, AddAssign},
};

use min_max_heap::MinMaxHeap;
use rustix::io::Errno;
use tracing::warn;
use vectors::{F32VectorCoding, QueryVectorDistance, VectorSimilarity};
use wt_mdb::{Error, Result};

use crate::{
    spann::{PostingKey, SessionIndexReader},
    vamana::{
        graph::{GraphSearchParams, GraphVectorIndexReader},
        search::{GraphSearchStats, GraphSearcher},
    },
    Neighbor,
};

// XXX centroid selection algorithm.
/// Tuning parameters for searching a SPANN index.
#[derive(Debug, Clone)]
pub struct SearchParams {
    /// Parameters for searching the head graph.
    /// NB: `head_params.beam_width` should be at least as large as `num_centroids`
    pub head_params: GraphSearchParams,
    /// The number of centroids to search.
    pub num_centroids: NonZero<usize>,
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
    /// Number of posting entries read.
    ///
    /// This may be greater than the number scored if vectors are replicated across centroids.
    pub posting_entries_read: usize,
    /// Number of posting entries scored.
    pub posting_entries_scored: usize,
    // XXX fast and slow scored entries.
}

impl Add for SearchStats {
    type Output = SearchStats;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            head: self.head + rhs.head,
            postings_read: self.postings_read + rhs.postings_read,
            posting_entries_read: self.posting_entries_read + rhs.posting_entries_read,
            posting_entries_scored: self.posting_entries_scored + rhs.posting_entries_scored,
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

        let mut centroids = self.head_searcher.search(query, &mut reader.head_reader)?;
        self.stats.head = self.head_searcher.stats();
        // TODO: be clever about choosing centroids.
        centroids.truncate(self.params.num_centroids.get());
        self.stats.postings_read = centroids.len();
        if centroids.is_empty() {
            return Ok(vec![]);
        }

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
            cursor.set_bounds(
                PostingKey::for_centroid(centroid_id)..PostingKey::for_centroid(centroid_id + 1),
            )?;
            while let Some(r) = unsafe { cursor.next_unsafe() } {
                self.stats.posting_entries_read += 1;
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
                self.stats.posting_entries_scored += 1;
            }
        }

        // XXX fix this so it is less stupid
        if self.params.num_rerank > 0 {
            let format = reader
                .head_reader
                .config()
                .rerank_format
                .ok_or(Error::Errno(Errno::NOTSUP))?;
            let query =
                format.query_vector_distance_f32(query, reader.head_reader.config().similarity);
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
        } else {
            Ok(result_queue.into_results())
        }
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
    lo: Option<ResultQueue<'a>>,
    hi: ResultQueue<'a>,
}

impl<'a> MultiResultQueue<'a> {
    fn new(
        query: &'a [f32],
        coding: F32VectorCoding,
        similarity: VectorSimilarity,
        limit: usize,
    ) -> Self {
        let lo = coding
            .query_vector_distance_f32_fast(query, similarity)
            .map(|d| ResultQueue::new(limit, d));
        let hi = ResultQueue::new(limit, coding.query_vector_distance_f32(query, similarity));
        Self { lo, hi }
    }

    fn push(&mut self, vertex: i64, vector: &[u8]) {
        if let Some(lo) = self.lo.as_mut() {
            // If we have a lo queue and the result doesn't rank, don't bother with the hi score.
            if !lo.push(vertex, vector) {
                return;
            }
        }
        self.hi.push(vertex, vector);
    }

    fn into_results(self) -> Vec<Neighbor> {
        self.hi.results.into_vec_asc()
    }
}
