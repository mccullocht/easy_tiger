//! Index search implementation, including graph search and re-ranking.

use std::ops::{Add, AddAssign};

use ahash::{AHashSet, HashMap};
use serde::{Deserialize, Serialize};

use super::{Graph, GraphSearchParams, GraphVectorIndex, GraphVectorStore};
use crate::{Neighbor, vamana::PatienceParams};

use vectors::QueryVectorDistance;
use wt_mdb::{Error, Result};

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Serialize)]
pub struct GraphSearchStats {
    /// Total number of candidates vertices seen and nav scored.
    pub candidates: usize,
    /// Total number of candidates successfully added to the candidates list.
    pub candidates_added: usize,
    /// Total number of graph vertices visited and traversed.
    pub visited: usize,
    /// Total number of candidates visited that did not match the filter predicate.
    pub filtered: usize,
    /// Number of candidates skipped due to a stale edge in a directed graph.
    pub skipped: usize,
}

impl Add for GraphSearchStats {
    type Output = GraphSearchStats;

    fn add(self, rhs: Self) -> Self::Output {
        GraphSearchStats {
            candidates: self.candidates + rhs.candidates,
            candidates_added: self.candidates_added + rhs.candidates_added,
            visited: self.visited + rhs.visited,
            filtered: self.filtered + rhs.filtered,
            skipped: self.skipped + rhs.skipped,
        }
    }
}

impl AddAssign for GraphSearchStats {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

/// Maintains information about the status of a single traced vertex during a graph search.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum VertexTrace {
    /// The requested id does not exist in the graph.
    NotFound,
    /// The vertex exists in the graph but was never scored during the search: graph traversal
    /// never reached it.
    Unseen,
    /// The vertex was scored during traversal at this distance (in nav/quantized space), but was
    /// dropped from the candidate list before the end of the search.
    Seen { distance: f64 },
    /// The vertex survived in the candidate list to the end of the search but fell beyond the
    /// `num_rerank` cut, so it was never reranked nor returned. The distance is in nav space.
    RerankDropped { distance: f64 },
    /// The vertex was returned at this rank in the result set. The distance is in rerank space if
    /// reranking was performed, otherwise in nav space.
    Found { rank: usize, distance: f64 },
}

/// The trace for a single traced vertex id.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct VertexIdTrace {
    /// The requested vertex id.
    pub id: i64,
    /// The trace of the vertex through the search.
    pub trace: VertexTrace,
}

/// A vertex scored during a traced search, with its nav-space query distance.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredVertex {
    /// The scored vertex id.
    pub id: i64,
    /// The nav-space distance from the query to this vertex.
    pub distance: f64,
}

/// The trace of a single graph search, with one entry per requested id in request order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSearchTrace {
    pub vectors: Vec<VertexIdTrace>,
    /// Every vertex scored during the search (entry point, seeds, and the edges of expanded
    /// vertices) with its nav-space query distance, in scoring order. Empty when the search was
    /// not traced.
    #[serde(default)]
    pub scored: Vec<ScoredVertex>,
}

/// Accumulates the state of in-flight traced vertex ids during a graph search.
struct TraceState {
    /// Traced vertex ids mapped to their position in the request and the nav-space distance they
    /// were scored at, if they were scored at all.
    ids: HashMap<i64, (usize, Option<f64>)>,
    /// Every scored vertex in scoring order (see [`GraphSearchTrace::scored`]).
    scored: Vec<ScoredVertex>,
}

impl TraceState {
    fn new(traced: &[i64]) -> Self {
        Self {
            ids: traced
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, (i, None)))
                .collect(),
            scored: Vec::new(),
        }
    }

    #[inline]
    fn observe_score(&mut self, vertex: i64, distance: f64) {
        self.scored.push(ScoredVertex {
            id: vertex,
            distance,
        });
        if let Some((_, scored)) = self.ids.get_mut(&vertex) {
            *scored = Some(distance);
        }
    }

    /// Resolve the final trace for every traced id given the search results, the candidate list at
    /// the end of traversal, and the nav vector store used to resolve vertex existence.
    fn finish(
        mut self,
        results: &[Neighbor],
        candidates: &CandidateList,
        nav: &mut impl GraphVectorStore,
    ) -> GraphSearchTrace {
        let mut vectors: Vec<(usize, i64, VertexTrace)> = self
            .ids
            .drain()
            .map(|(id, (rank, scored))| {
                let trace = match scored {
                    Some(distance) => match results.iter().position(|n| n.vertex() == id) {
                        Some(rank) => VertexTrace::Found {
                            rank,
                            distance: results[rank].distance(),
                        },
                        // Any candidate still in the list was returned unless it fell beyond the
                        // rerank cut (without rerank the full candidate list is the result).
                        None if candidates.iter().any(|c| c.neighbor.vertex() == id) => {
                            VertexTrace::RerankDropped { distance }
                        }
                        None => VertexTrace::Seen { distance },
                    },
                    None => match nav.get(id) {
                        Some(_) => VertexTrace::Unseen,
                        None => VertexTrace::NotFound,
                    },
                };
                (rank, id, trace)
            })
            .collect();
        vectors.sort_unstable_by_key(|(rank, _, _)| *rank);
        GraphSearchTrace {
            vectors: vectors
                .into_iter()
                .map(|(_, id, trace)| VertexIdTrace { id, trace })
                .collect(),
            scored: self.scored,
        }
    }
}

struct Patience {
    params: PatienceParams,
    candidates_added: usize,
    saturation_count: usize,
}

impl Patience {
    fn clear(&mut self) {
        self.candidates_added = 0;
        self.saturation_count = 0;
    }

    /// Update state with the number of candidates added in the most recent round.
    /// Returns true if patience has been exceeded.
    fn update(&mut self, candidates_added: usize) -> bool {
        let ratio =
            self.candidates_added as f64 / (self.candidates_added + candidates_added) as f64;
        self.candidates_added += candidates_added;
        if ratio >= self.params.saturation_threshold {
            self.saturation_count += 1;
            if self.saturation_count >= self.params.patience_count {
                return true;
            }
        } else {
            self.saturation_count = 0;
        }
        false
    }
}

/// Options for a graph search.
pub struct Options<F: FnMut(i64) -> bool> {
    filter: F,
    seeds: smallvec::SmallVec<[i64; 4]>,
    result_scratch: Option<Vec<Neighbor>>,
}

impl Default for Options<fn(i64) -> bool> {
    fn default() -> Self {
        Self {
            filter: (|_| true) as fn(i64) -> bool,
            seeds: smallvec::SmallVec::new(),
            result_scratch: None,
        }
    }
}

impl<F: FnMut(i64) -> bool> Options<F> {
    /// Create options with a pre-filter function that is applied to each result before it is
    /// returned.
    pub fn with_filter(filter: F) -> Self {
        Self {
            filter,
            seeds: smallvec::SmallVec::new(),
            result_scratch: None,
        }
    }

    /// Set seed vector ids for this request. Seeds are added as initial candidates along with the
    /// entry point to the graph. Any seed id that cannot be found is ignored.
    pub fn with_seeds(mut self, seeds: impl IntoIterator<Item = i64>) -> Self {
        self.seeds = seeds.into_iter().collect();
        self
    }

    /// A pre-allocated buffer of neighbors. This may be used to buffer results returned by a
    /// search_with_options() call.
    pub fn with_result_scratch(mut self, scratch: Vec<Neighbor>) -> Self {
        self.result_scratch = Some(scratch);
        self
    }
}

/// Helper to search a Vamana graph.
pub struct GraphSearcher {
    params: GraphSearchParams,
    patience: Option<Patience>,

    candidates: CandidateList,
    seen: AHashSet<i64>,
    candidates_added: usize,
    visited: usize,
    filtered: usize,
    skipped: usize,
}

impl GraphSearcher {
    /// Create a new, reusable graph searcher.
    pub fn new(params: GraphSearchParams) -> Self {
        let patience = params.patience.map(|p| Patience {
            params: p,
            candidates_added: 0,
            saturation_count: 0,
        });
        Self {
            params,
            patience,
            candidates: CandidateList::new(params.beam_width.get()),
            seen: AHashSet::new(),
            candidates_added: 0,
            visited: 0,
            filtered: 0,
            skipped: 0,
        }
    }

    /// Return the search params.
    pub fn params(&self) -> &GraphSearchParams {
        &self.params
    }

    /// Return stats for the last search that completed.
    pub fn stats(&self) -> GraphSearchStats {
        GraphSearchStats {
            candidates: self.seen.len(),
            candidates_added: self.candidates_added,
            visited: self.visited,
            filtered: self.filtered,
            skipped: self.skipped,
        }
    }

    /// Search for `query` in the given graph `reader`. The reader will search in quantized space
    /// before optionally re-ranking based on higher fidelity vectors stored in the graph.
    ///
    /// Returns an approximate list of neighbors with the highest scores.
    pub fn search(
        &mut self,
        query: &[f32],
        reader: &impl GraphVectorIndex,
    ) -> Result<Vec<Neighbor>> {
        self.seen.clear();
        self.search_internal(query, Options::default(), None, reader)
            .map(|(results, _)| results)
    }

    /// Search for `query` in graph `reader` with `options`.
    ///
    /// Returns a an approximate list of the closest neighbors matching any specified filter
    /// predicate.
    pub fn search_with_options<F: FnMut(i64) -> bool>(
        &mut self,
        query: &[f32],
        options: Options<F>,
        reader: &impl GraphVectorIndex,
    ) -> Result<Vec<Neighbor>> {
        self.seen.clear();
        self.search_internal(query, options, None, reader)
            .map(|(results, _)| results)
    }

    /// Search for `query` in the given graph `reader`, tracing the outcome of each id in `traced`
    /// through the search.
    ///
    /// Returns the search results alongside a trace with one entry per requested id, in request
    /// order. See [`VertexTrace`] for the possible outcomes.
    pub fn search_with_trace(
        &mut self,
        query: &[f32],
        reader: &impl GraphVectorIndex,
        traced: &[i64],
    ) -> Result<(Vec<Neighbor>, GraphSearchTrace)> {
        self.seen.clear();
        let trace = (!traced.is_empty()).then(|| TraceState::new(traced));
        self.search_internal(query, Options::default(), trace, reader)
            .map(|(results, trace)| {
                (
                    results,
                    trace.unwrap_or(GraphSearchTrace {
                        vectors: Vec::new(),
                        scored: Vec::new(),
                    }),
                )
            })
    }

    /// Search for the vector at `vertex_id` and return matching candidates.
    pub fn search_for_insert(
        &mut self,
        vertex_id: i64,
        reader: &impl GraphVectorIndex,
    ) -> Result<Vec<Neighbor>> {
        self.seen.clear();
        // Insertions may be concurrent and there could already be backlinks to this vertex in the graph.
        // Marking this vertex as seen ensures we don't traverse or score ourselves (should be identity score).
        self.seen.insert(vertex_id);

        // Always encode/quantize the nav query. There are some codings that support f32 x quantized
        // and we do not want to enter that path here because it is so expensive.
        let nav_query_rep = reader
            .nav_vectors()?
            .get(vertex_id)
            .unwrap_or_else(|| Err(Error::not_found_error()))?
            .to_vec();
        let nav_query = reader
            .config()
            .nav_format
            .query_distance_symmetric(reader.config().similarity, &nav_query_rep);

        let rerank_query = if self.params.num_rerank > 0 {
            if let Some(vectors) = reader.rerank_vectors() {
                let mut vectors = vectors?;
                let query = vectors
                    .get(vertex_id)
                    .unwrap_or_else(|| Err(Error::not_found_error()))?
                    .to_vec();
                Some(
                    vectors
                        .format()
                        .query_distance_symmetric(vectors.similarity(), query),
                )
            } else {
                None
            }
        } else {
            None
        };

        self.search_graph_and_rerank(
            nav_query.as_ref(),
            Options::default(),
            rerank_query.as_ref().map(|q| q.as_ref()),
            reader,
            None,
        )
        .map(|(results, _)| results)
    }

    fn search_internal<F: FnMut(i64) -> bool>(
        &mut self,
        query: &[f32],
        options: Options<F>,
        trace: Option<TraceState>,
        reader: &impl GraphVectorIndex,
    ) -> Result<(Vec<Neighbor>, Option<GraphSearchTrace>)> {
        // Center the query the same way the stored vectors were; every downstream query distance
        // consumes it.
        let query: &[f32] =
            &vectors::prepare_vector(query, None, false, reader.config().centroid.as_deref());
        let nav_query = reader
            .config()
            .nav_format
            .query_distance_asymmetric(reader.config().similarity, query);
        let rerank_query = if self.params.num_rerank > 0 {
            reader
                .config()
                .rerank_format
                .map(|f| f.query_distance_asymmetric(reader.config().similarity, query))
        } else {
            None
        };

        self.search_graph_and_rerank(
            nav_query.as_ref(),
            options,
            rerank_query.as_ref().map(|q| q.as_ref()),
            reader,
            trace,
        )
    }

    fn search_graph_and_rerank<F: FnMut(i64) -> bool>(
        &mut self,
        nav_query: &dyn QueryVectorDistance,
        mut options: Options<F>,
        rerank_query: Option<&dyn QueryVectorDistance>,
        reader: &impl GraphVectorIndex,
        mut trace: Option<TraceState>,
    ) -> Result<(Vec<Neighbor>, Option<GraphSearchTrace>)> {
        // TODO: come up with a better way of managing re-used state.
        self.candidates.clear();
        if let Some(p) = self.patience.as_mut() {
            p.clear()
        }
        self.candidates_added = 0;
        self.visited = 0;
        self.filtered = 0;

        let mut graph = reader.graph()?;
        let mut nav = reader.nav_vectors()?;
        if let Some(epr) = graph.entry_point() {
            let entry_point = epr?;
            let entry_vector = nav
                .get(entry_point)
                .unwrap_or_else(|| Err(Error::not_found_error()))?;
            let entry_distance = nav_query.distance(entry_vector);
            if let Some(t) = trace.as_mut() {
                t.observe_score(entry_point, entry_distance);
            }
            if self
                .candidates
                .add_unvisited(Neighbor::new(entry_point, entry_distance))
            {
                self.candidates_added += 1;
            }
            self.seen.insert(entry_point);
        }

        for seed in options.seeds {
            if !self.seen.insert(seed) {
                continue;
            }
            let seed_vector = match nav.get(seed) {
                Some(Ok(v)) => v,
                // Silently skip any seed that cannot be found in the graph.
                _ => continue,
            };
            let seed_distance = nav_query.distance(seed_vector);
            if let Some(t) = trace.as_mut() {
                t.observe_score(seed, seed_distance);
            }
            if self
                .candidates
                .add_unvisited(Neighbor::new(seed, seed_distance))
            {
                self.candidates_added += 1;
            }
        }

        while let Some(best_candidate) = self.candidates.next_unvisited() {
            self.visited += 1;
            let vertex_id = best_candidate.neighbor().vertex();
            if (options.filter)(vertex_id) {
                best_candidate.visit();
            } else {
                best_candidate.remove();
                self.filtered += 1;
            }

            let mut added = 0;
            for edge in graph
                .edges(vertex_id)
                .unwrap_or_else(|| Err(Error::not_found_error()))?
            {
                if !self.seen.insert(edge) {
                    continue;
                }
                let Some(vec) = nav.get(edge).transpose()? else {
                    self.skipped += 1;
                    continue;
                };
                let edge_distance = nav_query.distance(vec);
                if let Some(t) = trace.as_mut() {
                    t.observe_score(edge, edge_distance);
                }
                if self
                    .candidates
                    .add_unvisited(Neighbor::new(edge, edge_distance))
                {
                    added += 1;
                }
            }
            self.candidates_added += added;

            if self
                .patience
                .as_mut()
                .map(|p| p.update(added))
                .unwrap_or(false)
            {
                break;
            }
        }

        // Reuse result_scratch if present to avoid reallocation of the result vec.
        let mut results = options
            .result_scratch
            .take()
            .map(|mut v| {
                v.clear();
                v
            })
            .unwrap_or_default();
        if let Some(rerank_query) = rerank_query {
            let mut rerank_vectors = reader.rerank_vectors().expect("rerank enabled")?;
            results.reserve(self.params.num_rerank);
            for c in self.candidates.iter().take(self.params.num_rerank) {
                let vertex = c.neighbor.vertex();
                results.push(
                    rerank_vectors
                        .get(vertex)
                        .expect("row exists")
                        .map(|rv| Neighbor::new(vertex, rerank_query.distance(rv)))?,
                );
            }
            results.sort_unstable();
        } else {
            results.reserve(self.candidates.len());
            results.extend(self.candidates.iter().map(|c| c.neighbor));
        }
        let trace = trace.map(|t| t.finish(&results, &self.candidates, &mut nav));
        Ok((results, trace))
    }
}

/// A candidate in the search list. Once visited, the candidate becomes a result.
#[derive(Debug)]
struct Candidate {
    neighbor: Neighbor,
    visited: bool,
}

impl From<Neighbor> for Candidate {
    fn from(neighbor: Neighbor) -> Self {
        Candidate {
            neighbor,
            visited: false,
        }
    }
}

/// An ordered set of `Candidate` as a sort of priority queue.
///
/// Results are ordered by `Neighbor` value and the set is capped to a fixed capacity. Callers may
/// iterate over unvisited candidates, a core part of the Vamana search algorithm.
struct CandidateList {
    candidates: Vec<Candidate>,
    next_unvisited: usize,
}

impl CandidateList {
    /// Create a new candidate list with the given capacity. The list will never be longer
    /// than this capacity.
    fn new(capacity: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(capacity),
            next_unvisited: 0,
        }
    }

    /// Add a new candidate as an unvisited entry in the list.
    ///
    /// This maintains the list at a length <= capacity so the neighbor may not be inserted _or_ it
    /// may cause another neighbor to be dropped.
    fn add_unvisited(&mut self, neighbor: Neighbor) -> bool {
        // If the queue is full and the candidate is not competitive then drop it.
        if self.candidates.len() >= self.candidates.capacity()
            && neighbor >= self.candidates.last().unwrap().neighbor
        {
            return false;
        }

        if let Some(insert_idx) = self
            .candidates
            .binary_search_by_key(&neighbor, |c| c.neighbor)
            .err()
        {
            if self.candidates.len() >= self.candidates.capacity() {
                self.candidates.pop();
            }
            self.candidates.insert(insert_idx, neighbor.into());
            self.next_unvisited = std::cmp::min(self.next_unvisited, insert_idx);
            true
        } else {
            false
        }
    }

    /// Return the next unvisited neighbor, or None if all neighbors have been visited.
    fn next_unvisited(&mut self) -> Option<VisitCandidateGuard<'_>> {
        if self.next_unvisited < self.candidates.len() {
            Some(VisitCandidateGuard::new(self))
        } else {
            None
        }
    }

    /// Iterate over all candidates.
    fn iter(&self) -> impl Iterator<Item = &'_ Candidate> {
        self.candidates.iter()
    }

    /// Reset the candidate list to an empty state.
    fn clear(&mut self) {
        self.candidates.clear();
        self.next_unvisited = 0;
    }

    fn len(&self) -> usize {
        self.candidates.len()
    }
}

struct VisitCandidateGuard<'a> {
    list: &'a mut CandidateList,
    index: usize,
}

impl<'a> VisitCandidateGuard<'a> {
    fn new(list: &'a mut CandidateList) -> Self {
        let index = list.next_unvisited;
        Self { list, index }
    }

    /// The current neighbor we are visiting.
    fn neighbor(&self) -> Neighbor {
        self.list.candidates[self.index].neighbor
    }

    /// Mark this candidate as visited and update the full fidelity vector in the candidate list.
    fn visit(mut self) {
        self.list.candidates[self.index].visited = true;
        self.update_next_unvisited(self.index + 1)
    }

    /// Remove this candidate from the list. May happen if filter check is not passed.
    fn remove(mut self) {
        self.list.candidates.remove(self.index);
        self.update_next_unvisited(self.index);
    }

    fn update_next_unvisited(&mut self, start: usize) {
        self.list.next_unvisited = self
            .list
            .candidates
            .iter()
            .enumerate()
            .skip(start)
            .find_map(|(i, c)| if c.visited { None } else { Some(i) })
            .unwrap_or(self.list.candidates.len());
    }
}

#[cfg(test)]
mod test {
    use std::num::NonZero;

    use rustix::io::Errno;
    use vectors::{F32VectorCoding, F32VectorDistance, VectorSimilarity};
    use wt_mdb::{Error, Result};

    use crate::Neighbor;
    use crate::vamana::{
        EdgePruningConfig, EdgeType, Graph, GraphConfig, GraphVectorIndex, GraphVectorStore,
    };

    use super::{GraphSearchParams, GraphSearcher, Options, VertexTrace};

    #[derive(Debug)]
    struct TestVector {
        /// Raw vector, used to build the graph topology from f32 distances.
        vector: Vec<f32>,
        /// Vector as prepared for storage (centered against the configured centroid, if any).
        prepared: Vec<f32>,
        nav_vector: Vec<u8>,
        edges: Vec<i64>,
    }

    #[derive(Debug)]
    pub struct TestGraphVectorIndex {
        data: Vec<TestVector>,
        config: GraphConfig,
    }

    impl TestGraphVectorIndex {
        pub fn new<T, V>(
            max_edges: NonZero<usize>,
            distance_fn: Box<dyn F32VectorDistance>,
            iter: T,
        ) -> Self
        where
            T: IntoIterator<Item = V>,
            V: Into<Vec<f32>>,
        {
            Self::new_with_centroid(max_edges, distance_fn, iter, None)
        }

        pub fn new_with_centroid<T, V>(
            max_edges: NonZero<usize>,
            distance_fn: Box<dyn F32VectorDistance>,
            iter: T,
            centroid: Option<Vec<f32>>,
        ) -> Self
        where
            T: IntoIterator<Item = V>,
            V: Into<Vec<f32>>,
        {
            let coder = F32VectorCoding::BinaryQuantized.coder();
            let mut rep = iter
                .into_iter()
                .map(|x| {
                    let v: Vec<f32> = x.into();
                    // Euclidean fixture: prepare = subtract the centroid (no normalization).
                    let prepared = vectors::prepare_vector(&v, None, false, centroid.as_deref());
                    let b = coder.encode(&prepared);
                    TestVector {
                        vector: v,
                        prepared,
                        nav_vector: b,
                        edges: Vec::new(),
                    }
                })
                .collect::<Vec<_>>();

            for i in 0..rep.len() {
                rep[i].edges = Self::compute_edges(&rep, i, max_edges, distance_fn.as_ref());
            }
            let config = GraphConfig {
                dimensions: NonZero::new(rep.first().map(|v| v.vector.len()).unwrap_or(1)).unwrap(),
                similarity: VectorSimilarity::Euclidean,
                nav_format: F32VectorCoding::BinaryQuantized,
                rerank_format: Some(F32VectorCoding::F32),
                pruning: EdgePruningConfig::new(max_edges),
                index_search_params: GraphSearchParams {
                    beam_width: NonZero::new(usize::MAX).unwrap(),
                    num_rerank: usize::MAX,
                    patience: None,
                },
                centroid,
                edge_type: EdgeType::Undirected,
            };
            Self { data: rep, config }
        }

        pub fn reader(&self) -> TestGraphVectorIndexReader<'_> {
            TestGraphVectorIndexReader(self)
        }

        fn compute_edges(
            graph: &[TestVector],
            index: usize,
            max_edges: NonZero<usize>,
            distance_fn: &dyn F32VectorDistance,
        ) -> Vec<i64> {
            let q = &graph[index].vector;
            let mut scored = graph
                .iter()
                .enumerate()
                .filter_map(|(i, n)| {
                    if i != index {
                        Some(Neighbor::new(
                            i as i64,
                            distance_fn.distance_f32(q, &n.vector),
                        ))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            scored.sort();
            if scored.is_empty() {
                return vec![];
            }

            let mut selected = Vec::with_capacity(std::cmp::min(scored.len(), max_edges.get()));
            selected.push(scored[0]);
            // RNG prune: select edges that are closer to the vertex than they are to any of the other
            // nodes we've already selected an edge to.
            for n in scored.iter().skip(1) {
                if selected.len() == max_edges.get() {
                    break;
                }

                let q = &graph[n.vertex() as usize].vector;
                if !selected.iter().any(|p| {
                    distance_fn.distance_f32(q, &graph[p.vertex() as usize].vector) < n.distance()
                }) {
                    selected.push(*n);
                }
            }
            selected.into_iter().map(|n| n.vertex()).collect()
        }
    }

    #[derive(Debug)]
    pub struct TestGraphVectorIndexReader<'a>(&'a TestGraphVectorIndex);

    impl GraphVectorIndex for TestGraphVectorIndexReader<'_> {
        type Graph<'b>
            = TestGraphAccess<'b>
        where
            Self: 'b;
        type VectorStore<'b>
            = TestVectorStore<'b>
        where
            Self: 'b;

        fn config(&self) -> &GraphConfig {
            &self.0.config
        }

        fn graph(&self) -> Result<Self::Graph<'_>> {
            Ok(TestGraphAccess(self.0))
        }

        fn nav_vectors(&self) -> Result<Self::VectorStore<'_>> {
            Ok(TestVectorStore(self.0, TestVectorStoreType::Nav))
        }

        fn rerank_vectors(&self) -> Option<Result<Self::VectorStore<'_>>> {
            Some(Ok(TestVectorStore(self.0, TestVectorStoreType::Rerank)))
        }
    }

    #[derive(Debug)]
    pub struct TestGraphAccess<'a>(&'a TestGraphVectorIndex);

    impl Graph for TestGraphAccess<'_> {
        type EdgeIterator<'c>
            = std::iter::Copied<std::slice::Iter<'c, i64>>
        where
            Self: 'c;

        fn entry_point(&mut self) -> Option<Result<i64>> {
            if !self.0.data.is_empty() {
                Some(Ok(0))
            } else {
                None
            }
        }

        fn edges(&mut self, vertex_id: i64) -> Option<Result<Self::EdgeIterator<'_>>> {
            if vertex_id >= 0 && (vertex_id as usize) < self.0.data.len() {
                Some(Ok(self.0.data[vertex_id as usize].edges.iter().copied()))
            } else {
                None
            }
        }

        fn estimated_vertex_count(&mut self) -> Result<usize> {
            Ok(self.0.data.len())
        }

        fn set_entry_point(&mut self, _: i64) -> Result<()> {
            Err(Error::errno(Errno::NOTSUP))
        }

        fn remove_entry_point(&mut self) -> Result<()> {
            Err(Error::errno(Errno::NOTSUP))
        }

        fn set_edges(&mut self, _: i64, _: impl Into<Vec<i64>>) -> Result<()> {
            Err(Error::errno(Errno::NOTSUP))
        }

        fn remove_vertex(&mut self, _: i64) -> Result<Vec<i64>> {
            Err(Error::errno(Errno::NOTSUP))
        }

        fn next_available_vertex_id(&mut self) -> Result<i64> {
            Err(Error::errno(Errno::NOTSUP))
        }
    }

    enum TestVectorStoreType {
        Nav,
        Rerank,
    }

    pub struct TestVectorStore<'a>(&'a TestGraphVectorIndex, TestVectorStoreType);

    impl GraphVectorStore for TestVectorStore<'_> {
        fn format(&self) -> F32VectorCoding {
            match self.1 {
                TestVectorStoreType::Nav => self.0.config.nav_format,
                TestVectorStoreType::Rerank => self.0.config.rerank_format.unwrap(),
            }
        }

        fn similarity(&self) -> VectorSimilarity {
            self.0.config.similarity
        }

        fn centroid(&self) -> Option<&[f32]> {
            self.0.config.centroid.as_deref()
        }

        fn get(&mut self, vertex_id: i64) -> Option<Result<&[u8]>> {
            self.0.data.get(vertex_id as usize).map(|v| {
                Ok(match self.1 {
                    TestVectorStoreType::Nav => v.nav_vector.as_ref(),
                    TestVectorStoreType::Rerank => bytemuck::cast_slice(v.prepared.as_ref()),
                })
            })
        }

        fn set(&mut self, _: i64, _: impl AsRef<[u8]>) -> Result<()> {
            Err(Error::errno(Errno::NOTSUP))
        }

        fn remove(&mut self, _: i64) -> Result<Vec<u8>> {
            Err(Error::errno(Errno::NOTSUP))
        }
    }

    fn build_test_graph(max_edges: usize) -> TestGraphVectorIndex {
        let dim_values = [-0.25, -0.125, 0.125, 0.25];
        TestGraphVectorIndex::new(
            NonZero::new(max_edges).unwrap(),
            VectorSimilarity::Dot.distance_f32(),
            (0..256).map(|v| {
                Vec::from([
                    dim_values[v & 0x3],
                    dim_values[(v >> 2) & 0x3],
                    dim_values[(v >> 4) & 0x3],
                    dim_values[(v >> 6) & 0x3],
                ])
            }),
        )
    }

    fn build_test_graph_with_centroid(
        max_edges: usize,
        centroid: Vec<f32>,
    ) -> TestGraphVectorIndex {
        let dim_values = [-0.25, -0.125, 0.125, 0.25];
        TestGraphVectorIndex::new_with_centroid(
            NonZero::new(max_edges).unwrap(),
            VectorSimilarity::Dot.distance_f32(),
            (0..256).map(|v| {
                Vec::from([
                    dim_values[v & 0x3],
                    dim_values[(v >> 2) & 0x3],
                    dim_values[(v >> 4) & 0x3],
                    dim_values[(v >> 6) & 0x3],
                ])
            }),
            Some(centroid),
        )
    }

    fn normalize_scores(mut results: Vec<Neighbor>) -> Vec<Neighbor> {
        for n in results.iter_mut() {
            n.distance = (n.distance * 100000.0).round() / 100000.0;
        }
        results
    }

    #[test]
    fn basic_no_rerank() {
        let index = build_test_graph(4);
        let mut searcher = GraphSearcher::new(GraphSearchParams {
            beam_width: NonZero::new(4).unwrap(),
            num_rerank: 0,
            patience: None,
        });
        assert_eq!(
            searcher
                .search(&[-0.1, -0.1, -0.1, -0.1], &index.reader())
                .unwrap(),
            vec![
                Neighbor::new(0, 0.47999999940395355),
                Neighbor::new(1, 0.47999999940395355),
                Neighbor::new(4, 0.47999999940395355),
                Neighbor::new(16, 0.47999999940395355),
            ]
        );
    }

    #[test]
    fn basic_rerank() {
        let index = build_test_graph(4);
        let mut searcher = GraphSearcher::new(GraphSearchParams {
            beam_width: NonZero::new(4).unwrap(),
            num_rerank: 4,
            patience: None,
        });
        assert_eq!(
            normalize_scores(
                searcher
                    .search(&[-0.1, -0.1, -0.1, -0.1], &index.reader())
                    .unwrap()
            ),
            vec![
                Neighbor::new(1, 0.06813),
                Neighbor::new(4, 0.06813),
                Neighbor::new(16, 0.06813),
                Neighbor::new(0, 0.09),
            ]
        );
    }

    #[test]
    fn rerank_requested_but_not_supported() {
        let mut index = build_test_graph(4);
        index.config.rerank_format = None;

        let mut searcher = GraphSearcher::new(GraphSearchParams {
            beam_width: NonZero::new(4).unwrap(),
            num_rerank: 4,
            patience: None,
        });
        assert_eq!(
            searcher
                .search(&[-0.1, -0.1, -0.1, -0.1], &index.reader())
                .unwrap(),
            vec![
                Neighbor::new(0, 0.47999999940395355),
                Neighbor::new(1, 0.47999999940395355),
                Neighbor::new(4, 0.47999999940395355),
                Neighbor::new(16, 0.47999999940395355),
            ]
        );
    }

    #[test]
    fn stats_collection() {
        let index = build_test_graph(4);
        let mut searcher = GraphSearcher::new(GraphSearchParams {
            beam_width: NonZero::new(4).unwrap(),
            num_rerank: 0,
            patience: None,
        });

        // Search with a filter that rejects vertex 1
        let _ = searcher
            .search_with_options(
                &[-0.1, -0.1, -0.1, -0.1],
                Options::with_filter(|v| v != 1),
                &index.reader(),
            )
            .unwrap();

        let stats = searcher.stats();
        // We expect some visited nodes
        assert!(stats.visited > 0);
        // We expect candidates to be added (at least entry point and neighbors)
        assert!(stats.candidates_added > 0);
        // We filtered out vertex 1, so if it was visited, filtered count should be > 0.
        // In this small graph, vertex 1 is likely to be visited.
        assert!(
            stats.filtered >= 1,
            "Expected at least 1 filtered candidate, got {}",
            stats.filtered
        );
    }

    /// Nav-only search with a centroid exercises query_distance_asymmetric with centroid.
    /// Verifies that the graph can be traversed and returns a sorted candidate list.
    #[test]
    fn centroid_no_rerank() {
        let centroid = vec![0.0625, 0.0625, 0.0625, 0.0625];
        let index = build_test_graph_with_centroid(4, centroid);
        let mut searcher = GraphSearcher::new(GraphSearchParams {
            beam_width: NonZero::new(4).unwrap(),
            num_rerank: 0,
            patience: None,
        });
        let results = searcher
            .search(&[-0.1, -0.1, -0.1, -0.1], &index.reader())
            .unwrap();
        // Results must be non-empty and sorted by increasing BQ distance.
        assert!(!results.is_empty());
        for w in results.windows(2) {
            assert!(
                w[0].distance() <= w[1].distance(),
                "BQ distances not sorted: {w:?}"
            );
        }
    }

    /// Rerank search with a centroid exercises query_distance_symmetric (nav, encoded query) and
    /// query_distance_asymmetric (rerank, f32 query) both with centroid.
    ///
    /// Vertices 1, 4, 16, 64 each differ from the query [-0.1, -0.1, -0.1, -0.1] in exactly one
    /// dimension (at -0.25 vs -0.1) and are the nearest reachable vertices under raw F32 distance.
    /// With F32 reranking they should appear first with squared-Euclidean distance ≈ 0.0681.
    #[test]
    fn centroid_with_rerank() {
        let centroid = vec![0.0625, 0.0625, 0.0625, 0.0625];
        let index = build_test_graph_with_centroid(4, centroid);
        let mut searcher = GraphSearcher::new(GraphSearchParams {
            beam_width: NonZero::new(4).unwrap(),
            num_rerank: 4,
            patience: None,
        });
        let results = searcher
            .search(&[-0.1, -0.1, -0.1, -0.1], &index.reader())
            .unwrap();
        assert!(!results.is_empty());
        // Results must be sorted by increasing F32 squared-Euclidean distance.
        for w in results.windows(2) {
            assert!(
                w[0].distance() <= w[1].distance(),
                "rerank distances not sorted: {w:?}"
            );
        }
        // Top result distance should match the known nearest reachable vertex (squared dist ≈ 0.0681).
        let top_dist = normalize_scores(results)[0].distance();
        assert_eq!(top_dist, 0.06813, "unexpected top rerank distance");
    }

    fn traced_ids() -> Vec<i64> {
        let mut traced: Vec<i64> = (0..256).collect();
        traced.push(300);
        traced
    }

    /// With a candidate list as large as the graph no scored vertex can be dropped from the list
    /// and (without rerank) the whole list is returned: no vertex can be Seen or RerankDropped.
    /// Also verifies traces are returned in request order.
    #[test]
    fn trace_all_found() {
        let index = build_test_graph(4);
        let mut searcher = GraphSearcher::new(GraphSearchParams {
            beam_width: NonZero::new(256).unwrap(),
            num_rerank: 0,
            patience: None,
        });
        let traced = traced_ids();
        let (results, trace) = searcher
            .search_with_trace(&[-0.1, -0.1, -0.1, -0.1], &index.reader(), &traced)
            .unwrap();

        assert_eq!(trace.vectors.len(), traced.len());
        let mut found = 0;
        for (i, t) in trace.vectors.iter().enumerate() {
            assert_eq!(t.id, traced[i], "traces not in request order");
            if t.id == 300 {
                assert_eq!(t.trace, VertexTrace::NotFound);
                continue;
            }
            match t.trace {
                VertexTrace::Found { .. } => found += 1,
                VertexTrace::Unseen => {}
                other => panic!("unreachable state {other:?}"),
            }
        }
        assert_eq!(found, results.len());
        // Every result is marked found at its rank with the result distance.
        for (rank, r) in results.iter().enumerate() {
            match trace.vectors[r.vertex() as usize].trace {
                VertexTrace::Found {
                    rank: found_rank,
                    distance,
                } => {
                    assert_eq!(found_rank, rank);
                    assert_eq!(distance, r.distance());
                }
                other => panic!("result vertex traced as {other:?}"),
            }
        }
    }

    /// With a candidate list of 8 and only the top 2 reranked, the vertices ranked 3-8 in the
    /// final candidate list are dropped by the rerank stage and the rest are merely seen.
    #[test]
    fn trace_rerank_and_seen_states() {
        let index = build_test_graph(4);
        let mut searcher = GraphSearcher::new(GraphSearchParams {
            beam_width: NonZero::new(8).unwrap(),
            num_rerank: 2,
            patience: None,
        });
        let traced = traced_ids();
        let (results, trace) = searcher
            .search_with_trace(&[-0.1, -0.1, -0.1, -0.1], &index.reader(), &traced)
            .unwrap();

        assert_eq!(results.len(), 2);
        let mut found = 0;
        let mut rerank_dropped = 0;
        for t in &trace.vectors {
            match t.trace {
                VertexTrace::Found { rank, distance } => {
                    assert_eq!(results[rank].vertex(), t.id);
                    assert_eq!(distance, results[rank].distance());
                    found += 1;
                }
                VertexTrace::RerankDropped { distance } => {
                    assert!(distance >= 0.0);
                    rerank_dropped += 1;
                }
                VertexTrace::Seen { distance } => assert!(distance >= 0.0),
                VertexTrace::Unseen | VertexTrace::NotFound => {
                    assert!(t.id == 300 || t.trace == VertexTrace::Unseen)
                }
            }
        }
        assert_eq!(found, 2);
        assert_eq!(rerank_dropped, 6);
    }

    /// The scored list captures the entry point and every edge of every expanded vertex, each once,
    /// with nav-space distances that match the returned (nav-space) results when rerank is off.
    #[test]
    fn trace_scored_vertices() {
        let index = build_test_graph(4);
        let mut searcher = GraphSearcher::new(GraphSearchParams {
            beam_width: NonZero::new(4).unwrap(),
            num_rerank: 0,
            patience: None,
        });
        let (results, trace) = searcher
            .search_with_trace(&[-0.1, -0.1, -0.1, -0.1], &index.reader(), &[5])
            .unwrap();

        // The entry point (vertex 0 in the fixture) is always scored first.
        assert_eq!(trace.scored.first().map(|s| s.id), Some(0));
        // Each vertex is scored at most once: the seen set prevents rescoring.
        let mut ids: Vec<i64> = trace.scored.iter().map(|s| s.id).collect();
        let len = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), len, "duplicate scored vertex ids");
        // With num_rerank == 0 the result distances are nav-space, so they must match the scored
        // entry for the same vertex.
        for r in &results {
            let scored = trace
                .scored
                .iter()
                .find(|s| s.id == r.vertex())
                .unwrap_or_else(|| panic!("result vertex {} was not scored", r.vertex()));
            assert_eq!(scored.distance, r.distance());
        }
        assert!(trace.scored.len() >= results.len());
    }
}
