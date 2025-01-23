//! Index search implementation, including graph search and re-ranking.

use std::{
    collections::HashSet,
    ops::{Add, AddAssign},
};

use crate::{
    graph::{
        Graph, GraphSearchParams, GraphVectorIndexReader, GraphVertex, NavVectorStore,
        RawVectorStore,
    },
    quantization::Quantizer,
    scoring::QuantizedVectorScorer,
    Neighbor,
};

use wt_mdb::{Error, Result};

#[derive(Debug, Copy, Clone, Default)]
pub struct GraphSearchStats {
    /// Total number of candidates vertices seen and nav scored.
    pub candidates: usize,
    /// Total number of graph vertices visited and traversed.
    pub visited: usize,
}

impl Add for GraphSearchStats {
    type Output = GraphSearchStats;

    fn add(self, rhs: Self) -> Self::Output {
        GraphSearchStats {
            candidates: self.candidates + rhs.candidates,
            visited: self.visited + rhs.visited,
        }
    }
}

impl AddAssign for GraphSearchStats {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

/// Helper to search a Vamana graph.
pub struct GraphSearcher {
    params: GraphSearchParams,

    candidates: CandidateList,
    seen: HashSet<i64>, // TODO: use a more efficient hash function (ahash?)
    visited: usize,
}

impl GraphSearcher {
    /// Create a new, reusable graph searcher.
    pub fn new(params: GraphSearchParams) -> Self {
        Self {
            params,
            candidates: CandidateList::new(params.beam_width.get()),
            seen: HashSet::new(),
            visited: 0,
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
            visited: self.visited,
        }
    }

    /// Search for `query` in the given graph `reader`. The reader will search in quantized space
    /// before optionally re-ranking based on higher fidelity vectors stored in the graph.
    ///
    /// Returns an approximate list of neighbors with the highest scores.
    pub fn search(
        &mut self,
        query: &[f32],
        reader: &mut impl GraphVectorIndexReader,
    ) -> Result<Vec<Neighbor>> {
        self.seen.clear();
        self.search_internal(query, reader)
    }

    /// Search for the vector at `vertex_id` and return matching candidates.
    pub fn search_for_insert(
        &mut self,
        vertex_id: i64,
        reader: &mut impl GraphVectorIndexReader,
    ) -> Result<Vec<Neighbor>> {
        self.seen.clear();
        // Insertions may be concurrent and there could already be backlinks to this vertex in the graph.
        // Marking this vertex as seen ensures we don't traverse or score ourselves (should be identity score).
        self.seen.insert(vertex_id);

        // NB: if inserting in a WT backed graph this will create a cursor that we immediately discard.
        let query = reader
            .raw_vectors()?
            .get_raw_vector(vertex_id)
            .unwrap_or(Err(Error::not_found_error()))?
            .to_vec();
        self.search_internal(&query, reader)
    }

    fn search_internal(
        &mut self,
        query: &[f32],
        reader: &mut impl GraphVectorIndexReader,
    ) -> Result<Vec<Neighbor>> {
        // TODO: come up with a better way of managing re-used state.
        self.candidates.clear();
        self.visited = 0;

        let mut graph = reader.graph()?;
        let mut nav = reader.nav_vectors()?;
        let quantizer = reader.config().new_quantizer();
        let nav_scorer = reader.config().new_nav_scorer();
        let nav_query = self.init_candidates(
            query,
            &mut graph,
            &mut nav,
            quantizer.as_ref(),
            nav_scorer.as_ref(),
        )?;

        while let Some(mut best_candidate) = self.candidates.next_unvisited() {
            self.visited += 1;
            let node = graph
                .get_vertex(best_candidate.neighbor().vertex())
                .unwrap_or_else(|| Err(Error::not_found_error()))?;
            // If we aren't reranking we don't need to copy the actual vector.
            best_candidate.visit(if self.params.num_rerank > 0 {
                node.vector().map(|v| v.to_vec())
            } else {
                None
            });

            for edge in node.edges() {
                if !self.seen.insert(edge) {
                    continue;
                }
                let vec = nav
                    .get_nav_vector(edge)
                    .unwrap_or(Err(Error::not_found_error()))?;
                self.candidates
                    .add_unvisited(Neighbor::new(edge, nav_scorer.score(&nav_query, &vec)));
            }
        }

        self.extract_results(query, reader)
    }

    // Initialize the candidate queue and return the binary quantized query.
    fn init_candidates<G, N>(
        &mut self,
        query: &[f32],
        graph: &mut G,
        nav: &mut N,
        quantizer: &dyn Quantizer,
        nav_scorer: &dyn QuantizedVectorScorer,
    ) -> Result<Vec<u8>>
    where
        G: Graph,
        N: NavVectorStore,
    {
        let nav_query = quantizer.for_query(query);
        if let Some(epr) = graph.entry_point() {
            let entry_point = epr?;
            let entry_vector = nav
                .get_nav_vector(entry_point)
                .unwrap_or(Err(Error::not_found_error()))?;
            self.candidates.add_unvisited(Neighbor::new(
                entry_point,
                nav_scorer.score(&nav_query, &entry_vector),
            ));
            self.seen.insert(entry_point);
        }
        // We don't treat failing to obtain an entry point as an error because
        // the graph may be empty.
        Ok(nav_query)
    }

    fn extract_results<R: GraphVectorIndexReader>(
        &mut self,
        query: &[f32],
        reader: &R,
    ) -> Result<Vec<Neighbor>> {
        if self.params.num_rerank == 0 {
            return Ok(self.candidates.iter().map(|c| c.neighbor).collect());
        }

        let scorer = reader.config().new_scorer();
        let query = scorer.normalize_vector(query.into());
        // TODO: this may unnecessarily create a cursor. lazily creating it it difficult because
        // the result is falliable.
        let mut raw_vectors = reader.raw_vectors()?;
        let rescored = self
            .candidates
            .iter()
            .take(self.params.num_rerank)
            .map(|c| {
                let vertex = c.neighbor.vertex();
                let score = if let Some(candidate_vector) = c.state.vector() {
                    Ok(scorer.score(&query, &candidate_vector))
                } else {
                    raw_vectors
                        .get_raw_vector(vertex)
                        .expect("row exists")
                        .map(|rv| scorer.score(&query, &rv))
                };
                score.map(|s| Neighbor::new(vertex, s))
            })
            .collect::<Result<Vec<_>>>();
        rescored.map(|mut r| {
            r.sort();
            r
        })
    }
}

#[derive(Debug, PartialEq)]
enum CandidateState {
    Unvisited,
    Visited(Option<Vec<f32>>),
}

impl CandidateState {
    fn vector(&self) -> Option<&[f32]> {
        match self {
            CandidateState::Visited(v) => v.as_ref().map(|x| x.as_slice()),
            _ => None,
        }
    }
}

/// A candidate in the search list. Once visited, the candidate becomes a result.
#[derive(Debug)]
struct Candidate {
    neighbor: Neighbor,
    state: CandidateState,
}

impl From<Neighbor> for Candidate {
    fn from(neighbor: Neighbor) -> Self {
        Candidate {
            neighbor,
            state: CandidateState::Unvisited,
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
    fn add_unvisited(&mut self, neighbor: Neighbor) {
        // If the queue is full and the candidate is not competitive then drop it.
        if self.candidates.len() >= self.candidates.capacity()
            && neighbor >= self.candidates.last().unwrap().neighbor
        {
            return;
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
    fn visit(&mut self, vector: Option<impl Into<Vec<f32>>>) {
        self.list.candidates[self.index].state = CandidateState::Visited(vector.map(|v| v.into()));
        self.list.next_unvisited = self
            .list
            .candidates
            .iter()
            .enumerate()
            .skip(self.index + 1)
            .find_map(|(i, c)| {
                if c.state == CandidateState::Unvisited {
                    Some(i)
                } else {
                    None
                }
            })
            .unwrap_or(self.list.candidates.len());
    }
}

#[cfg(test)]
mod test {
    use std::{borrow::Cow, num::NonZero};

    use wt_mdb::Result;

    use crate::{
        graph::{
            Graph, GraphConfig, GraphVectorIndexReader, GraphVertex, NavVectorStore, RawVector,
            RawVectorStore,
        },
        quantization::{BinaryQuantizer, Quantizer, VectorQuantizer},
        scoring::{DotProductScorer, F32VectorScorer, VectorSimilarity},
        Neighbor,
    };

    use super::{GraphSearchParams, GraphSearcher};

    #[derive(Debug)]
    struct TestVector {
        vector: Vec<f32>,
        nav_vector: Vec<u8>,
        edges: Vec<i64>,
    }

    #[derive(Debug)]
    pub struct TestGraphVectorIndex {
        data: Vec<TestVector>,
        config: GraphConfig,
    }

    impl TestGraphVectorIndex {
        pub fn new<S, T, V>(max_edges: NonZero<usize>, scorer: S, iter: T) -> Self
        where
            S: F32VectorScorer,
            T: IntoIterator<Item = V>,
            V: Into<Vec<f32>>,
        {
            let mut rep = iter
                .into_iter()
                .map(|x| {
                    let v = x.into();
                    let b = BinaryQuantizer.for_doc(&v);
                    TestVector {
                        vector: v,
                        nav_vector: b,
                        edges: Vec::new(),
                    }
                })
                .collect::<Vec<_>>();

            for i in 0..rep.len() {
                rep[i].edges = Self::compute_edges(&rep, i, max_edges, &scorer);
            }
            let config = GraphConfig {
                dimensions: NonZero::new(rep.first().map(|v| v.vector.len()).unwrap_or(1)).unwrap(),
                similarity: VectorSimilarity::Euclidean,
                quantizer: VectorQuantizer::Binary,
                max_edges: max_edges,
                index_search_params: GraphSearchParams {
                    beam_width: NonZero::new(usize::MAX).unwrap(),
                    num_rerank: usize::MAX,
                },
            };
            Self { data: rep, config }
        }

        pub fn reader(&self) -> TestGraphVectorIndexReader {
            TestGraphVectorIndexReader(self)
        }

        fn compute_edges<S>(
            graph: &[TestVector],
            index: usize,
            max_edges: NonZero<usize>,
            scorer: &S,
        ) -> Vec<i64>
        where
            S: F32VectorScorer,
        {
            let q = &graph[index].vector;
            let mut scored = graph
                .iter()
                .enumerate()
                .filter_map(|(i, n)| {
                    if i != index {
                        Some(Neighbor::new(i as i64, scorer.score(q, &n.vector)))
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
                if !selected
                    .iter()
                    .any(|p| scorer.score(q, &graph[p.vertex() as usize].vector) > n.score())
                {
                    selected.push(*n);
                }
            }
            selected.into_iter().map(|n| n.vertex()).collect()
        }
    }

    #[derive(Debug)]
    pub struct TestGraphVectorIndexReader<'a>(&'a TestGraphVectorIndex);

    impl<'a> GraphVectorIndexReader for TestGraphVectorIndexReader<'a> {
        type Graph<'b>
            = TestGraphAccess<'b>
        where
            Self: 'b;
        type RawVectorStore<'b>
            = TestGraphAccess<'b>
        where
            Self: 'b;
        type NavVectorStore<'b>
            = TestGraphAccess<'b>
        where
            Self: 'b;

        fn config(&self) -> &GraphConfig {
            &self.0.config
        }

        fn graph(&self) -> Result<Self::Graph<'_>> {
            Ok(TestGraphAccess(self.0))
        }

        fn raw_vectors(&self) -> Result<Self::RawVectorStore<'_>> {
            Ok(TestGraphAccess(self.0))
        }

        fn nav_vectors(&self) -> Result<Self::NavVectorStore<'_>> {
            Ok(TestGraphAccess(self.0))
        }
    }

    #[derive(Debug)]
    pub struct TestGraphAccess<'a>(&'a TestGraphVectorIndex);

    impl<'a> Graph for TestGraphAccess<'a> {
        type Vertex<'c>
            = TestGraphVertex<'c>
        where
            Self: 'c;

        fn entry_point(&mut self) -> Option<Result<i64>> {
            if !self.0.data.is_empty() {
                Some(Ok(0))
            } else {
                None
            }
        }

        fn get_vertex(&mut self, vertex_id: i64) -> Option<Result<Self::Vertex<'_>>> {
            if vertex_id >= 0 && (vertex_id as usize) < self.0.data.len() {
                Some(Ok(TestGraphVertex(&self.0.data[vertex_id as usize])))
            } else {
                None
            }
        }
    }

    impl<'a> RawVectorStore for TestGraphAccess<'a> {
        fn get_raw_vector(&mut self, vertex_id: i64) -> Option<Result<RawVector<'_>>> {
            if vertex_id >= 0 && (vertex_id as usize) < self.0.data.len() {
                Some(Ok((&*self.0.data[vertex_id as usize].vector).into()))
            } else {
                None
            }
        }
    }

    impl<'a> NavVectorStore for TestGraphAccess<'a> {
        fn get_nav_vector(&mut self, vertex_id: i64) -> Option<Result<Cow<'_, [u8]>>> {
            if vertex_id >= 0 && (vertex_id as usize) < self.0.data.len() {
                Some(Ok(Cow::from(&self.0.data[vertex_id as usize].nav_vector)))
            } else {
                None
            }
        }
    }

    #[derive(Debug)]
    pub struct TestGraph<'a>(&'a TestGraphVectorIndex);

    impl<'a> Graph for TestGraph<'a> {
        type Vertex<'c>
            = TestGraphVertex<'c>
        where
            Self: 'c;

        fn entry_point(&mut self) -> Option<Result<i64>> {
            if self.0.data.is_empty() {
                None
            } else {
                Some(Ok(0))
            }
        }

        fn get_vertex(&mut self, vertex_id: i64) -> Option<Result<Self::Vertex<'_>>> {
            if vertex_id < 0 || vertex_id as usize >= self.0.data.len() {
                None
            } else {
                Some(Ok(TestGraphVertex(&self.0.data[vertex_id as usize])))
            }
        }
    }

    pub struct TestGraphVertex<'a>(&'a TestVector);

    impl<'a> GraphVertex for TestGraphVertex<'a> {
        type EdgeIterator<'c>
            = std::iter::Copied<std::slice::Iter<'c, i64>>
        where
            Self: 'c;

        fn vector(&self) -> Option<Cow<'_, [f32]>> {
            Some(Cow::from(&self.0.vector))
        }

        fn edges(&self) -> Self::EdgeIterator<'_> {
            self.0.edges.iter().copied()
        }
    }

    fn build_test_graph(max_edges: usize) -> TestGraphVectorIndex {
        let dim_values = [-0.25, -0.125, 0.125, 0.25];
        TestGraphVectorIndex::new(
            NonZero::new(max_edges).unwrap(),
            DotProductScorer,
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

    fn normalize_scores(mut results: Vec<Neighbor>) -> Vec<Neighbor> {
        for n in results.iter_mut() {
            n.score = (n.score * 100000.0).round() / 100000.0;
        }
        results
    }

    #[test]
    fn basic_no_rerank() {
        let index = build_test_graph(4);
        let mut searcher = GraphSearcher::new(GraphSearchParams {
            beam_width: NonZero::new(4).unwrap(),
            num_rerank: 0,
        });
        assert_eq!(
            searcher
                .search(&[-0.1, -0.1, -0.1, -0.1], &mut index.reader())
                .unwrap(),
            vec![
                Neighbor::new(0, 1.0),
                Neighbor::new(1, 1.0),
                Neighbor::new(4, 1.0),
                Neighbor::new(16, 1.0)
            ]
        );
    }

    #[test]
    fn basic_rerank() {
        let index = build_test_graph(4);
        let mut searcher = GraphSearcher::new(GraphSearchParams {
            beam_width: NonZero::new(4).unwrap(),
            num_rerank: 4,
        });
        assert_eq!(
            normalize_scores(
                searcher
                    .search(&[-0.1, -0.1, -0.1, -0.1], &mut index.reader())
                    .unwrap()
            ),
            vec![
                Neighbor::new(1, 0.93622),
                Neighbor::new(4, 0.93622),
                Neighbor::new(16, 0.93622),
                Neighbor::new(0, 0.91743),
            ]
        );
    }
}
