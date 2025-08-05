//! Index search implementation, including graph search and re-ranking.

use std::{
    collections::HashSet,
    ops::{Add, AddAssign},
};

use crate::{
    graph::{Graph, GraphSearchParams, GraphVectorIndexReader, GraphVectorStore, GraphVertex},
    vectors::{
        new_query_vector_distance_f32, new_query_vector_distance_indexing, QueryVectorDistance,
    },
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
        self.search_internal(query, |_| true, reader)
    }

    /// Search for `query` in the given graph `reader`, with oracle function `filter_predicate` dictating which
    /// vertex ids are valid results. The returned results will only include vertices that match `filter_predicate`
    /// and will assume that for any vertex id that all calls will return the same value.
    ///
    /// Returns an approximate list of neighbors matching `filter_predicate` with the highest scores.
    pub fn search_with_filter(
        &mut self,
        query: &[f32],
        filter_predicate: impl FnMut(i64) -> bool,
        reader: &mut impl GraphVectorIndexReader,
    ) -> Result<Vec<Neighbor>> {
        self.seen.clear();
        self.search_internal(query, filter_predicate, reader)
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

        // Always encode/quantize the nav query. There are some codings that support f32 x quantized
        // and we do not want to enter that path here because it is so expensive.
        let nav_query_rep = reader
            .nav_vectors()?
            .get(vertex_id)
            .unwrap_or(Err(Error::not_found_error()))?
            .to_vec();
        let nav_query = new_query_vector_distance_indexing(
            &nav_query_rep,
            reader.config().similarity,
            reader.config().nav_format,
        );

        let rerank_query_rep = if self.params.num_rerank > 0 {
            Some(
                reader
                    .rerank_vectors()?
                    .get(vertex_id)
                    .unwrap_or(Err(Error::not_found_error()))?
                    .to_vec(),
            )
        } else {
            None
        };
        let rerank_query = rerank_query_rep.as_ref().map(|q| {
            new_query_vector_distance_indexing(
                q,
                reader.config().similarity,
                reader.config().rerank_format,
            )
        });

        self.search_graph_and_rerank(
            nav_query.as_ref(),
            |_| true,
            rerank_query.as_ref().map(|q| q.as_ref()),
            reader,
        )
    }

    fn search_internal(
        &mut self,
        query: &[f32],
        filter_predicate: impl FnMut(i64) -> bool,
        reader: &mut impl GraphVectorIndexReader,
    ) -> Result<Vec<Neighbor>> {
        let nav_query = new_query_vector_distance_f32(
            query,
            reader.config().similarity,
            reader.config().nav_format,
        );
        let rerank_query = if self.params.num_rerank > 0 {
            Some(new_query_vector_distance_f32(
                query,
                reader.config().similarity,
                reader.config().rerank_format,
            ))
        } else {
            None
        };

        self.search_graph_and_rerank(
            nav_query.as_ref(),
            filter_predicate,
            rerank_query.as_ref().map(|q| q.as_ref()),
            reader,
        )
    }

    fn search_graph_and_rerank(
        &mut self,
        nav_query: &dyn QueryVectorDistance,
        mut filter_predicate: impl FnMut(i64) -> bool,
        rerank_query: Option<&dyn QueryVectorDistance>,
        reader: &mut impl GraphVectorIndexReader,
    ) -> Result<Vec<Neighbor>> {
        // TODO: come up with a better way of managing re-used state.
        self.candidates.clear();
        self.visited = 0;

        let mut graph = reader.graph()?;
        let mut nav = reader.nav_vectors()?;
        if let Some(epr) = graph.entry_point() {
            let entry_point = epr?;
            let entry_vector = nav
                .get(entry_point)
                .unwrap_or(Err(Error::not_found_error()))?;
            self.candidates
                .add_unvisited(Neighbor::new(entry_point, nav_query.distance(entry_vector)));
            self.seen.insert(entry_point);
        }

        while let Some(best_candidate) = self.candidates.next_unvisited() {
            self.visited += 1;
            let vertex_id = best_candidate.neighbor().vertex();
            let node = graph
                .get_vertex(vertex_id)
                .unwrap_or_else(|| Err(Error::not_found_error()))?;
            if filter_predicate(vertex_id) {
                best_candidate.visit();
            } else {
                best_candidate.remove();
            }

            for edge in node.edges() {
                if !self.seen.insert(edge) {
                    continue;
                }
                let vec = nav.get(edge).unwrap_or(Err(Error::not_found_error()))?;
                self.candidates
                    .add_unvisited(Neighbor::new(edge, nav_query.distance(vec)));
            }
        }

        if rerank_query.is_none() {
            return Ok(self.candidates.iter().map(|c| c.neighbor).collect());
        }

        let rerank_query = rerank_query.unwrap();
        let mut rerank_vectors = reader.rerank_vectors()?;
        let rescored = self
            .candidates
            .iter()
            .take(self.params.num_rerank)
            .map(|c| {
                let vertex = c.neighbor.vertex();
                rerank_vectors
                    .get(vertex)
                    .expect("row exists")
                    .map(|rv| Neighbor::new(vertex, rerank_query.distance(rv)))
            })
            .collect::<Result<Vec<_>>>();
        rescored.map(|mut r| {
            r.sort();
            r
        })
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
    fn visit(mut self) {
        self.list.candidates[self.index].visited = true;
        self.update_next_unvisited(self.index + 1)
    }

    /// Rmove this candidate from the list. May happen if filter check is not passed.
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

    use wt_mdb::Result;

    use crate::{
        graph::{
            Graph, GraphConfig, GraphLayout, GraphVectorIndexReader, GraphVectorStore, GraphVertex,
        },
        vectors::{F32VectorCoding, F32VectorDistance, VectorSimilarity},
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
        pub fn new<T, V>(
            max_edges: NonZero<usize>,
            distance_fn: Box<dyn F32VectorDistance>,
            iter: T,
        ) -> Self
        where
            T: IntoIterator<Item = V>,
            V: Into<Vec<f32>>,
        {
            let coder = F32VectorCoding::BinaryQuantized.new_coder(VectorSimilarity::Euclidean);
            let mut rep = iter
                .into_iter()
                .map(|x| {
                    let v = x.into();
                    let b = coder.encode(&v);
                    TestVector {
                        vector: v,
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
                rerank_format: F32VectorCoding::F32,
                layout: GraphLayout::Split,
                max_edges,
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

    impl GraphVectorIndexReader for TestGraphVectorIndexReader<'_> {
        type Graph<'b>
            = TestGraphAccess<'b>
        where
            Self: 'b;
        type NavVectorStore<'b>
            = TestNavVectorStore<'b>
        where
            Self: 'b;
        type RerankVectorStore<'b>
            = TestRerankVectorStore<'b>
        where
            Self: 'b;

        fn config(&self) -> &GraphConfig {
            &self.0.config
        }

        fn graph(&self) -> Result<Self::Graph<'_>> {
            Ok(TestGraphAccess(self.0))
        }

        fn nav_vectors(&self) -> Result<Self::NavVectorStore<'_>> {
            Ok(TestNavVectorStore(self.0))
        }

        fn rerank_vectors(&self) -> Result<Self::RerankVectorStore<'_>> {
            Ok(TestRerankVectorStore(self.0))
        }
    }

    #[derive(Debug)]
    pub struct TestGraphAccess<'a>(&'a TestGraphVectorIndex);

    impl Graph for TestGraphAccess<'_> {
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

    pub struct TestRerankVectorStore<'a>(&'a TestGraphVectorIndex);

    impl GraphVectorStore for TestRerankVectorStore<'_> {
        fn format(&self) -> F32VectorCoding {
            self.0.config.rerank_format
        }

        fn get(&mut self, vertex_id: i64) -> Option<Result<&[u8]>> {
            self.0
                .data
                .get(vertex_id as usize)
                .map(|vertex| Ok(bytemuck::cast_slice(vertex.vector.as_ref())))
        }
    }

    pub struct TestNavVectorStore<'a>(&'a TestGraphVectorIndex);

    impl GraphVectorStore for TestNavVectorStore<'_> {
        fn format(&self) -> F32VectorCoding {
            self.0.config.nav_format
        }

        fn get(&mut self, vertex_id: i64) -> Option<Result<&[u8]>> {
            self.0
                .data
                .get(vertex_id as usize)
                .map(|vertex| Ok(vertex.nav_vector.as_ref()))
        }
    }

    #[derive(Debug)]
    pub struct TestGraph<'a>(&'a TestGraphVectorIndex);

    impl Graph for TestGraph<'_> {
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

    impl GraphVertex for TestGraphVertex<'_> {
        type EdgeIterator<'c>
            = std::iter::Copied<std::slice::Iter<'c, i64>>
        where
            Self: 'c;

        fn edges(&self) -> Self::EdgeIterator<'_> {
            self.0.edges.iter().copied()
        }
    }

    fn build_test_graph(max_edges: usize) -> TestGraphVectorIndex {
        let dim_values = [-0.25, -0.125, 0.125, 0.25];
        TestGraphVectorIndex::new(
            NonZero::new(max_edges).unwrap(),
            VectorSimilarity::Dot.new_distance_function(),
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
        });
        assert_eq!(
            searcher
                .search(&[-0.1, -0.1, -0.1, -0.1], &mut index.reader())
                .unwrap(),
            vec![
                Neighbor::new(0, 0.0),
                Neighbor::new(1, 0.0),
                Neighbor::new(4, 0.0),
                Neighbor::new(16, 0.0)
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
                Neighbor::new(1, 0.06813),
                Neighbor::new(4, 0.06813),
                Neighbor::new(16, 0.06813),
                Neighbor::new(0, 0.09),
            ]
        );
    }
}
