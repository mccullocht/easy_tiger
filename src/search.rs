use std::{collections::HashSet, num::NonZero, sync::mpsc::channel};

use crate::{
    graph::{
        Graph, GraphSearchParams, GraphVectorIndexReader, GraphVertex, NavVectorStore,
        ParallelGraphVectorIndexReader,
    },
    quantization::binary_quantize,
    Neighbor,
};

use wt_mdb::{Error, Result, WiredTigerError};

/// Helper to search a Vamana graph.
pub struct GraphSearcher {
    candidates: CandidateList,
    seen: HashSet<i64>, // TODO: use a more efficient hash function (ahash?)
    params: GraphSearchParams,
}

impl GraphSearcher {
    /// Create a new, reusable graph searcher.
    pub fn new(params: GraphSearchParams) -> Self {
        Self {
            candidates: CandidateList::new(params.beam_width.get()),
            seen: HashSet::new(),
            params,
        }
    }

    /// Return the search params.
    pub fn params(&self) -> &GraphSearchParams {
        &self.params
    }

    /// Search for `query` in the given graph `reader`. The reader will search in quantized space
    /// before optionally re-ranking based on higher fidelity vectors stored in the graph.
    ///
    /// Returns an approximate list of neighbors with the highest scores.
    pub fn search<R: GraphVectorIndexReader>(
        &mut self,
        query: &[f32],
        reader: &mut R,
    ) -> Result<Vec<Neighbor>> {
        self.seen.clear();
        self.search_internal(query, reader)
    }

    /// Search for the vector at `vertex_id` and return matching candidates.
    pub fn search_for_insert<R: GraphVectorIndexReader>(
        &mut self,
        vertex_id: i64,
        reader: &mut R,
    ) -> Result<Vec<Neighbor>> {
        self.seen.clear();
        // Insertions may be concurrent and there could already be backlinks to this vertex in the graph.
        // Marking this vertex as seen ensures we don't traverse or score ourselves (should be identity score).
        self.seen.insert(vertex_id);

        // NB: if inserting in a WT backed graph this will create a cursor that we immediately discard.
        let query = reader
            .graph()?
            .get(vertex_id)
            .unwrap_or(Err(Error::WiredTiger(WiredTigerError::NotFound)))?
            .vector()
            .to_vec();
        self.search_internal(&query, reader)
    }

    fn search_internal<R: GraphVectorIndexReader>(
        &mut self,
        query: &[f32],
        reader: &mut R,
    ) -> Result<Vec<Neighbor>> {
        self.candidates.clear();

        let mut graph = reader.graph()?;
        let mut nav = reader.nav_vectors()?;
        let nav_scorer = reader.metadata().new_nav_scorer();
        let nav_query = if let Some(epr) = graph.entry_point() {
            let entry_point = epr?;
            let nav_query = binary_quantize(query);
            let entry_vector = nav
                .get(entry_point)
                .unwrap_or(Err(Error::WiredTiger(WiredTigerError::NotFound)))?;
            self.candidates.add_unvisited(Neighbor::new(
                entry_point,
                nav_scorer.score(&nav_query, &entry_vector),
            ));
            self.seen.insert(entry_point);
            nav_query
        } else {
            return Ok(vec![]);
        };

        while let Some(mut best_candidate) = self.candidates.next_unvisited() {
            let node = graph
                .get(best_candidate.neighbor().vertex())
                .unwrap_or_else(|| Err(Error::WiredTiger(WiredTigerError::NotFound)))?;
            // If we aren't reranking we don't need to copy the actual vector.
            best_candidate.visit(if self.params.num_rerank > 0 {
                node.vector().into()
            } else {
                Vec::new()
            });

            for edge in node.edges() {
                if !self.seen.insert(edge) {
                    continue;
                }
                let vec = nav
                    .get(edge)
                    .unwrap_or(Err(Error::WiredTiger(WiredTigerError::NotFound)))?;
                self.candidates
                    .add_unvisited(Neighbor::new(edge, nav_scorer.score(&nav_query, &vec)));
            }
        }

        Ok(self.extract_results(query, reader))
    }

    /// Search `reader` for `query` with up to `max_concurrent` vector node lookups in flight at
    /// any time.
    ///
    /// Return the top matching candidates.
    pub fn search_concurrently<R>(
        &mut self,
        query: &[f32],
        reader: &mut R,
        max_concurrent: NonZero<usize>,
    ) -> Result<Vec<Neighbor>>
    where
        R: ParallelGraphVectorIndexReader,
    {
        self.seen.clear();
        self.candidates.clear();

        // XXX we could probably factor this part out into a shared helper.
        let mut graph = reader.graph()?;
        let mut nav = reader.nav_vectors()?;
        let nav_scorer = reader.metadata().new_nav_scorer();
        let nav_query = if let Some(epr) = graph.entry_point() {
            let entry_point = epr?;
            let nav_query = binary_quantize(query);
            let entry_vector = nav
                .get(entry_point)
                .unwrap_or(Err(Error::WiredTiger(WiredTigerError::NotFound)))?;
            self.candidates.add_unvisited(Neighbor::new(
                entry_point,
                nav_scorer.score(&nav_query, &entry_vector),
            ));
            self.seen.insert(entry_point);
            nav_query
        } else {
            return Ok(vec![]);
        };

        let mut num_concurrent = 0;
        let (send, recv) = channel();
        loop {
            for neighbor in self
                .candidates
                .unvisited_iter()
                .take(max_concurrent.get() - num_concurrent)
                .copied()
            {
                let graph_send = send.clone();
                reader.lookup(neighbor.vertex(), move |vertex| {
                    graph_send
                        .send(vertex.map(|result| {
                            result.map(|v| {
                                (neighbor, v.vector().to_vec(), v.edges().collect::<Vec<_>>())
                            })
                        }))
                        .unwrap();
                });
                num_concurrent += 1;
            }

            // If we have no outstanding reads at this point then all of the members of the
            // candidate list have been visited and we've converged on the result set.
            if num_concurrent == 0 {
                break;
            }

            for lookup_result in std::iter::once(recv.recv().unwrap()).chain(recv.try_iter()) {
                num_concurrent -= 1;
                // XXX do something to make this not found business suck less. either getting to
                // notfound quicker or maybe implementing unwrap_or_not_found()?
                let (neighbor, vector, edges) =
                    lookup_result.unwrap_or(Err(Error::WiredTiger(WiredTigerError::NotFound)))?;
                self.candidates.visit_candidate(neighbor, vector);
                for edge in edges {
                    if self.seen.insert(edge) {
                        let doc = nav
                            .get(edge)
                            .unwrap_or(Err(Error::WiredTiger(WiredTigerError::NotFound)))?;
                        self.candidates
                            .add_unvisited(Neighbor::new(edge, nav_scorer.score(&nav_query, &doc)));
                    }
                }
            }
        }

        Ok(self.extract_results(query, reader))
    }

    fn extract_results<R: GraphVectorIndexReader>(
        &mut self,
        query: &[f32],
        reader: &R,
    ) -> Vec<Neighbor> {
        if self.params.num_rerank > 0 {
            let mut normalized_query = query.to_vec();
            let scorer = reader.metadata().new_scorer();
            scorer.normalize(&mut normalized_query);
            self.candidates
                .iter()
                .take(self.params.num_rerank)
                .map(|c| {
                    Neighbor::new(
                        c.neighbor.vertex(),
                        scorer.score(query, c.state.vector().expect("node visited")),
                    )
                })
                .collect()
        } else {
            self.candidates.iter().map(|c| c.neighbor).collect()
        }
    }
}

#[derive(Debug, PartialEq)]
enum CandidateState {
    Unvisited,
    Pending,
    Visited(Vec<f32>),
}

impl CandidateState {
    fn vector(&self) -> Option<&[f32]> {
        match self {
            CandidateState::Visited(v) => Some(v),
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

    /// Return an iterator over all unvisited candidates, marking any return as Pending.
    fn unvisited_iter(&mut self) -> impl Iterator<Item = &'_ Neighbor> {
        self.candidates
            .iter_mut()
            .skip(self.next_unvisited)
            .filter_map(|c| match c.state {
                CandidateState::Unvisited => {
                    c.state = CandidateState::Pending;
                    Some(&c.neighbor)
                }
                _ => None,
            })
    }

    /// Mark `neighbor` as visited and insert associated `vector`.
    ///
    /// Returns true if `neighbor` was successfully updated.
    fn visit_candidate(&mut self, neighbor: Neighbor, vector: Vec<f32>) -> bool {
        match self
            .candidates
            .binary_search_by_key(&neighbor, |c| c.neighbor)
        {
            Ok(index) => {
                self.candidates[index].state = CandidateState::Visited(vector);
                true
            }
            Err(_) => false,
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
    fn visit(&mut self, vector: impl Into<Vec<f32>>) {
        self.list.candidates[self.index].state = CandidateState::Visited(vector.into());
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
    use std::num::NonZero;

    use crate::{scoring::DotProductScorer, test::TestGraphVectorIndex, Neighbor};

    use super::{GraphSearchParams, GraphSearcher};

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
                Neighbor::new(5, 1.0)
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
                Neighbor::new(0, 0.6),
                Neighbor::new(1, 0.59707),
                Neighbor::new(4, 0.59707),
                Neighbor::new(5, 0.59487)
            ]
        );
    }
}
