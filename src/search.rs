use std::num::NonZero;

use crate::{
    graph::{Graph, GraphNode, NavVectorStore},
    quantization::binary_quantize,
    Neighbor,
};

/// Parameters for a search over a Vamana graph.
pub struct GraphSearchParams {
    /// Width of the graph search beam -- the number of candidates considered.
    /// We will return this many results.
    pub beam_width: NonZero<usize>,
    /// Number of results to re-rank using the vectors in the graph.
    pub num_rerank: usize,
}

/// Helper to search a Vamana graph.
pub struct GraphSearcher {
    candidates: CandidateList,
    params: GraphSearchParams,
}

impl GraphSearcher {
    pub fn new(params: GraphSearchParams) -> Self {
        Self {
            candidates: CandidateList::new(params.beam_width.get()),
            params,
        }
    }

    pub fn params(&self) -> &GraphSearchParams {
        &self.params
    }

    // NB: graph and nav have to be mutable, which means that we can only use one thread internally for
    // searching. A freelist or generator would be necessary to do a multi-threaded search that pops
    // multiple candidates at once. This would also require significant changes to CandidateList.
    // XXX need an error handling aware signature and
    pub fn search<G, N>(&mut self, query: &[f32], graph: &mut G, nav: &mut N) -> Vec<Neighbor>
    where
        G: Graph,
        N: NavVectorStore,
    {
        let nav_query = if let Some(entry_point) = graph.entry_point() {
            let nav_query = binary_quantize(query);
            let entry_vector = nav.get(entry_point).unwrap().unwrap();
            self.candidates.add_unvisited(Neighbor::new(
                entry_point,
                Self::nav_score(&nav_query, &entry_vector),
            ));
            nav_query
        } else {
            return vec![];
        };

        while let Some(mut best_candidate) = self.candidates.next_unvisited() {
            let node = graph
                .get(best_candidate.neighbor().node())
                .unwrap()
                .unwrap();
            // If we aren't reranking we don't need to read the
            best_candidate.visit(if self.params.num_rerank > 0 {
                node.vector().into()
            } else {
                Vec::new()
            });

            for edge in node.edges() {
                let vec = nav.get(edge).unwrap().unwrap();
                self.candidates
                    .add_unvisited(Neighbor::new(edge, Self::nav_score(&nav_query, &vec)));
            }
        }

        let results = if self.params.num_rerank > 0 {
            self.candidates
                .iter()
                .take(self.params.num_rerank)
                .map(|c| {
                    Neighbor::new(
                        c.neighbor.node(),
                        Self::score(query, c.vector.as_ref().expect("node visited")),
                    )
                })
                .collect()
        } else {
            self.candidates.iter().map(|c| c.neighbor).collect()
        };

        self.candidates.clear();

        results
    }

    fn score(q: &[f32], d: &[f32]) -> f64 {
        simsimd::SpatialSimilarity::dot(q, d).unwrap()
    }

    fn nav_score(q: &[u8], d: &[u8]) -> f64 {
        simsimd::BinarySimilarity::hamming(q, d).unwrap()
    }
}

/// A candidate in the search list. Once visited, the candidate becomes a result.
#[derive(Debug)]
struct Candidate {
    neighbor: Neighbor,
    // If set, we've visited this neighbor.
    vector: Option<Vec<f32>>,
}

impl From<Neighbor> for Candidate {
    fn from(neighbor: Neighbor) -> Self {
        Candidate {
            neighbor,
            vector: None,
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

    fn iter(&self) -> impl Iterator<Item = &'_ Candidate> {
        self.candidates.iter()
    }

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
        self.list.candidates[self.index].vector = Some(vector.into());
        self.list.next_unvisited = self
            .list
            .candidates
            .iter()
            .enumerate()
            .skip(self.index + 1)
            .find_map(|(i, c)| if c.vector.is_some() { None } else { Some(i) })
            .unwrap_or(self.list.candidates.len());
    }
}

// XXX when I get the next unvisited node I want to update it with a vector
// I can't represent the iteration as an iterator because it would need a reference and i wouldn't
// be able to insert new items while I have the iterator outstanding.
