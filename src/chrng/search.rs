//! Routines to search a CHRNG graph index.

use min_max_heap::MinMaxHeap;
use std::{
    cmp::Ordering,
    collections::HashSet,
    ops::{Add, AddAssign},
};
use wt_mdb::Result;

use crate::{
    chrng::{
        ClusterKey, HeadGraphCursor, HeadVectorDistanceCursor, IndexReader, TailGraphCursor,
        TailVectorDistanceCursor,
    },
    Neighbor,
};

#[derive(Debug, Clone, Copy)]
struct ClusterNeighbor {
    vertex_id: ClusterKey,
    distance: f64,
}

impl ClusterNeighbor {
    fn new(vertex_id: ClusterKey, distance: f64) -> Self {
        Self {
            vertex_id,
            distance,
        }
    }
}

impl PartialEq for ClusterNeighbor {
    fn eq(&self, other: &Self) -> bool {
        self.vertex_id == other.vertex_id && self.distance.total_cmp(&other.distance).is_eq()
    }
}

impl Eq for ClusterNeighbor {}

impl PartialOrd for ClusterNeighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ClusterNeighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.vertex_id.cmp(&other.vertex_id))
    }
}

impl From<ClusterNeighbor> for Neighbor {
    fn from(value: ClusterNeighbor) -> Self {
        Neighbor::new(value.vertex_id.vector_id, value.distance)
    }
}

/// Searches a CHRNG index.
pub struct Searcher {
    beam_width: usize,
    // We consume candidates as a max heap, but will push on to min and bound to beam_width.
    candidates: MinMaxHeap<ClusterNeighbor>,
    // Accepted candidates are min pushed, bound to beam_width.
    results: MinMaxHeap<Neighbor>,

    // Set of seen vertexes during the search of the head index.
    seen_head_vertexes: HashSet<i64>,

    // As we traverse the graph we may visit individual keys when exploring, but for each candidate
    // that is consumed we score the entire cluster. The visited cluster set is likely to be much
    // smaller on average than the list of all visited vertexes.
    seen_tail_clusters: HashSet<u32>,
    seen_tail_vertexes: HashSet<ClusterKey>,

    stats: Stats,
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Stats {
    /// Number of vertexes we examined during search of the head index.
    pub head_seen_vertexes: usize,
    /// The number of clusters we scored completely during the search.
    pub tail_seen_clusters: usize,
    /// The number of individual vertexes we scored by lookup.
    pub tail_seen_vertexes: usize,
    /// The number of vectors we performed distance computation on.
    pub tail_distance_computed_count: usize,
    /// The number of graph vertexes visited during the tail search.
    ///
    /// This value will be at least as many as beam_width.
    pub tail_visited: usize,
}

impl Add for Stats {
    type Output = Stats;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            head_seen_vertexes: self.head_seen_vertexes + rhs.head_seen_vertexes,
            tail_seen_clusters: self.tail_seen_clusters + rhs.tail_seen_clusters,
            tail_seen_vertexes: self.tail_seen_vertexes + rhs.tail_seen_vertexes,
            tail_distance_computed_count: self.tail_distance_computed_count
                + rhs.tail_distance_computed_count,
            tail_visited: self.tail_visited + rhs.tail_visited,
        }
    }
}

impl AddAssign for Stats {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Searcher {
    pub fn new(beam_width: usize) -> Self {
        Self {
            beam_width,
            candidates: MinMaxHeap::with_capacity(beam_width),
            results: MinMaxHeap::with_capacity(beam_width),
            seen_head_vertexes: HashSet::new(),
            seen_tail_clusters: HashSet::new(),
            seen_tail_vertexes: HashSet::new(),
            stats: Stats::default(),
        }
    }

    pub fn search(
        &mut self,
        query: impl Into<Vec<f32>>,
        index: &impl IndexReader,
    ) -> Result<Vec<Neighbor>> {
        self.clear();
        let query = query.into();
        let entry_cluster = self.find_best_entry_cluster(&query, index)?;
        let mut graph_cursor = index.tail_graph_cursor()?;
        let mut vec_cursor = index.tail_vector_distance_cursor(query)?;
        self.push_cluster_candidates(entry_cluster, &mut vec_cursor)?;
        // Score every cluster immediately adjacent to the best result in the entry cluster.
        if let Some(best_candidate) = self.candidates.peek_min() {
            for e in graph_cursor.edges(best_candidate.vertex_id)? {
                if self.seen_tail_clusters.insert(e.cluster_id) {
                    self.push_cluster_candidates(e.cluster_id, &mut vec_cursor)?;
                }
            }
        }

        // Run the traditional search loop after seeding the candidate queue.
        while let Some(candidate) = self.candidates.pop_min() {
            self.stats.tail_visited += 1;
            if self.results.len() >= self.beam_width
                && self
                    .results
                    .peek_max()
                    .map(|n| n.distance)
                    .unwrap()
                    .total_cmp(&candidate.distance)
                    .is_lt()
            {
                // If the candidate is worse than the worst result, break.
                break;
            }

            // Push the candidate into results, obeying beam_width.
            if self.results.len() < self.beam_width {
                self.results.push(candidate.into());
            } else {
                self.results.push_pop_max(candidate.into());
            }

            // Add unseen outbound edges from this vertex to the candidate queue.
            // If removed latency halves but so does recall.
            for e in graph_cursor.edges(candidate.vertex_id)? {
                if !self.seen_tail_clusters.contains(&e.cluster_id)
                    && self.seen_tail_vertexes.insert(e)
                {
                    self.push_candidate(e, &mut vec_cursor)?;
                }
            }
        }

        self.stats.head_seen_vertexes = self.seen_head_vertexes.len();
        self.stats.tail_seen_vertexes = self.seen_tail_vertexes.len();
        self.stats.tail_seen_clusters = self.seen_tail_clusters.len();

        Ok(self.results.drain_asc().collect::<Vec<_>>())
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    pub fn clear(&mut self) {
        self.candidates.clear();
        self.results.clear();
        self.seen_head_vertexes.clear();
        self.seen_tail_clusters.clear();
        self.seen_tail_vertexes.clear();
        self.stats = Stats::default();
    }

    fn find_best_entry_cluster(&mut self, query: &[f32], index: &impl IndexReader) -> Result<u32> {
        let mut graph_cursor = index.head_graph_cursor()?;
        let mut vec_cursor = index.head_vector_distance_cursor(query)?;

        let ep = graph_cursor.entry_point()?;
        let mut best_neighbor = Neighbor::new(ep, vec_cursor.distance(ep)?);
        self.seen_head_vertexes.insert(ep);
        let mut best_updated = true;
        while best_updated {
            best_updated = false;
            for e in graph_cursor.edges(best_neighbor.vertex)? {
                if self.seen_head_vertexes.insert(e) {
                    let e_neighbor = Neighbor::new(e, vec_cursor.distance(e)?);
                    if e_neighbor < best_neighbor {
                        best_neighbor = e_neighbor;
                        best_updated = true;
                    }
                }
            }
        }
        Ok(best_neighbor.vertex() as u32)
    }

    fn push_candidate(
        &mut self,
        vertex_id: ClusterKey,
        vectors: &mut impl TailVectorDistanceCursor,
    ) -> Result<()> {
        self.stats.tail_distance_computed_count += 1;
        let n = ClusterNeighbor::new(vertex_id, vectors.distance(vertex_id)?);
        if self.candidates.len() < self.beam_width {
            self.candidates.push(n);
        } else {
            self.candidates.push_pop_max(n);
        }
        Ok(())
    }

    /// Returns the number of values successfully inserted.
    fn push_cluster_candidates(
        &mut self,
        cluster_id: u32,
        vectors: &mut impl TailVectorDistanceCursor,
    ) -> Result<usize> {
        let mut num_inserted = 0usize;
        for r in vectors.cluster_distance(cluster_id, |k| self.seen_tail_vertexes.contains(&k)) {
            self.stats.tail_distance_computed_count += 1;
            let n = r.map(|(k, d)| ClusterNeighbor::new(k, d))?;
            if self.candidates.len() < self.beam_width {
                self.candidates.push(n);
                num_inserted += 1;
            } else if self.candidates.push_pop_max(n).vertex_id == n.vertex_id {
                num_inserted += 1;
            }
        }
        Ok(num_inserted)
    }
}
