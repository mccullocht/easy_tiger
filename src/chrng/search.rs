//! Routines to search a CHRNG graph index.

use min_max_heap::MinMaxHeap;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
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
        self.seen_tail_clusters.insert(entry_cluster);
        self.push_cluster_candidates(entry_cluster, &mut vec_cursor)?;

        // XXX with composite keying I could put each vector in two clusters if I really wanted to.
        // this can be graph aware (which would be _very_ annoying) or graph unaware (only the nav
        // vector is duplicated). I would want to implement SOAR scoring for this.
        //
        // XXX graph unaware smearing
        // * I need a coherent strategy for cluster scoring becuase that's the only way I'm going to
        //   find this thing.
        // * I need to be able to canonicalize to the primary cluster to pivot back into the graph.
        //
        // XXX graph aware
        // * Easy enough to canonicalize references back to the original vector when searching,
        //   which also avoids duplicating results in the queue.
        // * They do need to point to each other to handle updates correctly.

        let mut rounds = 1usize;
        loop {
            // XXX at 16 we do 10-12 rounds
            // XXX at 32 we do 7-9 rounds
            // XXX at 64 we do 6-7 rounds
            // XXX this is the number of rounds of IO we'd have to do assuming unlimited capacity to
            // actually do them and the penalty on object storage is ~200ms per. This does not
            // count graph access which we are assuming(!) is free.
            let num_bulk_candidates = 32;
            let bulk_score_min_edges = 4;
            let mut candidates = Vec::with_capacity(num_bulk_candidates);
            while let Some(candidate) = self.candidates.pop_min() {
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

                candidates.push(candidate);
                if candidates.len() == num_bulk_candidates {
                    break;
                }
            }

            // Either there are no more candidates or no more candidates better than any results.
            if candidates.is_empty() {
                break;
            }

            // XXX it would be worth counting under full candidate lists.

            self.stats.tail_visited += candidates.len();
            let mut clustered_candidates: HashMap<u32, HashSet<ClusterKey>> = HashMap::new();
            for c in candidates {
                if self.results.len() < self.beam_width {
                    self.results.push(c.into());
                } else {
                    self.results.push_pop_max(c.into());
                }

                for e in graph_cursor.edges(c.vertex_id)? {
                    if self.seen_tail_clusters.contains(&e.cluster_id)
                        || self.seen_tail_vertexes.contains(&e)
                    {
                        continue;
                    }

                    clustered_candidates
                        .entry(e.cluster_id)
                        .or_default()
                        .insert(e);
                }
            }

            let mut clusters_scored = 0usize;
            let mut vertexes_scored = 0usize;
            for (cluster_id, edges) in clustered_candidates {
                if edges.len() >= bulk_score_min_edges {
                    self.push_cluster_candidates(cluster_id, &mut vec_cursor)?;
                    clusters_scored += 1;
                    self.seen_tail_clusters.insert(cluster_id);
                } else {
                    vertexes_scored += edges.len();
                    for e in edges {
                        self.push_candidate(e, &mut vec_cursor)?;
                        self.seen_tail_vertexes.insert(e);
                    }
                }
            }

            println!("  round={rounds:2} clusters_scored={clusters_scored:2} vertexes_scored={vertexes_scored:3}");
            rounds += 1;
        }

        self.stats.head_seen_vertexes = self.seen_head_vertexes.len();
        self.stats.tail_seen_vertexes = self.seen_tail_vertexes.len();
        self.stats.tail_seen_clusters = self.seen_tail_clusters.len();

        // XXX we've concluded that many vertexes are scored individually before cluster scoring.

        let tail_not_cluster_scored = self
            .seen_tail_vertexes
            .iter()
            .copied()
            .filter(|k| !self.seen_tail_clusters.contains(&k.cluster_id))
            .collect::<HashSet<ClusterKey>>();
        println!(
            "  rounds {} seen clusters {} seen vertexes {} seen vertexes not cluster scored {}",
            rounds,
            self.seen_tail_clusters.len(),
            self.seen_tail_vertexes.len(),
            tail_not_cluster_scored.len()
        );

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
