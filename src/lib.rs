use std::cmp::Ordering;

pub mod bulk;
pub mod graph;
pub mod input;
pub mod quantization;
pub mod scoring;
pub mod search;
#[cfg(test)]
mod test;
pub mod worker_pool;
pub mod wt;

/// `Neighbor` is a node and a distance relative to some other node.
///
/// During a search score might be relative to the query vector.
/// In a graph index, score might be relative to another node in the index.
///
/// When compared `Neighbor`s are ordered first in descending order by score,
/// then in ascending order by vertex id.
#[derive(Debug, Copy, Clone)]
pub struct Neighbor {
    vertex: i64,
    score: f64,
}

impl Neighbor {
    pub fn new(vertex: i64, score: f64) -> Self {
        Self { vertex, score }
    }

    pub fn vertex(&self) -> i64 {
        self.vertex
    }

    pub fn score(&self) -> f64 {
        self.score
    }
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.vertex == other.vertex && self.score.total_cmp(&other.score).is_eq()
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        let c = self.score.total_cmp(&other.score).reverse();
        match c {
            Ordering::Equal => self.vertex.cmp(&other.vertex),
            _ => c,
        }
    }
}
