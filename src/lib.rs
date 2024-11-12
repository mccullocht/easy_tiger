use std::cmp::Ordering;

pub mod bulk;
pub mod graph;
pub mod input;
pub mod quantization;
pub mod scoring;
pub mod search;
#[cfg(test)]
mod test;
pub mod wt;

/// `Neighbor` is a node and a distance relative to some other node.
///
/// During a search score might be relative to the query vector.
/// In a graph index, score might be relative to another node in the index.
///
/// When compared `Neighbor`s are ordered first in descending order by score,
/// then in ascending order by node.
#[derive(Debug, Copy, Clone)]
pub struct Neighbor {
    node: i64,
    score: f64,
}

impl Neighbor {
    pub fn new(node: i64, score: f64) -> Self {
        Self { node, score }
    }

    pub fn node(&self) -> i64 {
        self.node
    }

    pub fn score(&self) -> f64 {
        self.score
    }
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.score.total_cmp(&other.score).is_eq()
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
            Ordering::Equal => self.node.cmp(&other.node),
            _ => c,
        }
    }
}
