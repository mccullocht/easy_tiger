use std::cmp::Ordering;

mod graph;
pub mod input;
pub mod quantization;
mod scoring;
pub mod search;
#[cfg(test)]
mod test;

/// `Neighbor` is a node and a distance relative to some other node.
///
/// During a search distance might be relative to the query vector.
/// In a graph index, distance might be relative to another node in the index.
///
/// When sorted, a slice of nodes is ordered first by distance, then by node, both in increasing
/// order.
#[derive(Debug, Copy, Clone)]
pub struct Neighbor {
    node: i64,
    dist: f64,
}

impl Neighbor {
    pub fn new(node: i64, distance: f64) -> Self {
        Self {
            node,
            dist: distance,
        }
    }

    pub fn node(&self) -> i64 {
        self.node
    }

    pub fn distance(&self) -> f64 {
        self.dist
    }
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.dist.total_cmp(&other.dist) == Ordering::Equal
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
        match self.dist.total_cmp(&other.dist) {
            Ordering::Equal => self.node.cmp(&other.node),
            Ordering::Less => Ordering::Less,
            Ordering::Greater => Ordering::Greater,
        }
    }
}
