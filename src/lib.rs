use std::cmp::Ordering;

pub mod bulk;
pub mod crud;
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

#[cfg(test)]
mod test_lib {
    use std::cmp::Ordering;

    use crate::Neighbor;

    #[test]
    fn neighbor_eq() {
        assert_eq!(Neighbor::new(1, 1.0), Neighbor::new(1, 1.0));
        assert_ne!(Neighbor::new(1, 1.0), Neighbor::new(1, 1.01));
    }

    #[test]
    fn neighbor_ord() {
        assert_eq!(
            Neighbor::new(1, 1.0).cmp(&Neighbor::new(1, 1.0)),
            Ordering::Equal
        );
        assert_eq!(
            Neighbor::new(1, 1.1).cmp(&Neighbor::new(1, 1.0)),
            Ordering::Less
        );
        assert_eq!(
            Neighbor::new(1, 1.0).cmp(&Neighbor::new(1, 1.1)),
            Ordering::Greater
        );
        assert_eq!(
            Neighbor::new(1, 1.0).cmp(&Neighbor::new(2, 1.0)),
            Ordering::Less
        );
        assert_eq!(
            Neighbor::new(2, 1.0).cmp(&Neighbor::new(1, 1.0)),
            Ordering::Greater
        );
    }
}
