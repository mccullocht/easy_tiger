//! EasyTiger is a scheme for indexing dense vectors backed by WiredTiger storage.
//!
//! This is built on top of WiredTiger APIs and designed to be used in parallel to
//! another table as a projection of vector data from that source. Underneath a
//! graph-based DiskANN index is used to serve vector similarity queries.

pub mod graph;
pub mod graph_clustering;
pub mod input;
pub mod kmeans;
pub mod search;
pub mod spann;
pub mod vamana;
pub mod wt;

use std::cmp::Ordering;

/// `Neighbor` is a node and a distance relative to some other node.
///
/// During a search distance might be relative to the query vector.
/// In a graph index, distance might be relative to another node in the index.
///
/// When compared `Neighbor`s are ordered first by distance then by vertex id.
/// then in ascending order by vertex id.
#[derive(Debug, Copy, Clone)]
pub struct Neighbor {
    vertex: i64,
    distance: f64,
}

impl Neighbor {
    pub fn new(vertex: i64, distance: f64) -> Self {
        Self { vertex, distance }
    }

    pub fn vertex(&self) -> i64 {
        self.vertex
    }

    pub fn distance(&self) -> f64 {
        self.distance
    }
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.vertex == other.vertex && self.distance.total_cmp(&other.distance).is_eq()
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
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.vertex.cmp(&other.vertex))
    }
}

impl From<[u8; 16]> for Neighbor {
    fn from(value: [u8; 16]) -> Self {
        let c = value.as_chunks::<8>().0;
        Self::new(i64::from_le_bytes(c[0]), f64::from_le_bytes(c[1]))
    }
}

impl From<Neighbor> for [u8; 16] {
    fn from(value: Neighbor) -> Self {
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&value.vertex().to_le_bytes());
        bytes[8..].copy_from_slice(&value.distance().to_le_bytes());
        bytes
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
            Ordering::Greater
        );
        assert_eq!(
            Neighbor::new(1, 1.0).cmp(&Neighbor::new(1, 1.1)),
            Ordering::Less
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
