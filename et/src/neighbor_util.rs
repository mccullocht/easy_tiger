use std::sync::{atomic::AtomicU64, Mutex};

use crossbeam_utils::CachePadded;
use easy_tiger::Neighbor;

/// Maintain a fixed size list of top [`Neighbor`]s with minimal locking.
pub(crate) struct TopNeighbors {
    // A locked list of neighbors and the target number of results.
    rep: CachePadded<(Mutex<Vec<Neighbor>>, usize)>,
    // An f64 value containing maximum competitive distance. This value can be consulted without
    // locking to eliminate non-competitive values.
    max_dist: CachePadded<AtomicU64>,
}

impl TopNeighbors {
    /// Create a new neighbor list with up to `n` values.
    pub fn new(n: usize) -> Self {
        Self {
            rep: CachePadded::new((Mutex::new(Vec::with_capacity(n * 2)), n)),
            max_dist: CachePadded::new(AtomicU64::new(f64::MAX.to_bits())),
        }
    }

    /// Add a new neighbor to the list. The neighbor may be discarded if it is not competitive.
    pub fn add(&self, neighbor: Neighbor) {
        use std::sync::atomic::Ordering;

        // Skip non-competitive values without locking. We use relaxed ordering when accessing this
        // value because there's no correctness penalty to being incorrect, just a small performance
        // penalty caused by unncessarily locking.
        if f64::from_bits(self.max_dist.load(Ordering::Relaxed)) < neighbor.distance() {
            return;
        }

        let mut neighbors = self.rep.0.lock().unwrap();
        neighbors.push(neighbor);
        if neighbors.len() == self.rep.1 * 2 {
            // Order the list to keep the top N and record a new (lower) max distance threshold,
            // then truncate the list to make room for more dedicated results.
            let (_, t, _) = neighbors.select_nth_unstable(self.rep.1 - 1);
            self.max_dist
                .store(t.distance().to_bits(), Ordering::Relaxed);
            neighbors.truncate(self.rep.1);
        }
    }

    /// Extract the list of the top N neighbors.
    pub fn into_neighbors(self) -> Vec<Neighbor> {
        let (neighbors_mu, n) = self.rep.into_inner();
        let mut neighbors = neighbors_mu.into_inner().unwrap();
        neighbors.sort_unstable();
        neighbors.truncate(n);
        neighbors
    }
}
