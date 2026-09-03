use std::{
    cmp::Ordering,
    sync::{Mutex, atomic::AtomicU64},
};

use crossbeam_utils::CachePadded;
use easy_tiger::Neighbor;
use vectors::EstimatedDistance;

#[derive(Debug, Copy, Clone)]
struct BoundNeighbor {
    n: Neighbor,
    e: EstimatedDistance,
}

impl BoundNeighbor {
    fn from_lower(vector_id: i64, e: EstimatedDistance) -> Self {
        Self {
            n: Neighbor::new(vector_id, e.distance - e.error),
            e,
        }
    }

    fn from_upper(vector_id: i64, e: EstimatedDistance) -> Self {
        Self {
            n: Neighbor::new(vector_id, e.distance + e.error),
            e,
        }
    }
}

impl PartialEq for BoundNeighbor {
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n
    }
}

impl Eq for BoundNeighbor {}

impl PartialOrd for BoundNeighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BoundNeighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.n.cmp(&other.n)
    }
}

struct Inner {
    n: usize,
    results: Vec<BoundNeighbor>,
    overflow: Vec<BoundNeighbor>,
}

impl Inner {
    fn new(n: usize) -> Self {
        Self {
            n,
            results: Vec::with_capacity(n * 2),
            overflow: Vec::with_capacity(n * 2),
        }
    }

    fn add(&mut self, n: BoundNeighbor) -> Option<f64> {
        self.results.push(n);
        if self.results.len() < self.n * 2 {
            return None;
        }

        self.prune()
    }

    fn prune(&mut self) -> Option<f64> {
        let (_, t, overflow_candidates) = self.results.select_nth_unstable(self.n - 1);
        let ub = t.n.distance();
        // Discard any overflow candidates that are no longer competitive.
        self.overflow.retain(|n| n.n.distance() <= ub);
        // Examine all the remaining results and save them in overflow if they are competitive.
        for n in overflow_candidates {
            let n = BoundNeighbor::from_lower(n.n.vertex(), n.e);
            if n.n.distance() <= ub {
                self.overflow.push(n);
            }
        }
        self.results.truncate(self.n);
        Some(ub)
    }

    fn into_neighbors(mut self) -> Vec<Neighbor> {
        self.prune();
        let mut neighbors = std::iter::chain(self.results, self.overflow)
            .map(|bn| Neighbor::new(bn.n.vertex(), bn.e.distance))
            .collect::<Vec<_>>();
        neighbors.sort_unstable();
        neighbors
    }
}

/// Maintain a fixed size list of top [`Neighbor`]s with minimal locking.
pub(crate) struct TopNeighbors {
    // A locked list of neighbors and the target number of results.
    rep: CachePadded<Mutex<Inner>>,
    // An f64 value containing maximum competitive distance. This value can be consulted without
    // locking to eliminate non-competitive values.
    max_dist: CachePadded<AtomicU64>,
}

impl TopNeighbors {
    /// Create a new neighbor list with up to `n` values.
    pub fn new(n: usize) -> Self {
        Self {
            rep: CachePadded::new(Mutex::new(Inner::new(n))),
            max_dist: CachePadded::new(AtomicU64::new(f64::MAX.to_bits())),
        }
    }

    pub fn add(&self, neighbor: Neighbor) {
        self.add_estimate(
            neighbor.vertex(),
            EstimatedDistance {
                distance: neighbor.distance(),
                error: 0.0,
            },
        );
    }

    /// Add a new neighbor to the list. The neighbor may be discarded if it is not competitive.
    pub fn add_estimate(&self, vector_id: i64, estimate: EstimatedDistance) {
        use std::sync::atomic::Ordering;

        let min_dist = estimate.distance - estimate.error;

        // Skip non-competitive values without locking. We use relaxed ordering when accessing this
        // value because there's no correctness penalty to being incorrect, just a small performance
        // penalty caused by unncessarily locking.
        if f64::from_bits(self.max_dist.load(Ordering::Relaxed)) < min_dist {
            return;
        }

        let mut inner = self.rep.lock().unwrap();
        if let Some(ub) = inner.add(BoundNeighbor::from_upper(vector_id, estimate)) {
            self.max_dist.store(ub.to_bits(), Ordering::Relaxed);
        }
    }

    /// Extract the list of the top N neighbors.
    pub fn into_neighbors(self) -> Vec<Neighbor> {
        self.rep.into_inner().into_inner().unwrap().into_neighbors()
    }
}
