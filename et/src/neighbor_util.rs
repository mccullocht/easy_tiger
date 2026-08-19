use std::{
    ops::RangeInclusive,
    sync::{Mutex, atomic::AtomicU64},
};

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

    /// Return the maximum competitive distance.
    pub fn max_distance(&self) -> f64 {
        f64::from_bits(self.max_dist.load(std::sync::atomic::Ordering::Relaxed))
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

/// A candidate scored with a lower and upper bound on its true distance.
#[derive(Debug, Copy, Clone)]
pub(crate) struct BoundedNeighbor {
    vertex: i64,
    lower: f64,
    upper: f64,
}

impl BoundedNeighbor {
    pub fn new(vertex: i64, bounds: RangeInclusive<f64>) -> Self {
        Self {
            vertex,
            lower: *bounds.start(),
            upper: *bounds.end(),
        }
    }
}

impl From<BoundedNeighbor> for Neighbor {
    /// Represent the candidate by its upper bound distance, the value it is ordered by.
    fn from(value: BoundedNeighbor) -> Self {
        Neighbor::new(value.vertex, value.upper)
    }
}

struct BoundedNeighborsRep {
    candidates: Vec<BoundedNeighbor>,
    /// Prune once the candidate list reaches this length.
    prune_at: usize,
}

/// Maintain the set of candidates that could still enter the top `k` given per-candidate distance
/// bounds.
///
/// Candidates are ordered by their upper bound distance; any candidate whose *lower* bound is no
/// greater than the upper bound of the `k`-th candidate is retained, since a more accurate distance
/// computation could still promote it into the top `k`. The size of this set is the depth a
/// reranking pass must reach to be sure it has not missed a true top-`k` neighbor.
pub(crate) struct BoundedNeighbors {
    rep: CachePadded<(Mutex<BoundedNeighborsRep>, usize)>,
    // The upper bound distance of the k-th candidate. Candidates with a larger lower bound can be
    // discarded without locking.
    threshold: CachePadded<AtomicU64>,
}

impl BoundedNeighbors {
    /// Create a list tracking everything competitive with the top `k`.
    pub fn new(k: usize) -> Self {
        Self {
            rep: CachePadded::new((
                Mutex::new(BoundedNeighborsRep {
                    candidates: Vec::with_capacity(k * 2),
                    prune_at: k * 2,
                }),
                k,
            )),
            threshold: CachePadded::new(AtomicU64::new(f64::MAX.to_bits())),
        }
    }

    /// Add a new candidate. It may be discarded if it cannot enter the top `k`.
    pub fn add(&self, candidate: BoundedNeighbor) {
        use std::sync::atomic::Ordering;

        // Skip non-competitive values without locking. Like `TopNeighbors` a stale threshold only
        // costs us an unnecessary lock, never correctness.
        if f64::from_bits(self.threshold.load(Ordering::Relaxed)) < candidate.lower {
            return;
        }

        let mut rep = self.rep.0.lock().unwrap();
        rep.candidates.push(candidate);
        if rep.candidates.len() >= rep.prune_at {
            let threshold = Self::prune(&mut rep, self.rep.1);
            self.threshold.store(threshold.to_bits(), Ordering::Relaxed);
        }
    }

    /// Drop candidates that cannot enter the top `k`, returning the new threshold.
    fn prune(rep: &mut BoundedNeighborsRep, k: usize) -> f64 {
        if rep.candidates.len() <= k {
            return f64::MAX;
        }

        let (_, kth, _) = rep
            .candidates
            .select_nth_unstable_by(k - 1, |a, b| a.upper.total_cmp(&b.upper));
        let threshold = kth.upper;
        rep.candidates.retain(|c| c.lower <= threshold);
        // Amortize pruning: never prune more often than once per doubling of the retained set.
        rep.prune_at = (k * 2).max(rep.candidates.len() * 2);
        threshold
    }

    /// Extract the competitive candidates ordered by upper bound distance.
    pub fn into_neighbors(self) -> Vec<Neighbor> {
        let (rep_mu, k) = self.rep.into_inner();
        let mut rep = rep_mu.into_inner().unwrap();
        Self::prune(&mut rep, k);
        rep.candidates.sort_unstable_by(|a, b| {
            a.upper
                .total_cmp(&b.upper)
                .then_with(|| a.vertex.cmp(&b.vertex))
        });
        rep.candidates.into_iter().map(Neighbor::from).collect()
    }
}
