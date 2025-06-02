use std::{io, num::NonZero};

use easy_tiger::{input::VectorStore, Neighbor};

/// Computes the recall for a query from a golden file.
// TODO: add an option for NDGC recall computation.
pub struct RecallComputer<N> {
    k: usize,
    neighbors: N,
}

impl<N> RecallComputer<N>
where
    N: VectorStore<Elem = u32>,
{
    /// Create a new RecallComputer that examines the first `k` results of `neighbors`.
    pub fn new(k: NonZero<usize>, neighbors: N) -> io::Result<Self> {
        if k.get() <= neighbors.elem_stride() {
            Ok(Self {
                k: k.get(),
                neighbors,
            })
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "recall k must be <= neighbors_len",
            ))
        }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// Compute the recall based on golden data for `query_index` given `query_results`.
    ///
    /// *Panics* if `query_index` is out of bounds in the golden file.
    pub fn compute_recall(&self, query_index: usize, query_results: &[Neighbor]) -> f64 {
        let mut expected = self.neighbors[query_index][..self.k].to_vec();
        expected.sort_unstable();
        let count = query_results
            .iter()
            .take(self.k)
            .filter(|n| expected.binary_search(&(n.vertex() as u32)).is_ok())
            .count();
        count as f64 / self.k as f64
    }
}
