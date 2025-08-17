use std::{
    collections::HashSet,
    fs::File,
    io,
    num::NonZero,
    path::{Path, PathBuf},
};

use clap::Args;
use easy_tiger::{
    input::{DerefVectorStore, VectorStore},
    Neighbor,
};
use memmap2::Mmap;

// XXX the computation method/type will be an enum
// XXX only need the one struct.
// XXX neighbors can be loaded from a path to make life easier.

#[derive(Args)]
pub struct RecallArgs {
    /// Compute recall@k. Must be <= neighbors_len.
    #[arg(long)]
    recall_k: Option<NonZero<usize>>,
    /// Path buf to formatted [`Neighbor`] vectors.
    /// This should include one row of length neighbors_len for each vector in the query set.
    #[arg(long)]
    neighbors: Option<PathBuf>,
    /// Number of neighbors for each query in the neighbors file.
    #[arg(long, default_value_t = NonZero::new(100).unwrap())]
    neighbors_len: NonZero<usize>,
}

/// Computes the recall for a query from a golden file.
// TODO: add an option for NDGC recall computation.
pub struct RecallComputer {
    k: usize,
    neighbors: DerefVectorStore<u8, Mmap>,
}

impl RecallComputer {
    const NEIGHBOR_LEN: usize = 16;

    pub fn from_args(args: RecallArgs) -> io::Result<Option<Self>> {
        if let Some((neighbors, k)) = args.neighbors.zip(args.recall_k) {
            Ok(Some(RecallComputer::new(
                k,
                &neighbors,
                args.neighbors_len,
            )?))
        } else {
            Ok(None)
        }
    }

    /// Create a new RecallComputer that examines the first `k` results of `neighbors`.
    pub fn new(
        k: NonZero<usize>,
        neighbors: &Path,
        neighbors_len: NonZero<usize>,
    ) -> io::Result<Self> {
        let elem_stride = Self::NEIGHBOR_LEN * neighbors_len.get();
        let neighbors: DerefVectorStore<u8, Mmap> = DerefVectorStore::<u8, _>::new(
            unsafe { Mmap::map(&File::open(neighbors)?)? },
            NonZero::new(elem_stride).unwrap(),
        )?;

        if k.get() <= neighbors_len.get() {
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

    pub fn neighbors_len(&self) -> usize {
        self.neighbors.len()
    }

    /// Compute the recall based on golden data for `query_index` given `query_results`.
    ///
    /// *Panics* if `query_index` is out of bounds in the golden file.
    pub fn compute_recall(&self, query_index: usize, query_results: &[Neighbor]) -> f64 {
        let expected = self
            .query_neighbors(query_index)
            .take(self.k)
            .map(|n| n.vertex())
            .collect::<HashSet<_>>();
        let count = query_results
            .iter()
            .take(self.k)
            .filter(|n| expected.contains(&n.vertex()))
            .count();
        count as f64 / self.k as f64
    }

    fn query_neighbors(&self, query_index: usize) -> impl Iterator<Item = Neighbor> + use<'_> {
        self.neighbors[query_index]
            .as_chunks::<{ Self::NEIGHBOR_LEN }>()
            .0
            .iter()
            .map(|n| Neighbor::from(*n))
    }
}
