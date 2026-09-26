//! Access to absolute centroid vectors for the SPANN tail index.

use vectors::F32VectorCoder;
use wt_mdb::{Error, Result};

use crate::vamana::{
    GraphVectorIndex, GraphVectorStore,
    wt::{CursorVectorStore, TransactionGraphVectorIndex},
};

/// Reads absolute f32 centroid vectors from a SPANN head index's high-fidelity vector table.
///
/// All posting-residual computations (ingress centering, per-centroid query adjustment, and
/// rebalance re-centering) must agree on the definition of a centroid vector: the head
/// high-fidelity vector decoded to f32, with the head graph's shared centroid (if any)
/// re-added to produce an absolute vector.
pub struct CentroidVectorSource<'a> {
    store: CursorVectorStore<'a>,
    coder: Box<dyn F32VectorCoder>,
}

impl<'a> CentroidVectorSource<'a> {
    /// Create a source reading centroid vectors from `head`.
    pub fn new(head: &'a TransactionGraphVectorIndex) -> Result<Self> {
        let store = head.high_fidelity_vectors()?;
        let coder = store.new_coder();
        Ok(Self { store, coder })
    }

    /// Return the absolute f32 vector of `centroid_id`.
    ///
    /// The stored vector is decoded and, if the head graph stores vectors as residuals
    /// against a shared centroid, that centroid is re-added.
    pub fn centroid_vector(&mut self, centroid_id: u32) -> Result<Vec<f32>> {
        let encoded = self
            .store
            .get(centroid_id as i64)
            .unwrap_or_else(|| Err(Error::not_found_error()))?;
        let mut vector = self.coder.decode(encoded);
        if let Some(center) = self.store.centroid() {
            for (d, c) in vector.iter_mut().zip(center.iter()) {
                *d += c;
            }
        }
        Ok(vector)
    }
}
