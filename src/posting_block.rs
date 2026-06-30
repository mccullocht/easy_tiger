//! Format for a block of vector postings.
//!
//! Each posting consists of a 64-bit identifier and a fixed byte size vector.
//!
//! Within each block the identifier and vector are fixed size, so the number of entries in the
//! block is inferred from the byte size of the block. Identifiers are stored before the rest of
//! the vector to allow mutation without having to visit every byte in the block.

use vectors::F32VectorCoder;

#[derive(Debug, Clone)]
pub struct PostingBlock<'a> {
    ids: &'a [[u8; 8]],
    vectors: &'a [u8],
    vector_len: usize,
}

impl<'a> PostingBlock<'a> {
    /// Create a new posting block from some input where each vector fills `vector_len` bytes.
    ///
    /// Returns `None` if the input data length does not align with an integer entry count.
    pub fn new(data: &'a [u8], vector_len: usize) -> Option<Self> {
        let entry_len = vector_len + std::mem::size_of::<i64>();
        if !data.len().is_multiple_of(entry_len) {
            return None;
        }
        let len = data.len() / entry_len;
        let split = len * std::mem::size_of::<i64>();
        let (ids, vectors) = data.split_at(split);
        Some(Self {
            ids: ids.as_chunks::<{ std::mem::size_of::<i64>() }>().0,
            vectors,
            vector_len,
        })
    }

    /// Return an iterator over the vectors in the posting block and their ids.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (i64, &'a [u8])> {
        self.ids
            .iter()
            .copied()
            .map(i64::from_le_bytes)
            .zip(self.vectors.chunks(self.vector_len))
    }
}

/// Encode a series of input `(id, float vector)` tuples where each vector is of `dim` length using
/// `coder` and pack them into a posting block.
pub fn encode_f32(
    vectors: impl ExactSizeIterator<Item = (i64, impl AsRef<[f32]>)>,
    coder: &dyn F32VectorCoder,
    dim: usize,
) -> Vec<u8> {
    let ids_len = std::mem::size_of::<i64>() * vectors.len();
    let vec_len = coder.byte_len(dim);
    let mut out = vec![0u8; ids_len + vectors.len() * vec_len];
    let (out_ids, out_vecs) = out.split_at_mut(ids_len);
    let out_it = out_ids
        .as_chunks_mut::<{ std::mem::size_of::<i64>() }>()
        .0
        .iter_mut()
        .zip(out_vecs.chunks_mut(vec_len));
    for (i, o) in vectors.zip(out_it) {
        *o.0 = i.0.to_le_bytes();
        coder.encode_to(i.1.as_ref(), o.1);
    }
    out
}

/// Return the expected `leaf_page_max` and `leaf_value_max` sizes that should be used in
/// WiredTiger given a maximum block size, the length of each vector, and WT's allocation size.
pub fn leaf_page_max(block_size: usize, vector_len: usize, allocation_size: usize) -> usize {
    (block_size * (vector_len + std::mem::size_of::<i64>())).next_multiple_of(allocation_size)
}
