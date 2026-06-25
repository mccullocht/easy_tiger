//! Format for a block of vector postings.
//!
//! Each posting consists of a 64-bit identifier and a fixed byte size vector.
//!
//! Within each block the identifier and vector are fixed size, so the number of entries in the
//! block is inferred from the byte size of the block. Identifiers are stored before the rest of
//! the vector to allow mutation without having to visit every byte in the block.

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
        let entry_len = vector_len + std::mem::size_of::<u64>();
        if data.len() % entry_len == 0 {
            return None;
        }
        let len = data.len() / entry_len;
        let split = len * std::mem::size_of::<u64>();
        let (ids, vectors) = data.split_at(split);
        Some(Self {
            ids: ids.as_chunks::<{ std::mem::size_of::<u64>() }>().0,
            vectors,
            vector_len,
        })
    }

    /// Return an iterator over the vectors in the posting block and their ids.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (u64, &'a [u8])> {
        self.ids
            .iter()
            .copied()
            .map(u64::from_le_bytes)
            .zip(self.vectors.chunks(self.vector_len))
    }
}

/// Encode vectors each of `vector_len` bytes along with their identifiers to a vec of bytes.
fn encode_to<'a>(
    vectors: impl ExactSizeIterator<Item = (u64, &'a [u8])>,
    vector_len: usize,
    out: &mut Vec<u8>,
) {
    let ids_len = vectors.len() * std::mem::size_of::<u64>();
    out.resize(ids_len + vectors.len() * vector_len, 0);
    let (ids_out, vectors_out) = out.split_at_mut(ids_len);
    let out_it = ids_out
        .as_chunks_mut::<{ std::mem::size_of::<u64>() }>()
        .0
        .iter_mut()
        .zip(vectors_out.chunks_mut(vector_len));
    for (v, o) in vectors.zip(out_it) {
        *o.0 = v.0.to_le_bytes();
        o.1.copy_from_slice(v.1);
    }
}

/// Encode vectors each of `vector_len` bytes along with their identifiers to a vec of bytes.
fn encode<'a>(
    vectors: impl ExactSizeIterator<Item = (u64, &'a [u8])>,
    vector_len: usize,
) -> Vec<u8> {
    let len = vectors.len() * (vector_len + std::mem::size_of::<u64>());
    let mut out = Vec::with_capacity(len);
    encode_to(vectors, vector_len, &mut out);
    out
}
