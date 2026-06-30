//! Format for a block of vector postings.
//!
//! Each posting consists of a 64-bit identifier and a fixed byte size vector.
//!
//! Within each block the identifier and vector are fixed size, so the number of entries in the
//! block is inferred from the byte size of the block. Identifiers are stored before the rest of
//! the vector to allow mutation without having to visit every byte in the block.

use vectors::F32VectorCoder;

/// A view over a block of vector posting data: (id, vector) tuples.
///
/// Vectors are expected to be fixed size but the format of the data is otherwise undefined.
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

    /// Lookup a vector by `id`, returning `None` if `id` is not present in the block.
    pub fn lookup(&self, id: i64) -> Option<&[u8]> {
        self.ids
            .binary_search(&id.to_le_bytes())
            .map(|i| {
                let start = self.vector_len * i;
                let end = start + self.vector_len;
                &self.vectors[start..end]
            })
            .ok()
    }

    /// Return the number of entries in the block.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Builder for a block of vector posting data.
///
/// Vectors are expected to be fixed size but the format of the data is otherwise undefined.
#[derive(Debug, Clone, Default)]
pub struct PostingBlockBuilder {
    entries: Vec<(i64, Vec<u8>)>,
    initial_entries: usize,
    dirty: Vec<i64>,
}

// XXX this all needs docs.
impl PostingBlockBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    fn from_entries(entries: Vec<(i64, Vec<u8>)>) -> Self {
        Self {
            initial_entries: entries.len(),
            entries,
            dirty: vec![],
        }
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (i64, &[u8])> {
        self.entries.iter().map(|(i, v)| (*i, v.as_slice()))
    }

    pub fn lookup(&self, id: i64) -> Option<&[u8]> {
        self.entries
            .binary_search_by_key(&id, |(i, _)| *i)
            .map(|i| self.entries[i].1.as_slice())
            .ok()
    }

    pub fn upsert(&mut self, id: i64, vector: impl Into<Vec<u8>>) {
        match self.entries.binary_search_by_key(&id, |(i, _)| *i) {
            Ok(i) => self.entries[i].1 = vector.into(),
            Err(i) => self.entries.insert(i, (id, vector.into())),
        }
        self.mark_dirty(id);
    }

    pub fn delete(&mut self, id: i64) {
        if let Ok(i) = self.entries.binary_search_by_key(&id, |(i, _)| *i) {
            self.entries.remove(i);
            self.mark_dirty(id);
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // XXX tri-state encode:
    // * If it's empty, should result in the key being removed.
    // * If there are too many mutations (~10% of max(entries, dirty)) then flush completely and set
    // * Otherwise produce a diff.
    //
    // I think the state I have is sufficient to decide between 2 and 3.

    fn mark_dirty(&mut self, id: i64) {
        if let Err(i) = self.dirty.binary_search(&id) {
            self.dirty.insert(i, id);
        }
    }
}

impl<B: Into<Vec<u8>>> FromIterator<(i64, B)> for PostingBlockBuilder {
    fn from_iter<T: IntoIterator<Item = (i64, B)>>(iter: T) -> Self {
        let entries = iter.into_iter().map(|(id, v)| (id, v.into())).collect();
        Self::from_entries(entries)
    }
}

impl From<PostingBlock<'_>> for PostingBlockBuilder {
    fn from(value: PostingBlock<'_>) -> Self {
        Self::from_iter(value.iter())
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
