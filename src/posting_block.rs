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

// XXX this proposal cannot distinguish between an update and a clean entry.
#[derive(Debug, Clone)]
struct BuilderEntry {
    id: i64,
    // usize::MAX for an insert of a new entry.
    base_index: usize,
    // Empty to indicate a delete.
    vector: Vec<u8>,
    // True if this entry has been modified.
    dirty: bool,
}

impl BuilderEntry {
    fn vector_tuple(&self) -> Option<(i64, &[u8])> {
        if !self.vector.is_empty() {
            Some((self.id, self.vector.as_slice()))
        } else {
            None
        }
    }
}

/// Builder for a block of vector posting data.
///
/// Vectors are expected to be fixed size but the format of the data is otherwise undefined.
#[derive(Debug, Clone, Default)]
pub struct PostingBlockBuilder(Vec<BuilderEntry>);

// XXX this all needs docs.
impl PostingBlockBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn iter(&self) -> impl Iterator<Item = (i64, &[u8])> {
        self.0.iter().filter_map(BuilderEntry::vector_tuple)
    }

    pub fn lookup(&self, id: i64) -> Option<&[u8]> {
        self.0
            .binary_search_by_key(&id, |e| e.id)
            .map(|i| self.0[i].vector.as_slice())
            .ok()
            .filter(|v| !v.is_empty())
    }

    pub fn upsert(&mut self, id: i64, vector: impl Into<Vec<u8>>) {
        match self.0.binary_search_by_key(&id, |e| e.id) {
            Ok(i) => {
                let e = &mut self.0[i];
                e.vector = vector.into();
                e.dirty = true;
            }
            Err(i) => {
                self.0.insert(
                    i,
                    BuilderEntry {
                        id,
                        base_index: usize::MAX,
                        vector: vector.into(),
                        dirty: true,
                    },
                );
            }
        }
    }

    pub fn delete(&mut self, id: i64) {
        if let Ok(i) = self.0.binary_search_by_key(&id, |e| e.id) {
            let e = &mut self.0[i];
            e.vector.clear();
            // XXX I should remove the value entirely if base_index is unset.
            // it will make it easier to fix up later.
            e.dirty = e.base_index != usize::MAX;
        }
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn build(self) -> EncodedPostingBlock {
        if self.is_empty() {
            return EncodedPostingBlock::Remove;
        }

        let vector_len = self.0[0].vector.len();
        let mut out = vec![0u8; self.len() * (std::mem::size_of::<i64>() + vector_len)];
        let (ids_out, vectors_out) = out.split_at_mut(self.len() * std::mem::size_of::<i64>());
        let out_it = ids_out
            .as_chunks_mut::<{ std::mem::size_of::<i64>() }>()
            .0
            .iter_mut()
            .zip(vectors_out.chunks_mut(vector_len));
        for (i, o) in self.iter().zip(out_it) {
            *o.0 = i.0.to_le_bytes();
            o.1.copy_from_slice(&i.1);
        }
        EncodedPostingBlock::Replace(out)
    }
}

impl<B: Into<Vec<u8>>> FromIterator<(i64, B)> for PostingBlockBuilder {
    fn from_iter<T: IntoIterator<Item = (i64, B)>>(iter: T) -> Self {
        Self(
            iter.into_iter()
                .enumerate()
                .map(|(i, (id, v))| BuilderEntry {
                    id: id,
                    base_index: i,
                    vector: v.into(),
                    dirty: false,
                })
                .collect(),
        )
    }
}

impl From<PostingBlock<'_>> for PostingBlockBuilder {
    fn from(value: PostingBlock<'_>) -> Self {
        Self::from_iter(value.iter())
    }
}

/// An encoded posting block produced by [`PostingBlockBuilder`].
///
/// This may represent a remove, set/overwrite, or a delta/modify call.
#[derive(Debug, Clone)]
pub enum EncodedPostingBlock {
    /// This block no longer contains any postings so remove it.
    Remove,
    /// Replace the contents of this block with a different value.
    Replace(Vec<u8>),
}

enum PostingDelta {
    Upsert(usize, [u8; 8], Vec<u8>),
    Delete(usize),
}

pub struct PostingBlockDelta(Vec<PostingDelta>);

impl PostingBlockDelta {
    fn from_builder(builder: PostingBlockBuilder) -> Self {
        // XXX the indexes are wrong here.
        // i need to track the base index and the output index.
        // i need to distinguish between insert and update
        // * inserts only need to know the next output index.
        // * updates and deletes need to track where base_index is in the output
        // XXX is it easier if I iterate over every entry?
        // delete => delete at current index, do not increment index.
        // insert => insert at current index, increment index.
        // update => replace at current index, increment index.
        PostingBlockDelta(
            builder
                .0
                .into_iter()
                .filter(|e| e.dirty)
                .map(|e| {
                    if e.vector.is_empty() {
                        assert_ne!(e.base_index, usize::MAX);
                        PostingDelta::Delete(e.base_index)
                    } else {
                        PostingDelta::Upsert(e.base_index, e.id.to_le_bytes(), e.vector)
                    }
                })
                .collect(),
        )
    }

    pub fn value_deltas(&self) -> Vec<wt_mdb::session::ValueDelta<'_>> {
        todo!()
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
