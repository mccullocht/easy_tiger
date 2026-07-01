//! Format for a block of vector postings.
//!
//! Each posting consists of a 64-bit identifier and a fixed byte size vector.
//!
//! Within each block the identifier and vector are fixed size, so the number of entries in the
//! block is inferred from the byte size of the block. Identifiers are stored before the rest of
//! the vector to allow mutation without having to visit every byte in the block.

use std::collections::BTreeMap;

use vectors::F32VectorCoder;

/// Internal model for handling pre-encoded blocks.
///
/// This is created with a fixed raw block length and vector length, validates settings, and
/// provides routines to examine the raw block data without maintaing a slice reference.
#[derive(Debug, Copy, Clone)]
struct PostingBlockMeta {
    raw_len: usize,
    len: usize,
    id_split: usize,
    vector_len: usize,
}

impl PostingBlockMeta {
    /// Create a new meta with a raw block length and a fixed vector length.
    ///
    /// Returns `None` if `raw_len` doesn't divide evenly into entries.
    fn new(raw_len: usize, vector_len: usize) -> Option<Self> {
        let entry_len = vector_len + std::mem::size_of::<i64>();
        if raw_len.is_multiple_of(entry_len) {
            let len = raw_len / entry_len;
            let id_split = len * std::mem::size_of::<i64>();
            Some(Self {
                raw_len,
                len,
                id_split,
                vector_len,
            })
        } else {
            None
        }
    }

    /// Presents ids as a slice of 8 byte entries.
    ///
    /// Each entry is a little-endian encoded `i64`. Entries are not guaranteed to be aligned.
    ///
    /// *Panics* if `raw_block` is not the same size as `raw_len` passed at construction.
    #[inline]
    fn raw_ids<'b>(&self, raw_block: &'b [u8]) -> &'b [[u8; 8]] {
        assert_eq!(self.raw_len, raw_block.len());
        &raw_block.as_chunks::<{ std::mem::size_of::<i64>() }>().0[..self.len]
    }

    /// Iterator over ids in `raw_block`
    ///
    /// *Panics* if `raw_block` is not the same size as `raw_len` passed at construction.
    #[inline]
    fn id_iter<'b>(&self, raw_block: &'b [u8]) -> impl ExactSizeIterator<Item = i64> + 'b {
        self.raw_ids(raw_block)
            .iter()
            .copied()
            .map(i64::from_le_bytes)
    }

    /// Iterator over vectors in `raw_block`
    ///
    /// *Panics* if `raw_block` is not the same size as `raw_len` passed at construction.
    #[inline]
    fn vector_iter<'b>(&self, raw_block: &'b [u8]) -> impl ExactSizeIterator<Item = &'b [u8]> {
        assert_eq!(self.raw_len, raw_block.len());
        raw_block[self.id_split..].chunks(self.vector_len)
    }

    /// Iterator over (id, vector) in `raw_block`.
    ///
    /// *Panics* if `raw_block` is not the same size as `raw_len` passed at construction.
    #[inline]
    fn block_iter<'b>(
        &self,
        raw_block: &'b [u8],
    ) -> impl ExactSizeIterator<Item = (i64, &'b [u8])> {
        self.id_iter(raw_block).zip(self.vector_iter(raw_block))
    }

    /// Lookup `id` in `raw_block`, returning the vector if present.
    #[inline]
    fn lookup<'b>(&self, raw_block: &'b [u8], id: i64) -> Option<&'b [u8]> {
        let i = self
            .raw_ids(raw_block)
            .binary_search_by_key(&id, |r| i64::from_le_bytes(*r))
            .ok()?;
        Some(self.vector_index(raw_block, i))
    }

    /// Read the vector in `raw_block` at `index`.
    ///
    /// *Panics* if `raw_block` is not the same size as `raw_len` passed at construction or if
    /// `i >= len`.
    fn vector_index<'b>(&self, raw_block: &'b [u8], i: usize) -> &'b [u8] {
        assert_eq!(self.raw_len, raw_block.len());
        let offset = self.id_split + i * self.vector_len;
        &raw_block[offset..(offset + self.vector_len)]
    }

    fn encode_block(
        vector_len: usize,
        it: impl ExactSizeIterator<Item = (i64, impl AsRef<[u8]>)>,
    ) -> Vec<u8> {
        let id_split = it.len() * std::mem::size_of::<i64>();
        let mut out = vec![0u8; id_split + vector_len * it.len()];
        let (id_out, vector_out) = out.split_at_mut(id_split);
        let oit = id_out
            .as_chunks_mut::<{ std::mem::size_of::<i64>() }>()
            .0
            .iter_mut()
            .zip(vector_out.chunks_mut(vector_len));
        for (i, o) in it.zip(oit) {
            *o.0 = i.0.to_le_bytes();
            assert_eq!(i.1.as_ref().len(), vector_len);
            o.1.copy_from_slice(i.1.as_ref());
        }
        out
    }
}

/// A view over a block of vector posting data: (id, vector) tuples.
///
/// Vectors are expected to be fixed size but the format of the data is otherwise undefined.
#[derive(Debug, Clone)]
pub struct PostingBlock<'a> {
    data: &'a [u8],
    meta: PostingBlockMeta,
}

impl<'a> PostingBlock<'a> {
    /// Create a new posting block from some input where each vector fills `vector_len` bytes.
    ///
    /// Returns `None` if the input data length does not align with an integer entry count.
    pub fn new(data: &'a [u8], vector_len: usize) -> Option<Self> {
        let meta = PostingBlockMeta::new(data.len(), vector_len)?;
        Some(Self { data, meta })
    }

    /// Return an iterator over the vectors in the posting block and their ids.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (i64, &'a [u8])> {
        self.meta.block_iter(self.data)
    }

    /// Lookup a vector by `id`, returning `None` if `id` is not present in the block.
    pub fn lookup(&self, id: i64) -> Option<&[u8]> {
        self.meta.lookup(self.data, id)
    }

    /// Return the number of entries in the block.
    pub fn len(&self) -> usize {
        self.meta.len
    }

    pub fn is_empty(&self) -> bool {
        self.meta.len == 0
    }
}

enum EntrySource {
    /// Index into the base block's entry array.
    Base(usize),
    /// Owned encoded vector bytes for a new or replaced entry.
    New(Vec<u8>),
}

/// A mutable posting block that can be seeded from an existing block and mutated via inserts and
/// removals. The original data is preserved so callers can produce a diff (e.g. via
/// `wiredtiger_calc_modify`) against the serialized result.
pub struct PostingBlockMut {
    /// Base block we are working off of. May be empty.
    base_data: Vec<u8>,
    /// Metadata for base block.
    base_meta: PostingBlockMeta,
    /// Current entries sorted by id. Start pre-populated from base block.
    entries: BTreeMap<i64, EntrySource>,
}

impl PostingBlockMut {
    /// Create a new mutable block where each vector is expected be `vector_len` bytes.
    pub fn new(vector_len: usize) -> Self {
        Self {
            base_data: vec![],
            base_meta: PostingBlockMeta::new(0, vector_len).expect("0 divides evenly"),
            entries: BTreeMap::new(),
        }
    }

    /// Create a mutable block pre-populated with data from `block`.
    pub fn from_block(block: &PostingBlock<'_>) -> Self {
        let entries = block
            .meta
            .id_iter(block.data)
            .enumerate()
            .map(|(i, id)| (id, EntrySource::Base(i)))
            .collect();
        Self {
            base_data: block.data.to_vec(),
            base_meta: block.meta,
            entries,
        }
    }

    /// Return an iterator over the current postings in the block.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (i64, &[u8])> {
        self.entries.iter().map(|(&id, e)| (id, self.get_vector(e)))
    }

    /// Lookup the vector for the given `id`, returning `None` if not found.
    pub fn lookup(&self, id: i64) -> Option<&[u8]> {
        let e = self.entries.get(&id)?;
        Some(self.get_vector(e))
    }

    /// Insert or replace the entry for `id`.
    ///
    /// *Panics* if `vector.into().len()` is not the vector length passed at construction.
    pub fn insert(&mut self, id: i64, vector: impl Into<Vec<u8>>) {
        let vector = vector.into();
        assert_eq!(vector.len(), self.base_meta.vector_len);
        self.entries.insert(id, EntrySource::New(vector));
    }

    /// Remove the entry for `id`, returning `true` if it was present.
    pub fn remove(&mut self, id: i64) -> bool {
        self.entries.remove(&id).is_some()
    }

    /// The original block data passed to [`PostingBlockMut::from_block`], or an empty slice for
    /// blocks created with [`PostingBlockMut::new`].
    ///
    /// This information can be used to create deltas for binary blocks.
    pub fn base_block(&self) -> &[u8] {
        &self.base_data
    }

    /// Serialize the current state of the block into the same layout as [`PostingBlock`].
    pub fn serialize(&self) -> Vec<u8> {
        PostingBlockMeta::encode_block(self.base_meta.vector_len, self.iter())
    }

    /// Return the number of entries in the block.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn get_vector<'a>(&'a self, entry: &'a EntrySource) -> &'a [u8] {
        match entry {
            EntrySource::Base(i) => self.base_meta.vector_index(&self.base_data, *i),
            EntrySource::New(v) => v.as_slice(),
        }
    }
}

/// Encode a series of input `(id, float vector)` tuples where each vector is of `dim` length using
/// `coder` and pack them into a posting block.
pub fn encode_f32(
    vectors: impl ExactSizeIterator<Item = (i64, impl AsRef<[f32]>)>,
    coder: &dyn F32VectorCoder,
    dim: usize,
) -> Vec<u8> {
    let vector_len = coder.byte_len(dim);
    PostingBlockMeta::encode_block(
        vector_len,
        vectors.map(|(id, v)| (id, coder.encode(v.as_ref()))),
    )
}

/// Return the expected `leaf_page_max` and `leaf_value_max` sizes that should be used in
/// WiredTiger given a maximum block size, the length of each vector, and WT's allocation size.
pub fn leaf_page_max(block_size: usize, vector_len: usize, allocation_size: usize) -> usize {
    (block_size * (vector_len + std::mem::size_of::<i64>())).next_multiple_of(allocation_size)
}
