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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build raw posting block bytes from sorted `(id, vector)` pairs with the same layout used
    /// by `PostingBlockMeta::encode_block`: all ids first, then all vectors.
    fn make_block_data(entries: &[(i64, &[u8])]) -> Vec<u8> {
        let n = entries.len();
        let vector_len = entries.first().map_or(0, |(_, v)| v.len());
        let id_split = n * 8;
        let mut out = vec![0u8; id_split + vector_len * n];
        let (id_out, vector_out) = out.split_at_mut(id_split);
        for (i, (id, vec)) in entries.iter().enumerate() {
            id_out[i * 8..(i + 1) * 8].copy_from_slice(&id.to_le_bytes());
            vector_out[i * vector_len..(i + 1) * vector_len].copy_from_slice(vec);
        }
        out
    }

    // --- PostingBlock ---

    #[test]
    fn posting_block_empty() {
        let block = PostingBlock::new(&[], 4).unwrap();
        assert_eq!(block.len(), 0);
        assert!(block.is_empty());
        assert_eq!(block.iter().count(), 0);
        assert_eq!(block.lookup(1), None);
    }

    #[test]
    fn posting_block_invalid_alignment() {
        // entry_len = 4 + 8 = 12; 5 is not a multiple of 12
        assert!(PostingBlock::new(&[0u8; 5], 4).is_none());
        assert!(PostingBlock::new(&[0u8; 11], 4).is_none());
        assert!(PostingBlock::new(&[0u8; 13], 4).is_none());
    }

    #[test]
    fn posting_block_single_entry() {
        let data = make_block_data(&[(42, &[1, 2, 3, 4])]);
        let block = PostingBlock::new(&data, 4).unwrap();
        assert_eq!(block.len(), 1);
        assert!(!block.is_empty());
        let entries: Vec<_> = block.iter().collect();
        assert_eq!(entries, [(42, &[1u8, 2, 3, 4][..])]);
    }

    #[test]
    fn posting_block_multiple_entries() {
        let data = make_block_data(&[
            (1, &[10, 11, 12, 13]),
            (5, &[20, 21, 22, 23]),
            (10, &[30, 31, 32, 33]),
        ]);
        let block = PostingBlock::new(&data, 4).unwrap();
        assert_eq!(block.len(), 3);
        let entries: Vec<_> = block.iter().collect();
        assert_eq!(entries[0], (1, &[10u8, 11, 12, 13][..]));
        assert_eq!(entries[1], (5, &[20u8, 21, 22, 23][..]));
        assert_eq!(entries[2], (10, &[30u8, 31, 32, 33][..]));
    }

    #[test]
    fn posting_block_lookup_found() {
        let data = make_block_data(&[(1, &[10, 11, 12, 13]), (5, &[20, 21, 22, 23])]);
        let block = PostingBlock::new(&data, 4).unwrap();
        assert_eq!(block.lookup(1), Some(&[10u8, 11, 12, 13][..]));
        assert_eq!(block.lookup(5), Some(&[20u8, 21, 22, 23][..]));
    }

    #[test]
    fn posting_block_lookup_not_found() {
        let data = make_block_data(&[(1, &[10, 11, 12, 13])]);
        let block = PostingBlock::new(&data, 4).unwrap();
        assert_eq!(block.lookup(0), None);
        assert_eq!(block.lookup(2), None);
    }

    #[test]
    fn posting_block_negative_ids() {
        let data = make_block_data(&[(-10, &[1, 2, 3, 4]), (-1, &[5, 6, 7, 8])]);
        let block = PostingBlock::new(&data, 4).unwrap();
        assert_eq!(block.lookup(-10), Some(&[1u8, 2, 3, 4][..]));
        assert_eq!(block.lookup(-1), Some(&[5u8, 6, 7, 8][..]));
        assert_eq!(block.lookup(0), None);
    }

    // --- PostingBlockMut ---

    #[test]
    fn posting_block_mut_new_empty() {
        let mb = PostingBlockMut::new(4);
        assert_eq!(mb.len(), 0);
        assert!(mb.is_empty());
        assert_eq!(mb.iter().count(), 0);
        assert_eq!(mb.base_block(), &[] as &[u8]);
    }

    #[test]
    fn posting_block_mut_insert_single() {
        let mut mb = PostingBlockMut::new(4);
        mb.insert(42, vec![1, 2, 3, 4]);
        assert_eq!(mb.len(), 1);
        assert!(!mb.is_empty());
        assert_eq!(mb.lookup(42), Some(&[1u8, 2, 3, 4][..]));
        assert_eq!(mb.lookup(0), None);
    }

    #[test]
    fn posting_block_mut_insert_sorted_by_id() {
        let mut mb = PostingBlockMut::new(4);
        mb.insert(5, vec![5, 5, 5, 5]);
        mb.insert(1, vec![1, 1, 1, 1]);
        mb.insert(3, vec![3, 3, 3, 3]);
        let ids: Vec<i64> = mb.iter().map(|(id, _)| id).collect();
        assert_eq!(ids, [1, 3, 5]);
    }

    #[test]
    fn posting_block_mut_insert_replaces_existing() {
        let mut mb = PostingBlockMut::new(4);
        mb.insert(1, vec![1, 1, 1, 1]);
        mb.insert(1, vec![9, 9, 9, 9]);
        assert_eq!(mb.len(), 1);
        assert_eq!(mb.lookup(1), Some(&[9u8, 9, 9, 9][..]));
    }

    #[test]
    fn posting_block_mut_remove_existing() {
        let mut mb = PostingBlockMut::new(4);
        mb.insert(1, vec![1, 2, 3, 4]);
        assert!(mb.remove(1));
        assert_eq!(mb.len(), 0);
        assert_eq!(mb.lookup(1), None);
    }

    #[test]
    fn posting_block_mut_remove_missing_returns_false() {
        let mut mb = PostingBlockMut::new(4);
        mb.insert(1, vec![1, 2, 3, 4]);
        assert!(!mb.remove(2));
        assert_eq!(mb.len(), 1);
    }

    #[test]
    fn posting_block_mut_from_block_inherits_entries() {
        let data = make_block_data(&[(1, &[10, 11, 12, 13]), (5, &[20, 21, 22, 23])]);
        let block = PostingBlock::new(&data, 4).unwrap();
        let mb = PostingBlockMut::from_block(&block);
        assert_eq!(mb.len(), 2);
        assert_eq!(mb.lookup(1), Some(&[10u8, 11, 12, 13][..]));
        assert_eq!(mb.lookup(5), Some(&[20u8, 21, 22, 23][..]));
        assert_eq!(mb.base_block(), &data[..]);
    }

    #[test]
    fn posting_block_mut_from_block_then_insert() {
        let data = make_block_data(&[(1, &[10, 11, 12, 13])]);
        let block = PostingBlock::new(&data, 4).unwrap();
        let mut mb = PostingBlockMut::from_block(&block);
        mb.insert(3, vec![30, 31, 32, 33]);
        assert_eq!(mb.len(), 2);
        let entries: Vec<_> = mb.iter().collect();
        assert_eq!(entries[0], (1, &[10u8, 11, 12, 13][..]));
        assert_eq!(entries[1], (3, &[30u8, 31, 32, 33][..]));
    }

    #[test]
    fn posting_block_mut_from_block_then_remove() {
        let data = make_block_data(&[(1, &[10, 11, 12, 13]), (5, &[20, 21, 22, 23])]);
        let block = PostingBlock::new(&data, 4).unwrap();
        let mut mb = PostingBlockMut::from_block(&block);
        assert!(mb.remove(1));
        assert_eq!(mb.len(), 1);
        assert_eq!(mb.lookup(1), None);
        assert_eq!(mb.lookup(5), Some(&[20u8, 21, 22, 23][..]));
    }

    #[test]
    fn posting_block_mut_from_block_replace_base_entry() {
        let data = make_block_data(&[(1, &[10, 11, 12, 13]), (5, &[20, 21, 22, 23])]);
        let block = PostingBlock::new(&data, 4).unwrap();
        let mut mb = PostingBlockMut::from_block(&block);
        mb.insert(1, vec![99, 99, 99, 99]);
        assert_eq!(mb.len(), 2);
        assert_eq!(mb.lookup(1), Some(&[99u8, 99, 99, 99][..]));
        assert_eq!(mb.lookup(5), Some(&[20u8, 21, 22, 23][..]));
    }

    #[test]
    fn posting_block_mut_serialize_empty() {
        let mb = PostingBlockMut::new(4);
        let serialized = mb.serialize();
        assert!(serialized.is_empty());
        assert!(PostingBlock::new(&serialized, 4).unwrap().is_empty());
    }

    #[test]
    fn posting_block_mut_serialize_roundtrip() {
        let mut mb = PostingBlockMut::new(4);
        mb.insert(5, vec![5, 6, 7, 8]);
        mb.insert(1, vec![1, 2, 3, 4]);
        mb.insert(3, vec![3, 4, 5, 6]);
        let serialized = mb.serialize();
        let block = PostingBlock::new(&serialized, 4).unwrap();
        assert_eq!(block.len(), 3);
        let entries: Vec<_> = block.iter().collect();
        assert_eq!(entries[0], (1, &[1u8, 2, 3, 4][..]));
        assert_eq!(entries[1], (3, &[3u8, 4, 5, 6][..]));
        assert_eq!(entries[2], (5, &[5u8, 6, 7, 8][..]));
    }

    #[test]
    fn posting_block_mut_serialize_matches_base_when_unmodified() {
        let data = make_block_data(&[(1, &[10, 11, 12, 13]), (5, &[20, 21, 22, 23])]);
        let block = PostingBlock::new(&data, 4).unwrap();
        let mb = PostingBlockMut::from_block(&block);
        assert_eq!(mb.serialize(), data);
    }

    #[test]
    #[should_panic]
    fn posting_block_mut_insert_wrong_vector_len_panics() {
        let mut mb = PostingBlockMut::new(4);
        mb.insert(1, vec![1, 2, 3]); // one byte short
    }

    // --- encode_f32 ---

    fn decode_f32_vec(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn encode_f32_empty_produces_empty_bytes() {
        let coder = vectors::F32VectorCoding::F32.coder(vectors::VectorSimilarity::Euclidean, None);
        let input: Vec<(i64, Vec<f32>)> = vec![];
        let result = encode_f32(input.into_iter(), coder.as_ref(), 4);
        assert!(result.is_empty());
    }

    #[test]
    fn encode_f32_single_entry_roundtrip() {
        let coder = vectors::F32VectorCoding::F32.coder(vectors::VectorSimilarity::Euclidean, None);
        let input = vec![(1i64, vec![1.0f32, 2.0, 3.0, 4.0])];
        let result = encode_f32(input.into_iter(), coder.as_ref(), 4);
        let block = PostingBlock::new(&result, coder.byte_len(4)).unwrap();
        assert_eq!(block.len(), 1);
        let (id, vec_bytes) = block.iter().next().unwrap();
        assert_eq!(id, 1);
        assert_eq!(decode_f32_vec(vec_bytes), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn encode_f32_multiple_entries_roundtrip() {
        let coder = vectors::F32VectorCoding::F32.coder(vectors::VectorSimilarity::Euclidean, None);
        let input = vec![
            (1i64, vec![1.0f32, 0.0, 0.0, 0.0]),
            (5i64, vec![0.0, 2.0, 0.0, 0.0]),
            (10i64, vec![0.0, 0.0, 3.0, 0.0]),
        ];
        let result = encode_f32(input.into_iter(), coder.as_ref(), 4);
        let block = PostingBlock::new(&result, coder.byte_len(4)).unwrap();
        assert_eq!(block.len(), 3);
        let entries: Vec<_> = block.iter().collect();
        assert_eq!(entries[0].0, 1);
        assert_eq!(decode_f32_vec(entries[0].1), [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(entries[1].0, 5);
        assert_eq!(decode_f32_vec(entries[1].1), [0.0, 2.0, 0.0, 0.0]);
        assert_eq!(entries[2].0, 10);
        assert_eq!(decode_f32_vec(entries[2].1), [0.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn encode_f32_output_length_matches_entry_count() {
        let coder = vectors::F32VectorCoding::F32.coder(vectors::VectorSimilarity::Euclidean, None);
        let n = 5usize;
        let input: Vec<(i64, Vec<f32>)> = (0..n as i64).map(|i| (i, vec![i as f32; 4])).collect();
        let result = encode_f32(input.into_iter(), coder.as_ref(), 4);
        assert_eq!(result.len(), n * (coder.byte_len(4) + std::mem::size_of::<i64>()));
    }

    #[test]
    fn encode_f32_cosine_normalizes_stored_vector() {
        let coder = vectors::F32VectorCoding::F32.coder(vectors::VectorSimilarity::Cosine, None);
        // l2 norm = 5.0 → after normalization: [0.6, 0.8, 0.0, 0.0]
        let input = vec![(1i64, vec![3.0f32, 4.0, 0.0, 0.0])];
        let result = encode_f32(input.into_iter(), coder.as_ref(), 4);
        let block = PostingBlock::new(&result, coder.byte_len(4)).unwrap();
        let (_, vec_bytes) = block.iter().next().unwrap();
        let decoded = decode_f32_vec(vec_bytes);
        let norm_sq: f32 = decoded.iter().map(|x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-6, "expected unit norm, got norm_sq={norm_sq}");
        assert!((decoded[0] - 0.6).abs() < 1e-6, "decoded[0]={}", decoded[0]);
        assert!((decoded[1] - 0.8).abs() < 1e-6, "decoded[1]={}", decoded[1]);
    }

    #[test]
    fn encode_f32_lookup_on_result() {
        let coder = vectors::F32VectorCoding::F32.coder(vectors::VectorSimilarity::Euclidean, None);
        let input = vec![
            (1i64, vec![1.0f32, 0.0, 0.0, 0.0]),
            (3i64, vec![0.0, 1.0, 0.0, 0.0]),
            (7i64, vec![0.0, 0.0, 1.0, 0.0]),
        ];
        let result = encode_f32(input.into_iter(), coder.as_ref(), 4);
        let block = PostingBlock::new(&result, coder.byte_len(4)).unwrap();
        assert!(block.lookup(1).is_some());
        assert!(block.lookup(3).is_some());
        assert!(block.lookup(7).is_some());
        assert!(block.lookup(2).is_none());
        assert_eq!(decode_f32_vec(block.lookup(3).unwrap()), [0.0, 1.0, 0.0, 0.0]);
    }

    // --- leaf_page_max ---

    #[test]
    fn leaf_page_max_rounds_up_to_allocation_size() {
        // entry_len = 4 + 8 = 12; 1 * 12 = 12 → rounds up to 512
        assert_eq!(leaf_page_max(1, 4, 512), 512);
        // 100 * 12 = 1200 → rounds up to 1536 (3 * 512)
        assert_eq!(leaf_page_max(100, 4, 512), 1536);
        // Already a multiple: 128 * 8 = 1024 → stays 1024
        assert_eq!(leaf_page_max(128, 0, 1024), 1024);
    }
}
