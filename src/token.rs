//! Abstraction for token indexing.

use std::{collections::BTreeMap, io};

// XXX impl Error
#[derive(Debug)]
pub enum Error {
    UnknownDocFormat(u8),
    UnknownPayloadFormat(u8),
    Coding(&'static str),
    Io(io::Error),
}

pub const DOC_ITER_SENTINEL: i64 = i64::MAX;

pub trait DocIterator {
    /// Return the current doc, or [`DOC_ITER_SENTINEL`] if there are no more docs.
    fn doc(&mut self) -> i64;
    /// Return the payload value for the current doc, or `0` if `doc()` is the sentinel value.
    fn payload(&mut self) -> u64;

    /// Advance to the next document and returns its id.
    ///
    /// Returns [`DOC_ITER_SENTINEL`] if there are no more docs.
    /// It is legal to `advance()` after the sentinel value is returned; further calls should also
    /// return the sentinel value.
    fn advance(&mut self) -> i64;
    /// Advance to the next document at or after `target` and return its id.
    ///
    /// Returns [`DOC_ITER_SENTINEL`] if there are no more docs.
    /// It is legal to `advance_to()` after the sentinel value is returned; further calls should
    /// also return the sentinel value.
    fn advance_to(&mut self, target: i64) -> i64;

    /// Cost of this iterator, proportional to hit count and cost of access.
    fn cost(&self) -> u64;

    /// Any errors that occurred while decoding underlying data for this iterator.
    ///
    /// Call this when the sentinel value is returned.
    fn error(&self) -> Result<(), Error>;
}

/// Construct a key for a posting block given a field name, key, and the last possible docid that
/// can appear in the block.
///
/// It is recommended to use [`DOC_ITER_SENTINEL`] as the limit for short posting lists as it makes
/// them easy to find -- seeking to a target docid will always return that block.
pub fn doc_block_key(field: &str, posting_key: &[u8], limit_docid: i64) -> Vec<u8> {
    let mut key = Vec::with_capacity(field.len() + posting_key.len() + 10);
    key.extend_from_slice(field.as_bytes());
    key.push(b':');
    key.extend_from_slice(posting_key);
    key.push(b':');
    key.extend_from_slice(&limit_docid.to_be_bytes());
    key
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum DocFormat {
    LEB128 = 0,
}

impl From<DocFormat> for u8 {
    fn from(value: DocFormat) -> Self {
        value as u8
    }
}

impl TryFrom<u8> for DocFormat {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(DocFormat::LEB128),
            _ => Err(Error::UnknownDocFormat(value)),
        }
    }
}

impl DocFormat {
    /// Encode `docs` to `out`. `prev` is the doc immediately preceding the first entry in `docs`.
    ///
    /// `docs` must be strictly increasing and greater than `prev`.
    fn encode(self, mut prev: i64, docs: impl Iterator<Item = i64>, out: &mut Vec<u8>) {
        match self {
            DocFormat::LEB128 => {
                for doc in docs {
                    assert!(doc > prev, "docs must be strictly increasing");
                    leb128::write::unsigned(out, doc.abs_diff(prev)).unwrap();
                    prev = doc;
                }
            }
        }
    }

    /// Decode `count` docs from `data` and append them to `out`. `prev` is the doc immediately
    /// preceding the first encoded doc.
    fn decode(
        self,
        mut prev: i64,
        count: usize,
        mut data: &[u8],
        out: &mut Vec<i64>,
    ) -> Result<(), &'static str> {
        match self {
            DocFormat::LEB128 => {
                for _ in 0..count {
                    let delta =
                        leb128::read::unsigned(&mut data).map_err(|_| "invalid doc delta")?;
                    prev = prev
                        .checked_add_unsigned(delta)
                        .ok_or("doc delta overflow")?;
                    out.push(prev);
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum PayloadFormat {
    None = 0,
    LEB128 = 1,
}

impl From<PayloadFormat> for u8 {
    fn from(value: PayloadFormat) -> Self {
        value as u8
    }
}

impl TryFrom<u8> for PayloadFormat {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(PayloadFormat::None),
            1 => Ok(PayloadFormat::LEB128),
            _ => Err(Error::UnknownPayloadFormat(value)),
        }
    }
}

impl PayloadFormat {
    /// Encode `payloads` to `out`.
    fn encode(self, payloads: impl Iterator<Item = u64>, out: &mut Vec<u8>) {
        match self {
            PayloadFormat::None => {}
            PayloadFormat::LEB128 => {
                for payload in payloads {
                    leb128::write::unsigned(out, payload).unwrap();
                }
            }
        }
    }

    /// Decode the payload at rank `target` within the block.
    ///
    /// `data` begins at the payload at rank `rank` and is advanced past the `target` payload.
    /// `target` must be at least `rank`.
    fn decode_to(self, data: &mut &[u8], rank: usize, target: usize) -> Result<u64, &'static str> {
        debug_assert!(rank <= target);
        match self {
            PayloadFormat::None => Ok(0),
            PayloadFormat::LEB128 => {
                // Skip intervening values by looking for terminal bytes rather than decoding.
                for _ in rank..target {
                    let end = data
                        .iter()
                        .position(|b| b & 0x80 == 0)
                        .ok_or("invalid payload")?;
                    *data = &data[end + 1..];
                }
                leb128::read::unsigned(data).map_err(|_| "invalid payload")
            }
        }
    }
}

/// Block format:
/// * number of entries (unsigned LEB128)
/// * doc format (u8)
/// * payload format (u8)
/// * first docid (signed LEB128); omitted if there are no entries.
/// * docs len (unsigned LEB128); omitted if there are no entries or payload format is `None`.
/// * remaining docs encoded with [`DocFormat`]
/// * payloads encoded with [`PayloadFormat`]
///
/// Docs are decoded eagerly on construction; payloads are decoded lazily as they are read.
pub struct DocIteratorBlock<'a> {
    doc_format: DocFormat,
    payload_format: PayloadFormat,
    /// Decoded docs followed by [`DOC_ITER_SENTINEL`].
    docs: Vec<i64>,
    /// Index of the current doc in `docs`.
    index: usize,
    /// Remaining undecoded payload bytes.
    payloads: &'a [u8],
    /// Rank of the payload at the start of `payloads`.
    next_payload_rank: usize,
    payload: u64,
    error: Option<&'static str>,
}

impl<'a> DocIteratorBlock<'a> {
    /// Create a new iterator over the encoded block `data`, positioned on the first doc.
    ///
    /// Returns an error if the block header or docs are malformed.
    pub fn new(mut data: &'a [u8]) -> Result<Self, Error> {
        let len = usize::try_from(
            leb128::read::unsigned(&mut data).map_err(|_| Error::Coding("invalid entry count"))?,
        )
        .map_err(|_| Error::Coding("entry count overflow"))?;
        let doc_format = DocFormat::try_from(read_u8(&mut data)?)?;
        let payload_format = PayloadFormat::try_from(read_u8(&mut data)?)?;

        let mut docs = Vec::with_capacity(len.saturating_add(1).min(data.len() + 1));
        let mut payloads = &data[data.len()..];
        if len > 0 {
            let first_doc =
                leb128::read::signed(&mut data).map_err(|_| Error::Coding("invalid first doc"))?;
            docs.push(first_doc);
            let doc_data = if payload_format == PayloadFormat::None {
                data
            } else {
                let docs_len = leb128::read::unsigned(&mut data)
                    .map_err(|_| Error::Coding("invalid docs len"))?;
                let (doc_data, payload_data) = data
                    .split_at_checked(usize::try_from(docs_len).unwrap_or(usize::MAX))
                    .ok_or(Error::Coding("docs len exceeds block len"))?;
                payloads = payload_data;
                doc_data
            };
            doc_format
                .decode(first_doc, len - 1, doc_data, &mut docs)
                .map_err(Error::Coding)?;
        }
        docs.push(DOC_ITER_SENTINEL);

        Ok(Self {
            doc_format,
            payload_format,
            docs,
            index: 0,
            payloads,
            next_payload_rank: 0,
            payload: 0,
            error: None,
        })
    }

    pub fn doc_format(&self) -> DocFormat {
        self.doc_format
    }

    pub fn payload_format(&self) -> PayloadFormat {
        self.payload_format
    }

    /// Return the number of entries in the block.
    pub fn len(&self) -> usize {
        self.docs.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl DocIterator for DocIteratorBlock<'_> {
    fn doc(&mut self) -> i64 {
        self.docs[self.index]
    }

    fn payload(&mut self) -> u64 {
        if self.index >= self.len() || self.payload_format == PayloadFormat::None {
            return 0;
        }
        if self.next_payload_rank <= self.index {
            match self.payload_format.decode_to(
                &mut self.payloads,
                self.next_payload_rank,
                self.index,
            ) {
                Ok(p) => {
                    self.payload = p;
                    self.next_payload_rank = self.index + 1;
                }
                Err(e) => {
                    // Terminate iteration so the caller will check error().
                    self.error = Some(e);
                    self.index = self.len();
                    return 0;
                }
            }
        }
        self.payload
    }

    fn advance(&mut self) -> i64 {
        self.index = (self.index + 1).min(self.len());
        self.docs[self.index]
    }

    fn advance_to(&mut self, target: i64) -> i64 {
        // The trailing sentinel ensures this never moves past the end.
        self.index += self.docs[self.index..].partition_point(|&d| d < target);
        self.docs[self.index]
    }

    fn cost(&self) -> u64 {
        self.len() as u64
    }

    fn error(&self) -> Result<(), Error> {
        self.error.map_or(Ok(()), |e| Err(Error::Coding(e)))
    }
}

fn read_u8(data: &mut &[u8]) -> Result<u8, Error> {
    let (b, rest) = data
        .split_first()
        .ok_or(Error::Coding("unexpected end of block"))?;
    *data = rest;
    Ok(*b)
}

/// A mutable doc block that can be seeded from an encoded [`DocIteratorBlock`] and mutated via
/// inserts and removals.
pub struct DocIteratorBlockMut {
    doc_format: DocFormat,
    payload_format: PayloadFormat,
    /// Current entries mapping doc -> payload.
    entries: BTreeMap<i64, u64>,
    /// If true, block contents has been modified.
    dirty: bool,
}

impl DocIteratorBlockMut {
    /// Create a new empty block that will be encoded with the given formats.
    pub fn new(doc_format: DocFormat, payload_format: PayloadFormat) -> Self {
        Self {
            doc_format,
            payload_format,
            entries: BTreeMap::new(),
            dirty: false,
        }
    }

    /// Create a mutable block pre-populated with the contents of an encoded block.
    pub fn deserialize(data: &[u8]) -> Result<Self, Error> {
        let mut block = DocIteratorBlock::new(data)?;
        let mut entries = BTreeMap::new();
        while block.doc() != DOC_ITER_SENTINEL {
            entries.insert(block.doc(), block.payload());
            block.advance();
        }
        block.error()?;
        Ok(Self {
            doc_format: block.doc_format,
            payload_format: block.payload_format,
            entries,
            dirty: false,
        })
    }

    /// Return an iterator over the current (doc, payload) entries in the block.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (i64, u64)> {
        self.entries.iter().map(|(&d, &p)| (d, p))
    }

    /// Lookup the payload for `doc`, returning `None` if not found.
    pub fn lookup(&self, doc: i64) -> Option<u64> {
        self.entries.get(&doc).copied()
    }

    /// Insert or replace the entry for `doc`.
    ///
    /// `payload` is not serialized if the payload format is [`PayloadFormat::None`].
    ///
    /// *Panics* if `doc` is [`DOC_ITER_SENTINEL`].
    pub fn insert(&mut self, doc: i64, payload: u64) {
        assert_ne!(doc, DOC_ITER_SENTINEL);
        self.dirty = true;
        self.entries.insert(doc, payload);
    }

    /// Remove the entry for `doc`, returning its payload if it was present.
    pub fn remove(&mut self, doc: i64) -> Option<u64> {
        let result = self.entries.remove(&doc);
        self.dirty |= result.is_some();
        result
    }

    /// Serialize the current state of the block into the layout read by [`DocIteratorBlock`].
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        leb128::write::unsigned(&mut out, self.entries.len() as u64).unwrap();
        out.push(self.doc_format.into());
        out.push(self.payload_format.into());
        let Some(&first_doc) = self.entries.keys().next() else {
            return out;
        };
        leb128::write::signed(&mut out, first_doc).unwrap();

        if self.payload_format == PayloadFormat::None {
            self.doc_format
                .encode(first_doc, self.entries.keys().copied().skip(1), &mut out);
        } else {
            let mut docs = Vec::new();
            self.doc_format
                .encode(first_doc, self.entries.keys().copied().skip(1), &mut docs);
            leb128::write::unsigned(&mut out, docs.len() as u64).unwrap();
            out.extend_from_slice(&docs);
            self.payload_format
                .encode(self.entries.values().copied(), &mut out);
        }
        out
    }

    pub fn doc_format(&self) -> DocFormat {
        self.doc_format
    }

    pub fn payload_format(&self) -> PayloadFormat {
        self.payload_format
    }

    /// Return the number of entries in the block.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return true if the block has been modified.
    pub fn dirty(&self) -> bool {
        self.dirty
    }

    /// Mark this block as dirty.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn collect(data: &[u8]) -> Vec<(i64, u64)> {
        let mut it = DocIteratorBlock::new(data).unwrap();
        let mut out = vec![];
        while it.doc() != DOC_ITER_SENTINEL {
            out.push((it.doc(), it.payload()));
            it.advance();
        }
        it.error().unwrap();
        out
    }

    fn build(payload_format: PayloadFormat, entries: &[(i64, u64)]) -> DocIteratorBlockMut {
        let mut block = DocIteratorBlockMut::new(DocFormat::LEB128, payload_format);
        for &(d, p) in entries {
            block.insert(d, p);
        }
        block
    }

    const ENTRIES: &[(i64, u64)] = &[(-5, 7), (0, 0), (3, 1 << 40), (300, 2), (1 << 50, 9)];

    #[test]
    fn empty() {
        for pf in [PayloadFormat::None, PayloadFormat::LEB128] {
            let data = DocIteratorBlockMut::new(DocFormat::LEB128, pf).serialize();
            let mut it = DocIteratorBlock::new(&data).unwrap();
            assert!(it.is_empty());
            assert_eq!(it.doc(), DOC_ITER_SENTINEL);
            assert_eq!(it.payload(), 0);
            assert_eq!(it.advance(), DOC_ITER_SENTINEL);
            assert_eq!(it.advance_to(7), DOC_ITER_SENTINEL);
            it.error().unwrap();
        }
    }

    #[test]
    fn roundtrip_payloads() {
        let data = build(PayloadFormat::LEB128, ENTRIES).serialize();
        assert_eq!(collect(&data), ENTRIES);
        let block = DocIteratorBlockMut::deserialize(&data).unwrap();
        assert_eq!(block.payload_format(), PayloadFormat::LEB128);
        assert!(!block.dirty());
        assert_eq!(block.iter().collect::<Vec<_>>(), ENTRIES);
    }

    #[test]
    fn roundtrip_no_payloads() {
        let data = build(PayloadFormat::None, ENTRIES).serialize();
        let expected = ENTRIES.iter().map(|&(d, _)| (d, 0)).collect::<Vec<_>>();
        assert_eq!(collect(&data), expected);
        let block = DocIteratorBlockMut::deserialize(&data).unwrap();
        assert_eq!(block.payload_format(), PayloadFormat::None);
        assert_eq!(block.iter().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn sparse_payload_reads() {
        let data = build(PayloadFormat::LEB128, ENTRIES).serialize();
        let mut it = DocIteratorBlock::new(&data).unwrap();
        assert_eq!(it.advance_to(1), 3);
        assert_eq!(it.payload(), 1 << 40);
        assert_eq!(it.payload(), 1 << 40);
        assert_eq!(it.advance_to(1 << 50), 1 << 50);
        assert_eq!(it.payload(), 9);
        assert_eq!(it.advance(), DOC_ITER_SENTINEL);
        assert_eq!(it.payload(), 0);
        assert_eq!(it.advance(), DOC_ITER_SENTINEL);
        it.error().unwrap();
    }

    #[test]
    fn skip_multibyte_payload() {
        let data = build(PayloadFormat::LEB128, ENTRIES).serialize();
        let mut it = DocIteratorBlock::new(&data).unwrap();
        assert_eq!(it.advance_to(300), 300);
        assert_eq!(it.payload(), 2);
        it.error().unwrap();
    }

    #[test]
    fn advance_to() {
        let data = build(PayloadFormat::None, ENTRIES).serialize();
        let mut it = DocIteratorBlock::new(&data).unwrap();
        assert_eq!(it.cost(), ENTRIES.len() as u64);
        assert_eq!(it.advance_to(-10), -5);
        assert_eq!(it.advance_to(0), 0);
        assert_eq!(it.advance_to(0), 0);
        assert_eq!(it.advance_to(4), 300);
        assert_eq!(it.advance_to(301), 1 << 50);
        assert_eq!(it.advance_to((1 << 50) + 1), DOC_ITER_SENTINEL);
        assert_eq!(it.advance_to(0), DOC_ITER_SENTINEL);
    }

    #[test]
    fn mutate() {
        let data = build(PayloadFormat::LEB128, ENTRIES).serialize();
        let mut block = DocIteratorBlockMut::deserialize(&data).unwrap();
        assert_eq!(block.remove(1), None);
        assert!(!block.dirty());
        assert_eq!(block.remove(300), Some(2));
        assert!(block.dirty());
        block.insert(3, 4);
        block.insert(100, 5);
        assert_eq!(block.lookup(3), Some(4));
        assert_eq!(block.len(), 5);
        assert_eq!(
            collect(&block.serialize()),
            &[(-5, 7), (0, 0), (3, 4), (100, 5), (1 << 50, 9)]
        );
    }

    #[test]
    fn unknown_formats() {
        assert!(matches!(
            DocIteratorBlock::new(&[0, 9, 0]),
            Err(Error::UnknownDocFormat(9))
        ));
        assert!(matches!(
            DocIteratorBlock::new(&[0, 0, 9]),
            Err(Error::UnknownPayloadFormat(9))
        ));
    }

    #[test]
    fn truncated() {
        let data = build(PayloadFormat::LEB128, ENTRIES).serialize();
        // Truncating into the header fails at construction.
        assert!(matches!(
            DocIteratorBlock::new(&data[..2]),
            Err(Error::Coding(_))
        ));
        // Truncating into the docs fails at construction.
        let no_payloads = build(PayloadFormat::None, ENTRIES).serialize();
        assert!(matches!(
            DocIteratorBlock::new(&no_payloads[..no_payloads.len() - 1]),
            Err(Error::Coding(_))
        ));
        // Truncating into the payloads fails during iteration.
        let truncated = &data[..data.len() - 2];
        let mut it = DocIteratorBlock::new(truncated).unwrap();
        while it.doc() != DOC_ITER_SENTINEL {
            it.payload();
            it.advance();
        }
        assert!(matches!(it.error(), Err(Error::Coding(_))));
        assert!(DocIteratorBlockMut::deserialize(truncated).is_err());
    }
}
