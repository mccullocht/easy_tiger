//! Abstraction for token indexing.

use std::{borrow::Cow, collections::BTreeMap};

use wt_mdb::{IndexCursorGuard, Transaction};

// XXX impl Error
#[derive(Debug, Clone)]
pub enum Error {
    UnknownDocFormat(u8),
    UnknownPayloadFormat(u8),
    Coding(&'static str),
    WiredTiger(wt_mdb::Error),
}

impl From<wt_mdb::Error> for Error {
    fn from(value: wt_mdb::Error) -> Self {
        Self::WiredTiger(value)
    }
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

impl<T: DocIterator + ?Sized> DocIterator for Box<T> {
    fn doc(&mut self) -> i64 {
        (**self).doc()
    }

    fn payload(&mut self) -> u64 {
        (**self).payload()
    }

    fn advance(&mut self) -> i64 {
        (**self).advance()
    }

    fn advance_to(&mut self, target: i64) -> i64 {
        (**self).advance_to(target)
    }

    fn cost(&self) -> u64 {
        (**self).cost()
    }

    fn error(&self) -> Result<(), Error> {
        (**self).error()
    }
}

/// Construct a key for a posting block given a field name, key, and the last possible docid that
/// can appear in the block.
///
/// It is recommended to use [`DOC_ITER_SENTINEL`] as the limit for short posting lists as it makes
/// them easy to find -- seeking to a target docid will always return that block.
pub fn doc_block_key(field: &str, posting_key: &[u8], limit_docid: i64) -> Vec<u8> {
    doc_block_key_internal(field, posting_key, Some(limit_docid))
}

/// Construct a prefix that matches all of the blocks for a term posting.
pub fn doc_block_prefix(field: &str, posting_key: &[u8]) -> Vec<u8> {
    doc_block_key_internal(field, posting_key, None)
}

fn doc_block_key_internal(field: &str, posting_key: &[u8], limit_docid: Option<i64>) -> Vec<u8> {
    let mut key = Vec::with_capacity(
        field.len() + posting_key.len() + limit_docid.map(|_| 8).unwrap_or_default() + 2,
    );
    key.extend_from_slice(field.as_bytes());
    key.push(b':');
    key.extend_from_slice(posting_key);
    key.push(b':');
    if let Some(d) = limit_docid {
        key.extend_from_slice(&d.to_be_bytes());
    }
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
    /// Encoded block data.
    data: Cow<'a, [u8]>,
    /// Offset in `data` of the remaining undecoded payload bytes.
    payload_offset: usize,
    /// Rank of the payload at the start of `payloads`.
    next_payload_rank: usize,
    payload: u64,
    error: Option<&'static str>,
}

impl<'a> DocIteratorBlock<'a> {
    /// Create a new iterator over the encoded block `data`, positioned on the first doc.
    ///
    /// Returns an error if the block header or docs are malformed.
    pub fn new(data: &'a [u8]) -> Result<Self, Error> {
        Self::from_cow(data.into())
    }

    fn from_cow(block: Cow<'a, [u8]>) -> Result<Self, Error> {
        let mut data: &[u8] = &block;
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

        let payload_offset = block.len() - payloads.len();
        Ok(Self {
            doc_format,
            payload_format,
            docs,
            index: 0,
            data: block,
            payload_offset,
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

impl DocIteratorBlock<'static> {
    /// Create a new iterator over the encoded block `data`, positioned on the first doc.
    ///
    /// Returns an error if the block header or docs are malformed.
    pub fn from_vec(data: Vec<u8>) -> Result<Self, Error> {
        Self::from_cow(data.into())
    }

    /// Create an iterator with no entries.
    fn empty() -> Self {
        Self {
            doc_format: DocFormat::LEB128,
            payload_format: PayloadFormat::None,
            docs: vec![DOC_ITER_SENTINEL],
            index: 0,
            data: Cow::Borrowed(&[]),
            payload_offset: 0,
            next_payload_rank: 0,
            payload: 0,
            error: None,
        }
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
            let mut payloads = &self.data[self.payload_offset..];
            match self
                .payload_format
                .decode_to(&mut payloads, self.next_payload_rank, self.index)
            {
                Ok(p) => {
                    self.payload = p;
                    self.payload_offset = self.data.len() - payloads.len();
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

/// Iterates over all of the docs for a term by reading the sequence of [`DocIteratorBlock`]s
/// stored under keys produced by [`doc_block_key`].
///
/// Blocks are read in key order so block limit docids must be non-negative.
pub struct TermDocIterator<'a> {
    cursor: IndexCursorGuard<'a>,
    /// Key prefix shared by all blocks in the posting.
    prefix_len: usize,
    /// Upper bound key for all blocks in the posting.
    upper_bound: Vec<u8>,
    block: DocIteratorBlock<'static>,
    /// Limit docid for `block`; no doc after this value will appear in `block`.
    block_limit: i64,
    cost: u64,
    error: Option<Error>,
}

impl<'a> TermDocIterator<'a> {
    /// Create a new iterator over the blocks for `term` in `field` in `table_name`, positioned on
    /// the first doc.
    ///
    /// `cost` is returned by [`DocIterator::cost`].
    pub fn new(
        txn: &'a Transaction,
        table_name: &str,
        field: &str,
        term: &[u8],
        cost: u64,
    ) -> Result<Self, Error> {
        let mut cursor = txn.open_index_cursor(table_name)?;
        let mut upper_bound = doc_block_prefix(field, term);
        let prefix_len = upper_bound.len();
        upper_bound.extend_from_slice(&[0xff; 8]);
        cursor.set_bounds(&upper_bound[..prefix_len]..=upper_bound.as_slice())?;
        let mut it = Self {
            cursor,
            prefix_len,
            upper_bound,
            block: DocIteratorBlock::empty(),
            block_limit: i64::MIN,
            cost,
            error: None,
        };
        it.load_next_block()?;
        it.settle(i64::MIN);
        it.error()?;
        Ok(it)
    }

    /// Read the next block from the cursor, or an empty block if there are no more blocks.
    fn load_next_block(&mut self) -> Result<(), Error> {
        let Some((key, value)) = self.cursor.next().transpose()? else {
            self.block = DocIteratorBlock::empty();
            self.block_limit = DOC_ITER_SENTINEL;
            return Ok(());
        };
        let limit_bytes = key
            .get(self.prefix_len..)
            .and_then(|b| <[u8; 8]>::try_from(b).ok())
            .ok_or(Error::Coding("invalid doc block key"))?;
        self.block = DocIteratorBlock::from_vec(value)?;
        self.block_limit = i64::from_be_bytes(limit_bytes);
        Ok(())
    }

    /// Load the first block that may contain `target`, seeking if the next block does not.
    fn load_block_for(&mut self, target: i64) -> Result<(), Error> {
        self.load_next_block()?;
        if self.block_limit < target {
            let mut lower_bound = self.upper_bound[..self.prefix_len].to_vec();
            lower_bound.extend_from_slice(&target.to_be_bytes());
            self.cursor
                .set_bounds(lower_bound.as_slice()..=self.upper_bound.as_slice())?;
            self.load_next_block()?;
        }
        Ok(())
    }

    /// Advance to the first doc at or after `target`, loading blocks as needed.
    ///
    /// Maintains the invariant that `block` is only exhausted if the iterator is exhausted.
    fn settle(&mut self, target: i64) -> i64 {
        loop {
            let doc = self.block.advance_to(target);
            if doc != DOC_ITER_SENTINEL || self.block_limit == DOC_ITER_SENTINEL {
                return doc;
            }
            if let Err(e) = self
                .block
                .error()
                .and_then(|()| self.load_block_for(target))
            {
                self.error = Some(e);
                self.block = DocIteratorBlock::empty();
                self.block_limit = DOC_ITER_SENTINEL;
                return DOC_ITER_SENTINEL;
            }
        }
    }
}

impl DocIterator for TermDocIterator<'_> {
    fn doc(&mut self) -> i64 {
        self.block.doc()
    }

    fn payload(&mut self) -> u64 {
        self.block.payload()
    }

    fn advance(&mut self) -> i64 {
        self.block.advance();
        self.settle(i64::MIN)
    }

    fn advance_to(&mut self, target: i64) -> i64 {
        self.settle(target)
    }

    fn cost(&self) -> u64 {
        self.cost
    }

    fn error(&self) -> Result<(), Error> {
        match &self.error {
            Some(e) => Err(e.clone()),
            None => self.block.error(),
        }
    }
}

/// Yields only the docs that appear in every input iterator.
///
/// The two cheapest inputs lead iteration, advancing against each other until they agree on a doc.
/// The remaining inputs are only advanced to docs that both leads agree on, in ascending cost
/// order.
pub struct ConjunctionDocIterator<'a> {
    lead1: Box<dyn DocIterator + 'a>,
    lead2: Box<dyn DocIterator + 'a>,
    /// Remaining iterators in ascending cost order.
    others: Vec<Box<dyn DocIterator + 'a>>,
    doc: i64,
}

impl<'a> ConjunctionDocIterator<'a> {
    /// Create a new iterator over the intersection of `first`, `second`, and `rest`, positioned on
    /// the first matching doc.
    pub fn new(
        first: Box<dyn DocIterator + 'a>,
        second: Box<dyn DocIterator + 'a>,
        rest: Vec<Box<dyn DocIterator + 'a>>,
    ) -> Self {
        let mut others = rest;
        others.push(first);
        others.push(second);
        others.sort_by_key(|it| it.cost());
        let mut others = others.into_iter();
        let mut it = Self {
            lead1: others.next().unwrap(),
            lead2: others.next().unwrap(),
            others: others.collect(),
            doc: i64::MIN,
        };
        let target = it.lead1.doc();
        it.do_next(target);
        it
    }

    /// Find the first doc at or after `doc` that matches all iterators, where `doc` is the
    /// current doc of `lead1`.
    fn do_next(&mut self, mut doc: i64) -> i64 {
        'advance_leads: while doc != DOC_ITER_SENTINEL {
            let next2 = advance_if_behind(&mut *self.lead2, doc);
            if next2 != doc {
                doc = self.lead1.advance_to(next2);
                if doc != next2 {
                    continue;
                }
            }

            for other in self.others.iter_mut() {
                let next = advance_if_behind(&mut **other, doc);
                if next != doc {
                    doc = self.lead1.advance_to(next);
                    continue 'advance_leads;
                }
            }
            break;
        }
        self.doc = doc;
        doc
    }
}

/// Advance `it` to `target` if it is positioned before `target` and return its current doc.
fn advance_if_behind(it: &mut dyn DocIterator, target: i64) -> i64 {
    let doc = it.doc();
    if doc < target {
        it.advance_to(target)
    } else {
        doc
    }
}

impl DocIterator for ConjunctionDocIterator<'_> {
    fn doc(&mut self) -> i64 {
        self.doc
    }

    fn payload(&mut self) -> u64 {
        0
    }

    fn advance(&mut self) -> i64 {
        let doc = self.lead1.advance();
        self.do_next(doc)
    }

    fn advance_to(&mut self, target: i64) -> i64 {
        if target <= self.doc {
            return self.doc;
        }
        let doc = self.lead1.advance_to(target);
        self.do_next(doc)
    }

    fn cost(&self) -> u64 {
        self.lead1.cost()
    }

    fn error(&self) -> Result<(), Error> {
        self.lead1.error()?;
        self.lead2.error()?;
        self.others.iter().try_for_each(|it| it.error())
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

    struct TermFixture {
        // Must be dropped before `_dir`.
        conn: std::sync::Arc<wt_mdb::Connection>,
        _dir: tempfile::TempDir,
    }

    const TABLE: &str = "postings";

    /// (field, term, limit_docid, entries)
    type TestBlock<'a> = (&'a str, &'a [u8], i64, &'a [(i64, u64)]);

    impl TermFixture {
        /// Create a table containing `blocks`.
        fn new(blocks: &[TestBlock<'_>]) -> Self {
            let dir = tempfile::TempDir::new().unwrap();
            let conn = wt_mdb::Connection::open(
                dir.path().to_str().unwrap(),
                Some(
                    wt_mdb::connection::OptionsBuilder::default()
                        .create()
                        .into(),
                ),
            )
            .unwrap();
            conn.create_table(
                TABLE,
                Some(
                    wt_mdb::connection::CreateOptionsBuilder::default()
                        .key_format::<Vec<u8>>()
                        .value_format::<Vec<u8>>()
                        .into(),
                ),
            )
            .unwrap();
            let txn = conn.begin_transaction(None).unwrap();
            {
                let mut cursor = txn.open_index_cursor(TABLE).unwrap();
                for &(field, term, limit, entries) in blocks {
                    cursor
                        .set(
                            &doc_block_key(field, term, limit),
                            &build(PayloadFormat::LEB128, entries).serialize(),
                        )
                        .unwrap();
                }
            }
            txn.commit(None).unwrap();
            Self { conn, _dir: dir }
        }

        fn txn(&self) -> Transaction {
            self.conn.begin_transaction(None).unwrap()
        }
    }

    fn multi_block_fixture() -> TermFixture {
        TermFixture::new(&[
            ("f", b"a", 99, &[(1, 10), (5, 50), (50, 500)]),
            ("f", b"a", 199, &[(150, 1500)]),
            ("f", b"a", 299, &[]),
            ("f", b"a", 399, &[(300, 3000)]),
            ("f", b"a", DOC_ITER_SENTINEL, &[(400, 4000), (1000, 10000)]),
            // Neighboring keys that should not be visited.
            ("f", b"", DOC_ITER_SENTINEL, &[(2, 2)]),
            ("f", b"ab", DOC_ITER_SENTINEL, &[(3, 3)]),
            ("g", b"a", DOC_ITER_SENTINEL, &[(4, 4)]),
        ])
    }

    const MULTI_BLOCK_ENTRIES: &[(i64, u64)] = &[
        (1, 10),
        (5, 50),
        (50, 500),
        (150, 1500),
        (300, 3000),
        (400, 4000),
        (1000, 10000),
    ];

    #[test]
    fn term_iterate() {
        let fixture = multi_block_fixture();
        let txn = fixture.txn();
        let mut it = TermDocIterator::new(&txn, TABLE, "f", b"a", 7).unwrap();
        assert_eq!(it.cost(), 7);
        let mut out = vec![];
        while it.doc() != DOC_ITER_SENTINEL {
            out.push((it.doc(), it.payload()));
            it.advance();
        }
        it.error().unwrap();
        assert_eq!(out, MULTI_BLOCK_ENTRIES);
        assert_eq!(it.advance(), DOC_ITER_SENTINEL);
        assert_eq!(it.payload(), 0);
    }

    #[test]
    fn term_advance_to() {
        let fixture = multi_block_fixture();
        let txn = fixture.txn();
        let mut it = TermDocIterator::new(&txn, TABLE, "f", b"a", 7).unwrap();
        // Within the current block.
        assert_eq!(it.advance_to(5), 5);
        assert_eq!(it.payload(), 50);
        // Past the end of the current block into the next block.
        assert_eq!(it.advance_to(100), 150);
        assert_eq!(it.payload(), 1500);
        // Into the range of the empty block.
        assert_eq!(it.advance_to(250), 300);
        assert_eq!(it.payload(), 3000);
        // Doc in the current block is not before target.
        assert_eq!(it.advance_to(200), 300);
        assert_eq!(it.advance(), 400);
        assert_eq!(it.advance_to(1001), DOC_ITER_SENTINEL);
        it.error().unwrap();

        // Skip past multiple blocks with a seek.
        let mut it = TermDocIterator::new(&txn, TABLE, "f", b"a", 7).unwrap();
        assert_eq!(it.advance_to(400), 400);
        assert_eq!(it.payload(), 4000);
        assert_eq!(it.advance(), 1000);
        assert_eq!(it.advance(), DOC_ITER_SENTINEL);
        it.error().unwrap();
    }

    #[test]
    fn term_missing() {
        let fixture = multi_block_fixture();
        let txn = fixture.txn();
        let mut it = TermDocIterator::new(&txn, TABLE, "f", b"b", 0).unwrap();
        assert_eq!(it.doc(), DOC_ITER_SENTINEL);
        assert_eq!(it.cost(), 0);
        assert_eq!(it.advance(), DOC_ITER_SENTINEL);
        assert_eq!(it.advance_to(10), DOC_ITER_SENTINEL);
        it.error().unwrap();
    }

    #[test]
    fn term_neighbors() {
        let fixture = multi_block_fixture();
        let txn = fixture.txn();
        for (field, term, doc) in [("f", b"".as_slice(), 2), ("f", b"ab", 3), ("g", b"a", 4)] {
            let mut it = TermDocIterator::new(&txn, TABLE, field, term, 1).unwrap();
            assert_eq!(it.doc(), doc);
            assert_eq!(it.advance(), DOC_ITER_SENTINEL);
            it.error().unwrap();
        }
    }

    #[test]
    fn term_corrupt_block() {
        let fixture = TermFixture::new(&[
            ("f", b"a", 99, &[(1, 10)]),
            ("f", b"a", DOC_ITER_SENTINEL, &[(100, 1000)]),
        ]);
        {
            let txn = fixture.txn();
            txn.open_index_cursor(TABLE)
                .unwrap()
                .set(&doc_block_key("f", b"a", DOC_ITER_SENTINEL), &[1, 0, 9])
                .unwrap();
            txn.commit(None).unwrap();
        }
        let txn = fixture.txn();
        let mut it = TermDocIterator::new(&txn, TABLE, "f", b"a", 7).unwrap();
        assert_eq!(it.doc(), 1);
        assert_eq!(it.advance(), DOC_ITER_SENTINEL);
        assert!(matches!(it.error(), Err(Error::UnknownPayloadFormat(9))));
    }

    fn block_iter(docs: &[i64]) -> Box<dyn DocIterator> {
        let mut block = DocIteratorBlockMut::new(DocFormat::LEB128, PayloadFormat::LEB128);
        for &d in docs {
            block.insert(d, 1);
        }
        Box::new(DocIteratorBlock::from_vec(block.serialize()).unwrap())
    }

    fn collect_docs(it: &mut dyn DocIterator) -> Vec<i64> {
        let mut out = vec![];
        while it.doc() != DOC_ITER_SENTINEL {
            assert_eq!(it.payload(), 0);
            out.push(it.doc());
            it.advance();
        }
        it.error().unwrap();
        out
    }

    #[test]
    fn conjunction() {
        let mut it = ConjunctionDocIterator::new(
            block_iter(&[-3, 1, 2, 5, 8, 13, 21, 34]),
            block_iter(&[-3, 2, 3, 5, 13, 21, 22]),
            vec![block_iter(&[-3, 0, 2, 4, 5, 6, 13, 21, 100])],
        );
        assert_eq!(it.cost(), 7);
        assert_eq!(collect_docs(&mut it), vec![-3, 2, 5, 13, 21]);
        assert_eq!(it.advance(), DOC_ITER_SENTINEL);
    }

    #[test]
    fn conjunction_advance_to() {
        let mut it = ConjunctionDocIterator::new(
            block_iter(&[1, 2, 5, 8, 13, 21, 34]),
            block_iter(&[2, 3, 5, 13, 21, 22]),
            vec![],
        );
        assert_eq!(it.doc(), 2);
        assert_eq!(it.advance_to(2), 2);
        assert_eq!(it.advance_to(6), 13);
        assert_eq!(it.advance_to(13), 13);
        assert_eq!(it.advance_to(14), 21);
        assert_eq!(it.advance_to(22), DOC_ITER_SENTINEL);
        assert_eq!(it.advance_to(100), DOC_ITER_SENTINEL);
        it.error().unwrap();
    }

    #[test]
    fn conjunction_disjoint() {
        let mut it =
            ConjunctionDocIterator::new(block_iter(&[1, 3, 5]), block_iter(&[2, 4, 6]), vec![]);
        assert_eq!(it.doc(), DOC_ITER_SENTINEL);
        assert_eq!(it.advance(), DOC_ITER_SENTINEL);

        let mut it = ConjunctionDocIterator::new(block_iter(&[1, 3, 5]), block_iter(&[]), vec![]);
        assert_eq!(it.cost(), 0);
        assert_eq!(it.doc(), DOC_ITER_SENTINEL);
        it.error().unwrap();
    }

    #[test]
    fn conjunction_random() {
        use rand::prelude::*;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0x5eed);
        for _ in 0..100 {
            let num_iters = rng.random_range(2..6);
            let doc_lists: Vec<Vec<i64>> = (0..num_iters)
                .map(|_| {
                    let density = rng.random_range(0.1..0.9);
                    (0..500).filter(|_| rng.random_bool(density)).collect()
                })
                .collect();
            let expected: Vec<i64> = doc_lists[0]
                .iter()
                .copied()
                .filter(|d| doc_lists[1..].iter().all(|l| l.contains(d)))
                .collect();
            let mut iters = doc_lists.iter().map(|l| block_iter(l));
            let first = iters.next().unwrap();
            let second = iters.next().unwrap();
            let mut it = ConjunctionDocIterator::new(first, second, iters.collect());
            assert_eq!(collect_docs(&mut it), expected);
        }
    }

    #[test]
    fn conjunction_error() {
        let mut block = DocIteratorBlockMut::new(DocFormat::LEB128, PayloadFormat::LEB128);
        block.insert(1, 1 << 40);
        let mut data = block.serialize();
        data.truncate(data.len() - 2);
        // Reading the truncated payload terminates the input with an error.
        let mut failed = DocIteratorBlock::from_vec(data).unwrap();
        failed.payload();
        let mut it = ConjunctionDocIterator::new(
            block_iter(&[1, 2]),
            block_iter(&[1, 2, 3]),
            vec![Box::new(failed)],
        );
        assert_eq!(it.doc(), DOC_ITER_SENTINEL);
        assert!(matches!(it.error(), Err(Error::Coding(_))));
    }
}
