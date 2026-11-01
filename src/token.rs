//! Abstraction for token indexing.

use std::{
    borrow::Cow,
    collections::{BTreeMap, BinaryHeap},
};

use wt_mdb::{IndexCursorGuard, Transaction};

#[derive(Debug, Clone)]
pub enum Error {
    UnknownDocFormat(u8),
    UnknownPayloadFormat(u8),
    InvalidFieldName(String),
    Coding(&'static str),
    WiredTiger(wt_mdb::Error),
}

impl From<wt_mdb::Error> for Error {
    fn from(value: wt_mdb::Error) -> Self {
        Self::WiredTiger(value)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownDocFormat(t) => write!(f, "UnknownDocFormat({t})"),
            Self::UnknownPayloadFormat(t) => write!(f, "UnknownPayloadFormat({t})"),
            Self::InvalidFieldName(n) => write!(f, "InvalidFieldName({n:?})"),
            Self::Coding(s) => write!(f, "Coding({s})"),
            Self::WiredTiger(e) => write!(f, "WiredTiger({e})"),
        }
    }
}

impl std::error::Error for Error {}

pub type DocId = u64;
pub type Payload = u64;

/// Sentinel value returned at the end of iteration.
pub const DOC_ITER_SENTINEL: DocId = u64::MAX;

/// An iterator over a set of documents.
///
/// Each iterator could represent a single term list or some compound operation: conjunction,
/// disjunction, negation, or positional operations like phrases.
///
/// [`DocIterator`]s begin positioned at the first matching document. Callers can iterate using
/// [`DocIterator::advance`] or skip forward using [`DocIterator::advance_to`], and read any
/// associated payload value with each hit. If an error occurs during iteration the iterator will
/// be advanced to the end of the list ([`DOC_ITER_SENTINEL`]) and the error will be exposed via
/// the [`DocIterator::error`] method.
pub trait DocIterator {
    /// Return the current doc, or [`DOC_ITER_SENTINEL`] if there are no more docs.
    fn doc(&mut self) -> DocId;
    /// Return the payload value for the current doc, or `0` if `doc()` is the sentinel value.
    fn payload(&mut self) -> Payload;

    /// Advance to the next document and returns its id.
    ///
    /// Returns [`DOC_ITER_SENTINEL`] if there are no more docs.
    /// It is legal to `advance()` after the sentinel value is returned; further calls should also
    /// return the sentinel value.
    fn advance(&mut self) -> DocId;
    /// Advance to the next document at or after `target` and return its id.
    ///
    /// Returns [`DOC_ITER_SENTINEL`] if there are no more docs.
    ///
    /// If the iterator is position before `target` its positional will remain unchanged.
    /// It is legal to `advance_to()` after the sentinel value is returned; further calls should
    /// also return the sentinel value.
    fn advance_to(&mut self, target: DocId) -> DocId;

    /// Cost of this iterator, proportional to hit count and cost of access.
    fn cost(&self) -> u64;

    /// Any errors that occurred while decoding underlying data for this iterator.
    ///
    /// Call this when the sentinel value is returned.
    fn error(&self) -> Result<(), Error>;
}

impl<T: DocIterator + ?Sized> DocIterator for Box<T> {
    fn doc(&mut self) -> DocId {
        (**self).doc()
    }

    fn payload(&mut self) -> Payload {
        (**self).payload()
    }

    fn advance(&mut self) -> DocId {
        (**self).advance()
    }

    fn advance_to(&mut self, target: DocId) -> DocId {
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
/// Keys are `field ':' escape(posting_key) 0x00 0x00 limit_be`, where `escape` replaces each 0x00
/// with 0x00 0xff. This preserves lexicographic order of posting keys and ensures that no key for
/// one posting falls within the range of another. `field` may not contain ':'.
///
/// It is recommended to use [`DOC_ITER_SENTINEL`] as the limit for short posting lists as it makes
/// them easy to find -- seeking to a target docid will always return that block.
pub fn doc_block_key(
    field: &str,
    posting_key: &[u8],
    limit_docid: DocId,
) -> Result<Vec<u8>, Error> {
    let mut key = doc_block_prefix(field, posting_key)?;
    key.extend_from_slice(&limit_docid.to_be_bytes());
    Ok(key)
}

/// Construct a prefix that matches all of the blocks for a term posting.
pub fn doc_block_prefix(field: &str, posting_key: &[u8]) -> Result<Vec<u8>, Error> {
    if field.contains(':') {
        return Err(Error::InvalidFieldName(field.to_owned()));
    }
    let mut key = Vec::with_capacity(field.len() + posting_key.len() + 11);
    key.extend_from_slice(field.as_bytes());
    key.push(b':');
    for &b in posting_key {
        key.push(b);
        if b == 0 {
            key.push(0xff);
        }
    }
    key.extend_from_slice(&[0, 0]);
    Ok(key)
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
    fn encode(self, mut prev: DocId, docs: impl Iterator<Item = DocId>, out: &mut Vec<u8>) {
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
        mut prev: DocId,
        count: usize,
        mut data: &[u8],
        out: &mut Vec<DocId>,
    ) -> Result<(), &'static str> {
        match self {
            DocFormat::LEB128 => {
                for _ in 0..count {
                    let delta =
                        leb128::read::unsigned(&mut data).map_err(|_| "invalid doc delta")?;
                    prev = prev.checked_add(delta).ok_or("doc delta overflow")?;
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
    fn encode(self, payloads: impl Iterator<Item = Payload>, out: &mut Vec<u8>) {
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
    fn decode_to(
        self,
        data: &mut &[u8],
        rank: usize,
        target: usize,
    ) -> Result<Payload, &'static str> {
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
    docs: Vec<DocId>,
    /// Index of the current doc in `docs`.
    index: usize,
    /// Encoded block data.
    data: Cow<'a, [u8]>,
    /// Offset in `data` of the remaining undecoded payload bytes.
    payload_offset: usize,
    /// Rank of the payload at the start of `payloads`.
    next_payload_rank: usize,
    payload: Payload,
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
            let first_doc = leb128::read::unsigned(&mut data)
                .map_err(|_| Error::Coding("invalid first doc"))?;
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
    fn doc(&mut self) -> DocId {
        self.docs[self.index]
    }

    fn payload(&mut self) -> Payload {
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

    fn advance(&mut self) -> DocId {
        self.index = (self.index + 1).min(self.len());
        self.docs[self.index]
    }

    fn advance_to(&mut self, target: DocId) -> DocId {
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
pub struct TermDocIterator {
    cursor: IndexCursorGuard,
    /// Key prefix shared by all blocks in the posting.
    prefix_len: usize,
    /// Upper bound key for all blocks in the posting.
    upper_bound: Vec<u8>,
    block: DocIteratorBlock<'static>,
    /// Limit docid for `block`; no doc after this value will appear in `block`.
    block_limit: DocId,
    cost: u64,
    error: Option<Error>,
}

impl TermDocIterator {
    /// Create a new iterator over the blocks for `term` in `field` in `table_name`, positioned on
    /// the first doc.
    ///
    /// `cost` is returned by [`DocIterator::cost`].
    pub fn new(
        txn: &Transaction,
        table_name: &str,
        field: &str,
        term: &[u8],
        cost: u64,
    ) -> Result<Self, Error> {
        let mut cursor = txn.open_index_cursor(table_name)?;
        let mut upper_bound = doc_block_prefix(field, term)?;
        let prefix_len = upper_bound.len();
        upper_bound.extend_from_slice(&[0xff; 8]);
        cursor.set_bounds(&upper_bound[..prefix_len]..=upper_bound.as_slice())?;
        let mut it = Self {
            cursor,
            prefix_len,
            upper_bound,
            block: DocIteratorBlock::empty(),
            block_limit: 0,
            cost,
            error: None,
        };
        it.load_next_block(0)?;
        it.settle(0);
        it.error()?;
        Ok(it)
    }

    /// Read the next block from the cursor, or an empty block if there are no more blocks.
    ///
    /// If the next block ends before `target` it is not loaded and this returns false.
    fn load_next_block(&mut self, target: DocId) -> Result<bool, Error> {
        // Safety: the value is copied before the cursor is repositioned.
        let Some((key, value)) = unsafe { self.cursor.next_unsafe() }.transpose()? else {
            self.block = DocIteratorBlock::empty();
            self.block_limit = DOC_ITER_SENTINEL;
            return Ok(true);
        };
        let limit = key
            .get(self.prefix_len..)
            .and_then(|b| <[u8; 8]>::try_from(b).ok())
            .map(u64::from_be_bytes)
            .ok_or(Error::Coding("invalid doc block key"))?;
        if limit < target {
            return Ok(false);
        }
        self.block = DocIteratorBlock::from_vec(value.to_vec())?;
        self.block_limit = limit;
        Ok(true)
    }

    /// Load the first block that may contain `target`, seeking if the next block does not.
    fn load_block_for(&mut self, target: DocId) -> Result<(), Error> {
        if !self.load_next_block(target)? {
            let mut lower_bound = self.upper_bound[..self.prefix_len].to_vec();
            lower_bound.extend_from_slice(&target.to_be_bytes());
            self.cursor
                .set_bounds(lower_bound.as_slice()..=self.upper_bound.as_slice())?;
            self.load_next_block(0)?;
        }
        Ok(())
    }

    /// Advance to the first doc at or after `target`, loading blocks as needed.
    ///
    /// Maintains the invariant that `block` is only exhausted if the iterator is exhausted.
    fn settle(&mut self, target: DocId) -> DocId {
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

impl DocIterator for TermDocIterator {
    fn doc(&mut self) -> DocId {
        self.block.doc()
    }

    fn payload(&mut self) -> Payload {
        self.block.payload()
    }

    fn advance(&mut self) -> DocId {
        self.block.advance();
        self.settle(0)
    }

    fn advance_to(&mut self, target: DocId) -> DocId {
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
    doc: DocId,
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
            doc: 0,
        };
        let target = it.lead1.doc();
        it.do_next(target);
        it
    }

    /// Find the first doc at or after `doc` that matches all iterators, where `doc` is the
    /// current doc of `lead1`.
    fn do_next(&mut self, mut doc: DocId) -> DocId {
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
fn advance_if_behind(it: &mut dyn DocIterator, target: DocId) -> DocId {
    let doc = it.doc();
    if doc < target {
        it.advance_to(target)
    } else {
        doc
    }
}

impl DocIterator for ConjunctionDocIterator<'_> {
    fn doc(&mut self) -> DocId {
        self.doc
    }

    fn payload(&mut self) -> Payload {
        0
    }

    fn advance(&mut self) -> DocId {
        let doc = self.lead1.advance();
        self.do_next(doc)
    }

    fn advance_to(&mut self, target: DocId) -> DocId {
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

/// Yields the docs that appear in any input iterator.
///
/// Inputs are kept in a min-heap ordered by their current doc. Iteration terminates early if any
/// input fails.
pub struct DisjunctionDocIterator<'a> {
    heap: BinaryHeap<DisjunctionEntry<'a>>,
    doc: DocId,
    cost: u64,
}

/// An input to [`DisjunctionDocIterator`] ordered so that the lowest doc is at the top of the heap.
struct DisjunctionEntry<'a> {
    doc: DocId,
    it: Box<dyn DocIterator + 'a>,
}

impl DisjunctionEntry<'_> {
    /// Set the current doc to `doc` and return false if the input failed.
    fn update(&mut self, doc: DocId) -> bool {
        self.doc = doc;
        doc != DOC_ITER_SENTINEL || self.it.error().is_ok()
    }
}

impl PartialEq for DisjunctionEntry<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.doc == other.doc
    }
}

impl Eq for DisjunctionEntry<'_> {}

impl PartialOrd for DisjunctionEntry<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DisjunctionEntry<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.doc.cmp(&self.doc)
    }
}

impl<'a> DisjunctionDocIterator<'a> {
    /// Create a new iterator over the union of `iters`, positioned on the first matching doc.
    pub fn new(iters: Vec<Box<dyn DocIterator + 'a>>) -> Self {
        let cost = iters
            .iter()
            .map(|it| it.cost())
            .fold(0, u64::saturating_add);
        let mut failed = false;
        let heap = iters
            .into_iter()
            .map(|mut it| {
                let doc = it.doc();
                let mut entry = DisjunctionEntry { doc: 0, it };
                failed |= !entry.update(doc);
                entry
            })
            .collect();
        let mut it = Self {
            heap,
            doc: DOC_ITER_SENTINEL,
            cost,
        };
        if !failed {
            it.settle();
        }
        it
    }

    /// Position on the lowest doc among all inputs.
    fn settle(&mut self) -> DocId {
        self.doc = self.heap.peek().map_or(DOC_ITER_SENTINEL, |e| e.doc);
        self.doc
    }
}

impl DocIterator for DisjunctionDocIterator<'_> {
    fn doc(&mut self) -> DocId {
        self.doc
    }

    fn payload(&mut self) -> Payload {
        0
    }

    fn advance(&mut self) -> DocId {
        if self.doc == DOC_ITER_SENTINEL {
            return DOC_ITER_SENTINEL;
        }
        while let Some(mut top) = self.heap.peek_mut()
            && top.doc == self.doc
        {
            let next = top.it.advance();
            if !top.update(next) {
                drop(top);
                self.doc = DOC_ITER_SENTINEL;
                return DOC_ITER_SENTINEL;
            }
        }
        self.settle()
    }

    fn advance_to(&mut self, target: DocId) -> DocId {
        if target <= self.doc {
            return self.doc;
        }
        while let Some(mut top) = self.heap.peek_mut()
            && top.doc < target
        {
            let next = top.it.advance_to(target);
            if !top.update(next) {
                drop(top);
                self.doc = DOC_ITER_SENTINEL;
                return DOC_ITER_SENTINEL;
            }
        }
        self.settle()
    }

    fn cost(&self) -> u64 {
        self.cost
    }

    fn error(&self) -> Result<(), Error> {
        self.heap.iter().try_for_each(|e| e.it.error())
    }
}

/// Yields at most one doc from `input` for each centroid.
///
/// `assignments` is a centroid mapping posting containing a hit for each doc with a vector, where
/// the payload is the id of the centroid that vector is assigned to. Docs from `input` that do not
/// appear in `assignments` are skipped, as are docs assigned to a centroid that has already been
/// yielded. The payload for each doc is its centroid id.
///
/// Iteration ends early once every centroid has been yielded.
pub struct CentroidParentFilterDocIterator<'a> {
    input: Box<dyn DocIterator + 'a>,
    assignments: Box<dyn DocIterator + 'a>,
    /// Bitmap of centroid ids that have already been yielded.
    seen: Vec<u64>,
    /// Number of centroids that have not yet been yielded.
    unseen: u32,
    doc: DocId,
    centroid: u64,
    error: Option<Error>,
}

impl<'a> CentroidParentFilterDocIterator<'a> {
    /// Create a new iterator over `input` filtered by centroid assignments in `assignments`,
    /// positioned on the first matching doc.
    ///
    /// All centroid ids in `assignments` must be no greater than `max_centroid_id`.
    /// `num_centroids` is the number of distinct centroid ids that may appear in `assignments`;
    /// if it is too small some centroids will not be yielded.
    pub fn new(
        input: Box<dyn DocIterator + 'a>,
        assignments: Box<dyn DocIterator + 'a>,
        max_centroid_id: u32,
        num_centroids: u32,
    ) -> Self {
        let mut it = Self {
            input,
            assignments,
            seen: vec![0; max_centroid_id as usize / 64 + 1],
            unseen: num_centroids,
            doc: 0,
            centroid: 0,
            error: None,
        };
        let target = it.input.doc();
        it.do_next(target);
        it
    }

    /// Find the first doc at or after `doc` with an unseen centroid, where `doc` is the current
    /// doc of `input`.
    fn do_next(&mut self, mut doc: DocId) -> DocId {
        self.centroid = 0;
        while doc != DOC_ITER_SENTINEL {
            if self.unseen == 0 {
                // Every centroid has been yielded so no remaining doc can match.
                doc = DOC_ITER_SENTINEL;
                break;
            }
            let assigned = advance_if_behind(&mut *self.assignments, doc);
            if assigned != doc {
                // No assignment for doc; skip to the next doc that has one.
                doc = self.input.advance_to(assigned);
                continue;
            }

            let centroid = self.assignments.payload();
            if self.assignments.doc() != doc {
                // Reading the payload failed and terminated assignments; error() will report it.
                doc = DOC_ITER_SENTINEL;
                break;
            }
            let Some(word) = self.seen.get_mut((centroid / 64) as usize) else {
                self.error = Some(Error::Coding("centroid id exceeds max_centroid_id"));
                doc = DOC_ITER_SENTINEL;
                break;
            };
            let mask = 1u64 << (centroid % 64);
            if *word & mask == 0 {
                *word |= mask;
                self.unseen -= 1;
                self.centroid = centroid;
                break;
            }
            doc = self.input.advance();
        }
        self.doc = doc;
        doc
    }
}

impl DocIterator for CentroidParentFilterDocIterator<'_> {
    fn doc(&mut self) -> DocId {
        self.doc
    }

    fn payload(&mut self) -> Payload {
        self.centroid
    }

    fn advance(&mut self) -> DocId {
        if self.doc == DOC_ITER_SENTINEL {
            return DOC_ITER_SENTINEL;
        }
        let doc = self.input.advance();
        self.do_next(doc)
    }

    fn advance_to(&mut self, target: DocId) -> DocId {
        if target <= self.doc {
            return self.doc;
        }
        let doc = self.input.advance_to(target);
        self.do_next(doc)
    }

    fn cost(&self) -> u64 {
        self.input.cost()
    }

    fn error(&self) -> Result<(), Error> {
        if let Some(e) = &self.error {
            return Err(e.clone());
        }
        self.input.error()?;
        self.assignments.error()
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
    entries: BTreeMap<DocId, Payload>,
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
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (DocId, Payload)> {
        self.entries.iter().map(|(&d, &p)| (d, p))
    }

    /// Lookup the payload for `doc`, returning `None` if not found.
    pub fn lookup(&self, doc: DocId) -> Option<u64> {
        self.entries.get(&doc).copied()
    }

    /// Insert or replace the entry for `doc`.
    ///
    /// `payload` is not serialized if the payload format is [`PayloadFormat::None`].
    ///
    /// *Panics* if `doc` is [`DOC_ITER_SENTINEL`].
    pub fn insert(&mut self, doc: DocId, payload: Payload) {
        assert_ne!(doc, DOC_ITER_SENTINEL);
        self.dirty = true;
        self.entries.insert(doc, payload);
    }

    /// Remove the entry for `doc`, returning its payload if it was present.
    pub fn remove(&mut self, doc: DocId) -> Option<Payload> {
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
        leb128::write::unsigned(&mut out, first_doc).unwrap();

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

    fn collect(data: &[u8]) -> Vec<(DocId, Payload)> {
        let mut it = DocIteratorBlock::new(data).unwrap();
        let mut out = vec![];
        while it.doc() != DOC_ITER_SENTINEL {
            out.push((it.doc(), it.payload()));
            it.advance();
        }
        it.error().unwrap();
        out
    }

    fn build(payload_format: PayloadFormat, entries: &[(DocId, Payload)]) -> DocIteratorBlockMut {
        let mut block = DocIteratorBlockMut::new(DocFormat::LEB128, payload_format);
        for &(d, p) in entries {
            block.insert(d, p);
        }
        block
    }

    const ENTRIES: &[(DocId, Payload)] = &[(0, 0), (3, 1 << 40), (300, 2), (1 << 50, 9)];

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
        assert_eq!(block.len(), 4);
        assert_eq!(
            collect(&block.serialize()),
            &[(0, 0), (3, 4), (100, 5), (1 << 50, 9)]
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
    type TestBlock<'a> = (&'a str, &'a [u8], DocId, &'a [(DocId, Payload)]);

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
                            &doc_block_key(field, term, limit).unwrap(),
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
            ("f", b"a:b", DOC_ITER_SENTINEL, &[(5, 5)]),
            ("f", b"a:\0\0\0\0\0\0\0\x01", DOC_ITER_SENTINEL, &[(6, 6)]),
            ("f", b"a\0", DOC_ITER_SENTINEL, &[(7, 7)]),
            ("f", b"a\0\xff\xff\xff\xff\xff\xff\xff\xff", 99, &[(8, 8)]),
            ("f", b"\0", DOC_ITER_SENTINEL, &[(9, 9)]),
        ])
    }

    const MULTI_BLOCK_ENTRIES: &[(DocId, Payload)] = &[
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
        for (field, term, doc) in [
            ("f", b"".as_slice(), 2),
            ("f", b"ab", 3),
            ("g", b"a", 4),
            ("f", b"a:b", 5),
            ("f", b"a:\0\0\0\0\0\0\0\x01", 6),
            ("f", b"a\0", 7),
            ("f", b"a\0\xff\xff\xff\xff\xff\xff\xff\xff", 8),
            ("f", b"\0", 9),
        ] {
            let mut it = TermDocIterator::new(&txn, TABLE, field, term, 1).unwrap();
            assert_eq!(it.doc(), doc, "term {term:?}");
            assert_eq!(it.advance(), DOC_ITER_SENTINEL);
            it.error().unwrap();
        }
    }

    #[test]
    fn term_invalid_field_name() {
        let fixture = multi_block_fixture();
        let txn = fixture.txn();
        assert!(matches!(
            TermDocIterator::new(&txn, TABLE, "f:a", b"b", 1),
            Err(Error::InvalidFieldName(n)) if n == "f:a"
        ));
    }

    const KEY_TEST_TERMS: &[&[u8]] = &[
        b"",
        b"\0",
        b"\0\0",
        b"\0\x01",
        b"\0\xff",
        b"\x01",
        b":",
        b"a",
        b"a\0",
        b"a\0\0",
        b"a\0\xff",
        b"a\0\xff\xff\xff\xff\xff\xff\xff\xff",
        b"a\x01",
        b"a:",
        b"a:\0\0\0\0\0\0\0\x01",
        b"a:b",
        b"ab",
        b"a\xff",
        b"\xff",
        b"\xff\0",
    ];

    const KEY_TEST_LIMITS: &[DocId] = &[0, 1, 0xff, 1 << 56, DOC_ITER_SENTINEL];

    #[test]
    fn doc_block_key_escapes_term() {
        assert_eq!(
            doc_block_key("f", b"a\0b", 0x0102).unwrap(),
            b"f:a\0\xffb\0\0\0\0\0\0\0\0\x01\x02"
        );
        assert_eq!(doc_block_prefix("f", b"").unwrap(), b"f:\0\0");
        assert_eq!(
            doc_block_prefix("f", b"\0\0").unwrap(),
            b"f:\0\xff\0\xff\0\0"
        );
    }

    #[test]
    fn doc_block_key_invalid_field_name() {
        for field in [":", "f:", ":f", "f:g"] {
            assert!(matches!(
                doc_block_key(field, b"a", 0),
                Err(Error::InvalidFieldName(n)) if n == field
            ));
            assert!(matches!(
                doc_block_prefix(field, b"a"),
                Err(Error::InvalidFieldName(n)) if n == field
            ));
        }
    }

    #[test]
    fn doc_block_key_ranges_disjoint() {
        for &t1 in KEY_TEST_TERMS {
            let prefix = doc_block_prefix("f", t1).unwrap();
            let mut upper = prefix.clone();
            upper.extend_from_slice(&[0xff; 8]);
            for &t2 in KEY_TEST_TERMS.iter().filter(|&&t| t != t1) {
                for &limit in KEY_TEST_LIMITS {
                    let key = doc_block_key("f", t2, limit).unwrap();
                    assert!(
                        !(prefix.as_slice()..=upper.as_slice()).contains(&key.as_slice()),
                        "key for {t2:?} limit {limit} in range of {t1:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn doc_block_key_order() {
        let mut terms = KEY_TEST_TERMS.to_vec();
        terms.sort();
        let keys = terms
            .iter()
            .flat_map(|t| {
                KEY_TEST_LIMITS
                    .iter()
                    .map(|&l| doc_block_key("f", t, l).unwrap())
            })
            .collect::<Vec<_>>();
        assert!(keys.is_sorted());
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
                .set(
                    &doc_block_key("f", b"a", DOC_ITER_SENTINEL).unwrap(),
                    &[1, 0, 9],
                )
                .unwrap();
            txn.commit(None).unwrap();
        }
        let txn = fixture.txn();
        let mut it = TermDocIterator::new(&txn, TABLE, "f", b"a", 7).unwrap();
        assert_eq!(it.doc(), 1);
        assert_eq!(it.advance(), DOC_ITER_SENTINEL);
        assert!(matches!(it.error(), Err(Error::UnknownPayloadFormat(9))));
    }

    fn block_iter(docs: &[DocId]) -> Box<dyn DocIterator> {
        let mut block = DocIteratorBlockMut::new(DocFormat::LEB128, PayloadFormat::LEB128);
        for &d in docs {
            block.insert(d, 1);
        }
        Box::new(DocIteratorBlock::from_vec(block.serialize()).unwrap())
    }

    fn collect_docs(it: &mut dyn DocIterator) -> Vec<DocId> {
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
            block_iter(&[1, 2, 5, 8, 13, 21, 34]),
            block_iter(&[2, 3, 5, 13, 21, 22]),
            vec![block_iter(&[0, 2, 4, 5, 6, 13, 21, 100])],
        );
        assert_eq!(it.cost(), 6);
        assert_eq!(collect_docs(&mut it), vec![2, 5, 13, 21]);
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
            let doc_lists: Vec<Vec<DocId>> = (0..num_iters)
                .map(|_| {
                    let density = rng.random_range(0.1..0.9);
                    (0..500).filter(|_| rng.random_bool(density)).collect()
                })
                .collect();
            let expected: Vec<DocId> = doc_lists[0]
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

    #[test]
    fn disjunction() {
        let mut it = DisjunctionDocIterator::new(vec![
            block_iter(&[1, 2, 5, 8]),
            block_iter(&[2, 3, 5, 13]),
            block_iter(&[0, 5, 100]),
        ]);
        assert_eq!(it.cost(), 11);
        assert_eq!(collect_docs(&mut it), vec![0, 1, 2, 3, 5, 8, 13, 100]);
        assert_eq!(it.advance(), DOC_ITER_SENTINEL);
    }

    #[test]
    fn disjunction_advance_to() {
        let mut it =
            DisjunctionDocIterator::new(vec![block_iter(&[1, 5, 8, 21]), block_iter(&[2, 5, 13])]);
        assert_eq!(it.doc(), 1);
        assert_eq!(it.advance_to(1), 1);
        assert_eq!(it.advance_to(3), 5);
        assert_eq!(it.advance_to(5), 5);
        assert_eq!(it.advance(), 8);
        assert_eq!(it.advance_to(9), 13);
        assert_eq!(it.advance_to(22), DOC_ITER_SENTINEL);
        assert_eq!(it.advance_to(100), DOC_ITER_SENTINEL);
        it.error().unwrap();
    }

    #[test]
    fn disjunction_empty() {
        let mut it = DisjunctionDocIterator::new(vec![]);
        assert_eq!(it.cost(), 0);
        assert_eq!(it.doc(), DOC_ITER_SENTINEL);
        assert_eq!(it.advance(), DOC_ITER_SENTINEL);
        it.error().unwrap();

        let mut it = DisjunctionDocIterator::new(vec![block_iter(&[]), block_iter(&[])]);
        assert_eq!(it.doc(), DOC_ITER_SENTINEL);
        assert_eq!(it.advance_to(1), DOC_ITER_SENTINEL);
        it.error().unwrap();
    }

    #[test]
    fn disjunction_random() {
        use rand::prelude::*;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0x5eed);
        for _ in 0..100 {
            let num_iters = rng.random_range(1..6);
            let doc_lists: Vec<Vec<DocId>> = (0..num_iters)
                .map(|_| {
                    let density = rng.random_range(0.01..0.5);
                    (0..500).filter(|_| rng.random_bool(density)).collect()
                })
                .collect();
            let expected: Vec<DocId> = doc_lists
                .iter()
                .flatten()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            let mut it =
                DisjunctionDocIterator::new(doc_lists.iter().map(|l| block_iter(l)).collect());
            assert_eq!(collect_docs(&mut it), expected);

            let mut it =
                DisjunctionDocIterator::new(doc_lists.iter().map(|l| block_iter(l)).collect());
            let mut target = 0;
            loop {
                target += rng.random_range(1..20);
                let want = expected
                    .iter()
                    .copied()
                    .find(|&d| d >= target)
                    .unwrap_or(DOC_ITER_SENTINEL);
                assert_eq!(it.advance_to(target), want);
                if want == DOC_ITER_SENTINEL {
                    break;
                }
                target = want;
            }
            it.error().unwrap();
        }
    }

    #[test]
    fn disjunction_error() {
        let mut block = DocIteratorBlockMut::new(DocFormat::LEB128, PayloadFormat::LEB128);
        block.insert(3, 1 << 40);
        let mut data = block.serialize();
        data.truncate(data.len() - 2);
        let failed = || {
            // Reading the truncated payload terminates the input with an error.
            let mut it = DocIteratorBlock::from_vec(data.clone()).unwrap();
            it.payload();
            Box::new(it)
        };

        // Input failed before construction.
        let mut it = DisjunctionDocIterator::new(vec![block_iter(&[1, 2]), failed()]);
        assert_eq!(it.doc(), DOC_ITER_SENTINEL);
        assert!(matches!(it.error(), Err(Error::Coding(_))));

        // Input fails during iteration, before other inputs are exhausted.
        let mut it = DisjunctionDocIterator::new(vec![
            block_iter(&[1, 2, 5]),
            Box::new(FailingDocIterator(block_iter(&[3]))),
        ]);
        assert_eq!(it.doc(), 1);
        assert_eq!(it.advance(), 2);
        assert_eq!(it.advance(), 3);
        assert_eq!(it.advance(), DOC_ITER_SENTINEL);
        assert_eq!(it.advance(), DOC_ITER_SENTINEL);
        assert!(matches!(it.error(), Err(Error::Coding(_))));

        let mut it = DisjunctionDocIterator::new(vec![
            block_iter(&[1, 2, 5]),
            Box::new(FailingDocIterator(block_iter(&[3]))),
        ]);
        assert_eq!(it.advance_to(4), DOC_ITER_SENTINEL);
        assert!(matches!(it.error(), Err(Error::Coding(_))));
    }

    /// Wraps an iterator and reports an error once it is exhausted.
    struct FailingDocIterator(Box<dyn DocIterator>);

    impl DocIterator for FailingDocIterator {
        fn doc(&mut self) -> DocId {
            self.0.doc()
        }

        fn payload(&mut self) -> Payload {
            self.0.payload()
        }

        fn advance(&mut self) -> DocId {
            self.0.advance()
        }

        fn advance_to(&mut self, target: DocId) -> DocId {
            self.0.advance_to(target)
        }

        fn cost(&self) -> u64 {
            self.0.cost()
        }

        fn error(&self) -> Result<(), Error> {
            Err(Error::Coding("failed"))
        }
    }

    fn payload_iter(entries: &[(DocId, Payload)]) -> Box<dyn DocIterator> {
        Box::new(
            DocIteratorBlock::from_vec(build(PayloadFormat::LEB128, entries).serialize()).unwrap(),
        )
    }

    fn collect_entries(it: &mut dyn DocIterator) -> Vec<(DocId, Payload)> {
        let mut out = vec![];
        while it.doc() != DOC_ITER_SENTINEL {
            out.push((it.doc(), it.payload()));
            it.advance();
        }
        it.error().unwrap();
        out
    }

    const ASSIGNMENTS: &[(DocId, Payload)] = &[
        (1, 0),
        (2, 70),
        (3, 0),
        (5, 3),
        (6, 70),
        (8, 3),
        (9, 127),
        (10, 1),
    ];

    #[test]
    fn centroid_parent_filter() {
        let mut it = CentroidParentFilterDocIterator::new(
            block_iter(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            payload_iter(ASSIGNMENTS),
            127,
            128,
        );
        assert_eq!(it.cost(), 12);
        assert_eq!(
            collect_entries(&mut it),
            vec![(1, 0), (2, 70), (5, 3), (9, 127), (10, 1)]
        );
        assert_eq!(it.advance(), DOC_ITER_SENTINEL);
        assert_eq!(it.payload(), 0);
    }

    #[test]
    fn centroid_parent_filter_early_termination() {
        let input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        // num_centroids matches the number of distinct centroids in ASSIGNMENTS.
        let mut it = CentroidParentFilterDocIterator::new(
            block_iter(&input),
            payload_iter(ASSIGNMENTS),
            127,
            5,
        );
        assert_eq!(
            collect_entries(&mut it),
            vec![(1, 0), (2, 70), (5, 3), (9, 127), (10, 1)]
        );

        // Stops once num_centroids have been yielded.
        let mut it = CentroidParentFilterDocIterator::new(
            block_iter(&input),
            payload_iter(ASSIGNMENTS),
            127,
            2,
        );
        assert_eq!(collect_entries(&mut it), vec![(1, 0), (2, 70)]);
        assert_eq!(it.advance(), DOC_ITER_SENTINEL);
        assert_eq!(it.advance_to(100), DOC_ITER_SENTINEL);

        let mut it = CentroidParentFilterDocIterator::new(
            block_iter(&input),
            payload_iter(ASSIGNMENTS),
            127,
            0,
        );
        assert_eq!(it.doc(), DOC_ITER_SENTINEL);
        it.error().unwrap();
    }

    #[test]
    fn centroid_parent_filter_advance_to() {
        let mut it = CentroidParentFilterDocIterator::new(
            block_iter(&[1, 3, 4, 6, 8, 9, 10]),
            payload_iter(ASSIGNMENTS),
            127,
            128,
        );
        assert_eq!(it.doc(), 1);
        assert_eq!(it.advance_to(1), 1);
        // 4 is unassigned so the next doc is 6.
        assert_eq!(it.advance_to(4), 6);
        assert_eq!(it.payload(), 70);
        // Skipped docs do not mark their centroid as seen.
        assert_eq!(it.advance_to(7), 8);
        assert_eq!(it.payload(), 3);
        assert_eq!(it.advance_to(11), DOC_ITER_SENTINEL);
        assert_eq!(it.advance_to(100), DOC_ITER_SENTINEL);
        it.error().unwrap();
    }

    #[test]
    fn centroid_parent_filter_random() {
        use rand::prelude::*;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0x5eed);
        for _ in 0..100 {
            let max_centroid_id = rng.random_range(0..200u32);
            let mut assignments = vec![];
            for d in 0..500 {
                if rng.random_bool(0.7) {
                    assignments.push((d, rng.random_range(0..=max_centroid_id) as u64));
                }
            }
            let num_centroids = assignments
                .iter()
                .map(|&(_, c)| c)
                .collect::<std::collections::HashSet<_>>()
                .len() as u32;
            let input: Vec<DocId> = (0..500).filter(|_| rng.random_bool(0.5)).collect();
            let mut seen = std::collections::HashSet::new();
            let expected: Vec<(DocId, Payload)> = input
                .iter()
                .filter_map(|d| assignments.iter().find(|(a, _)| a == d).copied())
                .filter(|&(_, c)| seen.insert(c))
                .collect();
            let mut it = CentroidParentFilterDocIterator::new(
                block_iter(&input),
                payload_iter(&assignments),
                max_centroid_id,
                num_centroids,
            );
            assert_eq!(collect_entries(&mut it), expected);
        }
    }

    #[test]
    fn centroid_parent_filter_out_of_range() {
        let mut it = CentroidParentFilterDocIterator::new(
            block_iter(&[1, 2, 3]),
            payload_iter(&[(1, 0), (2, 64), (3, 1)]),
            63,
            64,
        );
        assert_eq!(it.doc(), 1);
        assert_eq!(it.advance(), DOC_ITER_SENTINEL);
        assert_eq!(it.advance(), DOC_ITER_SENTINEL);
        assert!(matches!(it.error(), Err(Error::Coding(_))));
    }
}
