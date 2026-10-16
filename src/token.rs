//! Abstraction for token indexing.

use std::io;

// XXX impl Error
pub enum Error {
    UnknownDocFormat(u8),
    UnknownPayloadFormat(u8),
    Coding(&'static str),
    Io(io::Error),
}

const DOC_ITER_SENTINEL: i64 = i64::MAX;

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

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
enum DocFormat {
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

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
enum PayloadFormat {
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

// XXX consider block format:
// * number of entries
// * doc format
// * payload format
// * <first_docid>
// * <docs len if payload format is not None>
// * <docs>
// * <payloads>
