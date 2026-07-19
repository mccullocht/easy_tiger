//! Block-based posting storage for SPANN.

use std::{
    borrow::Cow,
    collections::{
        hash_map::Entry::{Occupied, Vacant},
        HashMap,
    },
};

use wt_mdb::{Error, Result, TypedCursorGuard, WT_MODIFY};

use super::TransactionIndex;
use crate::{posting_block::{PostingBlock, PostingBlockMut}, spann::CentroidCounts};

/// Maximum byte difference in a block to modify instead of full replacement.
const MAX_DIFF_PCT: usize = 15;

enum BlockState {
    /// Block not found in the underlying repos.
    NotFound,
    /// Block removed by the caller.
    Removed,
    /// Block found. Caller may have mutated block content.
    Found(PostingBlockMut),
}

impl BlockState {
    fn to_block_mut(&mut self, vector_len: usize) -> &mut PostingBlockMut {
        match self {
            BlockState::NotFound | BlockState::Removed => {
                *self = BlockState::Found(PostingBlockMut::new(vector_len))
            }
            _ => {}
        };
        if let BlockState::Found(b) = self {
            b
        } else {
            unreachable!("coerced to found")
        }
    }
}

/// Posting mutations backed by a centroid-keyed [`PostingBlock`] table.
///
/// Each centroid maps to a single row whose value is a serialized [`PostingBlock`] containing all
/// vectors assigned to that centroid. Modified blocks are held in memory and written to storage
/// only when [`flush`] is called.
pub struct CentroidPostingsMut<'a> {
    posting_cursor: TypedCursorGuard<'a, u32, Vec<u8>>,
    stats_cursor: TypedCursorGuard<'a, u32, CentroidCounts>,
    vector_len: usize,
    blocks: HashMap<u32, BlockState>,
}

impl<'a> CentroidPostingsMut<'a> {
    pub fn new(
        posting_cursor: TypedCursorGuard<'a, u32, Vec<u8>>,
        stats_cursor: TypedCursorGuard<'a, u32, CentroidCounts>,
        vector_len: usize,
    ) -> Self {
        Self {
            posting_cursor,
            stats_cursor,
            vector_len,
            blocks: HashMap::new(),
        }
    }

    pub fn from_txn(txn_idx: &'a TransactionIndex) -> Result<Self> {
        let posting_cursor = txn_idx
            .transaction()
            .open_cursor::<u32, Vec<u8>>(txn_idx.index().postings_table_name())?;
        let stats_cursor = txn_idx
            .transaction()
            .open_cursor::<u32, CentroidCounts>(&txn_idx.index().table_names.centroid_stats)?;
        Ok(Self::new(posting_cursor, stats_cursor, txn_idx.index().posting_vector_len()))
    }

    /// Return the given vector or a NotFound error.
    pub fn get(&mut self, centroid_id: u32, record_id: i64) -> Result<Vec<u8>> {
        if let BlockState::Found(b) = self.get_or_fetch(centroid_id)? {
            b.lookup(record_id)
                .map(|v| v.to_vec())
                .ok_or_else(Error::not_found_error)
        } else {
            Err(Error::not_found_error())
        }
    }

    /// Insert or update the posting for `(centroid_id, record_id)`.
    pub fn insert(&mut self, centroid_id: u32, record_id: i64, vector: &[u8]) -> Result<()> {
        self.get_or_fetch_block_mut(centroid_id)
            .map(|b| b.insert(record_id, vector.to_vec()))
    }

    /// Remove the posting for `(centroid_id, record_id)`.
    ///
    /// Returns the posting vector or `None` if the centroid/record is not found.
    pub fn remove(&mut self, centroid_id: u32, record_id: i64) -> Result<Option<Cow<'_, [u8]>>> {
        match self.get_or_fetch(centroid_id)? {
            BlockState::NotFound | BlockState::Removed => Ok(None),
            BlockState::Found(b) => Ok(b.remove(record_id)),
        }
    }

    /// Return the number of vectors in centroid_id postings.
    ///
    /// This will return 0 if the centroid was not found.
    pub fn centroid_len(&mut self, centroid_id: u32) -> Result<usize> {
        match self.get_or_fetch(centroid_id)? {
            BlockState::Found(b) => Ok(b.len()),
            _ => Ok(0),
        }
    }

    /// Read all postings for `centroid_id`, returning NotFound error if the posting block doesn't
    /// exist.
    ///
    /// Reflects any pending mutations not yet flushed, so may return NotFound after
    /// remove_centroid().
    pub fn read_centroid(&mut self, centroid_id: u32) -> Result<Vec<(i64, Vec<u8>)>> {
        match self.get_or_fetch(centroid_id)? {
            BlockState::Found(b) => Ok(b.iter().map(|(id, v)| (id, v.to_vec())).collect()),
            _ => Err(Error::not_found_error()),
        }
    }

    /// Remove all postings for `centroid_id`.
    ///
    /// If centroid_id is not found this operation is silently ignored.
    pub fn remove_centroid(&mut self, centroid_id: u32) -> Result<()> {
        match self.get_or_fetch(centroid_id)? {
            BlockState::NotFound => {}
            bs => *bs = BlockState::Removed,
        };
        Ok(())
    }

    /// Return the next allocatable centroid id.
    pub fn next_centroid_id(&mut self) -> Result<u32> {
        Ok(self
            .posting_cursor
            .largest_key()
            .transpose()?
            .map(|x| x + 1)
            .unwrap_or(0))
    }

    /// Write all buffered changes to storage.
    // XXX this should accept 'mut self'
    pub fn flush(&mut self) -> Result<()> {
        let mut modify_buf = [WT_MODIFY::default(); 128];
        for (centroid_id, bs) in std::mem::take(&mut self.blocks) {
            match bs {
                BlockState::NotFound => continue,
                BlockState::Removed => {
                    self.posting_cursor.remove(centroid_id).or_else(|e| {
                        if e == Error::not_found_error() {
                            Ok(())
                        } else {
                            Err(e)
                        }
                    })?;
                    // Update stats: remove the centroid from stats table
                    self.stats_cursor.remove(centroid_id).or_else(|e| {
                        if e == Error::not_found_error() {
                            Ok(())
                        } else {
                            Err(e)
                        }
                    })?;
                }
                BlockState::Found(block) => {
                    let new_serialized = block.serialize();
                    let base = block.base_block();
                    if !base.is_empty() {
                        let max_diff = base.len() * MAX_DIFF_PCT / 100;
                        if let Some(deltas) = self.posting_cursor.calculate_modifications(
                            base,
                            &new_serialized,
                            max_diff,
                            &mut modify_buf,
                        ) {
                            // SAFETY: modify_buf[..n] holds WT_MODIFY entries whose data pointers
                            // point into new_serialized, which remains alive for this call.
                            unsafe { self.posting_cursor.modify_unsafe(centroid_id, deltas)? };
                            continue;
                        }
                    }
                    self.posting_cursor.set(centroid_id, new_serialized.as_slice())?;
                    // Update stats: write the new count of vectors for this centroid
                    let count = CentroidCounts {
                        primary: block.len() as u32,
                    };
                    self.stats_cursor.set(centroid_id, count)?;
                }
            }
        }
        Ok(())
    }

    fn get_or_fetch(&mut self, centroid_id: u32) -> Result<&mut BlockState> {
        match self.blocks.entry(centroid_id) {
            Occupied(e) => Ok(e.into_mut()),
            Vacant(e) => {
                // SAFETY: will perform no other cursor operations until this value is copied.
                let block = match unsafe { self.posting_cursor.seek_exact_unsafe(centroid_id) } {
                    None => BlockState::NotFound,
                    Some(r) => {
                        let raw_data = r?;
                        let pb = PostingBlock::new(raw_data, self.vector_len)
                            .ok_or(Error::wired_tiger(wt_mdb::WiredTigerError::Generic))?;
                        BlockState::Found(PostingBlockMut::from_block(&pb))
                    }
                };
                Ok(e.insert(block))
            }
        }
    }

    fn get_or_fetch_block_mut(&mut self, centroid_id: u32) -> Result<&mut PostingBlockMut> {
        let vector_len = self.vector_len;
        self.get_or_fetch(centroid_id)
            .map(|bs| bs.to_block_mut(vector_len))
    }
}
