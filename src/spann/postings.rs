//! Block-based posting storage for SPANN.

use std::collections::{
    hash_map::Entry::{Occupied, Vacant},
    HashMap,
};

use wt_mdb::{Error, Result, TypedCursorGuard, WT_MODIFY};

use super::TransactionIndex;
use crate::posting_block::{PostingBlock, PostingBlockMut};

/// Posting mutations backed by a centroid-keyed [`PostingBlock`] table.
///
/// Each centroid maps to a single row whose value is a serialized [`PostingBlock`] containing all
/// vectors assigned to that centroid. Modified blocks are held in memory and written to storage
/// only when [`flush`] is called.
pub struct BlockPostingsMut<'a> {
    // TODO: consider RefCell/inner mutability for the cursor since the cursor makes this struct
    // Send + !Sync anyway.
    cursor: TypedCursorGuard<'a, u32, Vec<u8>>,
    vector_len: usize,
    dirty: HashMap<u32, PostingBlockMut>,
}

impl<'a> BlockPostingsMut<'a> {
    pub fn new(cursor: TypedCursorGuard<'a, u32, Vec<u8>>, vector_len: usize) -> Self {
        Self {
            cursor,
            vector_len,
            dirty: HashMap::new(),
        }
    }

    pub fn from_txn(txn_idx: &'a TransactionIndex) -> Result<Self> {
        txn_idx
            .transaction()
            .open_cursor::<u32, Vec<u8>>(txn_idx.index().postings_table_name())
            .map(|c| Self::new(c, txn_idx.index().posting_vector_len()))
    }

    /// Return the given vector or a NotFound error.
    pub fn get(&mut self, centroid_id: u32, record_id: i64) -> Result<Vec<u8>> {
        let block = self.load_or_create(centroid_id)?;
        block
            .lookup(record_id)
            .map(|v| v.to_vec())
            .ok_or(Error::not_found_error())
    }

    /// Insert or update the posting for `(centroid_id, record_id)`.
    pub fn insert(&mut self, centroid_id: u32, record_id: i64, vector: &[u8]) -> Result<()> {
        self.load_or_create(centroid_id)?
            .insert(record_id, vector.to_vec());
        Ok(())
    }

    /// Remove the posting for `(centroid_id, record_id)`.
    ///
    /// Not-found is silently ignored.
    pub fn remove(&mut self, centroid_id: u32, record_id: i64) -> Result<()> {
        self.load_or_create(centroid_id)?.remove(record_id);
        Ok(())
    }

    /// Return the number of vectors in centroid_id postings.
    ///
    /// This will return 0 if the centroid was not found.
    pub fn centroid_len(&mut self, centroid_id: u32) -> Result<usize> {
        self.load_or_create(centroid_id).map(|b| b.len())
    }

    /// Read all postings for `centroid_id` without removing them.
    ///
    /// Reflects any pending mutations not yet flushed.
    pub fn read_centroid(&mut self, centroid_id: u32) -> Result<Vec<(i64, Vec<u8>)>> {
        Ok(self
            .load_or_create(centroid_id)?
            .iter()
            .map(|(id, v)| (id, v.to_vec()))
            .collect())
    }

    /// Remove all postings for `centroid_id`.
    ///
    /// Not-found is silently ignored.
    pub fn remove_centroid(&mut self, centroid_id: u32) -> Result<()> {
        // Overwrite with an empty block; flush will delete the row from storage.
        self.dirty
            .insert(centroid_id, PostingBlockMut::new(self.vector_len));
        Ok(())
    }

    /// Return the next allocatable centroid id.
    pub fn next_centroid_id(&mut self) -> Result<u32> {
        Ok(self
            .cursor
            .largest_key()
            .transpose()?
            .map(|x| x + 1)
            .unwrap_or(0))
    }

    /// Write all buffered changes to storage.
    pub fn flush(&mut self) -> Result<()> {
        let dirty = std::mem::take(&mut self.dirty);
        let mut modify_buf = [WT_MODIFY::default(); 128];
        for (centroid_id, block) in dirty {
            if block.is_empty() {
                self.cursor.remove(centroid_id).or_else(|e| {
                    if e == Error::not_found_error() {
                        Ok(())
                    } else {
                        Err(e)
                    }
                })?;
                continue;
            }
            let new_serialized = block.serialize();
            let base = block.base_block();
            if !base.is_empty() {
                let max_diff = base.len() * 15 / 100;
                if let Some(deltas) = self.cursor.calculate_modifications(
                    base,
                    &new_serialized,
                    max_diff,
                    &mut modify_buf,
                ) {
                    // SAFETY: modify_buf[..n] holds WT_MODIFY entries whose data pointers
                    // point into new_serialized, which remains alive for this call.
                    unsafe { self.cursor.modify_unsafe(centroid_id, deltas)? };
                    continue;
                }
            }
            self.cursor.set(centroid_id, new_serialized.as_slice())?;
        }
        Ok(())
    }

    fn load_or_create(&mut self, centroid_id: u32) -> Result<&mut PostingBlockMut> {
        match self.dirty.entry(centroid_id) {
            Occupied(e) => Ok(e.into_mut()),
            Vacant(e) => {
                let block = match self.cursor.seek_exact(centroid_id) {
                    Some(r) => {
                        let data = r?;
                        let pb = PostingBlock::new(&data, self.vector_len)
                            .ok_or(Error::wired_tiger(wt_mdb::WiredTigerError::Generic))?;
                        PostingBlockMut::from_block(&pb)
                    }
                    None => PostingBlockMut::new(self.vector_len),
                };
                Ok(e.insert(block))
            }
        }
    }
}
