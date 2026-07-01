//! Block-based posting storage for SPANN.

use std::collections::{
    hash_map::Entry::{Occupied, Vacant},
    HashMap,
};

use wt_mdb::{Error, Result, TypedCursorGuard};

use crate::posting_block::{PostingBlock, PostingBlockMut};

/// Posting mutations backed by a centroid-keyed [`PostingBlock`] table.
///
/// Each centroid maps to a single row whose value is a serialized [`PostingBlock`] containing all
/// vectors assigned to that centroid. Modified blocks are held in memory and written to storage
/// only when [`flush`] is called.
pub struct BlockPostingsMut<'a> {
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

    /// Remove all postings for `centroid_id`.
    ///
    /// Not-found is silently ignored.
    pub fn remove_centroid(&mut self, centroid_id: u32) -> Result<()> {
        // Overwrite with an empty block; flush will delete the row from storage.
        self.dirty
            .insert(centroid_id, PostingBlockMut::new(self.vector_len));
        Ok(())
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

    /// Write all buffered changes to storage.
    pub fn flush(&mut self) -> Result<()> {
        for (centroid_id, block) in self.dirty.drain() {
            if block.is_empty() {
                self.cursor.remove(centroid_id).or_else(|e| {
                    if e == Error::not_found_error() {
                        Ok(())
                    } else {
                        Err(e)
                    }
                })?;
            } else {
                self.cursor.set(centroid_id, block.serialize().as_slice())?;
            }
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
                            .ok_or_else(|| Error::WiredTiger(wt_mdb::WiredTigerError::Generic))?;
                        PostingBlockMut::from_block(&pb)
                    }
                    None => PostingBlockMut::new(self.vector_len),
                };
                Ok(e.insert(block))
            }
        }
    }
}
