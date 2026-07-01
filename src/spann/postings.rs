//! Trait and implementations for mutating SPANN postings.

use std::collections::HashMap;

use wt_mdb::{Error, Result, TypedCursorGuard};

use crate::posting_block::{PostingBlock, PostingBlockMut};

use super::PostingKey;

/// Interface for applying batched mutations to SPANN postings.
///
/// Callers issue any number of inserts, removals, and drains then call [`PostingsMut::flush`] once
/// to commit all pending changes to storage.
pub trait PostingsMut {
    /// Insert or update the posting for `(centroid_id, record_id)`.
    fn insert(&mut self, centroid_id: u32, record_id: i64, vector: &[u8]) -> Result<()>;

    /// Remove the posting for `(centroid_id, record_id)`.
    ///
    /// Not-found is silently ignored.
    fn remove(&mut self, centroid_id: u32, record_id: i64) -> Result<()>;

    /// Remove all postings for `centroid_id` and return them as `(record_id, vector)` pairs.
    fn drain(&mut self, centroid_id: u32) -> Result<Vec<(i64, Vec<u8>)>>;

    /// Read all postings for `centroid_id` without removing them.
    ///
    /// This is transactional -- it reflects all mutations made to each centroid.
    fn read_centroid(&mut self, centroid_id: u32) -> Result<Vec<(i64, Vec<u8>)>>;

    /// Write all buffered changes to storage.
    fn flush(&mut self) -> Result<()>;
}

/// Posting mutations backed by the row-per-vector table format.
///
/// Each `(centroid_id, record_id)` pair occupies its own WiredTiger row. Mutations go to storage
/// immediately; [`flush`][PostingsMut::flush] is a no-op.
pub struct RowPostingsMut<'a> {
    cursor: TypedCursorGuard<'a, PostingKey, Vec<u8>>,
}

impl<'a> RowPostingsMut<'a> {
    pub fn new(cursor: TypedCursorGuard<'a, PostingKey, Vec<u8>>) -> Self {
        Self { cursor }
    }
}

impl PostingsMut for RowPostingsMut<'_> {
    fn insert(&mut self, centroid_id: u32, record_id: i64, vector: &[u8]) -> Result<()> {
        self.cursor.set(
            PostingKey {
                centroid_id,
                record_id,
            },
            vector,
        )
    }

    fn remove(&mut self, centroid_id: u32, record_id: i64) -> Result<()> {
        self.cursor
            .remove(PostingKey {
                centroid_id,
                record_id,
            })
            .or_else(|e| {
                if e == Error::not_found_error() {
                    Ok(())
                } else {
                    Err(e)
                }
            })
    }

    fn drain(&mut self, centroid_id: u32) -> Result<Vec<(i64, Vec<u8>)>> {
        let mut vectors = vec![];
        self.cursor
            .set_bounds(PostingKey::centroid_range(centroid_id))?;
        while let Some(r) = self.cursor.next() {
            let (key, vector) = r?;
            vectors.push((key.record_id, vector));
            self.cursor.remove(key)?;
        }
        self.cursor.reset()?;
        Ok(vectors)
    }

    fn read_centroid(&mut self, centroid_id: u32) -> Result<Vec<(i64, Vec<u8>)>> {
        let mut entries = vec![];
        self.cursor
            .set_bounds(PostingKey::centroid_range(centroid_id))?;
        while let Some(r) = self.cursor.next() {
            let (key, vector) = r?;
            entries.push((key.record_id, vector));
        }
        self.cursor.reset()?;
        Ok(entries)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Posting mutations backed by a centroid-keyed [`PostingBlock`] table.
///
/// Each centroid maps to a single row whose value is a serialized [`PostingBlock`] containing all
/// vectors assigned to that centroid. Modified blocks are held in memory and written to storage
/// only when [`flush`][PostingsMut::flush] is called.
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

    fn load_or_create(&mut self, centroid_id: u32) -> Result<&mut PostingBlockMut> {
        if !self.dirty.contains_key(&centroid_id) {
            let block = match self.cursor.seek_exact(centroid_id) {
                Some(r) => {
                    let data = r?;
                    let pb = PostingBlock::new(&data, self.vector_len)
                        .ok_or_else(|| Error::WiredTiger(wt_mdb::WiredTigerError::Generic))?;
                    PostingBlockMut::from_block(&pb)
                }
                None => PostingBlockMut::new(self.vector_len),
            };
            self.dirty.insert(centroid_id, block);
        }
        Ok(self.dirty.get_mut(&centroid_id).unwrap())
    }
}

impl PostingsMut for BlockPostingsMut<'_> {
    fn insert(&mut self, centroid_id: u32, record_id: i64, vector: &[u8]) -> Result<()> {
        self.load_or_create(centroid_id)?
            .insert(record_id, vector.to_vec());
        Ok(())
    }

    fn remove(&mut self, centroid_id: u32, record_id: i64) -> Result<()> {
        self.load_or_create(centroid_id)?.remove(record_id);
        Ok(())
    }

    fn drain(&mut self, centroid_id: u32) -> Result<Vec<(i64, Vec<u8>)>> {
        let vector_len = self.vector_len;
        let block = self.load_or_create(centroid_id)?;
        let entries = block.iter().map(|(id, v)| (id, v.to_vec())).collect();
        *block = PostingBlockMut::new(vector_len);
        Ok(entries)
    }

    fn read_centroid(&mut self, centroid_id: u32) -> Result<Vec<(i64, Vec<u8>)>> {
        Ok(self
            .load_or_create(centroid_id)?
            .iter()
            .map(|(id, v)| (id, v.to_vec()))
            .collect())
    }

    fn flush(&mut self) -> Result<()> {
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
}
