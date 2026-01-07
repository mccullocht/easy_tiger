use std::collections::{hash_map::Entry, HashMap};
use tracing::error;

use wt_mdb::{
    session::{FormatString, Formatted},
    Error, Result, Session, TypedCursorGuard,
};

use crate::spann::{
    CentroidAssignment, CentroidAssignmentRef, CentroidAssignmentType, PostingKey, TableIndex,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct CentroidCounts {
    /// Number of vectors whose primary assigned centroid is this centroid.
    pub primary: u32,
    /// Number of vectors whose secondary+ assigned centroid is this centroid.
    pub secondary: u32,
}

impl CentroidCounts {
    /// Sum of primary and secondary assignments.
    pub fn total(&self) -> u32 {
        self.primary + self.secondary
    }
}

impl Formatted for CentroidCounts {
    const FORMAT: FormatString = FormatString::new(c"u");

    type Ref<'a> = Self;

    fn to_formatted_ref(&self) -> Self::Ref<'_> {
        *self
    }

    fn pack(value: Self::Ref<'_>, packed: &mut Vec<u8>) -> Result<()> {
        packed.resize(8, 0);
        let packed_entries = packed.as_chunks_mut::<4>().0;
        packed_entries[0] = value.primary.to_le_bytes();
        packed_entries[1] = value.secondary.to_le_bytes();
        Ok(())
    }

    fn unpack<'b>(packed: &'b [u8]) -> Result<Self::Ref<'b>> {
        Ok(Self {
            primary: u32::from_le_bytes(packed[0..4].try_into().unwrap()),
            secondary: u32::from_le_bytes(packed[4..8].try_into().unwrap()),
        })
    }
}

/// Tracks occupancy statistics for centroids.
#[derive(Debug, Clone)]
pub struct CentroidStats(Vec<Option<CentroidCounts>>);

impl CentroidStats {
    /// Compute centroid statistics from per-vector assignments.
    ///
    /// This does not validate the assignments themselves to avoid reading posting data, but it does
    /// require a full scan of the centroids table. Callers should prefer `from_index_stats`.
    ///
    /// Callers should open a transaction for this sequence of reads.
    pub fn from_centroid_assignments(session: &Session, index: &TableIndex) -> Result<Self> {
        let mut head_cursor = session.get_record_cursor(index.head.graph_table_name())?;
        let centroid_len = head_cursor.largest_key().unwrap_or(Ok(0))? as usize + 1;

        let mut counts: Vec<Option<CentroidCounts>> = vec![None; centroid_len];
        let centroid_cursor =
            session.get_or_create_typed_cursor::<i64, Vec<u8>>(&index.table_names.centroids)?;
        for r in centroid_cursor {
            let centroid_bytes = r.map(|(_, c)| c)?;
            let mut centroid_it = centroid_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b));
            if let Some(primary_id) = centroid_it.next() {
                counts[primary_id as usize].get_or_insert_default().primary += 1;
                for secondary_id in centroid_it {
                    counts[secondary_id as usize]
                        .get_or_insert_default()
                        .secondary += 1;
                }
            }
        }

        // Some centroids may have no assignments; ensure they are represented in case the caller is
        // interested in re-clustering.
        for (i, c) in counts.iter_mut().enumerate() {
            if c.is_none() && head_cursor.seek_exact(i as i64).is_some() {
                *c = Some(CentroidCounts::default());
            }
        }

        Ok(Self(counts))
    }

    /// Read centroid assignments from the pre-computed stats table.
    ///
    /// This does not validate the assignments in the centroids or postings tables to avoid reading
    /// a large amount of data.
    ///
    /// Callers should open a transaction for this sequence of reads.
    pub fn from_index_stats(session: &Session, index: &TableIndex) -> Result<Self> {
        let mut cursor = session
            .get_or_create_typed_cursor::<u32, CentroidCounts>(&index.table_names.centroid_stats)?;
        let centroid_len = cursor.largest_key().unwrap_or(Ok(0))? as usize + 1;
        let mut counts = vec![None; centroid_len];
        for r in cursor {
            let (id, cc) = r?;
            counts[id as usize] = Some(cc);
        }
        Ok(Self(counts))
    }

    /// Return the number of centroids in the index.
    pub fn centroid_count(&self) -> usize {
        self.counts_iter().count()
    }

    /// Return the number of vector -> centroid assignments.
    pub fn vector_count(&self) -> usize {
        self.assignment_counts_iter()
            .map(|(_, c)| c as usize)
            .sum::<usize>()
    }

    /// Iterate over a list of centroid identifiers and the number of assigned vectors for each.
    pub fn assignment_counts_iter(&self) -> impl Iterator<Item = (usize, u32)> + '_ {
        self.counts_iter().map(|(i, c)| (i, c.total()))
    }

    /// Get the assignment counts for a specific centroid or `None` if the centroid does not exist.
    pub fn assignment_counts(&self, centroid_id: usize) -> Option<CentroidCounts> {
        *self.0.get(centroid_id).unwrap_or(&None)
    }

    /// Iterate over a list of centroid identifiers and the number of primary assigned vectors for each.
    pub fn primary_assignment_counts_iter(&self) -> impl Iterator<Item = (usize, u32)> + '_ {
        self.counts_iter().map(|(i, c)| (i, c.primary))
    }

    /// Iterate over a list of centroid identifiers and the number of secondary assigned vectors for each.
    pub fn secondary_assignment_counts_iter(&self) -> impl Iterator<Item = (usize, u32)> + '_ {
        self.counts_iter().map(|(i, c)| (i, c.secondary))
    }

    /// Iterate over counts for all centroids with at least one assignment.
    fn counts_iter(&self) -> impl Iterator<Item = (usize, CentroidCounts)> + '_ {
        self.0
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.as_ref().map(|counts| (i, *counts)))
    }

    /// Iterate over available centroid ids. The returned iterator is effectively unbounded (up to
    /// `u32::MAX`) so callers should `take()` this iterator to mint the number of ids they need.
    pub fn available_centroid_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.0
            .iter()
            .enumerate()
            .filter(|(_, c)| c.is_none())
            .map(|(i, _)| i)
            .chain(self.0.len()..)
    }
}

pub struct CentroidAssignmentUpdater<'a> {
    assignments_cursor: TypedCursorGuard<'a, i64, CentroidAssignment>,
    stats: CentroidStatsCache<'a>,
    replica_count: usize,
}

impl<'a> CentroidAssignmentUpdater<'a> {
    /// Create a new updater for `index` using `session` for WT access.
    pub fn new(index: &TableIndex, session: &'a Session) -> Result<Self> {
        let assignments_cursor = session
            .get_or_create_typed_cursor::<i64, CentroidAssignment>(&index.table_names.centroids)?;
        let stats =
            CentroidStatsCache::new(session.get_or_create_typed_cursor::<u32, CentroidCounts>(
                &index.table_names.centroid_stats,
            )?);
        Ok(Self {
            assignments_cursor,
            stats,
            replica_count: index.config().replica_count,
        })
    }

    /// Returns true if `key` is for the primary assigned for `record_id`.
    pub fn is_primary(&mut self, key: PostingKey) -> Result<bool> {
        if self.replica_count <= 1 {
            return Ok(true);
        } else {
            self.assignments_cursor
                .seek_exact(key.record_id)
                .map(|r| r.map(|a| a.primary_id == key.centroid_id))
                .unwrap_or(Ok(false))
        }
    }

    /// Read the assignments for `record_id`.
    pub fn read(&mut self, record_id: i64) -> Option<Result<CentroidAssignment>> {
        self.assignments_cursor.seek_exact(record_id)
    }

    /// Insert centroid assignments for a new record.
    ///
    /// Returns a DuplicateKey error if the record already exists.
    pub fn insert(&mut self, record_id: i64, assignments: CentroidAssignmentRef<'_>) -> Result<()> {
        if let Some(_existing_assignment) = self.assignments_cursor.seek_exact(record_id) {
            error!("attempted to insert duplicate record id {}", record_id);
            return Err(Error::WiredTiger(wt_mdb::WiredTigerError::DuplicateKey));
        }
        self.assignments_cursor.set(record_id, assignments)?;
        for (assignment_type, centroid_id) in assignments.iter() {
            self.stats.increment_count(centroid_id, assignment_type)?;
        }
        Ok(())
    }

    /// Delete centroid assignments for an existing record.
    ///
    /// Returns a NotFound error if the record does not exist.
    pub fn delete(&mut self, record_id: i64) -> Result<CentroidAssignment> {
        let existing_assignment = self
            .assignments_cursor
            .seek_exact(record_id)
            .unwrap_or(Err(Error::not_found_error()))?;
        for (assignment_type, centroid_id) in existing_assignment.iter() {
            self.stats.decrement_count(centroid_id, assignment_type)?;
        }
        self.assignments_cursor.remove(record_id)?;
        Ok(existing_assignment)
    }

    /// Update centroid assignments for an existing record, returning the previous assignments.
    ///
    /// Returns a NotFound error if the record does not exist.
    pub fn update(
        &mut self,
        record_id: i64,
        assignments: CentroidAssignmentRef<'_>,
    ) -> Result<CentroidAssignment> {
        let existing_assignment = self
            .assignments_cursor
            .seek_exact(record_id)
            .unwrap_or(Err(Error::not_found_error()))?;
        for (assignment_type, centroid_id) in existing_assignment.iter() {
            self.stats.decrement_count(centroid_id, assignment_type)?;
        }
        for (assignment_type, centroid_id) in assignments.iter() {
            self.stats.increment_count(centroid_id, assignment_type)?;
        }
        self.assignments_cursor
            .set(record_id, assignments)
            .map(|()| existing_assignment)
    }

    /// Flush buffered stats updates back to the database.
    ///
    /// Dropping the updater will also flush any buffered updates, but prefer this method to allow
    /// observation of any errors.
    pub fn flush(&mut self) -> Result<()> {
        self.stats.flush()
    }
}

impl Drop for CentroidAssignmentUpdater<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.flush() {
            error!("Error flushing centroid assignment stats: {e}")
        }
    }
}

struct CentroidStatsCache<'a> {
    cursor: TypedCursorGuard<'a, u32, CentroidCounts>,
    cache: HashMap<u32, CentroidCounts>,
}

impl<'a> CentroidStatsCache<'a> {
    fn new(cursor: TypedCursorGuard<'a, u32, CentroidCounts>) -> Self {
        Self {
            cursor,
            cache: HashMap::new(),
        }
    }

    fn increment_count(&mut self, id: u32, ctype: CentroidAssignmentType) -> Result<()> {
        let counts = self.get_counts(id)?;
        match ctype {
            CentroidAssignmentType::Primary => counts.primary += 1,
            CentroidAssignmentType::Secondary => counts.secondary += 1,
        };
        Ok(())
    }

    fn decrement_count(&mut self, id: u32, ctype: CentroidAssignmentType) -> Result<()> {
        let counts = self.get_counts(id)?;
        match ctype {
            CentroidAssignmentType::Primary => counts.primary -= 1,
            CentroidAssignmentType::Secondary => counts.secondary -= 1,
        };
        Ok(())
    }

    fn get_counts(&mut self, id: u32) -> Result<&'_ mut CentroidCounts> {
        let e = match self.cache.entry(id) {
            Entry::Occupied(e) => e,
            Entry::Vacant(e) => {
                let cursor_counts = self
                    .cursor
                    .seek_exact(id)
                    .unwrap_or(Ok(CentroidCounts::default()))?;
                e.insert_entry(cursor_counts)
            }
        };
        Ok(e.into_mut())
    }

    fn flush(&mut self) -> Result<()> {
        for (id, counts) in self.cache.drain() {
            self.cursor.set(id, counts)?;
        }
        Ok(())
    }
}
