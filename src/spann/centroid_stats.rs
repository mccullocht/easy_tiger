use std::collections::{HashMap, hash_map::Entry};
use tracing::error;

use wt_mdb::{
    Error, Result, Transaction, TypedCursorGuard,
    session::{FormatString, Formatted},
};

use crate::spann::{CentroidAssignment, TableIndex, TransactionIndex};

#[derive(Debug, Default, Clone, Copy)]
pub struct CentroidCounts {
    /// Number of vectors assigned to this centroid.
    pub primary: u32,
}

impl CentroidCounts {
    pub fn total(&self) -> u32 {
        self.primary
    }
}

impl Formatted for CentroidCounts {
    const FORMAT: FormatString = FormatString::new(c"u");

    type Ref<'a> = Self;

    fn to_formatted_ref(&self) -> Self::Ref<'_> {
        *self
    }

    fn pack(value: Self::Ref<'_>, packed: &mut Vec<u8>) -> Result<()> {
        packed.resize(4, 0);
        packed.copy_from_slice(&value.primary.to_le_bytes());
        Ok(())
    }

    fn unpack<'b>(packed: &'b [u8]) -> Result<Self::Ref<'b>> {
        Ok(Self {
            primary: u32::from_le_bytes(packed[0..4].try_into().unwrap()),
        })
    }
}

/// Tracks occupancy statistics for centroids.
#[derive(Debug, Clone)]
pub struct CentroidStats {
    counts: HashMap<u32, CentroidCounts>,
    max_centroid_id: u32,
}

impl CentroidStats {
    /// Compute centroid statistics from per-vector assignments.
    ///
    /// This does not validate the assignments themselves to avoid reading posting data, but it does
    /// require a full scan of the centroids table. Callers should prefer `from_index_stats`.
    ///
    /// Callers should open a transaction for this sequence of reads.
    pub fn from_centroid_assignments(txn: &Transaction, index: &TableIndex) -> Result<Self> {
        let head_cursor = txn.open_record_cursor(index.head.graph_table_name())?;
        let mut counts: HashMap<u32, CentroidCounts> = HashMap::new();
        let mut max_centroid_id = 0u32;

        let centroid_cursor =
            txn.open_cursor::<i64, CentroidAssignment>(&index.table_names.centroids)?;
        for r in centroid_cursor {
            let (_, a) = r?;
            let entry = counts
                .entry(a.primary_id)
                .or_insert_with(CentroidCounts::default);
            entry.primary += 1;
            max_centroid_id = max_centroid_id.max(a.primary_id);
        }

        // Some centroids may have no assignments; ensure they are represented in case the caller is
        // interested in re-clustering.
        for r in head_cursor {
            let (id, _) = r?;
            if !counts.contains_key(&(id as u32)) {
                counts.insert(id as u32, CentroidCounts::default());
            }
        }

        Ok(Self {
            counts,
            max_centroid_id,
        })
    }

    /// Read centroid assignments from the pre-computed stats table.
    ///
    /// This does not validate the assignments in the centroids or postings tables to avoid reading
    /// a large amount of data.
    ///
    /// Callers should open a transaction for this sequence of reads.
    pub fn from_index_stats(txn_idx: &TransactionIndex) -> Result<Self> {
        let cursor = txn_idx
            .transaction()
            .open_cursor::<u32, CentroidCounts>(&txn_idx.index().table_names.centroid_stats)?;
        let mut counts: HashMap<u32, CentroidCounts> = HashMap::new();
        let mut max_centroid_id = 0u32;

        for r in cursor {
            let (id, cc) = r?;
            counts.insert(id, cc);
            max_centroid_id = max_centroid_id.max(id);
        }
        Ok(Self {
            counts,
            max_centroid_id,
        })
    }

    /// Return the number of centroids in the index.
    pub fn centroid_count(&self) -> usize {
        self.counts.values().filter(|c| c.total() > 0).count()
    }

    /// Return the number of vector -> centroid assignments.
    pub fn vector_count(&self) -> usize {
        self.counts
            .values()
            .map(|c| c.total() as usize)
            .sum::<usize>()
    }

    /// Iterate over a list of centroid identifiers and the number of assigned vectors for each.
    pub fn assignment_counts_iter(&self) -> impl Iterator<Item = (usize, u32)> + '_ {
        self.counts.iter().filter_map(|(&id, counts)| {
            if counts.total() > 0 {
                Some((id as usize, counts.total()))
            } else {
                None
            }
        })
    }

    /// Get the assignment counts for a specific centroid or `None` if the centroid does not exist.
    pub fn assignment_counts(&self, centroid_id: usize) -> Option<CentroidCounts> {
        self.counts.get(&(centroid_id as u32)).copied()
    }

    /// Iterate over a list of centroid identifiers and the number of assigned vectors for each.
    pub fn primary_assignment_counts_iter(&self) -> impl Iterator<Item = (usize, u32)> + '_ {
        self.counts.iter().filter_map(|(&id, counts)| {
            if counts.primary > 0 {
                Some((id as usize, counts.primary))
            } else {
                None
            }
        })
    }

    /// Iterate over available centroid ids. The returned iterator is effectively unbounded (up to
    /// `u32::MAX`) so callers should `take()` this iterator to mint the number of ids they need.
    pub fn available_centroid_ids(&self) -> impl Iterator<Item = usize> + '_ {
        (self.max_centroid_id + 1..).map(|x| x as usize)
    }
}

pub struct CentroidAssignmentUpdater<'a> {
    assignments_cursor: TypedCursorGuard<'a, i64, CentroidAssignment>,
    stats: CentroidStatsCache<'a>,
}

impl<'a> CentroidAssignmentUpdater<'a> {
    /// Create a new updater for `index` using `session` for WT access.
    pub fn new(txn_idx: &'a TransactionIndex) -> Result<Self> {
        let assignments_cursor = txn_idx
            .transaction()
            .open_cursor::<i64, CentroidAssignment>(&txn_idx.index().table_names.centroids)?;
        let stats = CentroidStatsCache::new(
            txn_idx
                .transaction()
                .open_cursor::<u32, CentroidCounts>(&txn_idx.index().table_names.centroid_stats)?,
        );
        Ok(Self {
            assignments_cursor,
            stats,
        })
    }

    /// Read the assignments for `record_id`.
    pub fn read(&mut self, record_id: i64) -> Option<Result<CentroidAssignment>> {
        self.assignments_cursor.seek_exact(record_id)
    }

    /// Insert centroid assignments for a new record.
    ///
    /// Returns a DuplicateKey error if the record already exists.
    pub fn insert(&mut self, record_id: i64, assignment: CentroidAssignment) -> Result<()> {
        if let Some(_existing_assignment) = self.assignments_cursor.seek_exact(record_id) {
            error!("attempted to insert duplicate record id {}", record_id);
            return Err(Error::wired_tiger(wt_mdb::WiredTigerError::DuplicateKey));
        }
        self.assignments_cursor.set(record_id, assignment)?;
        self.stats.increment_count(assignment.primary_id)?;
        Ok(())
    }

    /// Delete centroid assignments for an existing record.
    ///
    /// Returns a NotFound error if the record does not exist.
    pub fn delete(&mut self, record_id: i64) -> Result<CentroidAssignment> {
        let existing_assignment = self
            .assignments_cursor
            .seek_exact(record_id)
            .unwrap_or_else(|| Err(Error::not_found_error()))?;
        self.stats.decrement_count(existing_assignment.primary_id)?;
        self.assignments_cursor.remove(record_id)?;
        Ok(existing_assignment)
    }

    /// Overwrite the assignment for `record_id`, without consider previous assignment.
    pub fn overwrite(&mut self, record_id: i64, assignment: CentroidAssignment) -> Result<()> {
        self.assignments_cursor.set(record_id, assignment)?;
        self.stats.increment_count(assignment.primary_id)
    }

    /// Update centroid assignments for an existing record, returning the previous assignments.
    ///
    /// Returns a NotFound error if the record does not exist.
    pub fn update(
        &mut self,
        record_id: i64,
        assignment: CentroidAssignment,
    ) -> Result<CentroidAssignment> {
        let existing_assignment = self
            .assignments_cursor
            .seek_exact(record_id)
            .unwrap_or_else(|| Err(Error::not_found_error()))?;
        self.stats.decrement_count(existing_assignment.primary_id)?;
        self.stats.increment_count(assignment.primary_id)?;
        self.assignments_cursor
            .set(record_id, assignment)
            .map(|()| existing_assignment)
    }

    /// Return the number of vectors for `centroid_id`, or 0 if the centroid isn't found.
    pub fn centroid_len(&mut self, centroid_id: u32) -> Result<u32> {
        self.stats.count(centroid_id)
    }

    /// Flush buffered stats updates back to the database.
    /// This must be called in order to push updates into the database.
    pub fn flush(&mut self) -> Result<()> {
        self.stats.flush()
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

    fn count(&mut self, id: u32) -> Result<u32> {
        self.get_counts(id).map(|c| c.primary)
    }

    fn increment_count(&mut self, id: u32) -> Result<()> {
        self.get_counts(id)?.primary += 1;
        Ok(())
    }

    fn decrement_count(&mut self, id: u32) -> Result<()> {
        self.get_counts(id)?.primary -= 1;
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
