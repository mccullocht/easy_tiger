use wt_mdb::{Result, Session};

use crate::spann::TableIndex;

/// Tracks occupancy statistics for centroids.
pub struct CentroidStats {
    primary: Vec<i64>,
    secondary: Vec<i64>,
}

impl CentroidStats {
    /// Generate statistics from the index, using `session` to read the data.
    ///
    /// Callers should open a transaction for this sequence of reads.
    pub fn from_index(session: &Session, index: &TableIndex) -> Result<Self> {
        let mut primary = Self::initial_counts(session, index)?;
        let mut secondary = primary.clone();
        Self::update_counts(session, index, &mut primary, &mut secondary)?;
        Ok(Self { primary, secondary })
    }

    /// Return the number of centroids in the index.
    pub fn len(&self) -> usize {
        self.counts_iter().count()
    }

    /// Return the number of vector -> centroid assignments.
    pub fn assigned(&self) -> usize {
        self.assignment_counts_iter().map(|(_, c)| c).sum::<u64>() as usize
    }

    /// Iterate over a list of centroid identifiers and the number of assigned vectors for each.
    pub fn assignment_counts_iter(&self) -> impl Iterator<Item = (u32, u64)> + '_ {
        self.counts_iter().map(|(i, (p, s))| (i, (p + s)))
    }

    /// Iterate over a list of centroid identifiers and the number of primary assigned vectors for each.
    pub fn primary_assignment_counts_iter(&self) -> impl Iterator<Item = (u32, u64)> + '_ {
        self.counts_iter().map(|(i, (p, _))| (i, p))
    }

    /// Iterate over a list of centroid identifiers and the number of secondary assigned vectors for each.
    pub fn secondary_assignment_counts_iter(&self) -> impl Iterator<Item = (u32, u64)> + '_ {
        self.counts_iter().map(|(i, (_, s))| (i, s))
    }

    /// Iterate over
    fn counts_iter(&self) -> impl Iterator<Item = (u32, (u64, u64))> + '_ {
        self.primary
            .iter()
            .copied()
            .zip(self.secondary.iter().copied())
            .enumerate()
            .filter(|(_, (p, _))| *p >= 0)
            .map(|(i, (p, s))| (i as u32, (p as u64, s as u64)))
    }

    /// Iterate over available centroid ids. The returned iterator is effectively unbounded (up to
    /// `u32::MAX`) so callers should `take()` this iterator to mint the number of ids they need.
    pub fn available_centroid_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.primary
            .iter()
            .enumerate()
            .filter(|(_, c)| **c < 0)
            .map(|(i, _)| i as u32)
            .chain(self.primary.len() as u32..)
    }

    /// Initialize a per-centroid counts list with -1 if the centroid is unused or zero if it is.
    fn initial_counts(session: &Session, index: &TableIndex) -> Result<Vec<i64>> {
        let mut head_cursor = session.get_record_cursor(index.head.graph_table_name())?;
        let centroid_len = head_cursor.largest_key().unwrap_or(Ok(0))? as usize + 1;
        let mut counts = vec![-1; centroid_len];
        head_cursor.set_bounds(0..)?;
        for r in head_cursor {
            counts[r.map(|(c, _)| c)? as usize] = 0;
        }
        Ok(counts)
    }

    /// Update primary and secondary centroid counts based on assignments in the centroids table.
    /// The first centroid id read out of each entry is considered the primary and all others are
    /// secondaries.
    fn update_counts(
        session: &Session,
        index: &TableIndex,
        primary: &mut [i64],
        secondary: &mut [i64],
    ) -> Result<()> {
        let centroid_cursor =
            session.get_or_create_typed_cursor::<i64, Vec<u8>>(&index.table_names.centroids)?;
        for r in centroid_cursor {
            let centroids = r.map(|(_, c)| c)?;
            if let Some((p, s)) = centroids.as_chunks::<4>().0.split_first() {
                primary[u32::from_le_bytes(*p) as usize] += 1;
                for s in s.iter().copied().map(u32::from_le_bytes) {
                    secondary[s as usize] += 1;
                }
            }
        }
        Ok(())
    }
}
