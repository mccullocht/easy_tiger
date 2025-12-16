use wt_mdb::{
    session::{FormatString, Formatted},
    Result, Session,
};

use crate::spann::TableIndex;

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
    /// Generate statistics from the index, using `session` to read the data.
    ///
    /// Callers should open a transaction for this sequence of reads.
    pub fn from_index(session: &Session, index: &TableIndex) -> Result<Self> {
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

    /// Return the number of centroids in the index.
    pub fn centroid_count(&self) -> usize {
        self.counts_iter().count()
    }

    /// Return the number of vector -> centroid assignments.
    pub fn vector_count(&self) -> usize {
        self.assignment_counts_iter().map(|(_, c)| c).sum::<u32>() as usize
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
