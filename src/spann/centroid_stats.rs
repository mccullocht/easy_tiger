use std::collections::HashMap;

use wt_mdb::{
    Result, 
    session::{FormatString, Formatted},
};

use crate::spann::{TransactionIndex};

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
