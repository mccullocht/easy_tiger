//! Exhaustive (flat) vector search.

use std::{io, num::NonZero};

use vectors::QueryVectorDistance;
use wt_mdb::RecordCursorGuard;

use crate::posting_block::PostingBlock;
use crate::Neighbor;

/// Search `cursor` exhaustively and return the `k` nearest neighbors by `distance_fn`.
pub fn exhaustive_search(
    mut cursor: RecordCursorGuard<'_>,
    vector_len: usize,
    distance_fn: &dyn QueryVectorDistance,
    k: NonZero<usize>,
) -> io::Result<Vec<Neighbor>> {
    // Accumulate the top k values. When the buffer is full we select the top N and store a min
    // competitive result to act as a ratchet. This is much cheaper than a heap.
    let mut results = Vec::with_capacity(k.get() * 2);
    let mut min_competitive = Neighbor::new(i64::MAX, f64::MAX);
    while let Some(entry) = unsafe { cursor.next_unsafe() } {
        let (_, block_bytes) = entry.map_err(io::Error::from)?;
        let block = PostingBlock::new(block_bytes, vector_len).expect("valid block");
        for (id, vector) in block.iter() {
            let dist = distance_fn.distance(vector);
            let candidate = Neighbor::new(id, dist);
            if candidate > min_competitive {
                continue;
            }
            results.push(candidate);
            if results.len() == results.capacity() {
                results.select_nth_unstable(k.get());
                min_competitive = results[k.get()];
                results.truncate(k.get());
            }
        }
    }
    results.sort_unstable();
    results.truncate(k.get());
    Ok(results)
}
