//! Timestamp management routines. Used to set the read and commit timestamps for transactions.

use std::sync::atomic::AtomicU64;

use crate::Connection;

/// A generic timestamp provider.
///
/// Timestamps can be be used to get a timestamp for reading or choose a timestamp for committing a
/// transaction. The exact semantics of the timestamps are determined by the implementation, with
/// the understanding that timestamps are expected to be monotonically increasing and that
/// `current()` will always be less than `next()`.
pub trait Timestamp {
    fn current(&self) -> u64;
    fn next(&self) -> u64;
}

pub struct MontonicTimestamp(AtomicU64);

impl TryFrom<&Connection> for MontonicTimestamp {
    type Error = crate::Error;

    fn try_from(conn: &Connection) -> Result<Self, Self::Error> {
        let mut ts =
            conn.query_timestamp(crate::connection::QueryGlobalTimestampType::AllDurable)?;
        if ts == 0 {
            // Start at 1 if there are no timestamps in the system since 0 is a reserved value.
            // Reading at zero may indicate "most recent" in some contexts, so we want to avoid that.
            ts = 1;
        }
        Ok(Self(AtomicU64::new(ts + 1)))
    }
}

impl Timestamp for MontonicTimestamp {
    fn current(&self) -> u64 {
        self.0.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn next(&self) -> u64 {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1
    }
}
