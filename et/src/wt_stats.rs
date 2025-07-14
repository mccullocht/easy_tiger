use std::sync::Arc;

use wt_mdb::{Connection, Session};
use wt_sys::{WT_STAT_CONN_BLOCK_BYTE_READ, WT_STAT_CONN_BLOCK_READ, WT_STAT_CONN_CURSOR_SEARCH};

pub struct WiredTigerConnectionStats {
    /// Cursor search() calls.
    pub search_calls: i64,
    /// Read IO calls (pread).
    pub read_ios: i64,
    /// Block bytes read.
    pub read_bytes: i64,
}

impl TryFrom<&Session> for WiredTigerConnectionStats {
    type Error = wt_mdb::Error;

    fn try_from(value: &Session) -> Result<Self, Self::Error> {
        let mut stat_cursor = value.new_stats_cursor(wt_mdb::options::Statistics::Fast, None)?;
        let search_calls = stat_cursor
            .seek_exact(WT_STAT_CONN_CURSOR_SEARCH as i32)
            .expect("WT_STAT_CONN_CURSOR_SEARCH")?
            .value;
        let read_ios = stat_cursor
            .seek_exact(WT_STAT_CONN_BLOCK_READ as i32)
            .expect("WT_STAT_CONN_BLOCK_READ")?
            .value;
        let read_bytes = stat_cursor
            .seek_exact(WT_STAT_CONN_BLOCK_BYTE_READ as i32)
            .expect("WT_STAT_CONN_BLOCK_BYTE_READ")?
            .value;
        Ok(Self {
            search_calls,
            read_ios,
            read_bytes,
        })
    }
}

impl TryFrom<&Arc<Connection>> for WiredTigerConnectionStats {
    type Error = wt_mdb::Error;

    fn try_from(value: &Arc<Connection>) -> Result<Self, Self::Error> {
        let session = value.open_session()?;
        (&session).try_into()
    }
}
