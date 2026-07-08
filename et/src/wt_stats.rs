use std::{ops::Sub, sync::Arc};

use wt_mdb::{Connection, Transaction};
use wt_sys::{
    WT_STAT_CONN_BLOCK_BYTE_READ, WT_STAT_CONN_BLOCK_BYTE_WRITE,
    WT_STAT_CONN_BLOCK_BYTE_WRITE_CHECKPOINT, WT_STAT_CONN_BLOCK_READ, WT_STAT_CONN_BLOCK_WRITE,
    WT_STAT_CONN_CURSOR_INSERT_BYTES, WT_STAT_CONN_CURSOR_MODIFY,
    WT_STAT_CONN_CURSOR_MODIFY_BYTES, WT_STAT_CONN_CURSOR_MODIFY_BYTES_TOUCH,
    WT_STAT_CONN_CURSOR_REMOVE_BYTES, WT_STAT_CONN_CURSOR_SEARCH,
    WT_STAT_CONN_CURSOR_UPDATE_BYTES, WT_STAT_CONN_LOG_BYTES_PAYLOAD,
    WT_STAT_CONN_LOG_BYTES_WRITTEN, WT_STAT_CONN_LOG_WRITES,
};

pub struct WiredTigerConnectionStats {
    /// Cursor search() calls.
    pub search_calls: i64,
    /// Read IO calls (pread).
    pub read_ios: i64,
    /// Block bytes read.
    pub read_bytes: i64,

    /// Log records written (transactions committed).
    pub log_writes: i64,
    /// Logical bytes written to log before compression.
    pub log_bytes_payload: i64,
    /// Actual bytes written to log files (after compression).
    pub log_bytes_written: i64,

    /// Block manager write IOs.
    pub block_writes: i64,
    /// Block manager bytes written (data files; checkpoints + eviction).
    pub block_bytes_written: i64,
    /// Block manager bytes written specifically during checkpoints.
    pub block_bytes_written_checkpoint: i64,

    /// cursor.modify() calls.
    pub cursor_modify_calls: i64,
    /// Total size of values that cursor.modify() was applied to (full block sizes, not deltas).
    pub cursor_modify_bytes: i64,
    /// Actual delta bytes written by cursor.modify() (the WT_MODIFY items).
    pub cursor_modify_bytes_touch: i64,

    /// Bytes written by cursor inserts (new keys, full value logged).
    pub cursor_insert_bytes: i64,
    /// Bytes written by cursor updates (existing keys, full value logged).
    pub cursor_update_bytes: i64,
    /// Bytes removed by cursor removes.
    pub cursor_remove_bytes: i64,
}

impl WiredTigerConnectionStats {
    fn read(txn: &Transaction) -> wt_mdb::Result<Self> {
        let mut c = txn.open_stats_cursor(wt_mdb::Statistics::Fast, None)?;
        macro_rules! stat {
            ($key:expr) => {
                c.seek_exact($key as i32).expect(stringify!($key))?.value
            };
        }
        Ok(Self {
            search_calls: stat!(WT_STAT_CONN_CURSOR_SEARCH),
            read_ios: stat!(WT_STAT_CONN_BLOCK_READ),
            read_bytes: stat!(WT_STAT_CONN_BLOCK_BYTE_READ),
            log_writes: stat!(WT_STAT_CONN_LOG_WRITES),
            log_bytes_payload: stat!(WT_STAT_CONN_LOG_BYTES_PAYLOAD),
            log_bytes_written: stat!(WT_STAT_CONN_LOG_BYTES_WRITTEN),
            block_writes: stat!(WT_STAT_CONN_BLOCK_WRITE),
            block_bytes_written: stat!(WT_STAT_CONN_BLOCK_BYTE_WRITE),
            block_bytes_written_checkpoint: stat!(WT_STAT_CONN_BLOCK_BYTE_WRITE_CHECKPOINT),
            cursor_modify_calls: stat!(WT_STAT_CONN_CURSOR_MODIFY),
            cursor_modify_bytes: stat!(WT_STAT_CONN_CURSOR_MODIFY_BYTES),
            cursor_modify_bytes_touch: stat!(WT_STAT_CONN_CURSOR_MODIFY_BYTES_TOUCH),
            cursor_insert_bytes: stat!(WT_STAT_CONN_CURSOR_INSERT_BYTES),
            cursor_update_bytes: stat!(WT_STAT_CONN_CURSOR_UPDATE_BYTES),
            cursor_remove_bytes: stat!(WT_STAT_CONN_CURSOR_REMOVE_BYTES),
        })
    }
}

impl TryFrom<&Transaction> for WiredTigerConnectionStats {
    type Error = wt_mdb::Error;

    fn try_from(value: &Transaction) -> Result<Self, Self::Error> {
        Self::read(value)
    }
}

impl TryFrom<&Arc<Connection>> for WiredTigerConnectionStats {
    type Error = wt_mdb::Error;

    fn try_from(value: &Arc<Connection>) -> Result<Self, Self::Error> {
        let txn = value.begin_transaction(None)?;
        Self::read(&txn)
    }
}

impl Sub for WiredTigerConnectionStats {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            search_calls: self.search_calls - rhs.search_calls,
            read_ios: self.read_ios - rhs.read_ios,
            read_bytes: self.read_bytes - rhs.read_bytes,
            log_writes: self.log_writes - rhs.log_writes,
            log_bytes_payload: self.log_bytes_payload - rhs.log_bytes_payload,
            log_bytes_written: self.log_bytes_written - rhs.log_bytes_written,
            block_writes: self.block_writes - rhs.block_writes,
            block_bytes_written: self.block_bytes_written - rhs.block_bytes_written,
            block_bytes_written_checkpoint: self.block_bytes_written_checkpoint
                - rhs.block_bytes_written_checkpoint,
            cursor_modify_calls: self.cursor_modify_calls - rhs.cursor_modify_calls,
            cursor_modify_bytes: self.cursor_modify_bytes - rhs.cursor_modify_bytes,
            cursor_modify_bytes_touch: self.cursor_modify_bytes_touch
                - rhs.cursor_modify_bytes_touch,
            cursor_insert_bytes: self.cursor_insert_bytes - rhs.cursor_insert_bytes,
            cursor_update_bytes: self.cursor_update_bytes - rhs.cursor_update_bytes,
            cursor_remove_bytes: self.cursor_remove_bytes - rhs.cursor_remove_bytes,
        }
    }
}
