use std::{ops::Sub, sync::Arc};

use wt_mdb::{Connection, Transaction};
use wt_sys::{
    WT_STAT_CONN_BLOCK_BYTE_READ, WT_STAT_CONN_BLOCK_BYTE_WRITE, WT_STAT_CONN_BLOCK_READ,
    WT_STAT_CONN_CURSOR_INSERT_BYTES, WT_STAT_CONN_CURSOR_MODIFY_BYTES,
    WT_STAT_CONN_CURSOR_MODIFY_BYTES_TOUCH, WT_STAT_CONN_CURSOR_REMOVE_BYTES,
    WT_STAT_CONN_CURSOR_SEARCH, WT_STAT_CONN_CURSOR_UPDATE_BYTES,
    WT_STAT_CONN_LOG_BYTES_WRITTEN, WT_STAT_CONN_TXN_ROLLBACK, WT_STAT_CONN_TXN_UPDATE_CONFLICT,
};

pub struct WiredTigerConnectionStats {
    /// Cursor search() calls.
    pub search_calls: i64,
    /// Read IO calls (pread).
    pub read_ios: i64,
    /// Block bytes read.
    pub read_bytes: i64,
}

impl TryFrom<&Transaction> for WiredTigerConnectionStats {
    type Error = wt_mdb::Error;

    fn try_from(value: &Transaction) -> Result<Self, Self::Error> {
        let mut stat_cursor = value.open_stats_cursor(wt_mdb::Statistics::Fast, None)?;
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
        let txn = value.begin_transaction(None)?;
        (&txn).try_into()
    }
}

pub struct WiredTigerWriteStats {
    /// WAL (log) bytes written.
    pub log_bytes: i64,
    /// Block data bytes written.
    pub data_bytes: i64,
    /// Cursor insert bytes.
    pub insert_bytes: i64,
    /// Cursor update bytes.
    pub update_bytes: i64,
    /// Cursor remove bytes.
    pub remove_bytes: i64,
    /// Cursor modify input bytes.
    pub modify_bytes: i64,
    /// Cursor modify delta bytes actually written.
    pub modify_bytes_touch: i64,
    /// OCC write-write conflicts detected.
    pub txn_update_conflicts: i64,
    /// Total transaction rollbacks (includes OCC and other causes).
    pub txn_rollbacks: i64,
}

impl TryFrom<&Transaction> for WiredTigerWriteStats {
    type Error = wt_mdb::Error;

    fn try_from(value: &Transaction) -> Result<Self, Self::Error> {
        let mut stat_cursor = value.open_stats_cursor(wt_mdb::Statistics::Fast, None)?;
        let log_bytes = stat_cursor
            .seek_exact(WT_STAT_CONN_LOG_BYTES_WRITTEN as i32)
            .expect("WT_STAT_CONN_LOG_BYTES_WRITTEN")?
            .value;
        let data_bytes = stat_cursor
            .seek_exact(WT_STAT_CONN_BLOCK_BYTE_WRITE as i32)
            .expect("WT_STAT_CONN_BLOCK_BYTE_WRITE")?
            .value;
        let insert_bytes = stat_cursor
            .seek_exact(WT_STAT_CONN_CURSOR_INSERT_BYTES as i32)
            .expect("WT_STAT_CONN_CURSOR_INSERT_BYTES")?
            .value;
        let update_bytes = stat_cursor
            .seek_exact(WT_STAT_CONN_CURSOR_UPDATE_BYTES as i32)
            .expect("WT_STAT_CONN_CURSOR_UPDATE_BYTES")?
            .value;
        let remove_bytes = stat_cursor
            .seek_exact(WT_STAT_CONN_CURSOR_REMOVE_BYTES as i32)
            .expect("WT_STAT_CONN_CURSOR_REMOVE_BYTES")?
            .value;
        let modify_bytes = stat_cursor
            .seek_exact(WT_STAT_CONN_CURSOR_MODIFY_BYTES as i32)
            .expect("WT_STAT_CONN_CURSOR_MODIFY_BYTES")?
            .value;
        let modify_bytes_touch = stat_cursor
            .seek_exact(WT_STAT_CONN_CURSOR_MODIFY_BYTES_TOUCH as i32)
            .expect("WT_STAT_CONN_CURSOR_MODIFY_BYTES_TOUCH")?
            .value;
        let txn_update_conflicts = stat_cursor
            .seek_exact(WT_STAT_CONN_TXN_UPDATE_CONFLICT as i32)
            .expect("WT_STAT_CONN_TXN_UPDATE_CONFLICT")?
            .value;
        let txn_rollbacks = stat_cursor
            .seek_exact(WT_STAT_CONN_TXN_ROLLBACK as i32)
            .expect("WT_STAT_CONN_TXN_ROLLBACK")?
            .value;
        Ok(Self {
            log_bytes,
            data_bytes,
            insert_bytes,
            update_bytes,
            remove_bytes,
            modify_bytes,
            modify_bytes_touch,
            txn_update_conflicts,
            txn_rollbacks,
        })
    }
}

impl TryFrom<&Arc<Connection>> for WiredTigerWriteStats {
    type Error = wt_mdb::Error;

    fn try_from(value: &Arc<Connection>) -> Result<Self, Self::Error> {
        let txn = value.begin_transaction(None)?;
        (&txn).try_into()
    }
}

impl Sub for WiredTigerWriteStats {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            log_bytes: self.log_bytes - rhs.log_bytes,
            data_bytes: self.data_bytes - rhs.data_bytes,
            insert_bytes: self.insert_bytes - rhs.insert_bytes,
            update_bytes: self.update_bytes - rhs.update_bytes,
            remove_bytes: self.remove_bytes - rhs.remove_bytes,
            modify_bytes: self.modify_bytes - rhs.modify_bytes,
            modify_bytes_touch: self.modify_bytes_touch - rhs.modify_bytes_touch,
            txn_update_conflicts: self.txn_update_conflicts - rhs.txn_update_conflicts,
            txn_rollbacks: self.txn_rollbacks - rhs.txn_rollbacks,
        }
    }
}
