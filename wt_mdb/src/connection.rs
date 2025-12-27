use crate::{
    make_result, options::ConnectionOptions, session::Session, wt_call, ConfigurationString, Error,
    Result,
};
use std::{
    ffi::{CStr, CString},
    ptr::{self, NonNull},
    sync::Arc,
};

use tracing::error;
use wt_sys::{wiredtiger_open, WT_CONNECTION, WT_SESSION};

/// Global timestamp types that may be written by application code.
pub enum SetGlobalTimestampType {
    Durable,
    Oldest,
    Stable,
}

impl SetGlobalTimestampType {
    fn config_str(&self, ts: u64) -> CString {
        let s = match self {
            Self::Durable => format!("durable_timestamp={:x}", ts),
            Self::Oldest => format!("oldest_timestamp={:x}", ts),
            Self::Stable => format!("stable_timestamp={:x}", ts),
        };
        CString::new(s).expect("no nulls")
    }
}

// Global timestamp types that may be read by application code.
pub enum QueryGlobalTimestampType {
    AllDurable,
    BackupCheckpoint,
    LastCheckpoint,
    Oldest,
    OldestReader,
    Pinned,
    Recovery,
    Stable,
}

impl QueryGlobalTimestampType {
    fn config_str(&self) -> &'static CStr {
        match self {
            Self::AllDurable => c"get=all_durable",
            Self::BackupCheckpoint => c"get=backup_checkpoint",
            Self::LastCheckpoint => c"get=last_checkpoint",
            Self::Oldest => c"get=oldest_timestamp",
            Self::OldestReader => c"get=oldest_reader",
            Self::Pinned => c"get=pinned",
            Self::Recovery => c"get=recovery",
            Self::Stable => c"get=stable_timestamp",
        }
    }
}

/// A connection to a WiredTiger database.
///
/// There is typically only one connection per database per process.
/// `Connection`s may be freely shared between threads and are safe for concurrent access.
pub struct Connection(NonNull<WT_CONNECTION>);

impl Connection {
    /// Open a new `Connection` to a WiredTiger database.
    pub fn open(filename: &str, options: Option<ConnectionOptions>) -> Result<Arc<Self>> {
        let mut connp: *mut WT_CONNECTION = ptr::null_mut();
        let dbpath = CString::new(filename).unwrap();
        make_result(
            unsafe {
                wiredtiger_open(
                    dbpath.as_ptr(),
                    ptr::null_mut(),
                    options.unwrap_or_default().as_config_ptr(),
                    &mut connp,
                )
            },
            (),
        )?;
        NonNull::new(connp)
            .ok_or(Error::generic_error())
            .map(|conn| Arc::new(Connection(conn)))
    }

    /// Create a new `Session`. These can be used to obtain cursors to read and write data
    /// as well as manage transaction.
    pub fn open_session(self: &Arc<Self>) -> Result<Session> {
        let mut sessionp: *mut WT_SESSION = ptr::null_mut();
        unsafe {
            wt_call!(
                self.0,
                open_session,
                std::ptr::null_mut(),
                std::ptr::null(),
                &mut sessionp
            )
        }?;
        NonNull::new(sessionp)
            .ok_or(Error::generic_error())
            .map(|session| Session::new(session, self))
    }

    pub fn query_timestamp(&self, tt: QueryGlobalTimestampType) -> Result<u64> {
        let mut buf = [0u8; 17];
        unsafe {
            wt_call!(
                self.0,
                query_timestamp,
                buf.as_mut_ptr() as *mut std::ffi::c_char,
                tt.config_str().as_ptr()
            )?;
        }
        Ok(u64::from_str_radix(
            CStr::from_bytes_until_nul(&buf)
                .expect("valid CStr")
                .to_str()
                .expect("valid_utf8"),
            16,
        )
        .expect("valid hex"))
    }

    pub fn set_timestamp(&self, tt: SetGlobalTimestampType, ts: u64) -> Result<()> {
        let config_str = tt.config_str(ts);
        unsafe { wt_call!(self.0, set_timestamp, config_str.as_ptr()) }
    }
}

unsafe impl Send for Connection {}
unsafe impl Sync for Connection {}

impl Drop for Connection {
    fn drop(&mut self) {
        if let Err(e) = unsafe { wt_call!(self.0, close, std::ptr::null()) } {
            error!("Failed to close WT_CONNECTION: {}", e);
        }
    }
}
