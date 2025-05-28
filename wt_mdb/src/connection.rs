use crate::{
    make_result,
    options::{ConfigurationString, ConnectionOptions},
    session::Session,
    wt_call, Error, Result,
};
use std::{
    ffi::CString,
    ptr::{self, NonNull},
    sync::Arc,
};

use tracing::error;
use wt_sys::{wiredtiger_open, WT_CONNECTION, WT_SESSION};

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
