use crate::{
    options::ConfigurationString, options::ConnectionOptions, session::Session, wrap_ptr_create,
    Result,
};
use std::{
    ffi::CString,
    ptr::{self, NonNull},
    sync::Arc,
};

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
        let result: i32;
        unsafe {
            result = wiredtiger_open(
                dbpath.as_ptr(),
                ptr::null_mut(),
                options.unwrap_or_default().as_config_ptr(),
                &mut connp,
            );
        };
        wrap_ptr_create(result, connp).map(|conn| Arc::new(Connection(conn)))
    }

    /// Create a new `Session`. These can be used to obtain cursors to read and write data
    /// as well as manage transaction.
    pub fn open_session(self: &Arc<Self>) -> Result<Session> {
        let mut sessionp: *mut WT_SESSION = ptr::null_mut();
        let result: i32;
        unsafe {
            result = (self.0.as_ref().open_session.unwrap())(
                self.0.as_ptr(),
                ptr::null_mut(),
                ptr::null(),
                &mut sessionp,
            );
        }
        wrap_ptr_create(result, sessionp).map(|session| Session::new(session, self))
    }
}

unsafe impl Send for Connection {}
unsafe impl Sync for Connection {}

impl Drop for Connection {
    fn drop(&mut self) {
        // TODO: log something when an error occurs here.
        // This would be unexpected as the connection can't be dropped until all cursors and sessions have also been dropped.
        unsafe { self.0.as_ref().close.unwrap()(self.0.as_ptr(), std::ptr::null()) };
    }
}
