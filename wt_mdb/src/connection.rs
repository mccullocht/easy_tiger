use crate::{make_result, session::Session, wrap_ptr_create, Result};
use std::{
    ffi::CString,
    ptr::{self, NonNull},
    sync::Arc,
};

use wt_sys::{wiredtiger_open, WT_CONNECTION, WT_SESSION};

/// Builder for options when connecting to a WiredTiger database.
#[derive(Default)]
pub struct ConnectionOptionsBuilder {
    create: bool,
}

impl ConnectionOptionsBuilder {
    pub fn create(mut self) -> Self {
        self.create = true;
        self
    }
}

/// Options when connecting to a WiredTiger database.
#[derive(Debug)]
pub struct ConnectionOptions {
    rep: CString,
}

impl From<ConnectionOptionsBuilder> for ConnectionOptions {
    fn from(value: ConnectionOptionsBuilder) -> Self {
        let rep = if value.create {
            c"create".into()
        } else {
            CString::default()
        };
        ConnectionOptions { rep }
    }
}

/// A connection to a WiredTiger database.
///
/// There is typically only one connection per process. `Connection`s are thread-safe but
/// `Session`s and `RecordCursor`s are not.
pub struct Connection {
    conn: NonNull<WT_CONNECTION>,
}

impl Connection {
    /// Open a new `Connection` to a WiredTiger database.
    pub fn open(filename: &str, options: &ConnectionOptions) -> Result<Arc<Self>> {
        let mut connp: *mut WT_CONNECTION = ptr::null_mut();
        let dbpath = CString::new(filename).unwrap();
        let result: i32;
        unsafe {
            result = wiredtiger_open(
                dbpath.as_ptr(),
                ptr::null_mut(),
                if options.rep.is_empty() {
                    ptr::null()
                } else {
                    options.rep.as_ptr()
                },
                &mut connp,
            );
        };
        wrap_ptr_create(result, connp).map(|conn| Arc::new(Connection { conn }))
    }

    /// Create a new `Session`. These can be used to obtain cursors to read and write data
    /// as well as manage transaction.
    pub fn open_session(self: &Arc<Self>) -> Result<Session> {
        let mut sessionp: *mut WT_SESSION = ptr::null_mut();
        let result: i32;
        unsafe {
            result = (self.conn.as_ref().open_session.unwrap())(
                self.conn.as_ptr(),
                ptr::null_mut(),
                ptr::null(),
                &mut sessionp,
            );
        }
        wrap_ptr_create(result, sessionp).map(|session| Session::new(session, self))
    }

    /// Close this database connection.
    pub fn close(self) -> Result<()> {
        self.close_internal()
    }

    fn close_internal(&self) -> Result<()> {
        unsafe {
            make_result(
                self.conn.as_ref().close.unwrap()(self.conn.as_ptr(), std::ptr::null()),
                (),
            )
        }
    }
}

unsafe impl Send for Connection {}
unsafe impl Sync for Connection {}

impl Drop for Connection {
    fn drop(&mut self) {
        let _ = self.close_internal();
    }
}
