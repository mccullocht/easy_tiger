use crate::{make_result, session::Session, wrap_ptr_create, Result};
use std::{
    ffi::CString,
    num::NonZero,
    ptr::{self, NonNull},
    sync::Arc,
};

use wt_sys::{wiredtiger_open, WT_CONNECTION, WT_SESSION};

/// Builder for options when connecting to a WiredTiger database.
#[derive(Default)]
pub struct ConnectionOptionsBuilder {
    create: bool,
    cache_size_mb: Option<NonZero<usize>>,
}

impl ConnectionOptionsBuilder {
    /// If set, create the database if it does not exist.
    pub fn create(mut self) -> Self {
        self.create = true;
        self
    }

    /// Maximum heap memory to allocate for the cache, in MB.
    pub fn cache_size_mb(mut self, size: NonZero<usize>) -> Self {
        self.cache_size_mb = Some(size);
        self
    }
}

/// Options when connecting to a WiredTiger database.
#[derive(Debug, Default)]
pub struct ConnectionOptions(Option<CString>);

impl From<ConnectionOptionsBuilder> for ConnectionOptions {
    fn from(value: ConnectionOptionsBuilder) -> Self {
        let mut options = Vec::new();
        if value.create {
            options.push("create".to_string())
        }
        if let Some(cache_size) = value.cache_size_mb {
            options.push(format!("cache_size={}", cache_size.get() << 20));
        }
        if options.is_empty() {
            Self(None)
        } else {
            Self(Some(
                CString::new(options.join(",")).expect("options does not contain null"),
            ))
        }
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
    pub fn open(filename: &str, options: Option<ConnectionOptions>) -> Result<Arc<Self>> {
        let mut connp: *mut WT_CONNECTION = ptr::null_mut();
        let dbpath = CString::new(filename).unwrap();
        let result: i32;
        unsafe {
            result = wiredtiger_open(
                dbpath.as_ptr(),
                ptr::null_mut(),
                options
                    .unwrap_or_default()
                    .0
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
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
    ///
    /// The connection will automatically be closed when this struct is dropped, consuming any error.
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
