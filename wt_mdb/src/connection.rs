use crate::{session::Session, wrap_ptr_create, Result};
use std::{
    ffi::CString,
    ptr::{self, NonNull},
};

use wt_sys::{wiredtiger_open, WT_CONNECTION, WT_SESSION};

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

pub struct Connection {
    conn: NonNull<WT_CONNECTION>,
}

impl Connection {
    pub fn open(filename: &str, options: &ConnectionOptions) -> Result<Self> {
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
        wrap_ptr_create(result, connp).map(|conn| Connection { conn })
    }

    pub fn open_session(&self) -> Result<Session> {
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
}

impl Drop for Connection {
    fn drop(&mut self) {
        unsafe {
            self.conn.as_ref().close.unwrap()(self.conn.as_ptr(), std::ptr::null());
        }
    }
}
