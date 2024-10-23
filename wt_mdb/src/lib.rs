use wt_sys::{wiredtiger_open, wiredtiger_strerror};
use wt_sys::{WT_CONNECTION, WT_CURSOR, WT_EVENT_HANDLER, WT_SESSION};
use wt_sys::{WT_ERROR, WT_NOTFOUND};

use std::ffi::{CStr, CString};
use std::io;
use std::os::raw;
use std::ptr;
use std::ptr::NonNull;

// TODO: provide real WT errors this is annoying.
fn get_error(result: i32) -> io::Error {
    let err_msg = unsafe { CStr::from_ptr(wiredtiger_strerror(result)) };
    io::Error::other(err_msg.to_str().unwrap().to_owned())
}

fn make_result<T>(result: i32, value: T) -> Result<T, io::Error> {
    if result == 0 {
        Ok(value)
    } else {
        Err(get_error(result))
    }
}

pub struct Connection {
    conn: NonNull<WT_CONNECTION>,
}

pub struct Session {
    session: NonNull<WT_SESSION>,
}

pub struct Cursor {
    cursor: NonNull<WT_CURSOR>,
}

impl Connection {
    pub fn open(filename: &str, options: &str) -> Result<Self, io::Error> {
        let mut connp: *mut WT_CONNECTION = ptr::null_mut();
        let options = CString::new(options).unwrap();
        let dbpath = CString::new(filename).unwrap();
        let event_handler: *const WT_EVENT_HANDLER = ptr::null();
        let result: i32;
        unsafe {
            result = wiredtiger_open(
                dbpath.as_ptr(),
                event_handler as *mut WT_EVENT_HANDLER,
                options.as_ptr(),
                &mut connp,
            );
        };
        if result == 0 {
            let conn = NonNull::new(connp).ok_or_else(|| get_error(WT_ERROR))?;
            Ok(Connection { conn })
        } else {
            Err(get_error(result))
        }
    }

    pub fn create_session(&self) -> Result<Session, io::Error> {
        let mut sessionp: *mut WT_SESSION = ptr::null_mut();
        let event_handler: *mut WT_EVENT_HANDLER = ptr::null_mut();
        let result: i32;
        unsafe {
            result = (self.conn.as_ref().open_session.unwrap())(
                self.conn.as_ptr(),
                event_handler,
                ptr::null(),
                &mut sessionp,
            );
        }
        if result == 0 {
            let session = NonNull::new(sessionp).ok_or_else(|| get_error(WT_ERROR))?;
            Ok(Session { session })
        } else {
            Err(get_error(result))
        }
    }
}

// TODO: implement transactions
impl Session {
    pub fn create(&self, name: &str, config: &str) -> Result<(), io::Error> {
        let name = CString::new(name).unwrap();
        let config = CString::new(config).unwrap();
        let result: i32;
        unsafe {
            result = (self.session.as_ref().create.unwrap())(
                self.session.as_ptr(),
                name.as_ptr(),
                config.as_ptr(),
            );
        }
        make_result(result, ())
    }

    pub fn cursor(&self, uri: &str) -> Result<Cursor, io::Error> {
        let uri = CString::new(uri).unwrap();
        let mut cursorp: *mut WT_CURSOR = ptr::null_mut();
        let result: i32;
        unsafe {
            result = (self.session.as_ref().open_cursor.unwrap())(
                self.session.as_ptr(),
                uri.as_ptr(),
                ptr::null_mut(),
                ptr::null(),
                &mut cursorp,
            );
        }
        if result == 0 {
            let cursor = NonNull::new(cursorp).ok_or_else(|| get_error(WT_ERROR))?;
            Ok(Cursor { cursor })
        } else {
            Err(get_error(result))
        }
    }
}

// TODO: flesh out cursor API.
impl Cursor {
    pub fn set(&self, key: &str, value: &str) -> Result<(), io::Error> {
        let result: i32;
        let ckey = CString::new(key).unwrap();
        let cval = CString::new(value).unwrap();
        unsafe {
            self.cursor.as_ref().set_key.unwrap()(self.cursor.as_ptr(), ckey.as_ptr());
            self.cursor.as_ref().set_value.unwrap()(self.cursor.as_ptr(), cval.as_ptr());
            result = self.cursor.as_ref().insert.unwrap()(self.cursor.as_ptr());
        }
        make_result(result, ())
    }

    pub fn search(&self, key: &str) -> Result<Option<String>, io::Error> {
        let mut result: i32;
        let ckey = CString::new(key).unwrap();
        let mut val: *mut raw::c_char = ptr::null_mut();
        unsafe {
            self.cursor.as_ref().set_key.unwrap()(self.cursor.as_ptr(), ckey.as_ptr());
            result = self.cursor.as_ref().search.unwrap()(self.cursor.as_ptr());
            if result == WT_NOTFOUND {
                return Ok(None);
            }
            if result != 0 {
                return Err(get_error(result));
            }
            result = self.cursor.as_ref().get_value.unwrap()(self.cursor.as_ptr(), &mut val);
            let owned_val = CStr::from_ptr(val).to_string_lossy().into_owned();
            make_result(result, Some(owned_val))
        }
    }
}

// TODO: close APIs
impl Drop for Connection {
    fn drop(&mut self) {
        unsafe {
            self.conn.as_ref().close.unwrap()(self.conn.as_ptr(), std::ptr::null());
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            self.session.as_ref().close.unwrap()(self.session.as_ptr(), std::ptr::null());
        }
    }
}

impl Drop for Cursor {
    fn drop(&mut self) {
        unsafe {
            self.cursor.as_ref().close.unwrap()(self.cursor.as_ptr());
        }
    }
}
