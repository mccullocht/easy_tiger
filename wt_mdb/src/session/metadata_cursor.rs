use std::{
    ffi::{c_char, CStr, CString},
    mem::ManuallyDrop,
    ops::{Bound, Deref, DerefMut, RangeBounds},
};

use rustix::io::Errno;

use crate::{map_not_found, wt_call, Error, Result};

use super::{InnerCursor, Session};

/// An `MetadataCursor` allows reading metadata associated with objects in the database.
pub struct MetadataCursor<'a> {
    inner: InnerCursor,
    session: &'a Session,
}

impl<'a> MetadataCursor<'a> {
    pub(super) fn new(inner: InnerCursor, session: &'a Session) -> Self {
        Self { inner, session }
    }

    pub fn session(&self) -> &Session {
        self.session
    }

    /// Seek to the for `key` and returns the associated value if present.
    pub fn seek_exact(&mut self, key: &str) -> Option<Result<String>> {
        let key = match Self::str_to_cstring(key) {
            Ok(k) => k,
            Err(e) => return Some(Err(e)),
        };
        map_not_found(
            unsafe {
                wt_call!(void self.inner.ptr, set_key, key.as_ptr())
                    .and_then(|()| wt_call!(self.inner.ptr, search))
            }
            .and_then(|()| self.get_value()),
        )
    }

    /// Set the bounds this cursor. This affects almost all positioning operations, so for instance
    /// a `seek_exact()` with a key out of bounds might yield `None`.
    ///
    /// Cursor bounds are removed by `reset()`.
    pub fn set_bounds<'b>(&mut self, bounds: impl RangeBounds<&'b str>) -> Result<()> {
        let (start_key, start_config_str) = match bounds.start_bound() {
            Bound::Included(key) => (Some(*key), c"bound=lower,action=set"),
            Bound::Excluded(key) => (Some(*key), c"bound=lower,action=set,inclusive=false"),
            Bound::Unbounded => (None, c"bound=lower,action=clear"),
        };
        if let Some(k) = start_key {
            let k = Self::str_to_cstring(k)?;
            unsafe { wt_call!(void self.inner.ptr, set_key, k.as_ptr()) }?;
        }
        unsafe { wt_call!(self.inner.ptr, bound, start_config_str.as_ptr())? };
        let (end_key, end_config_str) = match bounds.end_bound() {
            Bound::Included(key) => (Some(*key), c"bound=upper,action=set"),
            Bound::Excluded(key) => (Some(*key), c"bound=upper,action=set,inclusive=false"),
            Bound::Unbounded => (None, c"bound=upper,action=clear"),
        };
        if let Some(k) = end_key {
            let k = Self::str_to_cstring(k)?;
            unsafe { wt_call!(void self.inner.ptr, set_key, k.as_ptr()) }?;
        }
        unsafe { wt_call!(self.inner.ptr, bound, end_config_str.as_ptr()) }
    }

    /// Reset the cursor to an unpositioned state.
    pub fn reset(&mut self) -> Result<()> {
        unsafe { wt_call!(self.inner.ptr, reset) }
    }

    fn get_key(&self) -> Result<String> {
        let mut key: *const c_char = std::ptr::null();
        unsafe { wt_call!(self.inner.ptr, get_key, &mut key) }?;
        unsafe { Self::cstr_ptr_to_string(key) }
    }

    fn get_value(&self) -> Result<String> {
        let mut value: *const c_char = std::ptr::null();
        unsafe { wt_call!(self.inner.ptr, get_value, &mut value) }?;
        unsafe { Self::cstr_ptr_to_string(value) }
    }

    // NB: c_char varies depending on the platform; it may be u8 or i8.
    #[allow(clippy::unnecessary_cast)]
    unsafe fn cstr_ptr_to_string(ptr: *const c_char) -> Result<String> {
        CString::from(unsafe { CStr::from_ptr(ptr as *const i8) })
            .into_string()
            .map_err(|_| Error::Errno(Errno::INVAL))
    }

    fn str_to_cstring(s: &str) -> Result<CString> {
        let mut sbytes: Vec<u8> = s.to_owned().into();
        sbytes.push(0);
        CString::from_vec_with_nul(sbytes).map_err(|_| Error::Errno(Errno::INVAL))
    }
}

impl Iterator for MetadataCursor<'_> {
    type Item = Result<(String, String)>;

    /// Advance and return the next record.
    ///
    /// If this cursor is unpositioned, returns to the start of the collection.
    fn next(&mut self) -> Option<Self::Item> {
        map_not_found(
            unsafe { wt_call!(self.inner.ptr, next) }
                .and_then(|()| self.get_key())
                .and_then(|k| self.get_value().map(|v| (k, v))),
        )
    }
}

/// Holds the cursor and returns it to the session when out of scope.
pub struct MetadataCursorGuard<'a>(ManuallyDrop<MetadataCursor<'a>>);

impl<'a> MetadataCursorGuard<'a> {
    pub(super) fn new(cursor: MetadataCursor<'a>) -> Self {
        Self(ManuallyDrop::new(cursor))
    }
}

impl<'a> Deref for MetadataCursorGuard<'a> {
    type Target = MetadataCursor<'a>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for MetadataCursorGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Drop for MetadataCursorGuard<'_> {
    fn drop(&mut self) {
        let cursor = unsafe { ManuallyDrop::take(&mut self.0) };
        cursor.session.return_cursor(cursor.inner);
    }
}
