//! A cursor implementation that only works with raw keys and values in WiredTiger -- equivalent to
//! the 'u' format type. Opening cursors may also use a raw mode to work this way regardless of
//! other format settings.
//!
//! This cursor is used as the basis for other typed cursors.

use std::{
    ffi::CStr,
    ops::{Bound, RangeBounds},
};

use crate::{wt_call, Result};

use super::{InnerCursor, Item, Session};

/// Cursor over table data that expects raw byte strings for all keys and values.
///
/// Note that the underlying format of the table may vary
/// the table is byte-string keyed and byte-string valued.
pub struct RawCursor<'a> {
    inner: InnerCursor,
    session: &'a Session,
}

impl<'a> RawCursor<'a> {
    #[allow(unused)] // XXX
    pub(super) fn new(inner: InnerCursor, session: &'a Session) -> Self {
        Self { inner, session }
    }

    pub fn session(&self) -> &Session {
        self.session
    }

    /// Returns the name of the table.
    pub fn table_name(&self) -> &CStr {
        self.inner.uri.table_name()
    }

    /// Underlying format of keys.
    /// All keys passed are expected to be packed according to this format.
    pub fn key_format(&self) -> &CStr {
        self.inner.key_format()
    }

    /// Underlying format of values.
    /// All values passed are expected to be packed according to this format.
    pub fn value_format(&self) -> &CStr {
        self.inner.value_format()
    }

    /// Set the contents of `record` in the collection.
    pub fn set(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        // safety: the memory passed to set_{key,value} need only be valid until a modifying
        // call like insert().
        unsafe {
            wt_call!(void self.inner.ptr, set_key, &Item::from(key).0)?;
            wt_call!(void self.inner.ptr, set_value, &Item::from(value).0)?;
            wt_call!(self.inner.ptr, insert)
        }
    }

    /// Remove a record by `key`.
    ///
    /// This may return a `WiredTigerError::NotFound` if the key does not exist in the collection.
    pub fn remove(&mut self, key: &[u8]) -> Result<()> {
        unsafe {
            wt_call!(void self.inner.ptr, set_key, &Item::from(key).0)?;
            wt_call!(self.inner.ptr, remove)
        }
    }

    /// Advance and return the next record.
    ///
    /// # Safety
    /// If the cursor's parent session returns this record during a transaction and that transaction
    /// is rolled back, we cannot guarantee that view value data is safe to access. Use
    /// `Iterator.next()` to ensure safe access at the cost of a copy of the record value.
    pub unsafe fn next_unsafe(&mut self) -> Result<(&[u8], &[u8])> {
        unsafe { wt_call!(self.inner.ptr, next) }
            .and_then(|()| self.key())
            .and_then(|k| self.value().map(|v| (k, v)))
    }

    /// Seek to the for `key` and returns any associated value.
    ///
    /// # Safety
    /// If the cursor's parent session returns this record during a transaction and that transaction
    /// is rolled back, we cannot guarantee that view value data is safe to access. Use
    /// `seek_exact()` to ensure safe access at the cost of a copy of the record value.
    pub unsafe fn seek_exact_unsafe(&mut self, key: &[u8]) -> Result<&[u8]> {
        unsafe {
            wt_call!(void self.inner.ptr, set_key, &Item::from(key).0)
                .and_then(|()| wt_call!(self.inner.ptr, search))
        }
        .and_then(|()| self.value())
    }

    /// Return the largest key in the collection.
    ///
    /// # Safety
    /// If the cursor's parent session returns this record during a transaction and that transaction
    /// is rolled back, we cannot guarantee that view value data is safe to access. Use
    /// `seek_exact()` to ensure safe access at the cost of a copy of the record value.
    pub unsafe fn largest_key(&mut self) -> Result<&[u8]> {
        unsafe { wt_call!(self.inner.ptr, largest_key) }.and_then(|()| self.key())
    }

    /// Set the bounds this cursor. This affects almost all positioning operations, so for instance
    /// a `seek_exact()` with a key out of bounds might yield `None`.
    ///
    /// Cursor bounds are removed by `reset()`.
    pub fn set_bounds<'b>(&mut self, bounds: impl RangeBounds<&'b [u8]>) -> Result<()> {
        let (start_key, start_config_str) = match bounds.start_bound() {
            Bound::Included(key) => (Some(*key), c"bound=lower,action=set"),
            Bound::Excluded(key) => (Some(*key), c"bound=lower,action=set,inclusive=false"),
            Bound::Unbounded => (None, c"bound=lower,action=clear"),
        };
        if let Some(k) = start_key.map(Item::from) {
            unsafe { wt_call!(void self.inner.ptr, set_key, &k) }?;
        }
        unsafe { wt_call!(self.inner.ptr, bound, start_config_str.as_ptr())? };
        let (end_key, end_config_str) = match bounds.end_bound() {
            Bound::Included(key) => (Some(*key), c"bound=upper,action=set"),
            Bound::Excluded(key) => (Some(*key), c"bound=upper,action=set,inclusive=false"),
            Bound::Unbounded => (None, c"bound=upper,action=clear"),
        };
        if let Some(k) = end_key.map(Item::from) {
            unsafe { wt_call!(void self.inner.ptr, set_key, &k) }?;
        }
        unsafe { wt_call!(self.inner.ptr, bound, end_config_str.as_ptr()) }
    }

    pub fn set_bound(&mut self, bound: Bound<&[u8]>, upper: bool) -> Result<()> {
        let (key, config_str) = match bound {
            Bound::Included(key) => (
                Some(key),
                if upper {
                    c"bound=upper,action=set"
                } else {
                    c"bound=lower,action=set"
                },
            ),
            Bound::Excluded(key) => (
                Some(key),
                if upper {
                    c"bound=upper,action=set,inclusive=false"
                } else {
                    c"bound=lower,action=set,inclusive=false"
                },
            ),
            Bound::Unbounded => (
                None,
                if upper {
                    c"bound=upper,action=clear"
                } else {
                    c"bound=lower,action=clear"
                },
            ),
        };
        if let Some(k) = key.map(Item::from) {
            unsafe { wt_call!(void self.inner.ptr, set_key, &k) }?;
        }
        unsafe { wt_call!(self.inner.ptr, bound, config_str.as_ptr()) }
    }

    /// Reset the cursor to an unpositioned state.
    pub fn reset(&mut self) -> Result<()> {
        unsafe { wt_call!(self.inner.ptr, reset) }
    }

    pub(crate) fn return_to_session(self) {
        self.session.return_cursor(self.inner);
    }

    /// Return the current key. This assumes that we have just positioned the cursor
    /// and WT_NOTFOUND will not be returned.
    fn key(&self) -> Result<&[u8]> {
        let mut k = Item::default();
        unsafe { wt_call!(self.inner.ptr, get_key, &mut k.0).map(|()| k.into()) }
    }

    /// Return the current value. This assumes that we have just positioned the cursor
    /// and WT_NOTFOUND will not be returned.
    fn value(&self) -> Result<&[u8]> {
        let mut k = Item::default();
        unsafe { wt_call!(self.inner.ptr, get_value, &mut k.0).map(|()| k.into()) }
    }
}
