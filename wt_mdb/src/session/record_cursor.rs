use std::{
    ffi::CStr,
    mem::ManuallyDrop,
    ops::{Bound, Deref, DerefMut, RangeBounds},
};

use crate::{map_not_found, wt_call, Record, RecordView, Result};

use super::{InnerCursor, Item, Session};

/// A `RecordCursor` facilities viewing and mutating data in a WiredTiger table where
/// the table is `i64` keyed and byte-string valued.
pub struct RecordCursor<'a> {
    inner: InnerCursor,
    session: &'a Session,
}

impl<'a> RecordCursor<'a> {
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

    /// Set the contents of `record` in the collection.
    pub fn set(&mut self, record: &RecordView<'_>) -> Result<()> {
        // safety: the memory passed to set_{key,value} need only be valid until a modifying
        // call like insert().
        unsafe {
            wt_call!(void self.inner.ptr, set_key, record.key())?;
            wt_call!(void
                self.inner.ptr,
                set_value,
                &Item::from(record.value()).0
            )?;
            wt_call!(self.inner.ptr, insert)
        }
    }

    /// Remove a record by `key`.
    ///
    /// This may return a `WiredTigerError::NotFound` if the key does not exist in the collection.
    pub fn remove(&mut self, key: i64) -> Result<()> {
        unsafe {
            wt_call!(void self.inner.ptr, set_key, key)?;
            wt_call!(self.inner.ptr, remove)
        }
    }

    /// Advance and return the next record.
    ///
    /// If this cursor is unpositioned, returns to the start of the collection.
    ///
    /// Like a `FusedIterator`, this returns `None` when the end of the collection is reached and
    /// continues to return `None` until the cursor is re-positioned.
    ///
    /// # Safety
    /// If the cursor's parent session returns this record during a transaction and that transaction
    /// is rolled back, we cannot guarantee that view value data is safe to access. Use
    /// `Iterator.next()` to ensure safe access at the cost of a copy of the record value.
    pub unsafe fn next_unsafe(&mut self) -> Option<Result<RecordView<'_>>> {
        map_not_found(
            unsafe { wt_call!(self.inner.ptr, next) }.and_then(|()| self.record_view(None)),
        )
    }

    /// Seek to the for `key` and return any associated `RecordView` if present.
    ///
    /// # Safety
    /// If the cursor's parent session returns this record during a transaction and that transaction
    /// is rolled back, we cannot guarantee that view value data is safe to access. Use
    /// `seek_exact()` to ensure safe access at the cost of a copy of the record value.
    pub unsafe fn seek_exact_unsafe(&mut self, key: i64) -> Option<Result<RecordView<'_>>> {
        map_not_found(
            unsafe {
                wt_call!(void self.inner.ptr, set_key, key)
                    .and_then(|()| wt_call!(self.inner.ptr, search))
            }
            .and_then(|()| self.record_view(Some(key))),
        )
    }

    /// Seek to the for `key` and return any associated `Record` if present.
    pub fn seek_exact(&mut self, key: i64) -> Option<Result<Record>> {
        unsafe { self.seek_exact_unsafe(key) }.map(|r| r.map(|v| v.to_owned()))
    }

    /// Return the largest key in the collection or `None` if the collection is empty.
    pub fn largest_key(&mut self) -> Option<Result<i64>> {
        map_not_found(
            unsafe { wt_call!(self.inner.ptr, largest_key) }.and_then(|()| self.record_key()),
        )
    }

    /// Set the bounds this cursor. This affects almost all positioning operations, so for instance
    /// a `seek_exact()` with a key out of bounds might yield `None`.
    ///
    /// Cursor bounds are removed by `reset()`.
    pub fn set_bounds(&mut self, bounds: impl RangeBounds<i64>) -> Result<()> {
        let (start_key, start_config_str) = match bounds.start_bound() {
            Bound::Included(key) => (Some(*key), c"bound=lower,action=set"),
            Bound::Excluded(key) => (Some(*key), c"bound=lower,action=set,inclusive=false"),
            Bound::Unbounded => (None, c"bound=lower,action=clear"),
        };
        if let Some(k) = start_key {
            unsafe { wt_call!(void self.inner.ptr, set_key, k) }?;
        }
        unsafe { wt_call!(self.inner.ptr, bound, start_config_str.as_ptr())? };
        let (end_key, end_config_str) = match bounds.end_bound() {
            Bound::Included(key) => (Some(*key), c"bound=upper,action=set"),
            Bound::Excluded(key) => (Some(*key), c"bound=upper,action=set,inclusive=false"),
            Bound::Unbounded => (None, c"bound=upper,action=clear"),
        };
        if let Some(k) = end_key {
            unsafe { wt_call!(void self.inner.ptr, set_key, k) }?;
        }
        unsafe { wt_call!(self.inner.ptr, bound, end_config_str.as_ptr()) }
    }

    /// Reset the cursor to an unpositioned state.
    pub fn reset(&mut self) -> Result<()> {
        unsafe { wt_call!(self.inner.ptr, reset) }
    }

    pub(super) fn into_inner(self) -> InnerCursor {
        self.inner
    }

    /// Return the current record key. This assumes that we have just positioned the cursor
    /// and WT_NOTFOUND will not be returned.
    fn record_key(&self) -> Result<i64> {
        let mut k = 0i64;
        unsafe { wt_call!(self.inner.ptr, get_key, &mut k).map(|()| k) }
    }

    /// Return the current record view. This assumes that we have just positioned the cursor
    /// and WT_NOTFOUND will not be returned.
    ///
    /// A `known_key` may be provided in cases where we seeked by key that will be used in
    /// the returned record view rather than examining the cursor.
    fn record_view(&self, known_key: Option<i64>) -> Result<RecordView<'_>> {
        let key = known_key.map(Ok).unwrap_or_else(|| self.record_key())?;

        let value: &[u8] = unsafe {
            let mut item = Item::default();
            wt_call!(self.inner.ptr, get_value, &mut item.0).map(|()| item.into())?
        };

        Ok(RecordView::new(key, value))
    }
}

impl Iterator for RecordCursor<'_> {
    type Item = Result<Record>;

    /// Advance and return the next record.
    ///
    /// If this cursor is unpositioned, returns to the start of the collection.
    fn next(&mut self) -> Option<Self::Item> {
        unsafe { self.next_unsafe() }.map(|r| r.map(RecordView::to_owned))
    }
}

pub struct RecordCursorGuard<'a> {
    session: &'a Session,
    // On drop we will take the value and return it to session.
    cursor: ManuallyDrop<RecordCursor<'a>>,
}

impl<'a> RecordCursorGuard<'a> {
    pub(super) fn new(session: &'a Session, cursor: RecordCursor<'a>) -> Self {
        Self {
            session,
            cursor: ManuallyDrop::new(cursor),
        }
    }
}

impl<'a> Deref for RecordCursorGuard<'a> {
    type Target = RecordCursor<'a>;

    fn deref(&self) -> &Self::Target {
        &self.cursor
    }
}

impl DerefMut for RecordCursorGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.cursor
    }
}

impl Drop for RecordCursorGuard<'_> {
    fn drop(&mut self) {
        // Safety: we never intend to allow RecordCursorGuard to drop the value.
        self.session
            .return_record_cursor(unsafe { ManuallyDrop::take(&mut self.cursor) });
    }
}
