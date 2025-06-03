use std::{
    borrow::Cow,
    ffi::CStr,
    mem::ManuallyDrop,
    ops::{Bound, Deref, DerefMut, RangeBounds},
};

use crate::{map_not_found, wt_call, Result};

use super::{InnerCursor, Item, Session};

/// A `IndexRecordView` in a WiredTiger table with an i64 key and a byte array value.
///
/// The underlying byte array may or may not be owned, the `IndexRecord` type alias may be more
/// convenient when the data is owned.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct IndexRecordView<'a, 'b> {
    key: Cow<'a, [u8]>,
    value: Cow<'b, [u8]>,
}

impl<'a, 'b> IndexRecordView<'a, 'b> {
    /// Create a new `IndexRecordView` from a key and an unowned byte array value.
    pub fn new<K: Into<Cow<'a, [u8]>>, V: Into<Cow<'b, [u8]>>>(key: K, value: V) -> Self {
        IndexRecordView {
            key: key.into(),
            value: value.into(),
        }
    }

    /// Return the key.
    pub fn key(&self) -> &[u8] {
        self.key.as_ref()
    }

    /// Return the value.
    pub fn value(&self) -> &[u8] {
        self.value.as_ref()
    }

    /// Ensure that this IndexRecordView owns the underlying value.
    pub fn to_owned(self) -> IndexRecord {
        IndexRecord::new(self.key().to_vec(), self.value.to_vec())
    }

    /// Returns the inner key and value within the `IndexRecordView`.
    pub fn into_inner(self) -> (Cow<'a, [u8]>, Cow<'b, [u8]>) {
        (self.key, self.value)
    }
}

/// An alias for `IndexRecordView` with `'static` lifetime, may be more convenient when the value is
/// actually owned.
pub type IndexRecord = IndexRecordView<'static, 'static>;

/// An `IndexCursor` facilities viewing and mutating data in a WiredTiger table where
/// the table is byte-string keyed and byte-string valued.
pub struct IndexCursor<'a> {
    inner: InnerCursor,
    session: &'a Session,
}

impl<'a> IndexCursor<'a> {
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
    pub fn set(&mut self, record: &IndexRecordView<'_, '_>) -> Result<()> {
        // safety: the memory passed to set_{key,value} need only be valid until a modifying
        // call like insert().
        unsafe {
            wt_call!(void self.inner.ptr, set_key, &Item::from(record.key()).0)?;
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
    pub fn remove(&mut self, key: &[u8]) -> Result<()> {
        unsafe {
            wt_call!(void self.inner.ptr, set_key, &Item::from(key).0)?;
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
    pub unsafe fn next_unsafe(&mut self) -> Option<Result<IndexRecordView<'_, '_>>> {
        map_not_found(unsafe { wt_call!(self.inner.ptr, next) }.and_then(|()| self.record_view()))
    }

    /// Seek to the for `key` and return any associated `RecordView` if present.
    ///
    /// # Safety
    /// If the cursor's parent session returns this record during a transaction and that transaction
    /// is rolled back, we cannot guarantee that view value data is safe to access. Use
    /// `seek_exact()` to ensure safe access at the cost of a copy of the record value.
    pub unsafe fn seek_exact_unsafe<'k>(
        &mut self,
        key: &'k [u8],
    ) -> Option<Result<IndexRecordView<'k, '_>>> {
        map_not_found(
            unsafe {
                wt_call!(void self.inner.ptr, set_key, &Item::from(key).0)
                    .and_then(|()| wt_call!(self.inner.ptr, search))
            }
            .and_then(|()| self.record_value().map(|v| IndexRecordView::new(key, v))),
        )
    }

    /// Seek to the for `key` and return any associated `Record` if present.
    pub fn seek_exact(&mut self, key: &[u8]) -> Option<Result<IndexRecord>> {
        unsafe { self.seek_exact_unsafe(key) }.map(|r| r.map(|v| v.to_owned()))
    }

    /// Return the largest key in the collection or `None` if the collection is empty.
    pub fn largest_key(&mut self) -> Option<Result<&[u8]>> {
        map_not_found(
            unsafe { wt_call!(self.inner.ptr, largest_key) }.and_then(|()| self.record_key()),
        )
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

    /// Reset the cursor to an unpositioned state.
    pub fn reset(&mut self) -> Result<()> {
        unsafe { wt_call!(self.inner.ptr, reset) }
    }

    /// Return the current record key. This assumes that we have just positioned the cursor
    /// and WT_NOTFOUND will not be returned.
    fn record_key(&self) -> Result<&[u8]> {
        let mut k = Item::default();
        unsafe { wt_call!(self.inner.ptr, get_key, &mut k.0).map(|()| k.into()) }
    }

    /// Return the current record value. This assumes that we have just positioned the cursor
    /// and WT_NOTFOUND will not be returned.
    fn record_value(&self) -> Result<&[u8]> {
        let mut k = Item::default();
        unsafe { wt_call!(self.inner.ptr, get_value, &mut k.0).map(|()| k.into()) }
    }

    /// Return the current record view. This assumes that we have just positioned the cursor
    /// and WT_NOTFOUND will not be returned.
    fn record_view(&self) -> Result<IndexRecordView<'_, '_>> {
        let key = self.record_key()?;
        let value = self.record_value()?;
        Ok(IndexRecordView::new(key, value))
    }
}

impl Iterator for IndexCursor<'_> {
    type Item = Result<IndexRecord>;

    /// Advance and return the next record.
    ///
    /// If this cursor is unpositioned, returns to the start of the collection.
    fn next(&mut self) -> Option<Self::Item> {
        unsafe { self.next_unsafe() }.map(|r| r.map(IndexRecordView::to_owned))
    }
}

pub struct IndexCursorGuard<'a> {
    session: &'a Session,
    // On drop we will take the value and return it to session.
    cursor: ManuallyDrop<IndexCursor<'a>>,
}

impl<'a> IndexCursorGuard<'a> {
    pub(super) fn new(session: &'a Session, cursor: IndexCursor<'a>) -> Self {
        Self {
            session,
            cursor: ManuallyDrop::new(cursor),
        }
    }
}

impl<'a> Deref for IndexCursorGuard<'a> {
    type Target = IndexCursor<'a>;

    fn deref(&self) -> &Self::Target {
        &self.cursor
    }
}

impl DerefMut for IndexCursorGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.cursor
    }
}

impl Drop for IndexCursorGuard<'_> {
    fn drop(&mut self) {
        // Safety: we never intend to allow IndexCursorGuard to drop the value.
        self.session
            .return_cursor(unsafe { ManuallyDrop::take(&mut self.cursor).inner });
    }
}
