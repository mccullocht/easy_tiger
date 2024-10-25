use std::{
    ffi::c_void,
    iter::FusedIterator,
    num::NonZero,
    ptr::{self, NonNull},
    slice,
};

use wt_sys::{WT_CURSOR, WT_ITEM, WT_NOTFOUND};

use crate::{make_result, session::Session, Error, Result};

/// A `Record`` in a WiredTiger table with an i64 key and a byte array value.
///
/// `Record`` owns the underlying byte array; use `RecordView`` for unowned records.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Record {
    key: i64,
    value: Vec<u8>,
}

impl Record {
    /// Create a new `Record` from a key and an owned byte array value.
    pub fn new(key: i64, value: impl Into<Vec<u8>>) -> Self {
        Record {
            key,
            value: value.into(),
        }
    }

    /// Return the key.
    pub fn key(&self) -> i64 {
        self.key
    }

    /// Return the value.
    pub fn value(&self) -> &[u8] {
        self.value.as_ref()
    }
}

impl<'a> From<&RecordView<'a>> for Record {
    fn from(value: &RecordView<'a>) -> Self {
        Record::new(value.key, value.value)
    }
}

impl<'a> From<RecordView<'a>> for Record {
    fn from(value: RecordView<'a>) -> Self {
        Self::from(&value)
    }
}

impl PartialEq<RecordView<'_>> for Record {
    fn eq(&self, other: &RecordView<'_>) -> bool {
        RecordView::from(self) == *other
    }
}

/// A `RecordView`` in a WiredTiger table with an i64 key and a byte array value.
///
/// `RecordView` does not own the underlying byte array, use `Record` for owned records.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct RecordView<'a> {
    key: i64,
    value: &'a [u8],
}

impl<'a> RecordView<'a> {
    /// Create a new `RecordView` from a key and an unowned byte array value.
    pub fn new(key: i64, value: &'a [u8]) -> Self {
        RecordView { key, value }
    }

    /// Return the key.
    pub fn key(&self) -> i64 {
        self.key
    }

    /// Return the value.
    pub fn value(&self) -> &[u8] {
        self.value
    }
}

impl<'a> From<&'a Record> for RecordView<'a> {
    fn from(value: &'a Record) -> Self {
        RecordView {
            key: value.key,
            value: value.value.as_ref(),
        }
    }
}

impl<'a> PartialEq<Record> for RecordView<'a> {
    fn eq(&self, other: &Record) -> bool {
        *self == RecordView::from(other)
    }
}

/// A `RecordCursor` facilities viewing and mutating data in a WiredTiger table where
/// the table is `i64` keyed and byte-string valued.
pub struct RecordCursor<'a> {
    cursor: NonNull<WT_CURSOR>,
    session: &'a Session<'a>,
}

impl<'a> RecordCursor<'a> {
    pub(crate) fn new(cursor: NonNull<WT_CURSOR>, session: &'a Session<'a>) -> Self {
        Self { cursor, session }
    }

    pub fn session(&self) -> &Session {
        self.session
    }

    /// Set the contents of `record` in the collection.
    pub fn set<'b, R>(&mut self, record: R) -> Result<()>
    where
        R: Into<RecordView<'b>>,
    {
        // safety: the memory passed to set_{key,value} need only be valid until a modifying
        // call like insert().
        let view = record.into();
        unsafe {
            self.cursor.as_ref().set_key.unwrap()(self.cursor.as_ptr(), view.key());
            self.cursor.as_ref().set_value.unwrap()(
                self.cursor.as_ptr(),
                &Self::item_from_value(view.value()),
            );
            make_result(
                self.cursor.as_ref().insert.unwrap()(self.cursor.as_ptr()),
                (),
            )
        }
    }

    /// Remove a record by `key`.
    ///
    /// This may return a `WiredTigerError::NotFound` if the key does not exist in the collection.
    pub fn remove(&mut self, key: i64) -> Result<()> {
        unsafe {
            self.cursor.as_ref().set_key.unwrap()(self.cursor.as_ptr(), key);
            make_result(
                self.cursor.as_ref().remove.unwrap()(self.cursor.as_ptr()),
                (),
            )
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
        unsafe {
            match NonZero::new(self.cursor.as_ref().next.unwrap()(self.cursor.as_ptr())) {
                None => Some(self.record_view(None)),
                Some(code) if code.get() == WT_NOTFOUND => None,
                Some(code) => Some(Err(Error::from(code))),
            }
        }
    }

    /// Seek to the for `key` and return any associated `RecordView` if present.
    /// Leaves the cursor positioned at `key` if `Some(Ok(_))`` value is returned, otherwise
    /// the cursor is unpositioned.
    /// TODO: double check that this is correct.
    ///
    /// # Safety
    /// If the cursor's parent session returns this record during a transaction and that transaction
    /// is rolled back, we cannot guarantee that view value data is safe to access. Use
    /// `seek_exact()` to ensure safe access at the cost of a copy of the record value.
    pub unsafe fn seek_exact_unsafe(&mut self, key: i64) -> Option<Result<RecordView<'_>>> {
        unsafe {
            self.cursor.as_ref().set_key.unwrap()(self.cursor.as_ptr(), key);
            match NonZero::new(self.cursor.as_ref().search.unwrap()(self.cursor.as_ptr())) {
                None => Some(self.record_view(Some(key))),
                Some(code) if code.get() == WT_NOTFOUND => None,
                Some(code) => Some(Err(Error::from(code))),
            }
        }
    }

    /// Seek to the for `key` and return any associated `Record` if present.
    /// Leaves the cursor positioned at `key` if `Some(Ok(_))`` value is returned, otherwise
    /// the cursor is unpositioned.
    /// TODO: double check that this is correct.
    pub fn seek_exact(&mut self, key: i64) -> Option<Result<Record>> {
        unsafe { self.seek_exact_unsafe(key) }.map(|r| r.map(|v| v.into()))
    }

    /// Return the largest key in the collection or `None` if the collection is empty.
    pub fn largest_key(&mut self) -> Option<Result<i64>> {
        unsafe {
            match NonZero::new(self.cursor.as_ref().largest_key.unwrap()(
                self.cursor.as_ptr(),
            )) {
                None => Some(self.record_key()),
                Some(code) if code.get() == WT_NOTFOUND => None,
                Some(code) => Some(Err(Error::from(code))),
            }
        }
    }

    /// Reset the cursor to an unpositioned state.
    pub fn reset(&mut self) -> Result<()> {
        unsafe {
            make_result(
                self.cursor.as_ref().reset.unwrap()(self.cursor.as_ptr()),
                (),
            )
        }
    }

    /// Close this cursor.
    pub fn close(mut self) -> Result<()> {
        self.close_internal()
    }

    /// Return the current record key. This assumes that we have just positioned the cursor
    /// and WT_NOTFOUND will not be returned.
    fn record_key(&self) -> Result<i64> {
        unsafe {
            let mut k = 0i64;
            make_result(
                self.cursor.as_ref().get_key.unwrap()(self.cursor.as_ptr(), &mut k),
                (),
            )
            .map(|_| k)
        }
    }

    /// Return the current record view. This assumes that we have just positioned the cursor
    /// and WT_NOTFOUND will not be returned.
    ///
    /// A `known_key` may be provided in cases where we seeked by key that will be used in
    /// the returned record view rather than examining the cursor.
    fn record_view(&self, known_key: Option<i64>) -> Result<RecordView<'_>> {
        let key = known_key.map(Ok).unwrap_or_else(|| self.record_key())?;

        let value = unsafe {
            let mut item = Self::default_item();
            make_result(
                self.cursor.as_ref().get_value.unwrap()(self.cursor.as_ptr(), &mut item),
                (),
            )
            .map(|_| slice::from_raw_parts(item.data as *const u8, item.size))?
        };

        Ok(RecordView::new(key, value))
    }

    /// Return a default, empty WT_ITEM for fetching values.
    fn default_item() -> WT_ITEM {
        Self::item_from_value(&[])
    }

    /// Return a WT_ITEM that points to the contents of the value slice.
    fn item_from_value(value: &[u8]) -> WT_ITEM {
        WT_ITEM {
            data: value.as_ptr() as *const c_void,
            size: value.len(),
            mem: ptr::null_mut(),
            memsize: 0,
            flags: 0,
        }
    }

    /// Close a mutable instance. This is invoked by both `close()` and `drop()`, although
    /// drop ignores the returned value.
    fn close_internal(&mut self) -> Result<()> {
        unsafe {
            make_result(
                self.cursor.as_ref().close.unwrap()(self.cursor.as_ptr()),
                (),
            )
        }
    }
}

impl<'a> Drop for RecordCursor<'a> {
    fn drop(&mut self) {
        let _ = self.close_internal();
    }
}

impl<'a> Iterator for RecordCursor<'a> {
    type Item = Result<Record>;

    /// Advance and return the next record.
    ///
    /// If this cursor is unpositioned, returns to the start of the collection.
    fn next(&mut self) -> Option<Self::Item> {
        unsafe { self.next_unsafe() }.map(|r| r.map(|v| v.into()))
    }
}

impl<'a> FusedIterator for RecordCursor<'a> {}
