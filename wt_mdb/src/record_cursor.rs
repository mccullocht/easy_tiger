use std::{
    borrow::Cow,
    ffi::c_void,
    iter::FusedIterator,
    num::NonZero,
    ptr::{self, NonNull},
    slice,
};

use wt_sys::{WT_CURSOR, WT_ITEM, WT_NOTFOUND};

use crate::{make_result, session::Session, Error, Result};

/// A `RecordView` in a WiredTiger table with an i64 key and a byte array value.
///
/// The underlying byte array may or may not be owned, the `Record` type alias may be more
/// convenient when the data is owned.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct RecordView<'a> {
    key: i64,
    value: Cow<'a, [u8]>,
}

impl<'a> RecordView<'a> {
    /// Create a new `RecordView` from a key and an unowned byte array value.
    pub fn new<V>(key: i64, value: V) -> Self
    where
        V: Into<Cow<'a, [u8]>>,
    {
        RecordView {
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

    /// Ensure that this RecordView owns the underlying value.
    pub fn to_owned(self) -> Record {
        let k = self.key();
        match self.value {
            Cow::Borrowed(v) => Record::new(k, v.to_owned()),
            Cow::Owned(v) => Record::new(k, v),
        }
    }
}

/// An alias for `RecordView` with `'static` lifetime, may be more convenient when the value is
/// actually owned.
pub type Record = RecordView<'static>;

/// A `RecordCursor` facilities viewing and mutating data in a WiredTiger table where
/// the table is `i64` keyed and byte-string valued.
pub struct RecordCursor<'a> {
    cursor: NonNull<WT_CURSOR>,
    session: &'a Session,
}

impl<'a> RecordCursor<'a> {
    pub(crate) fn new(cursor: NonNull<WT_CURSOR>, session: &'a Session) -> Self {
        Self { cursor, session }
    }

    pub fn session(&self) -> &Session {
        self.session
    }

    /// Set the contents of `record` in the collection.
    pub fn set(&mut self, record: &RecordView<'_>) -> Result<()> {
        // safety: the memory passed to set_{key,value} need only be valid until a modifying
        // call like insert().
        unsafe {
            self.cursor.as_ref().set_key.unwrap()(self.cursor.as_ptr(), record.key());
            self.cursor.as_ref().set_value.unwrap()(
                self.cursor.as_ptr(),
                &Self::item_from_value(record.value()),
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
        unsafe { self.seek_exact_unsafe(key) }.map(|r| r.map(|v| v.to_owned()))
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

/// It is safe to send a `RecordCursor` to another thread to use.
/// It is not safe to reference a `RecordCursor` from another thread without synchronization.
unsafe impl<'a> Send for RecordCursor<'a> {}

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
        unsafe { self.next_unsafe() }.map(|r| r.map(|v| v.to_owned()))
    }
}

impl<'a> FusedIterator for RecordCursor<'a> {}
