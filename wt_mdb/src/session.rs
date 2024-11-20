use std::{
    cell::RefCell,
    ffi::{c_void, CStr, CString},
    mem::ManuallyDrop,
    num::NonZero,
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
    slice,
    sync::Arc,
};

use wt_sys::{WT_CURSOR, WT_ITEM, WT_NOTFOUND, WT_SESSION};

use crate::{
    connection::Connection,
    make_result,
    options::{
        BeginTransactionOptions, CommitTransactionOptions, ConfigurationString, CreateOptions,
        DropOptions, RollbackTransactionOptions,
    },
    wrap_ptr_create, Error, Record, RecordView, Result,
};

/// URI of a WT table encoded as a CString.
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct TableUri(CString);

impl TableUri {
    fn table_name(&self) -> &CStr {
        // magic number 6 comes from length of string "table:"
        &(self.0.as_c_str()[6usize..])
    }
}

impl Deref for TableUri {
    type Target = CStr;
    fn deref(&self) -> &Self::Target {
        self.0.as_c_str()
    }
}

impl From<&str> for TableUri {
    fn from(value: &str) -> Self {
        Self(CString::new(format!("table:{}", value)).expect("no nulls in table name"))
    }
}

/// A WiredTiger session.
///
/// `Session`s are used to create cursors to view and mutate data and manage transaction state.
///
/// `Session` is `Send` and may be freely passed to other threads but it is not `Sync` as it is
/// unsafe to access without synchronization. `RecordCursor`s reference their parent `Session` so
/// it is not possible to `Send` a `Session` with open cursors. Some `Session` APIs support cursor
/// caching to try to mitigate the costs of opening/closing cursors to perform a `Send`.
pub struct Session {
    ptr: NonNull<WT_SESSION>,
    connection: Arc<Connection>,
    cached_cursors: RefCell<Vec<InnerCursor>>,
}

impl Session {
    pub(crate) fn new(session: NonNull<WT_SESSION>, connection: &Arc<Connection>) -> Self {
        Self {
            ptr: session,
            connection: connection.clone(),
            cached_cursors: RefCell::new(vec![]),
        }
    }

    /// Return the `Connection` this session belongs to.
    pub fn connection(&self) -> &Arc<Connection> {
        &self.connection
    }

    /// Create a new record table.
    pub fn create_record_table(
        &self,
        table_name: &str,
        config: Option<CreateOptions>,
    ) -> Result<()> {
        let uri = TableUri::from(table_name);
        unsafe {
            make_result(
                (self.ptr.as_ref().create.unwrap())(
                    self.ptr.as_ptr(),
                    uri.as_ptr(),
                    config.unwrap_or_default().as_config_ptr(),
                ),
                (),
            )
        }
    }

    /// Drop a record table.
    ///
    /// This requires exclusive access -- if any cursors are open on the specified table the call will fail
    /// and return an EBUSY posix error.
    pub fn drop_record_table(&self, table_name: &str, config: Option<DropOptions>) -> Result<()> {
        let uri = TableUri::from(table_name);
        unsafe {
            make_result(
                self.ptr.as_ref().drop.unwrap()(
                    self.ptr.as_ptr(),
                    uri.as_ptr(),
                    config.unwrap_or_default().as_config_ptr(),
                ),
                (),
            )
        }
    }

    /// Open a record cursor over the named table.
    pub fn open_record_cursor(&self, table_name: &str) -> Result<RecordCursor> {
        self.open_record_cursor_with_options(table_name, None)
    }

    fn open_record_cursor_with_options(
        &self,
        table_name: &str,
        options: Option<&CStr>,
    ) -> Result<RecordCursor> {
        let uri = TableUri::from(table_name);
        let mut cursorp: *mut WT_CURSOR = ptr::null_mut();
        let result: i32;
        unsafe {
            result = (self.ptr.as_ref().open_cursor.unwrap())(
                self.ptr.as_ptr(),
                uri.0.as_ptr(),
                ptr::null_mut(),
                options.map(|s| s.as_ptr()).unwrap_or(std::ptr::null()),
                &mut cursorp,
            );
        }
        wrap_ptr_create(result, cursorp)
            .map(|ptr| RecordCursor::new(InnerCursor { ptr, uri }, self))
    }

    /// Get a cached cursor or create a new cursor over `table_name`.
    pub fn get_record_cursor(&self, table_name: &str) -> Result<RecordCursorGuard<'_>> {
        let mut cursor_cache = self.cached_cursors.borrow_mut();
        cursor_cache
            .iter()
            .position(|c| c.uri.table_name().to_bytes() == table_name.as_bytes())
            .map(|i| {
                let inner = cursor_cache.remove(i);
                Ok(RecordCursor::new(inner, self))
            })
            .unwrap_or_else(|| self.open_record_cursor(table_name))
            .map(|c| RecordCursorGuard::new(self, c))
    }

    /// Return a `RecordCursor` to the cache for future re-use.
    pub fn return_record_cursor(&self, cursor: RecordCursor) {
        self.cached_cursors.borrow_mut().push(cursor.into_inner())
    }

    /// Remove all cached cursors.
    pub fn clear_cursor_cache(&self) {
        self.cached_cursors.borrow_mut().clear();
    }

    /// Starts a transaction in this session.
    ///
    /// The transaction remains active until `commit_transaction` or `rollback_transaction` are called.
    /// Operations performed on cursors already opened or opened before the transaction ends happen
    /// within the context of this transaction.
    ///
    /// This may not be called on a session with an active transaction or an error will be returned (EINVAL)
    /// but otherwise behavior is unspecified.
    pub fn begin_transaction(&self, options: Option<&BeginTransactionOptions>) -> Result<()> {
        unsafe {
            make_result(
                self.ptr.as_ref().begin_transaction.unwrap()(
                    self.ptr.as_ptr(),
                    options.as_config_ptr(),
                ),
                (),
            )
        }
    }

    /// Commit the current transaction.
    ///
    /// If this method returns an error, the transaction was rolled back and all cursors associated with the session
    /// are reset.
    ///
    /// A transaction must be in progress when this method is called or an error will be returned (EINVAL) but behavior
    /// is otherwise unspecified.
    pub fn commit_transaction(&self, options: Option<&CommitTransactionOptions>) -> Result<()> {
        unsafe {
            make_result(
                self.ptr.as_ref().commit_transaction.unwrap()(
                    self.ptr.as_ptr(),
                    options.as_config_ptr(),
                ),
                (),
            )
        }
    }

    /// Rollback the current transaction.
    ///
    /// All cursors associated with the session are reset.
    ///
    /// A transaction must be in progress when this method is called or an error will be returned (EINVAL) but behavior
    /// is otherwise unspecified.
    pub fn rollback_transaction(&self, options: Option<&RollbackTransactionOptions>) -> Result<()> {
        unsafe {
            make_result(
                self.ptr.as_ref().rollback_transaction.unwrap()(
                    self.ptr.as_ptr(),
                    options.as_config_ptr(),
                ),
                (),
            )
        }
    }

    /// Create a new table called `table_name` and bulk load entries from `iter`.
    ///
    /// Bulk load requires that `table_name` not exist or be empty and that `iter` yields records in
    /// order by `key()`.
    pub fn bulk_load<'a, I>(
        &self,
        table_name: &str,
        options: Option<CreateOptions>,
        iter: I,
    ) -> Result<()>
    where
        I: Iterator<Item = RecordView<'a>>,
    {
        self.create_record_table(table_name, options)?;
        let mut cursor = self.open_record_cursor_with_options(table_name, Some(c"bulk=true"))?;
        for record in iter {
            cursor.set(&record)?;
        }
        Ok(())
    }

    /// Reset this session, which also resets any outstanding cursors.
    pub fn reset(&self) -> Result<()> {
        unsafe { make_result(self.ptr.as_ref().reset.unwrap()(self.ptr.as_ptr()), ()) }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Empty the cursor cache otherwise closing the session may fail.
        self.clear_cursor_cache();
        // TODO: print something if this returns an error.
        unsafe { self.ptr.as_ref().close.unwrap()(self.ptr.as_ptr(), std::ptr::null()) };
    }
}

unsafe impl Send for Session {}

/// Inner representation of a cursor.
///
/// This inner representation is used by RecordCursor but also may be cached by Session.
struct InnerCursor {
    ptr: NonNull<WT_CURSOR>,
    uri: TableUri,
}

impl Drop for InnerCursor {
    fn drop(&mut self) {
        // TODO: log this.
        let _ = unsafe { self.ptr.as_ref().close.unwrap()(self.ptr.as_ptr()) };
    }
}

/// A `RecordCursor` facilities viewing and mutating data in a WiredTiger table where
/// the table is `i64` keyed and byte-string valued.
pub struct RecordCursor<'a> {
    inner: InnerCursor,
    session: &'a Session,
}

impl<'a> RecordCursor<'a> {
    fn new(inner: InnerCursor, session: &'a Session) -> Self {
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
            self.inner.ptr.as_ref().set_key.unwrap()(self.inner.ptr.as_ptr(), record.key());
            self.inner.ptr.as_ref().set_value.unwrap()(
                self.inner.ptr.as_ptr(),
                &Self::item_from_value(record.value()),
            );
            make_result(
                self.inner.ptr.as_ref().insert.unwrap()(self.inner.ptr.as_ptr()),
                (),
            )
        }
    }

    /// Remove a record by `key`.
    ///
    /// This may return a `WiredTigerError::NotFound` if the key does not exist in the collection.
    pub fn remove(&mut self, key: i64) -> Result<()> {
        unsafe {
            self.inner.ptr.as_ref().set_key.unwrap()(self.inner.ptr.as_ptr(), key);
            make_result(
                self.inner.ptr.as_ref().remove.unwrap()(self.inner.ptr.as_ptr()),
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
            match NonZero::new(self.inner.ptr.as_ref().next.unwrap()(
                self.inner.ptr.as_ptr(),
            )) {
                None => Some(self.record_view(None)),
                Some(code) if code.get() == WT_NOTFOUND => None,
                Some(code) => Some(Err(Error::from(code))),
            }
        }
    }

    /// Seek to the for `key` and return any associated `RecordView` if present.
    ///
    /// # Safety
    /// If the cursor's parent session returns this record during a transaction and that transaction
    /// is rolled back, we cannot guarantee that view value data is safe to access. Use
    /// `seek_exact()` to ensure safe access at the cost of a copy of the record value.
    pub unsafe fn seek_exact_unsafe(&mut self, key: i64) -> Option<Result<RecordView<'_>>> {
        unsafe {
            self.inner.ptr.as_ref().set_key.unwrap()(self.inner.ptr.as_ptr(), key);
            match NonZero::new(self.inner.ptr.as_ref().search.unwrap()(
                self.inner.ptr.as_ptr(),
            )) {
                None => Some(self.record_view(Some(key))),
                Some(code) if code.get() == WT_NOTFOUND => None,
                Some(code) => Some(Err(Error::from(code))),
            }
        }
    }

    /// Seek to the for `key` and return any associated `Record` if present.
    pub fn seek_exact(&mut self, key: i64) -> Option<Result<Record>> {
        unsafe { self.seek_exact_unsafe(key) }.map(|r| r.map(|v| v.to_owned()))
    }

    /// Return the largest key in the collection or `None` if the collection is empty.
    pub fn largest_key(&mut self) -> Option<Result<i64>> {
        unsafe {
            match NonZero::new(self.inner.ptr.as_ref().largest_key.unwrap()(
                self.inner.ptr.as_ptr(),
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
                self.inner.ptr.as_ref().reset.unwrap()(self.inner.ptr.as_ptr()),
                (),
            )
        }
    }

    fn into_inner(self) -> InnerCursor {
        self.inner
    }

    /// Return the current record key. This assumes that we have just positioned the cursor
    /// and WT_NOTFOUND will not be returned.
    fn record_key(&self) -> Result<i64> {
        unsafe {
            let mut k = 0i64;
            make_result(
                self.inner.ptr.as_ref().get_key.unwrap()(self.inner.ptr.as_ptr(), &mut k),
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
                self.inner.ptr.as_ref().get_value.unwrap()(self.inner.ptr.as_ptr(), &mut item),
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
}

impl<'a> Iterator for RecordCursor<'a> {
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
    fn new(session: &'a Session, cursor: RecordCursor<'a>) -> Self {
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

impl<'a> DerefMut for RecordCursorGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.cursor
    }
}

impl<'a> Drop for RecordCursorGuard<'a> {
    fn drop(&mut self) {
        // Safety: we never intend to allow RecordCursorGuard to drop the value.
        self.session
            .return_record_cursor(unsafe { ManuallyDrop::take(&mut self.cursor) });
    }
}
