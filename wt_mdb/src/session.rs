// TODO:
// * integrate cursor pooling.

use std::{
    ffi::{c_void, CStr},
    num::NonZero,
    ptr::{self, NonNull},
    slice,
    sync::Arc,
};

use wt_sys::{WT_CURSOR, WT_ITEM, WT_NOTFOUND, WT_SESSION};

use crate::{
    connection::Connection,
    make_result, make_table_uri,
    options::{
        BeginTransactionOptions, CommitTransactionOptions, ConfigurationString, CreateOptions,
        DropOptions, RollbackTransactionOptions,
    },
    wrap_ptr_create, Error, Record, RecordView, Result,
};

/// Inner state of a `Session`.
///
/// Mark this as Send+Sync so it can use meaningfully from an Arc.
/// *Safety*
/// - Session has an Arc<InnerCursor> but only access it through mut methods, ensuring that
///   we will not have concurrent access.
/// - RecordCursor has an Arc<InnerCursor> reference but does not use it.
struct InnerSession {
    ptr: NonNull<WT_SESSION>,
    conn: Arc<Connection>,
}

unsafe impl Send for InnerSession {}
unsafe impl Sync for InnerSession {}

/// Close the underlying WiredTiger session.
impl Drop for InnerSession {
    fn drop(&mut self) {
        // TODO: print something if this returns an error.
        // I would not expect this to happen as we have structure things to guarantee that
        // `InnerSession` is only dropped when all cursors are closed.
        unsafe { self.ptr.as_ref().close.unwrap()(self.ptr.as_ptr(), std::ptr::null()) };
    }
}

/// A WiredTiger session.
///
/// Sessions are used to create cursors to view and mutate data and manage
/// transaction state.
pub struct Session(Arc<InnerSession>);

impl Session {
    pub(crate) fn new(session: NonNull<WT_SESSION>, connection: &Arc<Connection>) -> Self {
        Self(Arc::new(InnerSession {
            ptr: session,
            conn: connection.clone(),
        }))
    }

    /// Return the `Connection` this session belongs to.
    pub fn connection(&self) -> &Arc<Connection> {
        &self.0.conn
    }

    /// Create a new record table.
    pub fn create_record_table(
        &mut self,
        table_name: &str,
        config: Option<CreateOptions>,
    ) -> Result<()> {
        let uri = make_table_uri(table_name);
        unsafe {
            make_result(
                (self.0.ptr.as_ref().create.unwrap())(
                    self.0.ptr.as_ptr(),
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
    pub fn drop_record_table(
        &mut self,
        table_name: &str,
        config: Option<DropOptions>,
    ) -> Result<()> {
        let uri = make_table_uri(table_name);
        unsafe {
            make_result(
                self.0.ptr.as_ref().drop.unwrap()(
                    self.0.ptr.as_ptr(),
                    uri.as_ptr(),
                    config.unwrap_or_default().as_config_ptr(),
                ),
                (),
            )
        }
    }

    /// Open a record cursor over the named table.
    pub fn open_record_cursor(&mut self, table_name: &str) -> Result<RecordCursor> {
        self.open_record_cursor_with_options(table_name, None)
    }

    fn open_record_cursor_with_options(
        &mut self,
        table_name: &str,
        options: Option<&CStr>,
    ) -> Result<RecordCursor> {
        let uri = make_table_uri(table_name);
        let mut cursorp: *mut WT_CURSOR = ptr::null_mut();
        let result: i32;
        unsafe {
            result = (self.0.ptr.as_ref().open_cursor.unwrap())(
                self.0.ptr.as_ptr(),
                uri.as_ptr(),
                ptr::null_mut(),
                options.map(|s| s.as_ptr()).unwrap_or(std::ptr::null()),
                &mut cursorp,
            );
        }
        wrap_ptr_create(result, cursorp).map(|cursor| RecordCursor::new(cursor, self.0.clone()))
    }

    /// Starts a transaction in this session.
    ///
    /// The transaction remains active until `commit_transaction` or `rollback_transaction` are called.
    /// Operations performed on cursors already opened or opened before the transaction ends happen
    /// within the context of this transaction.
    ///
    /// This may not be called on a session with an active transaction or an error will be returned (EINVAL)
    /// but otherwise behavior is unspecified.
    pub fn begin_transaction(&mut self, options: Option<&BeginTransactionOptions>) -> Result<()> {
        unsafe {
            make_result(
                self.0.ptr.as_ref().begin_transaction.unwrap()(
                    self.0.ptr.as_ptr(),
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
    pub fn commit_transaction(&mut self, options: Option<&CommitTransactionOptions>) -> Result<()> {
        unsafe {
            make_result(
                self.0.ptr.as_ref().commit_transaction.unwrap()(
                    self.0.ptr.as_ptr(),
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
    pub fn rollback_transaction(
        &mut self,
        options: Option<&RollbackTransactionOptions>,
    ) -> Result<()> {
        unsafe {
            make_result(
                self.0.ptr.as_ref().rollback_transaction.unwrap()(
                    self.0.ptr.as_ptr(),
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
        &mut self,
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
    pub fn reset(&mut self) -> Result<()> {
        unsafe { make_result(self.0.ptr.as_ref().reset.unwrap()(self.0.ptr.as_ptr()), ()) }
    }
}

/// A `RecordCursor` facilities viewing and mutating data in a WiredTiger table where
/// the table is `i64` keyed and byte-string valued.
pub struct RecordCursor {
    cursor: NonNull<WT_CURSOR>,
    // Ref the InnerSession, *DO NOT USE*.
    //
    // We maintain this reference to ensure that the underlying WT_SESSION outlives this cursor.
    //
    // We cannot use Arc<Session> because Arc<T> is only Send if T: Send + Sync. If Session
    // methods use &mut self then they cannot be used through an Arc, and if we make Session
    // Sync then it may be erroneously called from multiple threads.
    //
    // An alternative to this would be to use Arc<Mutex<InnerSession>> and allow concurrent
    // calls to Session methods, which requires lock acquisition for every Session call.
    //
    // Note that we do not allow access to the Session from RecordCursor; doing so would be
    // unsound as we could leak a reference to the underlying WT_SESSION to another thread.
    _session: Arc<InnerSession>,
}

impl RecordCursor {
    fn new(cursor: NonNull<WT_CURSOR>, session: Arc<InnerSession>) -> Self {
        Self {
            cursor,
            _session: session,
        }
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
unsafe impl Send for RecordCursor {}

impl Drop for RecordCursor {
    fn drop(&mut self) {
        let _ = self.close_internal();
    }
}

impl Iterator for RecordCursor {
    type Item = Result<Record>;

    /// Advance and return the next record.
    ///
    /// If this cursor is unpositioned, returns to the start of the collection.
    fn next(&mut self) -> Option<Self::Item> {
        unsafe { self.next_unsafe() }.map(|r| r.map(RecordView::to_owned))
    }
}
