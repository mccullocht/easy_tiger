// TODO:
// * Transaction managed as an object, mutable function to create a transaction and
//   consuming functions to commit and rollback. Factory for RecordCursors.
// * integrate cursor pooling.

use std::{
    ffi::CStr,
    ptr::{self, NonNull},
    sync::Arc,
};

use wt_sys::{WT_CURSOR, WT_SESSION};

use crate::{
    connection::Connection,
    make_result, make_table_uri,
    options::{
        BeginTransactionOptions, CommitTransactionOptions, ConfigurationString, CreateOptions,
        DropOptions, RollbackTransactionOptions,
    },
    record_cursor::RecordCursor,
    wrap_ptr_create, RecordView, Result,
};

/// Inner state of a `Session`.
///
/// Mark this as Send+Sync so it can use meaningfully from an Arc.
/// *Safety*
/// - Session has an Arc<InnerCursor> but only access it through mut methods, ensuring that
///   we will not have concurrent access.
/// - RecordCursor has an Arc<InnerCursor> reference but does not use it.
pub(crate) struct InnerSession {
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
