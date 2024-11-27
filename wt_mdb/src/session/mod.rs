mod record_cursor;
mod stat_cursor;

use std::{
    cell::RefCell,
    ffi::{CStr, CString},
    ops::Deref,
    ptr::{self, NonNull},
    sync::Arc,
};

use record_cursor::InnerCursor;
use wt_sys::{WT_CURSOR, WT_SESSION};

use crate::{
    connection::Connection,
    make_result,
    options::{
        BeginTransactionOptions, CommitTransactionOptions, ConfigurationString, CreateOptions,
        DropOptions, RollbackTransactionOptions, Statistics,
    },
    wrap_ptr_create, RecordView, Result,
};

pub use record_cursor::{RecordCursor, RecordCursorGuard};
pub use stat_cursor::StatCursor;

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
    fn return_record_cursor(&self, cursor: RecordCursor) {
        self.cached_cursors.borrow_mut().push(cursor.into_inner())
    }

    /// Remove all cached cursors.
    pub fn clear_cursor_cache(&self) {
        self.cached_cursors.borrow_mut().clear();
    }

    /// Return a new cursor that provides statistics by name.
    pub fn new_stats_cursor(
        &self,
        level: Statistics,
        table: Option<&str>,
    ) -> Result<StatCursor<'_>> {
        let table_stats_uri = table.map(|t| {
            CString::new(format!("statistics:table:{}", t)).expect("no nulls in table name")
        });
        let uri = table_stats_uri.as_deref().unwrap_or(c"statistics:");
        let options = level
            .to_config_string_clause()
            .map(|s| CString::new(s).expect("no nulls in stats options"));
        let mut cursorp: *mut WT_CURSOR = std::ptr::null_mut();
        unsafe {
            wrap_ptr_create(
                (self.ptr.as_ref().open_cursor.unwrap())(
                    self.ptr.as_ptr(),
                    uri.as_ptr(),
                    ptr::null_mut(),
                    options.as_ref().map(|o| o.as_ptr()).unwrap_or(std::ptr::null()),
                    &mut cursorp,
                ),
                cursorp,
            )
            .map(|ptr| StatCursor {
                ptr,
                _session: self,
            })
        }
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
