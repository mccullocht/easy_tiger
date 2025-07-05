mod format;
mod index_cursor;
mod metadata_cursor;
mod record_cursor;
mod stat_cursor;

use std::{
    cell::RefCell,
    ffi::{c_void, CStr, CString},
    ops::Deref,
    ptr::NonNull,
    sync::Arc,
};

use metadata_cursor::{MetadataCursor, MetadataCursorGuard};
use rustix::io::Errno;
use tracing::error;
use wt_sys::{WT_CURSOR, WT_ITEM, WT_SESSION};

use crate::{
    connection::Connection,
    options::{
        BeginTransactionOptions, CommitTransactionOptions, ConfigurationString, CreateOptions,
        DropOptions, RollbackTransactionOptions, Statistics, TableType,
    },
    wt_call, Error, Result,
};

pub use index_cursor::{IndexCursor, IndexCursorGuard, IndexRecord, IndexRecordView};
pub use record_cursor::{Record, RecordCursor, RecordCursorGuard, RecordView};
pub use stat_cursor::StatCursor;

const METADATA_URI: &CStr = c"metadata:";

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
        Self(CString::new(format!("table:{value}")).expect("no nulls in table name"))
    }
}

/// Wrapper around [wt_sys::WT_ITEM].
#[derive(Copy, Clone)]
struct Item(WT_ITEM);

// Use an empty slice so that the default pointer is not null.
const EMPTY_ITEM: &[u8] = &[];

impl Default for Item {
    fn default() -> Self {
        Self::from(EMPTY_ITEM)
    }
}

impl From<&[u8]> for Item {
    fn from(value: &[u8]) -> Self {
        Self(WT_ITEM {
            data: value.as_ptr() as *const c_void,
            size: value.len(),
            mem: std::ptr::null_mut(),
            memsize: 0,
            flags: 0,
        })
    }
}

impl From<Item> for &[u8] {
    fn from(value: Item) -> Self {
        unsafe { std::slice::from_raw_parts(value.0.data as *const u8, value.0.size) }
    }
}

/// Inner representation of a cursor.
///
/// This inner representation is used by RecordCursor but also may be cached by Session.
struct InnerCursor {
    pub ptr: NonNull<WT_CURSOR>,
    pub uri: TableUri,
}

impl InnerCursor {
    /// Interrogate the underlying cursor for the [TableType].
    ///
    /// Returns `None` if the table type cannot be determined.
    fn table_type(&self) -> Option<TableType> {
        let key_format = unsafe {
            NonNull::new(self.ptr.as_ref().key_format as *mut i8)
                .map(|p| CStr::from_ptr(p.as_ptr() as *const i8))
        }
        .unwrap_or(c"u");
        match key_format.to_bytes() {
            b"q" => Some(TableType::Record),
            b"u" => Some(TableType::Index),
            _ => None,
        }
    }

    /// Reset the underlying [wt_sys::WT_CURSOR].
    fn reset(&mut self) -> Result<()> {
        unsafe { wt_call!(self.ptr, reset) }
    }
}

impl Drop for InnerCursor {
    fn drop(&mut self) {
        if let Err(e) = unsafe { wt_call!(self.ptr, close) } {
            error!("Failed to close WT_CURSOR: {}", e);
        }
    }
}

unsafe impl Send for InnerCursor {}

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

    /// Create a new table.
    pub fn create_table(&self, table_name: &str, config: Option<CreateOptions>) -> Result<()> {
        let uri = TableUri::from(table_name);
        unsafe {
            wt_call!(
                self.ptr,
                create,
                uri.as_ptr(),
                config.unwrap_or_default().as_config_ptr()
            )
        }
    }

    /// Drop a table.
    ///
    /// This requires exclusive access -- if any cursors are open on the specified table the call will fail
    /// and return an EBUSY posix error.
    pub fn drop_table(&self, table_name: &str, config: Option<DropOptions>) -> Result<()> {
        let uri = TableUri::from(table_name);
        unsafe {
            wt_call!(
                self.ptr,
                drop,
                uri.as_ptr(),
                config.unwrap_or_default().as_config_ptr()
            )
        }
    }

    /// Open a record cursor over the named table.
    ///
    /// Returns [rustix::io::Errno::INVAL] if the underlying table is not a record table.
    pub fn open_record_cursor(&self, table_name: &str) -> Result<RecordCursor> {
        self.open_record_cursor_with_options(table_name, None)
    }

    fn open_record_cursor_with_options(
        &self,
        table_name: &str,
        options: Option<&CStr>,
    ) -> Result<RecordCursor> {
        self.open_typed_cursor(table_name, options, TableType::Record)
            .map(|c| RecordCursor::new(c, self))
    }

    /// Open an index cursor over the named table.
    ///
    /// Returns [rustix::io::Errno::INVAL] if the underlying table is not an index table.
    pub fn open_index_cursor(&self, table_name: &str) -> Result<IndexCursor> {
        self.open_typed_cursor(table_name, None, TableType::Index)
            .map(|c| IndexCursor::new(c, self))
    }

    fn open_typed_cursor(
        &self,
        table_name: &str,
        options: Option<&CStr>,
        expected_table_type: TableType,
    ) -> Result<InnerCursor> {
        let uri = TableUri::from(table_name);
        let inner = self
            .new_cursor_pointer(&uri.0, options)
            .map(|ptr| InnerCursor { ptr, uri })?;
        if inner.table_type().is_some_and(|t| t == expected_table_type) {
            Ok(inner)
        } else {
            Err(Error::Errno(Errno::INVAL))
        }
    }

    /// Open a cursor over database metadata.
    pub fn open_metadata_cursor(&self) -> Result<MetadataCursor> {
        self.new_cursor_pointer(METADATA_URI, None).map(|ptr| {
            MetadataCursor::new(
                InnerCursor {
                    ptr,
                    uri: TableUri(METADATA_URI.to_owned()),
                },
                self,
            )
        })
    }

    /// Get a cached [RecordCursor] or create a new cursor over `table_name`.
    pub fn get_record_cursor(&self, table_name: &str) -> Result<RecordCursorGuard<'_>> {
        self.get_typed_cursor(table_name, TableType::Record)
            .map(|c| RecordCursorGuard::new(self, RecordCursor::new(c, self)))
    }

    /// Get a cached [IndexCursor] or create a new cursor over `table_name`.
    pub fn get_index_cursor(&self, table_name: &str) -> Result<IndexCursorGuard<'_>> {
        self.get_typed_cursor(table_name, TableType::Index)
            .map(|c| IndexCursorGuard::new(self, IndexCursor::new(c, self)))
    }

    fn get_typed_cursor(
        &self,
        table_name: &str,
        expected_table_type: TableType,
    ) -> Result<InnerCursor> {
        let mut cursor_cache = self.cached_cursors.borrow_mut();
        cursor_cache
            .iter()
            .position(|c| c.uri.table_name().to_bytes() == table_name.as_bytes())
            .map(|i| {
                let inner = cursor_cache.remove(i);
                Ok(inner)
            })
            .unwrap_or_else(|| self.open_typed_cursor(table_name, None, expected_table_type))
    }

    pub fn get_metadata_cursor(&self) -> Result<MetadataCursorGuard<'_>> {
        let mut cursor_cache = self.cached_cursors.borrow_mut();
        cursor_cache
            .iter()
            .position(|c| c.uri.table_name() == METADATA_URI)
            .map(|i| Ok(MetadataCursor::new(cursor_cache.remove(i), self)))
            .unwrap_or_else(|| self.open_metadata_cursor())
            .map(MetadataCursorGuard::new)
    }

    /// Return an `InnerCursor` to the cache for future re-use.
    fn return_cursor(&self, mut cursor: InnerCursor) {
        let _ = cursor.reset();
        self.cached_cursors.borrow_mut().push(cursor)
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
            CString::new(format!("statistics:table:{t}")).expect("no nulls in table name")
        });
        let uri = table_stats_uri.as_deref().unwrap_or(c"statistics:");
        let options = level
            .to_config_string_clause()
            .map(|s| CString::new(s).expect("no nulls in stats options"));
        self.new_cursor_pointer(uri, options.as_deref())
            .map(|ptr| StatCursor {
                ptr,
                _session: self,
            })
    }

    fn new_cursor_pointer(&self, uri: &CStr, options: Option<&CStr>) -> Result<NonNull<WT_CURSOR>> {
        let mut cursorp: *mut WT_CURSOR = std::ptr::null_mut();
        unsafe {
            wt_call!(
                self.ptr,
                open_cursor,
                uri.as_ptr(),
                std::ptr::null_mut(),
                options.map(CStr::as_ptr).unwrap_or(std::ptr::null()),
                &mut cursorp
            )
        }?;
        NonNull::new(cursorp).ok_or(Error::generic_error())
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
        unsafe { wt_call!(self.ptr, begin_transaction, options.as_config_ptr()) }
    }

    /// Commit the current transaction.
    ///
    /// If this method returns an error, the transaction was rolled back and all cursors associated with the session
    /// are reset.
    ///
    /// A transaction must be in progress when this method is called or an error will be returned (EINVAL) but behavior
    /// is otherwise unspecified.
    pub fn commit_transaction(&self, options: Option<&CommitTransactionOptions>) -> Result<()> {
        unsafe { wt_call!(self.ptr, commit_transaction, options.as_config_ptr()) }
    }

    /// Rollback the current transaction.
    ///
    /// All cursors associated with the session are reset.
    ///
    /// A transaction must be in progress when this method is called or an error will be returned (EINVAL) but behavior
    /// is otherwise unspecified.
    pub fn rollback_transaction(&self, options: Option<&RollbackTransactionOptions>) -> Result<()> {
        unsafe { wt_call!(self.ptr, rollback_transaction, options.as_config_ptr()) }
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
        self.create_table(table_name, options)?;
        let mut cursor = self.open_record_cursor_with_options(table_name, Some(c"bulk=true"))?;
        for record in iter {
            cursor.set(&record)?;
        }
        Ok(())
    }

    /// Checkpoint the database.
    pub fn checkpoint(&self) -> Result<()> {
        unsafe { wt_call!(self.ptr, checkpoint, std::ptr::null()) }
    }

    /// Reset this session, which also resets any outstanding cursors.
    pub fn reset(&self) -> Result<()> {
        unsafe { wt_call!(self.ptr, reset) }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Empty the cursor cache otherwise closing the session may fail.
        self.clear_cursor_cache();
        if let Err(e) = unsafe { wt_call!(self.ptr, close, std::ptr::null()) } {
            error!("Failed to close WT_SESSION: {}", e);
        }
    }
}

unsafe impl Send for Session {}
