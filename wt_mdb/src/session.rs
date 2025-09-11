mod format;
mod typed_cursor;

use std::{
    borrow::Cow,
    cell::RefCell,
    ffi::{c_char, c_void, CStr, CString},
    ptr::NonNull,
    sync::Arc,
};

use tracing::error;
use wt_sys::{WT_CURSOR, WT_ITEM, WT_SESSION};

use crate::{
    connection::Connection,
    options::{
        BeginTransactionOptions, CommitTransactionOptions, ConfigurationString, CreateOptions,
        CreateOptionsBuilder, DropOptions, RollbackTransactionOptions, Statistics,
    },
    wt_call, Error, Result,
};

pub use format::{pack1, pack2, pack3, unpack1, unpack2, unpack3, FormatString, Formatted};
pub use typed_cursor::{TypedCursor, TypedCursorGuard};

const METADATA_URI: &CStr = c"metadata:";

fn table_uri(name: &str) -> CString {
    CString::new([b"table:", name.as_bytes()].concat()).expect("no nulls")
}

/// Wrapper around [wt_sys::WT_ITEM].
#[derive(Debug, Copy, Clone)]
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
/// This inner representation is used by TypedCursor but also may be cached by Session.
struct InnerCursor(NonNull<WT_CURSOR>);

impl InnerCursor {
    fn uri(&self) -> &CStr {
        unsafe { CStr::from_ptr(self.0.as_ref().uri) }
    }

    fn key_format(&self) -> &CStr {
        self.format_or_default(unsafe { self.0.as_ref().key_format })
    }

    fn value_format(&self) -> &CStr {
        self.format_or_default(unsafe { self.0.as_ref().value_format })
    }

    fn format_or_default(&self, format: *const c_char) -> &CStr {
        if format.is_null() {
            c"u"
        } else {
            unsafe { CStr::from_ptr(format) }
        }
    }

    /// Reset the underlying [wt_sys::WT_CURSOR].
    fn reset(&mut self) -> Result<()> {
        unsafe { wt_call!(self.0, reset) }
    }
}

impl Drop for InnerCursor {
    fn drop(&mut self) {
        if let Err(e) = unsafe { wt_call!(self.0, close) } {
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
        let uri = table_uri(table_name);
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
        let uri = table_uri(table_name);
        unsafe {
            wt_call!(
                self.ptr,
                drop,
                uri.as_ptr(),
                config.unwrap_or_default().as_config_ptr()
            )
        }
    }

    /// Rename a table.
    ///
    /// This requires exclusive access -- if any cursors are open on the specified table the call will fail
    /// and return an EBUSY posix error.
    pub fn rename_table(&self, table_name: &str, new_table_name: &str) -> Result<()> {
        let old_uri = table_uri(table_name);
        let new_uri = table_uri(new_table_name);
        unsafe {
            wt_call!(
                self.ptr,
                rename,
                old_uri.as_ptr(),
                new_uri.as_ptr(),
                std::ptr::null()
            )
        }
    }

    /// Open a record cursor over the named table.
    ///
    /// Returns [rustix::io::Errno::INVAL] if the underlying table is not a record table.
    pub fn open_record_cursor(&self, table_name: &str) -> Result<RecordCursor<'_>> {
        self.new_typed_cursor::<i64, Vec<u8>>(table_name, None)
    }

    /// Open an index cursor over the named table.
    ///
    /// Returns [rustix::io::Errno::INVAL] if the underlying table is not an index table.
    pub fn open_index_cursor(&self, table_name: &str) -> Result<IndexCursor<'_>> {
        self.new_typed_cursor::<Vec<u8>, Vec<u8>>(table_name, None)
    }

    /// Open a cursor over database metadata.
    pub fn open_metadata_cursor(&self) -> Result<MetadataCursor<'_>> {
        self.new_typed_cursor_uri::<CString, CString>(METADATA_URI, None)
    }

    /// Return a new cursor that provides statistics keyed by WT constants.
    ///
    /// Accepts a level which may be no higher than the connection level, and optionally a table
    /// name to observe the stats of.
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
            .map(|s| CString::new(s.into_bytes()).expect("no nulls"));
        self.new_typed_cursor_uri::<i32, StatValue>(uri, options.as_deref())
    }

    /// Create a new cursor over `table_name` with the given `options` where `K` and `V` are formats
    /// used for the key and value.
    ///
    /// May fail if the table does not exist or if the key or value formats do not match.
    pub fn new_typed_cursor<K: Formatted, V: Formatted>(
        &self,
        table_name: &str,
        options: Option<&CStr>,
    ) -> Result<TypedCursor<'_, K, V>> {
        let uri = table_uri(table_name);
        self.new_typed_cursor_uri::<K, V>(&uri, options)
    }

    /// Get a cached [RecordCursor] or create a new cursor over `table_name`.
    pub fn get_record_cursor(&self, table_name: &str) -> Result<RecordCursorGuard<'_>> {
        let table_uri = table_uri(table_name);
        self.get_or_create_typed_cursor_uri::<i64, Vec<u8>>(&table_uri)
    }

    /// Get a cached [IndexCursor] or create a new cursor over `table_name`.
    pub fn get_index_cursor(&self, table_name: &str) -> Result<IndexCursorGuard<'_>> {
        let table_uri = table_uri(table_name);
        self.get_or_create_typed_cursor_uri::<Vec<u8>, Vec<u8>>(&table_uri)
    }

    /// Get a cached [MetadataCursor] or create a new metadata cursor.
    pub fn get_metadata_cursor(&self) -> Result<MetadataCursorGuard<'_>> {
        self.get_or_create_typed_cursor_uri(METADATA_URI)
    }

    /// Get a cached typed cursor or create a new typed cursor.
    pub fn get_or_create_typed_cursor<K: Formatted, V: Formatted>(
        &self,
        table_name: &str,
    ) -> Result<TypedCursorGuard<'_, K, V>> {
        let table_uri = table_uri(table_name);
        self.get_or_create_typed_cursor_uri(&table_uri)
    }

    // NB: this doesn't accept options because we don't check options when serving from the cache.
    fn get_or_create_typed_cursor_uri<K: Formatted, V: Formatted>(
        &self,
        uri: &CStr,
    ) -> Result<TypedCursorGuard<'_, K, V>> {
        let mut cursor_cache = self.cached_cursors.borrow_mut();
        cursor_cache
            .iter()
            .position(|c| c.uri() == uri)
            .map(|i| TypedCursor::new(cursor_cache.remove(i), self))
            .unwrap_or_else(|| self.new_typed_cursor_uri::<K, V>(uri, None))
            .map(TypedCursorGuard::new)
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

    fn new_typed_cursor_uri<K: Formatted, V: Formatted>(
        &self,
        uri: &CStr,
        options: Option<&CStr>,
    ) -> Result<TypedCursor<'_, K, V>> {
        let options: Cow<'_, CStr> = if let Some(o) = options {
            CString::new([o.to_bytes(), b",raw"].concat())
                .expect("no nulls")
                .into()
        } else {
            c"raw".into()
        };
        let mut cursorp: *mut WT_CURSOR = std::ptr::null_mut();
        unsafe {
            wt_call!(
                self.ptr,
                open_cursor,
                uri.as_ptr(),
                std::ptr::null_mut(),
                options.as_ptr(),
                &mut cursorp
            )
        }?;
        let inner = NonNull::new(cursorp)
            .ok_or(Error::generic_error())
            .map(InnerCursor)?;
        TypedCursor::new(inner, self)
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

    /// Create a new table `table_name` and bulk load input from `iter` with key format `K` and
    /// value format `V`.
    ///
    /// This requires that `table_name` not exist or be empty and that `iter` yields records in
    /// order by `K` or an error may occur.
    pub fn bulk_load<K, V, I>(
        &self,
        table_name: &str,
        create_options: Option<CreateOptionsBuilder>,
        iter: I,
    ) -> Result<()>
    where
        K: Formatted,
        V: Formatted,
        I: Iterator<Item = (K, V)>,
    {
        let mut cursor = self.new_bulk_load_cursor::<K, V>(table_name, create_options)?;
        for (k, v) in iter {
            cursor.insert(k.to_formatted_ref(), v.to_formatted_ref())?;
        }
        Ok(())
    }

    /// Create a new table `table_name` with `create_options`, key format `K` and value format `V`,
    /// then return a cursor that may only be used to bulk insert records in order by key.
    pub fn new_bulk_load_cursor<K: Formatted, V: Formatted>(
        &self,
        table_name: &str,
        create_options: Option<CreateOptionsBuilder>,
    ) -> Result<BulkLoadCursor<'_, K, V>> {
        let create_options: CreateOptions = create_options
            .unwrap_or_default()
            .key_format::<K>()
            .value_format::<V>()
            .into();
        self.create_table(table_name, Some(create_options))?;
        self.new_typed_cursor::<K, V>(table_name, Some(c"bulk"))
            .map(BulkLoadCursor)
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

/// Owned value produced by a StatCursor.
#[derive(Debug, Clone)]
pub struct StatValue {
    pub description: &'static CStr,
    pub value_str: CString,
    pub value: i64,
}

/// Ref value produced by a StatCursor.
#[derive(Debug, Copy, Clone)]
pub struct StatValueRef<'b> {
    /// Description of this stat.
    pub description: &'static CStr,
    /// The value of the stat rendered as a string.
    pub value_str: &'b CStr,
    /// The value of the stat coerced to a number if possible.
    pub value: i64,
}

impl<'a> From<StatValueRef<'a>> for StatValue {
    fn from(value: StatValueRef<'a>) -> Self {
        StatValue {
            description: value.description,
            value_str: value.value_str.into(),
            value: value.value,
        }
    }
}

impl Formatted for StatValue {
    const FORMAT: FormatString = FormatString::new(c"SSq");

    type Ref<'a> = StatValueRef<'a>;

    fn to_formatted_ref(&self) -> Self::Ref<'_> {
        StatValueRef {
            description: self.description,
            value_str: &self.value_str,
            value: self.value,
        }
    }

    fn pack(value: Self::Ref<'_>, packed: &mut Vec<u8>) -> Result<()> {
        pack3::<CString, CString, i64>(
            Self::FORMAT,
            value.description,
            value.value_str,
            value.value,
            packed,
        )
    }

    fn unpack<'b>(packed: &'b [u8]) -> Result<Self::Ref<'b>> {
        unpack3::<CString, CString, i64>(Self::FORMAT, packed).map(move |(a, b, c)| StatValueRef {
            description: unsafe { CStr::from_ptr::<'static>(a.as_ptr()) },
            value_str: b,
            value: c,
        })
    }
}

pub type RecordCursor<'a> = TypedCursor<'a, i64, Vec<u8>>;
pub type RecordCursorGuard<'a> = TypedCursorGuard<'a, i64, Vec<u8>>;
pub type IndexCursor<'a> = TypedCursor<'a, Vec<u8>, Vec<u8>>;
pub type IndexCursorGuard<'a> = TypedCursorGuard<'a, Vec<u8>, Vec<u8>>;
pub type MetadataCursor<'a> = TypedCursor<'a, CString, CString>;
pub type MetadataCursorGuard<'a> = TypedCursorGuard<'a, CString, CString>;
pub type StatCursor<'a> = TypedCursor<'a, i32, StatValue>;

/// A bulk load cursor may only be used for inserting new entries in order by key.
pub struct BulkLoadCursor<'a, K, V>(TypedCursor<'a, K, V>);

impl<'a, K: Formatted, V: Formatted> BulkLoadCursor<'a, K, V> {
    pub fn session(&self) -> &'a Session {
        self.0.session()
    }

    pub fn insert(&mut self, key: K::Ref<'_>, value: V::Ref<'_>) -> Result<()> {
        self.0.set(key, value)
    }
}

enum TxnState<'a> {
    Open(&'a Session),
    Closed,
}

/// An RAII guard for a transaction.
///
/// When created the guard begins a transaction. The caller may explicitly commit the transaction;
/// if they do not then the transaction is rolled back when the guard goes out of scope.
///
/// Two notable features of this API are:
/// * [`Session`] still has methods to begin/commit/rollback transactions. This API does not
///   prevent you from using those methods and doing so will likely result in a broken invariant.
/// * This API does not prevent attempts to nest transactions at compile time -- these will manifest
///   as runtime errors generated by WiredTiger.
// TODO: consider making begin/commit/rollback transactions module private.
pub struct TransactionGuard<'a>(TxnState<'a>);

impl<'a> TransactionGuard<'a> {
    /// Create a new transaction in `session`, with any specified `options`.
    pub fn new(session: &'a Session, options: Option<&BeginTransactionOptions>) -> Result<Self> {
        session.begin_transaction(options)?;
        Ok(Self(TxnState::Open(session)))
    }

    /// Commit this transaction with any specified `options`.
    pub fn commit(mut self, options: Option<&CommitTransactionOptions>) -> Result<()> {
        if let TxnState::Open(session) = self.0 {
            self.0 = TxnState::Closed;
            session.commit_transaction(options)?;
        };
        Ok(())
    }

    /// Rollback this transaction with any specified `options`.
    ///
    /// Note that dropping the guard will also rollback the transaction, although you may not specify
    /// any options in that case.
    pub fn rollback(mut self, options: Option<&RollbackTransactionOptions>) -> Result<()> {
        if let TxnState::Open(session) = self.0 {
            self.0 = TxnState::Closed;
            session.rollback_transaction(options)?;
        };
        Ok(())
    }
}

impl<'a> Drop for TransactionGuard<'a> {
    fn drop(&mut self) {
        if let TxnState::Open(session) = self.0 {
            if let Err(e) = session.rollback_transaction(None) {
                error!("Failed to rollback transaction: {}", e);
            }
        };
    }
}
