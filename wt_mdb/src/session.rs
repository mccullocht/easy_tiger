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

use crate::{connection::Connection, wt_call, ConfigurationString, Error, Result, Statistics};

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

/// Used to select the type of query timestamp to read from the session.
pub enum QueryTransactionTimestampType {
    Commit,
    FirstCommit,
    Prepare,
    Read,
}

impl QueryTransactionTimestampType {
    fn config_str(&self) -> &'static CStr {
        match self {
            Self::Commit => c"get=commit",
            Self::FirstCommit => c"get=first_commit",
            Self::Prepare => c"get=prepare",
            Self::Read => c"get=read",
        }
    }
}

/// Types of timestamps that may be set on a transaction.
///
/// These timestamps can be set in between operations on the transaction, so it is possible for
/// callers to use different timestamps for different parts of the transaction.
pub enum SetTransactionTimestampType {
    Commit,
    Durable,
    Prepare,
    Read,
}

impl SetTransactionTimestampType {
    fn txn_type(&self) -> wt_sys::WT_TS_TXN_TYPE {
        match self {
            Self::Commit => wt_sys::WT_TS_TXN_TYPE_WT_TS_TXN_TYPE_COMMIT,
            Self::Durable => wt_sys::WT_TS_TXN_TYPE_WT_TS_TXN_TYPE_DURABLE,
            Self::Prepare => wt_sys::WT_TS_TXN_TYPE_WT_TS_TXN_TYPE_PREPARE,
            Self::Read => wt_sys::WT_TS_TXN_TYPE_WT_TS_TXN_TYPE_READ,
        }
    }
}

/// An options builder for creating a table, column group, index, or file in WiredTiger.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct CreateOptionsBuilder {
    key_format: FormatString,
    value_format: FormatString,
    app_metadata: Option<String>,
}

impl Default for CreateOptionsBuilder {
    fn default() -> Self {
        Self {
            key_format: FormatString::new(c"q"),
            value_format: FormatString::new(c"u"),
            app_metadata: None,
        }
    }
}

impl CreateOptionsBuilder {
    /// Set the format for the key.
    pub fn key_format<K: Formatted>(mut self) -> Self {
        self.key_format = K::FORMAT;
        self
    }

    /// Set the format for the value.
    pub fn value_format<V: Formatted>(mut self) -> Self {
        self.value_format = V::FORMAT;
        self
    }

    /// Attach metadata that can be read from the metadata table as the value for this table.
    pub fn app_metadata(mut self, metadata: &str) -> Self {
        assert!(
            !metadata.as_bytes().contains(&0),
            "metadata may not contain a NULL character"
        );
        self.app_metadata = Some(metadata.to_owned());
        self
    }
}

/// Options when creating a table, column group, index, or file in WiredTiger.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct CreateOptions(CString);

impl Default for CreateOptions {
    fn default() -> Self {
        CreateOptionsBuilder::default().into()
    }
}

impl From<CreateOptionsBuilder> for CreateOptions {
    fn from(value: CreateOptionsBuilder) -> Self {
        let mut parts = vec![
            format!("key_format={}", value.key_format.format_str()),
            format!("value_format={}", value.value_format.format_str()),
        ];
        if let Some(metadata) = value.app_metadata {
            parts.push(format!("app_metadata={metadata}"));
        }
        Self(CString::new(parts.join(",")).expect("no nulls"))
    }
}

impl ConfigurationString for CreateOptions {
    fn as_config_string(&self) -> Option<&CStr> {
        Some(self.0.as_c_str())
    }
}

/// An options builder for dropping a table, column group, index, or file in WiredTiger.
#[derive(Default)]
pub struct DropOptionsBuilder {
    force: bool,
}

impl DropOptionsBuilder {
    /// If set, return success even if the object does not exist.
    pub fn set_force(mut self) -> Self {
        self.force = true;
        self
    }
}

/// Options for dropping a table, column group, index, or file in WiredTiger.
#[derive(Default, Debug, Clone)]
pub struct DropOptions(Option<CString>);

impl From<DropOptionsBuilder> for DropOptions {
    fn from(value: DropOptionsBuilder) -> Self {
        DropOptions(if value.force {
            Some(CString::from(c"force=true"))
        } else {
            None
        })
    }
}

impl ConfigurationString for DropOptions {
    fn as_config_string(&self) -> Option<&CStr> {
        self.0.as_deref()
    }
}

/// Options builder when beginning a WiredTiger transaction.
#[derive(Default)]
pub struct BeginTransactionOptionsBuilder {
    name: Option<String>,
    read_timestamp: Option<u64>,
}

impl BeginTransactionOptionsBuilder {
    /// Set the name of the transaction, for debugging purposes.
    pub fn set_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_owned());
        self
    }

    /// Set the read timestamp for the transaction.
    ///
    /// Note that the read timestamp may only be set once per transaction; if it is set here then
    /// subsequent calls to set_transaction_timestamp for the read timestamp will fail.
    pub fn set_read_timestamp(mut self, ts: u64) -> Self {
        self.read_timestamp = Some(ts);
        self
    }
}

/// Options when beginning a new transaction.
#[derive(Debug, Hash, Eq, PartialEq, Clone, Default)]
pub struct BeginTransactionOptions {
    options: Option<CString>,
    read_timestamp: Option<u64>,
}

impl BeginTransactionOptions {
    pub fn with_read_timestamp(ts: u64) -> Self {
        Self {
            options: None,
            read_timestamp: Some(ts),
        }
    }
}

impl From<BeginTransactionOptionsBuilder> for BeginTransactionOptions {
    fn from(value: BeginTransactionOptionsBuilder) -> Self {
        Self {
            options: value
                .name
                .map(|n| CString::new(format!("name={n}")).expect("name has no nulls")),
            read_timestamp: value.read_timestamp,
        }
    }
}

impl ConfigurationString for BeginTransactionOptions {
    fn as_config_string(&self) -> Option<&CStr> {
        self.options.as_deref()
    }
}

/// Options build for commit_transaction() operations.
#[derive(Default)]
pub struct CommitTransactionOptionsBuilder {
    commit_timestamp: Option<u64>,
    operation_timeout_ms: Option<u32>,
}

impl CommitTransactionOptionsBuilder {
    /// Set the commit timestamp for the transaction.
    pub fn set_commit_timestamp(mut self, ts: u64) -> Self {
        self.commit_timestamp = Some(ts);
        self
    }

    /// When set to a non-zero value acts a requested time limit for the operations in ms.
    /// This is not a guarantee -- the operation may still take longer than the timeout.
    /// If the limit is reached the operation may be rolled back.
    pub fn operation_timeout_ms(mut self, timeout: Option<u32>) -> Self {
        self.operation_timeout_ms = timeout;
        self
    }
}

/// Options for commit_transaction() operations.
#[derive(Debug, Hash, Eq, PartialEq, Clone, Default)]
pub struct CommitTransactionOptions {
    options: Option<CString>,
    commit_timestamp: Option<u64>,
}

impl CommitTransactionOptions {
    pub fn with_commit_timestamp(ts: u64) -> Self {
        Self {
            options: None,
            commit_timestamp: Some(ts),
        }
    }
}

impl From<CommitTransactionOptionsBuilder> for CommitTransactionOptions {
    fn from(value: CommitTransactionOptionsBuilder) -> Self {
        Self {
            options: value
                .operation_timeout_ms
                .map(|t| CString::new(format!("operation_timeout_ms={t}")).expect("no nulls")),
            commit_timestamp: value.commit_timestamp,
        }
    }
}

impl ConfigurationString for CommitTransactionOptions {
    fn as_config_string(&self) -> Option<&CStr> {
        self.options.as_deref()
    }
}

/// Options build for rollback_transaction() operations.
#[derive(Default)]
pub struct RollbackTransactionOptionsBuilder {
    operation_timeout_ms: Option<u32>,
}

impl RollbackTransactionOptionsBuilder {
    /// When set to a non-zero value acts a requested time limit for the operations in ms.
    /// This is not a guarantee -- the operation may still take longer than the timeout.
    /// If the limit is reached the operation may be rolled back.
    pub fn operation_timeout_ms(mut self, timeout: Option<u32>) -> Self {
        self.operation_timeout_ms = timeout;
        self
    }
}

/// Options for rollback_transaction() operations.
#[derive(Debug, Hash, Eq, PartialEq, Clone, Default)]
pub struct RollbackTransactionOptions(Option<CString>);

impl From<RollbackTransactionOptionsBuilder> for RollbackTransactionOptions {
    fn from(value: RollbackTransactionOptionsBuilder) -> Self {
        Self(
            value
                .operation_timeout_ms
                .map(|t| CString::new(format!("operation_timeout_ms={t}")).expect("no nulls")),
        )
    }
}

impl ConfigurationString for RollbackTransactionOptions {
    fn as_config_string(&self) -> Option<&CStr> {
        self.0.as_deref()
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

    /// Open a new transaction on this session with options.
    pub fn transaction(
        &self,
        options: Option<BeginTransactionOptions>,
    ) -> Result<TransactionGuard<'_>> {
        TransactionGuard::new(self, options)
    }

    /// Starts a transaction in this session.
    ///
    /// The transaction remains active until `commit_transaction` or `rollback_transaction` are called.
    /// Operations performed on cursors already opened or opened before the transaction ends happen
    /// within the context of this transaction.
    ///
    /// This may not be called on a session with an active transaction or an error will be returned (EINVAL)
    /// but otherwise behavior is unspecified.
    pub fn begin_transaction(&self, options: Option<BeginTransactionOptions>) -> Result<()> {
        unsafe { wt_call!(self.ptr, begin_transaction, options.as_config_ptr()) }?;
        if let Some(ts) = options.and_then(|o| o.read_timestamp) {
            self.set_transaction_timestamp(SetTransactionTimestampType::Read, ts)?;
        }
        Ok(())
    }

    /// Commit the current transaction.
    ///
    /// If this method returns an error, the transaction was rolled back and all cursors associated with the session
    /// are reset.
    ///
    /// A transaction must be in progress when this method is called or an error will be returned (EINVAL) but behavior
    /// is otherwise unspecified.
    pub fn commit_transaction(&self, options: Option<CommitTransactionOptions>) -> Result<()> {
        if let Some(ts) = options.as_ref().and_then(|o| o.commit_timestamp) {
            self.set_transaction_timestamp(SetTransactionTimestampType::Commit, ts)?;
        }
        unsafe { wt_call!(self.ptr, commit_transaction, options.as_config_ptr()) }
    }

    /// Rollback the current transaction.
    ///
    /// All cursors associated with the session are reset.
    ///
    /// A transaction must be in progress when this method is called or an error will be returned (EINVAL) but behavior
    /// is otherwise unspecified.
    pub fn rollback_transaction(&self, options: Option<RollbackTransactionOptions>) -> Result<()> {
        unsafe { wt_call!(self.ptr, rollback_transaction, options.as_config_ptr()) }
    }

    /// Set the transaction timestamp for the current transaction.
    ///
    /// This should be called on a session with an active transaction. This may be called in the
    /// during a transaction to use a different commit timestamp for different parts of the
    /// transaction. For read timestamps this may only be called once per transaction, including
    /// any read timestamp set when the transaction begins.
    ///
    /// Timestamps are bound in part by global state; consult WiredTiger documentation for more
    /// information.
    pub fn set_transaction_timestamp(
        &self,
        txn_type: SetTransactionTimestampType,
        timestamp: u64,
    ) -> Result<()> {
        unsafe {
            wt_call!(
                self.ptr,
                timestamp_transaction_uint,
                txn_type.txn_type(),
                timestamp
            )
        }
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

    /// Query the session for transaction timestamp state. Callers may select one of the types of
    /// timestamps to query. Returns a 64-bit unsigned timestamp.
    pub fn query_transaction_timestamp(
        &self,
        timestamp: QueryTransactionTimestampType,
    ) -> Result<u64> {
        let mut buf = [0u8; 17];
        unsafe {
            wt_call!(
                self.ptr,
                query_timestamp,
                buf.as_mut_ptr() as *mut c_char,
                timestamp.config_str().as_ptr()
            )
        }?;
        Ok(u64::from_str_radix(
            CStr::from_bytes_until_nul(&buf)
                .expect("null terminated")
                .to_str()
                .expect("valid utf8"),
            16,
        )
        .expect("valid hex string"))
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
    pub fn new(session: &'a Session, options: Option<BeginTransactionOptions>) -> Result<Self> {
        session.begin_transaction(options)?;
        Ok(Self(TxnState::Open(session)))
    }

    /// Commit this transaction with any specified `options`.
    pub fn commit(mut self, options: Option<CommitTransactionOptions>) -> Result<()> {
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
    pub fn rollback(mut self, options: Option<RollbackTransactionOptions>) -> Result<()> {
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
