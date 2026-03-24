use crate::{
    make_result,
    session::{
        table_uri, BeginTransactionOptions, FormatString, Formatted, InnerCursor, Item, Session,
    },
    transaction::Transaction,
    wt_call, ConfigurationString, Error, Result, Statistics,
};
use std::{
    ffi::{CStr, CString},
    marker::PhantomData,
    num::NonZero,
    ptr::{self, NonNull},
    sync::Arc,
};

use tracing::error;
use wt_sys::{wiredtiger_open, WT_CONNECTION, WT_SESSION};

/// Global timestamp types that may be written by application code.
pub enum SetGlobalTimestampType {
    Durable,
    Oldest,
    Stable,
}

impl SetGlobalTimestampType {
    fn config_str(&self, ts: u64) -> CString {
        let s = match self {
            Self::Durable => format!("durable_timestamp={:x}", ts),
            Self::Oldest => format!("oldest_timestamp={:x}", ts),
            Self::Stable => format!("stable_timestamp={:x}", ts),
        };
        CString::new(s).expect("no nulls")
    }
}

// Global timestamp types that may be read by application code.
pub enum QueryGlobalTimestampType {
    AllDurable,
    BackupCheckpoint,
    LastCheckpoint,
    Oldest,
    OldestReader,
    Pinned,
    Recovery,
    Stable,
}

impl QueryGlobalTimestampType {
    fn config_str(&self) -> &'static CStr {
        match self {
            Self::AllDurable => c"get=all_durable",
            Self::BackupCheckpoint => c"get=backup_checkpoint",
            Self::LastCheckpoint => c"get=last_checkpoint",
            Self::Oldest => c"get=oldest_timestamp",
            Self::OldestReader => c"get=oldest_reader",
            Self::Pinned => c"get=pinned",
            Self::Recovery => c"get=recovery",
            Self::Stable => c"get=stable_timestamp",
        }
    }
}

/// Builder for options when connecting to a WiredTiger database.
#[derive(Default)]
pub struct OptionsBuilder {
    create: bool,
    cache_size_mb: Option<NonZero<usize>>,
    statistics: Statistics,
    checkpoint_log_size: usize,
    checkpoint_wait_seconds: usize,
    log: bool,
}

impl OptionsBuilder {
    /// If set, create the database if it does not exist.
    pub fn create(mut self) -> Self {
        self.create = true;
        self
    }

    /// Maximum heap memory to allocate for the cache, in MB.
    pub fn cache_size_mb(mut self, size: NonZero<usize>) -> Self {
        self.cache_size_mb = Some(size);
        self
    }

    /// Configure statistics collection.
    pub fn statistics(mut self, statistics: Statistics) -> Self {
        self.statistics = statistics;
        self
    }

    /// If non-zero, write a checkpoint every N bytes.
    pub fn checkpoint_log_size(mut self, log_size: usize) -> Self {
        self.checkpoint_log_size = log_size;
        self
    }

    /// If non-zero, write a checkpoint every N seconds.
    pub fn checkpoint_wait_seconds(mut self, wait_seconds: usize) -> Self {
        self.checkpoint_wait_seconds = wait_seconds;
        self
    }

    pub fn log(mut self, enable: bool) -> Self {
        self.log = enable;
        self
    }
}

/// Options when connecting to a WiredTiger database.
#[derive(Debug, Default)]
pub struct Options(Option<CString>);

impl From<OptionsBuilder> for Options {
    fn from(value: OptionsBuilder) -> Self {
        let mut options = Vec::new();
        if value.create {
            options.push("create".to_string())
        }
        if let Some(cache_size) = value.cache_size_mb {
            options.push(format!("cache_size={}", cache_size.get() << 20));
        }
        if let Some(clause) = value.statistics.to_config_string_clause() {
            options.push(clause);
        }
        if value.checkpoint_log_size > 0 || value.checkpoint_wait_seconds > 0 {
            options.push(format!(
                "checkpoint=(log_size={},wait={})",
                value.checkpoint_log_size, value.checkpoint_wait_seconds
            ))
        }
        if value.log {
            options.push("log=(enabled=true)".to_string());
        }
        if options.is_empty() {
            Self(None)
        } else {
            Self(Some(
                CString::new(options.join(",")).expect("options does not contain null"),
            ))
        }
    }
}

impl ConfigurationString for Options {
    fn as_config_string(&self) -> Option<&CStr> {
        self.0.as_deref()
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

/// A connection to a WiredTiger database.
///
/// There is typically only one connection per database per process.
/// `Connection`s may be freely shared between threads and are safe for concurrent access.
pub struct Connection(NonNull<WT_CONNECTION>);

impl Connection {
    /// Open a new `Connection` to a WiredTiger database.
    pub fn open(filename: &str, options: Option<Options>) -> Result<Arc<Self>> {
        let mut connp: *mut WT_CONNECTION = ptr::null_mut();
        let dbpath = CString::new(filename).unwrap();
        make_result(
            unsafe {
                wiredtiger_open(
                    dbpath.as_ptr(),
                    ptr::null_mut(),
                    options.unwrap_or_default().as_config_ptr(),
                    &mut connp,
                )
            },
            (),
        )?;
        NonNull::new(connp)
            .ok_or(Error::generic_error())
            .map(|conn| Arc::new(Connection(conn)))
    }

    /// Create a new `Session`. These can be used to obtain cursors to read and write data
    /// as well as manage transaction.
    pub fn open_session(self: &Arc<Self>) -> Result<Session> {
        let mut sessionp: *mut WT_SESSION = ptr::null_mut();
        unsafe {
            wt_call!(
                self.0,
                open_session,
                std::ptr::null_mut(),
                std::ptr::null(),
                &mut sessionp
            )
        }?;
        NonNull::new(sessionp)
            .ok_or(Error::generic_error())
            .map(|session| Session::new(session, self))
    }

    /// Query a connection global timestamp value maintained by WiredTiger.
    pub fn query_timestamp(&self, tt: QueryGlobalTimestampType) -> Result<u64> {
        let mut buf = [0u8; 17];
        unsafe {
            wt_call!(
                self.0,
                query_timestamp,
                buf.as_mut_ptr() as *mut std::ffi::c_char,
                tt.config_str().as_ptr()
            )?;
        }
        Ok(u64::from_str_radix(
            CStr::from_bytes_until_nul(&buf)
                .expect("valid CStr")
                .to_str()
                .expect("valid_utf8"),
            16,
        )
        .expect("valid hex"))
    }

    /// Set a connection global timestamp value maintained by WiredTiger.
    pub fn set_timestamp(&self, tt: SetGlobalTimestampType, ts: u64) -> Result<()> {
        let config_str = tt.config_str(ts);
        unsafe { wt_call!(self.0, set_timestamp, config_str.as_ptr()) }
    }

    /// Begin a new transaction with the specified options.
    pub fn begin_transaction(
        self: &Arc<Self>,
        // TODO(txn): move this into connection or transaction modules.
        options: Option<BeginTransactionOptions>,
    ) -> Result<Transaction> {
        Transaction::new(self, options)
    }

    /// Create a new table.
    pub fn create_table(
        self: &Arc<Self>,
        table_name: &str,
        config: Option<CreateOptions>,
    ) -> Result<()> {
        self.open_session()?.create_table(table_name, config)
    }

    /// Drop a table.
    ///
    /// This requires exclusive access -- if any cursors are open on the specified table the call will fail
    /// and return an EBUSY posix error.
    pub fn drop_table(
        self: &Arc<Self>,
        table_name: &str,
        config: Option<DropOptions>,
    ) -> Result<()> {
        self.open_session()?.drop_table(table_name, config)
    }

    /// Create a new table `table_name` with `create_options`, key format `K` and value format `V`,
    /// then return a cursor that may only be used to bulk insert records in order by key.
    pub fn new_bulk_load_cursor<K: Formatted, V: Formatted>(
        self: &Arc<Self>,
        table_name: &str,
        create_options: Option<CreateOptionsBuilder>,
    ) -> Result<BulkLoadCursor<K, V>> {
        let create_options: CreateOptions = create_options
            .unwrap_or_default()
            .key_format::<K>()
            .value_format::<V>()
            .into();
        let session = self.open_session()?;
        session.create_table(table_name, Some(create_options))?;
        let uri = table_uri(table_name);
        let inner = session.open_raw_cursor(&uri, Some(c"bulk"))?;
        Ok(BulkLoadCursor {
            inner,
            session,
            key_buf: vec![],
            value_buf: vec![],
            _km: PhantomData,
            _vm: PhantomData,
        })
    }

    /// Create a new table `table_name` and bulk load input from `iter` with key format `K` and
    /// value format `V`.
    ///
    /// This requires that `table_name` not exist or be empty and that `iter` yields records in
    /// order by `K` or an error may occur.
    pub fn bulk_load<K, V, I>(
        self: &Arc<Self>,
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
            cursor.append(k.to_formatted_ref(), v.to_formatted_ref())?;
        }
        Ok(())
    }
}

unsafe impl Send for Connection {}
unsafe impl Sync for Connection {}

/// A bulk load cursor may only be used for inserting new entries in order by key.
pub struct BulkLoadCursor<K, V> {
    inner: InnerCursor,
    // Session must be dropped after inner, so it is declared after inner.
    #[allow(dead_code)]
    session: Session,
    key_buf: Vec<u8>,
    value_buf: Vec<u8>,
    _km: PhantomData<K>,
    _vm: PhantomData<V>,
}

impl<K: Formatted, V: Formatted> BulkLoadCursor<K, V> {
    /// Append a new record to the table. Records must be inserted in order by key or an error may
    /// occur.
    pub fn append(&mut self, key: K::Ref<'_>, value: V::Ref<'_>) -> Result<()> {
        K::pack(key, &mut self.key_buf)?;
        V::pack(value, &mut self.value_buf)?;
        // safety: the memory passed to set_{key,value} need only be valid until a modifying
        // call like insert().
        unsafe {
            wt_call!(void self.inner.0, set_key, &Item::from(self.key_buf.as_slice()).0)?;
            wt_call!(void self.inner.0, set_value, &Item::from(self.value_buf.as_slice()).0)?;
            wt_call!(self.inner.0, insert)
        }
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        if let Err(e) = unsafe { wt_call!(self.0, close, std::ptr::null()) } {
            error!("Failed to close WT_CONNECTION: {}", e);
        }
    }
}
