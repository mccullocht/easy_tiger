use std::{
    ffi::{c_char, CStr, CString},
    num::NonZero,
    str::FromStr,
};

pub(crate) trait ConfigurationString {
    fn as_config_string(&self) -> Option<&CStr>;

    fn as_config_ptr(&self) -> *const c_char {
        self.as_config_string()
            .map(|c| c.as_ptr())
            .unwrap_or(std::ptr::null())
    }
}

impl<C> ConfigurationString for Option<&C>
where
    C: ConfigurationString,
{
    fn as_config_string(&self) -> Option<&CStr> {
        self.map(ConfigurationString::as_config_string).flatten()
    }
}

/// Builder for options when connecting to a WiredTiger database.
#[derive(Default)]
pub struct ConnectionOptionsBuilder {
    create: bool,
    cache_size_mb: Option<NonZero<usize>>,
    statistics: Statistics,
    checkpoint_log_size: usize,
    checkpoint_wait_seconds: usize,
}

impl ConnectionOptionsBuilder {
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
}

/// Level of statistics gathering.
///
/// This is set on both the connection and when accessing stats cursors.
/// Note that the level for a stats cursor must be less than the connection level
/// or an error may occur.
#[derive(Debug, PartialEq, Eq)]
pub enum Statistics {
    /// Collect no stats.
    None,
    /// Collect stats that are fast/cheap to collect.
    Fast,
    /// Collect all known stats, even if they are expensive.
    All,
}

impl Statistics {
    pub(crate) fn to_config_string_clause(&self) -> Option<String> {
        match self {
            Self::None => None,
            opt => Some(format!("statistics=({opt})")),
        }
    }
}

impl Default for Statistics {
    fn default() -> Self {
        Self::None
    }
}

impl std::fmt::Display for Statistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::None => "none",
                Self::Fast => "fast",
                Self::All => "all",
            }
        )
    }
}

impl FromStr for Statistics {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(Self::None),
            "fast" => Ok(Self::Fast),
            "all" => Ok(Self::All),
            _ => Err(format!("Unknown statistics type {s}")),
        }
    }
}

/// Options when connecting to a WiredTiger database.
#[derive(Debug, Default)]
pub struct ConnectionOptions(Option<CString>);

impl From<ConnectionOptionsBuilder> for ConnectionOptions {
    fn from(value: ConnectionOptionsBuilder) -> Self {
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
        if options.is_empty() {
            Self(None)
        } else {
            Self(Some(
                CString::new(options.join(",")).expect("options does not contain null"),
            ))
        }
    }
}

impl ConfigurationString for ConnectionOptions {
    fn as_config_string(&self) -> Option<&CStr> {
        self.0.as_deref()
    }
}

/// Type of WiredTiger table.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub enum TableType {
    /// Record-type tables are `i64` keyed and byte-array valued.
    Record,
    /// Index-type tables are byte-array keyed and values.
    Index,
}

impl Default for TableType {
    fn default() -> Self {
        Self::Record
    }
}

/// An options builder for creating a table, column group, index, or file in WiredTiger.
#[derive(Default, Debug, Hash, Eq, PartialEq, Clone)]
pub struct CreateOptionsBuilder {
    table_type: TableType,
    app_metadata: Option<String>,
}

impl CreateOptionsBuilder {
    /// Set the table type for this table.
    pub fn table_type(mut self, table_type: TableType) -> Self {
        self.table_type = table_type;
        self
    }

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
            match value.table_type {
                TableType::Record => "key_format=q".to_owned(),
                TableType::Index => "key_format=u".to_owned(),
            },
            "value_format=u".into(),
        ];
        if let Some(metadata) = value.app_metadata {
            parts.push(metadata);
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
}

impl BeginTransactionOptionsBuilder {
    /// Set the name of the transaction, for debugging purposes.
    pub fn set_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_owned());
        self
    }
}

/// Options when beginning a new transaction.
#[derive(Debug, Hash, Eq, PartialEq, Clone, Default)]
pub struct BeginTransactionOptions(Option<CString>);

impl From<BeginTransactionOptionsBuilder> for BeginTransactionOptions {
    fn from(value: BeginTransactionOptionsBuilder) -> Self {
        Self(
            value
                .name
                .map(|n| CString::new(format!("name={n}")).expect("name has no nulls")),
        )
    }
}

impl ConfigurationString for BeginTransactionOptions {
    fn as_config_string(&self) -> Option<&CStr> {
        self.0.as_deref()
    }
}

/// Options build for commit_transaction() operations.
#[derive(Default)]
pub struct CommitTransactionOptionsBuilder {
    operation_timeout_ms: Option<u32>,
}

impl CommitTransactionOptionsBuilder {
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
pub struct CommitTransactionOptions(Option<CString>);

impl From<CommitTransactionOptionsBuilder> for CommitTransactionOptions {
    fn from(value: CommitTransactionOptionsBuilder) -> Self {
        Self(
            value
                .operation_timeout_ms
                .map(|t| CString::new(format!("operation_timeout_ms={t}")).expect("no nulls")),
        )
    }
}

impl ConfigurationString for CommitTransactionOptions {
    fn as_config_string(&self) -> Option<&CStr> {
        self.0.as_deref()
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
