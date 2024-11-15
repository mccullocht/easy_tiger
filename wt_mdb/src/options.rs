use std::{
    ffi::{c_char, CStr, CString},
    num::NonZero,
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

/// An options builder for creating a table, column group, index, or file in WiredTiger.
#[derive(Default)]
pub struct CreateOptionsBuilder;

/// Options when creating a table, column group, index, or file in WiredTiger.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct CreateOptions(CString);

impl Default for CreateOptions {
    fn default() -> Self {
        CreateOptions(CString::from(c"key_format=q,value_format=u"))
    }
}

impl From<CreateOptionsBuilder> for CreateOptions {
    fn from(_value: CreateOptionsBuilder) -> Self {
        Self::default()
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
#[derive(Default, Debug)]
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
                .map(|n| CString::new(format!("name={}", n)).expect("name has no nulls")),
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
                .map(|t| CString::new(format!("operation_timeout_ms={}", t)).expect("no nulls")),
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
                .map(|t| CString::new(format!("operation_timeout_ms={}", t)).expect("no nulls")),
        )
    }
}

impl ConfigurationString for RollbackTransactionOptions {
    fn as_config_string(&self) -> Option<&CStr> {
        self.0.as_deref()
    }
}
