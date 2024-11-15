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
