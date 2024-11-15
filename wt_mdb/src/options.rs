use std::ffi::{c_char, CStr};

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