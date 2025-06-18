//! WiredTiger config parsing APIs.
//!
//! These are useful when writing WiredTiger extensions or reading db metadata.

use std::{ffi::CString, os::raw::c_char, ptr::NonNull};

use tracing::error;
use wt_sys::{WT_CONFIG_ITEM, WT_CONFIG_PARSER};

use crate::{make_result, map_not_found, wt_call, Error, Result};

/// A parser for WiredTiger config strings.
///
/// This struct allows both linear and random access over the config.
/// If random access is used, iteration behavior is not well defined.
pub struct ConfigParser<'a> {
    ptr: NonNull<wt_sys::WT_CONFIG_PARSER>,
    config: &'a str,
}

impl<'a> ConfigParser<'a> {
    /// Create a new config parser.
    ///
    /// Errors that occur will be logged to stderr.
    pub fn new(config: &'a str) -> Result<Self> {
        let mut handle: *mut WT_CONFIG_PARSER = std::ptr::null_mut();
        make_result(
            unsafe {
                wt_sys::wiredtiger_config_parser_open(
                    std::ptr::null_mut(),
                    config.as_bytes().as_ptr() as *const c_char,
                    config.len(),
                    &mut handle,
                )
            },
            (),
        )?;
        NonNull::new(handle)
            .ok_or(Error::generic_error())
            .map(|ptr| Self { ptr, config })
    }

    /// Return the config string being parsed.
    pub fn config_str(&self) -> &'a str {
        self.config
    }

    pub fn get(&mut self, key: &str) -> Option<Result<ConfigItem<'a>>> {
        // NB: the expected key is a null-terminated cstring.
        let mut key_bytes = key.to_string().into_bytes();
        key_bytes.push(0);
        let key = CString::from_vec_with_nul(key_bytes).expect("key doesn't contain nulls");
        let mut item = ConfigItem::default_wt_item();
        map_not_found(
            unsafe { wt_call!(self.ptr, get, key.as_ptr(), &mut item) }
                .and_then(|()| ConfigItem::from_config_item(&item).ok_or(Error::generic_error())),
        )
    }
}

impl<'a> Iterator for ConfigParser<'a> {
    type Item = Result<(&'a str, ConfigItem<'a>)>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut key = ConfigItem::default_wt_item();
        let mut value = ConfigItem::default_wt_item();
        map_not_found(
            unsafe { wt_call!(self.ptr, next, &mut key, &mut value) }
                .and_then(|()| ConfigItem::from_config_item(&value).ok_or(Error::generic_error()))
                .map(|v| (ConfigItem::get_str(&key), v)),
        )
    }
}

impl Drop for ConfigParser<'_> {
    fn drop(&mut self) {
        if let Err(e) = unsafe { wt_call!(self.ptr, close) } {
            error!("Failed to close config parser: {}", e);
        }
    }
}

unsafe impl Send for ConfigParser<'_> {}

/// Types of entries that can appear in config.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ConfigItem<'a> {
    /// A quoted string
    String(&'a str),
    /// A boolean literal ("true" or "false")
    Bool(bool),
    /// An unquoted string, possibly an identifier.
    Id(&'a str),
    /// An integer, but not a floating point number.
    Num(i64),
    /// A nested structure. Use another [ConfigParser] to parse it.
    Struct(&'a str),
}

impl<'a> ConfigItem<'a> {
    /// If this is a [ConfigItem::String](`String`) or [ConfigItem::Id](`Id`) then return the inner
    /// string value.
    pub fn str_or_id(&self) -> Option<&'a str> {
        match self {
            Self::String(s) => Some(s),
            Self::Id(s) => Some(s),
            _ => None,
        }
    }

    fn default_wt_item() -> WT_CONFIG_ITEM {
        WT_CONFIG_ITEM {
            str_: std::ptr::null(),
            len: 0,
            val: 0,
            type_: wt_sys::__wt_config_item_WT_CONFIG_ITEM_TYPE_WT_CONFIG_ITEM_STRING,
        }
    }

    fn get_str(item: &WT_CONFIG_ITEM) -> &'a str {
        // NB: we assume that because the source string is utf8 that a substring gnereated from it
        // will also be utf8.
        unsafe {
            std::str::from_utf8(std::slice::from_raw_parts::<'a, _>(
                item.str_ as *const u8,
                item.len,
            ))
            .expect("source string is also utf8")
        }
    }

    #[allow(non_upper_case_globals, non_snake_case)]
    fn from_config_item(item: &WT_CONFIG_ITEM) -> Option<Self> {
        match item.type_ {
            wt_sys::__wt_config_item_WT_CONFIG_ITEM_TYPE_WT_CONFIG_ITEM_STRING => {
                Some(Self::String(Self::get_str(item)))
            }
            wt_sys::__wt_config_item_WT_CONFIG_ITEM_TYPE_WT_CONFIG_ITEM_BOOL => {
                Some(Self::Bool(item.val != 0))
            }
            wt_sys::__wt_config_item_WT_CONFIG_ITEM_TYPE_WT_CONFIG_ITEM_ID => {
                Some(Self::Id(Self::get_str(item)))
            }
            wt_sys::__wt_config_item_WT_CONFIG_ITEM_TYPE_WT_CONFIG_ITEM_NUM => {
                Some(Self::Num(item.val))
            }
            wt_sys::__wt_config_item_WT_CONFIG_ITEM_TYPE_WT_CONFIG_ITEM_STRUCT => {
                Some(Self::Struct(Self::get_str(item)))
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::config::{ConfigItem, ConfigParser};

    const CONFIG: &'static str = "id=anid,str=\"astring\",enabled=true,disabled=false,pos_num=7,neg_num=-7,float_num=3.14,doc={ \"json\": true }";

    #[test]
    fn get_id() {
        let mut parser = ConfigParser::new(CONFIG).unwrap();
        assert_eq!(parser.get("id"), Some(Ok(ConfigItem::Id("anid"))));
    }

    #[test]
    fn get_string() {
        let mut parser = ConfigParser::new(CONFIG).unwrap();
        assert_eq!(parser.get("str"), Some(Ok(ConfigItem::String("astring"))));
    }

    #[test]
    fn get_bool() {
        let mut parser = ConfigParser::new(CONFIG).unwrap();
        assert_eq!(parser.get("enabled"), Some(Ok(ConfigItem::Bool(true))));
        assert_eq!(parser.get("disabled"), Some(Ok(ConfigItem::Bool(false))));
    }

    #[test]
    fn get_num() {
        let mut parser = ConfigParser::new(CONFIG).unwrap();
        assert_eq!(parser.get("pos_num"), Some(Ok(ConfigItem::Num(7))));
        assert_eq!(parser.get("neg_num"), Some(Ok(ConfigItem::Num(-7))));
        assert_eq!(parser.get("float_num"), Some(Ok(ConfigItem::Id("3.14"))));
    }

    #[test]
    fn get_struct() {
        let mut parser = ConfigParser::new(CONFIG).unwrap();
        assert_eq!(
            parser.get("doc"),
            Some(Ok(ConfigItem::Struct("{ \"json\": true }")))
        );
    }

    #[test]
    fn get_unknown() {
        let mut parser = ConfigParser::new(CONFIG).unwrap();
        assert_eq!(parser.get("unknown"), None);
    }

    #[test]
    fn iterate() {
        assert_eq!(
            ConfigParser::new(CONFIG).unwrap().collect::<Vec<_>>(),
            vec![
                Ok(("id", ConfigItem::Id("anid"))),
                Ok(("str", ConfigItem::String("astring"))),
                Ok(("enabled", ConfigItem::Bool(true))),
                Ok(("disabled", ConfigItem::Bool(false))),
                Ok(("pos_num", ConfigItem::Num(7))),
                Ok(("neg_num", ConfigItem::Num(-7))),
                Ok(("float_num", ConfigItem::Id("3.14"))),
                Ok(("doc", ConfigItem::Struct("{ \"json\": true }"))),
            ]
        );
    }

    #[test]
    fn parse_json() {
        let mut parser = ConfigParser::new("{ \"json\": true }").unwrap();
        assert_eq!(parser.get("json"), Some(Ok(ConfigItem::Bool(true))));
    }

    #[test]
    fn item_str_or_id() {
        assert_eq!(ConfigItem::String("s").str_or_id(), Some("s"));
        assert_eq!(ConfigItem::Id("s").str_or_id(), Some("s"));
        assert_eq!(ConfigItem::Struct("s").str_or_id(), None);
        assert_eq!(ConfigItem::Num(1).str_or_id(), None);
        assert_eq!(ConfigItem::Bool(true).str_or_id(), None);
    }
}
