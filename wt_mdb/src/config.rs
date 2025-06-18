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

    pub fn get(&self, key: &str) -> Option<Result<ConfigItem<'a>>> {
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
#[derive(Debug, Copy, Clone)]
pub enum ConfigItem<'a> {
    /// A string
    String(&'a str),
    /// A boolean literal ("true" or "false")
    Bool(bool),
    /// Something that is like a string but not exactly a string.
    Id(&'a str),
    /// An integer, but not a floating point number.
    Num(i64),
    /// A nested structure. Use another [ConfigParser] to parse it.
    Struct(&'a str),
}

impl<'a> ConfigItem<'a> {
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
            str::from_utf8(std::slice::from_raw_parts::<'a, _>(
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
