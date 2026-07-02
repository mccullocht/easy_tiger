//! Automatically generated bindings for the WiredTiger C library.
//!

#![allow(clippy::all)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unnecessary_transmutes)]
#![allow(rustdoc::invalid_html_tags)]
#![allow(rustdoc::broken_intra_doc_links)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

impl Default for __wt_item {
    fn default() -> Self {
        Self {
            data: std::ptr::null(),
            size: 0,
            mem: std::ptr::null_mut(),
            memsize: 0,
            flags: 0,
        }
    }
}

impl __wt_item {
    /// Construct a `WT_ITEM` pointing into `s`. The caller must ensure `s` outlives the item.
    pub unsafe fn from_slice(s: &[u8]) -> Self {
        Self {
            data: s.as_ptr() as *const _,
            size: s.len(),
            mem: std::ptr::null_mut(),
            memsize: 0,
            flags: 0,
        }
    }
}

impl Default for __wt_modify {
    fn default() -> Self {
        Self {
            data: __wt_item::default(),
            offset: 0,
            size: 0,
        }
    }
}
