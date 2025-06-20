use std::{
    ffi::{c_char, CStr},
    ptr::NonNull,
};

use crate::{map_not_found, wt_call, Error, Result};
use tracing::error;
use wt_sys::WT_CURSOR;

use super::Session;

/// Cursor over statistics, either for the entire connection or a specific URI.
pub struct StatCursor<'a> {
    pub(super) ptr: NonNull<WT_CURSOR>,
    pub(super) _session: &'a Session,
}

impl<'a> StatCursor<'a> {
    /// Seek to specific WT_STAT_.* and return the associated value if any.
    pub fn seek_exact(&mut self, wt_stat: u32) -> Option<Result<i64>> {
        map_not_found(
            unsafe {
                wt_call!(void self.ptr, set_key, wt_stat).and_then(|()| wt_call!(self.ptr, search))
            }
            .and_then(|()| self.read_stat().map(|(_, v)| v)),
        )
    }

    // Read the description into a string instead of aliasing the value.
    // If we alias it's hard to attach correct lifetimes to to the result to avoid
    // dangling pointers.
    fn read_stat(&self) -> Result<(String, i64)> {
        let (desc, value) = unsafe {
            let mut desc_ptr: *mut c_char = std::ptr::null_mut();
            let mut pvalue_ptr: *mut c_char = std::ptr::null_mut();
            let mut value = 0i64;
            wt_call!(
                self.ptr,
                get_value,
                &mut desc_ptr,
                &mut pvalue_ptr,
                &mut value
            )?;
            (CStr::from_ptr::<'a>(desc_ptr), value)
        };
        Ok((
            desc.to_str()
                .map(str::to_string)
                .map_err(|_| Error::generic_error())?,
            value,
        ))
    }
}

impl Iterator for StatCursor<'_> {
    type Item = Result<(String, i64)>;

    fn next(&mut self) -> Option<Self::Item> {
        map_not_found(unsafe { wt_call!(self.ptr, next) }.and_then(|()| self.read_stat()))
    }
}

impl Drop for StatCursor<'_> {
    fn drop(&mut self) {
        if let Err(e) = unsafe { wt_call!(self.ptr, close) } {
            error!("Failed to close statistics WT_CURSOR: {}", e);
        }
    }
}
