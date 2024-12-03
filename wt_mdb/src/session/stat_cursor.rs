use std::{
    ffi::{c_char, CStr},
    ptr::NonNull,
};

use crate::{map_not_found, wt_call, Error, Result};
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
                wt_call!(nocode self.ptr, set_key, wt_stat)
                    .and_then(|()| wt_call!(self.ptr, search))
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

impl<'a> Iterator for StatCursor<'a> {
    type Item = Result<(String, i64)>;

    fn next(&mut self) -> Option<Self::Item> {
        map_not_found(unsafe { wt_call!(self.ptr, next) }.and_then(|()| self.read_stat()))
    }
}

impl<'a> Drop for StatCursor<'a> {
    fn drop(&mut self) {
        // TODO: print something if this returns an error.
        let _ = unsafe { wt_call!(self.ptr, close) };
    }
}
