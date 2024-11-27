use std::{
    ffi::{c_char, CStr},
    num::NonZero,
    ptr::NonNull,
};

use crate::{make_result, Error, Result};
use wt_sys::{WT_CURSOR, WT_NOTFOUND};

use super::Session;

/// Cursor over statistics, either for the entire connection or a specific URI.
pub struct StatCursor<'a> {
    pub(super) ptr: NonNull<WT_CURSOR>,
    pub(super) _session: &'a Session,
}

impl<'a> StatCursor<'a> {
    /// Seek to specific WT_STAT_.* and return the associated value if any.
    pub fn seek_exact(&mut self, wt_stat: u32) -> Option<Result<i64>> {
        unsafe {
            self.ptr.as_ref().set_key.unwrap()(self.ptr.as_ptr(), wt_stat);
            match NonZero::new(self.ptr.as_ref().search.unwrap()(self.ptr.as_ptr())) {
                None => Some(self.read_stat().map(|(_, v)| v)),
                Some(code) if code.get() == WT_NOTFOUND => None,
                Some(code) => Some(Err(Error::from(code))),
            }
        }
    }

    // Read the description into a string instead of aliasing the value.
    // If we alias it's hard to attach correct lifetimes to to the result to avoid
    // dangling pointers.
    fn read_stat(&self) -> Result<(String, i64)> {
        let (desc, value) = unsafe {
            let mut desc_ptr: *mut c_char = std::ptr::null_mut();
            let mut pvalue_ptr: *mut c_char = std::ptr::null_mut();
            let mut value = 0i64;
            make_result(
                self.ptr.as_ref().get_value.unwrap()(
                    self.ptr.as_ptr(),
                    &mut desc_ptr,
                    &mut pvalue_ptr,
                    &mut value,
                ),
                (),
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
        unsafe {
            match NonZero::new(self.ptr.as_ref().next.unwrap()(self.ptr.as_ptr())) {
                None => Some(self.read_stat()),
                Some(code) if code.get() == WT_NOTFOUND => None,
                Some(code) => Some(Err(Error::from(code))),
            }
        }
    }
}

impl<'a> Drop for StatCursor<'a> {
    fn drop(&mut self) {
        // TODO: print something if this returns an error.
        unsafe { self.ptr.as_ref().close.unwrap()(self.ptr.as_ptr()) };
    }
}
