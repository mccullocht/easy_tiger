use std::{
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Bound, Deref, DerefMut, RangeBounds},
};

use rustix::io::Errno;

use crate::{
    map_not_found,
    session::{format::Formatted, InnerCursor, Item},
    wt_call, Error, Result,
};

use super::Session;

/// Format a variable using the passed formatter into buf (if necessary).
///
/// This is an end run around lifetime issues as Formatted::pack_trivial and the buffer will have
/// different lifetimes.
macro_rules! format_to_buf {
    ($var:expr, $formatter:ident, $buf:expr) => {{
        $formatter::pack($var, &mut $buf).map(|()| $buf.as_slice())
    }};
}

pub struct TypedCursor<'a, K, V> {
    inner: InnerCursor,
    session: &'a Session,
    // TODO: consider using SmallVec for buffers. This will make the cursor larger but will require
    // zero allocations, particularly for small keys.
    key_buf: Vec<u8>,
    value_buf: Vec<u8>,
    supports_get_raw_kv: bool,
    _km: PhantomData<&'a K>,
    _vm: PhantomData<&'a V>,
}

impl<'a, K: Formatted, V: Formatted> TypedCursor<'a, K, V> {
    pub(super) fn new(inner: InnerCursor, session: &'a Session) -> Result<Self> {
        if inner.key_format() == K::FORMAT && inner.value_format() == V::FORMAT {
            // We opt in certain uri types to this behavior. It is faster but not supported by all
            // cursor types -- in particular stats cursors do not support this call.
            let supports_get_raw_kv = inner.uri().to_str().unwrap().starts_with("table:");
            Ok(Self {
                inner,
                session,
                key_buf: vec![],
                value_buf: vec![],
                supports_get_raw_kv,
                _km: PhantomData,
                _vm: PhantomData,
            })
        } else {
            Err(Error::Errno(Errno::INVAL))
        }
    }

    pub fn session(&self) -> &'a Session {
        self.session
    }

    /// Set the contents of `record` in the collection.
    pub fn set(&mut self, key: K::Ref<'_>, value: V::Ref<'_>) -> Result<()> {
        let key = format_to_buf!(key, K, self.key_buf)?;
        let value = format_to_buf!(value, V, self.value_buf)?;
        // safety: the memory passed to set_{key,value} need only be valid until a modifying
        // call like insert().
        unsafe {
            wt_call!(void self.inner.0, set_key, &Item::from(key).0)?;
            wt_call!(void self.inner.0, set_value, &Item::from(value).0)?;
            wt_call!(self.inner.0, insert)
        }
    }

    /// Remove a record by `key`.
    ///
    /// This may return a `WiredTigerError::NotFound` if the key does not exist in the collection.
    pub fn remove(&mut self, key: K::Ref<'_>) -> Result<()> {
        let key = format_to_buf!(key, K, self.key_buf)?;
        unsafe {
            wt_call!(void self.inner.0, set_key, &Item::from(key).0)?;
            wt_call!(self.inner.0, remove)
        }
    }

    /// Advance and return the next record.
    ///
    /// If this cursor is unpositioned, returns to the start of the collection.
    ///
    /// Like a `FusedIterator`, this returns `None` when the end of the collection is reached and
    /// continues to return `None` until the cursor is re-positioned.
    ///
    /// # Safety
    /// If the cursor's parent session returns this record during a transaction and that transaction
    /// is rolled back, we cannot guarantee that view value data is safe to access. Use
    /// `Iterator.next()` to ensure safe access at the cost of a copy of the record value.
    pub unsafe fn next_unsafe(&mut self) -> Option<Result<(K::Ref<'_>, V::Ref<'_>)>> {
        map_not_found(unsafe { wt_call!(self.inner.0, next) }.and_then(|()| self.key_and_value()))
    }

    /// Seek to the for `key` and return any associated `RecordView` if present.
    ///
    /// # Safety
    /// If the cursor's parent session returns this record during a transaction and that transaction
    /// is rolled back, we cannot guarantee that view value data is safe to access. Use
    /// `seek_exact()` to ensure safe access at the cost of a copy of the record value.
    pub unsafe fn seek_exact_unsafe(&mut self, key: K::Ref<'_>) -> Option<Result<V::Ref<'_>>> {
        let key = match format_to_buf!(key, K, self.key_buf) {
            Ok(k) => k,
            Err(e) => return Some(Err(e)),
        };
        map_not_found(
            unsafe {
                wt_call!(void self.inner.0, set_key, &Item::from(key).0)
                    .and_then(|()| wt_call!(self.inner.0, search))
            }
            .and_then(|()| self.value()),
        )
    }

    /// Seek to the for `key` and return any associated `Record` if present.
    pub fn seek_exact(&mut self, key: K::Ref<'_>) -> Option<Result<V>> {
        unsafe { self.seek_exact_unsafe(key) }.map(|r| r.map(|v| v.into()))
    }

    /// Return the largest key in the collection or `None` if the collection is empty.
    pub fn largest_key(&mut self) -> Option<Result<K>> {
        map_not_found(
            unsafe { wt_call!(self.inner.0, largest_key) }
                .and_then(|()| self.key().map(|k| k.into())),
        )
    }

    /// Set the bounds this cursor.
    ///
    /// This resets the cursor to an unpositioned state. Once imposed the bounds affect all
    /// positioning operations, so for instance a `seek_exact()` with a key out of bounds would
    /// yield `None`, `next()` will yield the first entry after the lower bound, etc.
    pub fn set_bounds<'b>(&mut self, bounds: impl RangeBounds<K::Ref<'b>>) -> Result<()> {
        // Reset to an unpositioned state and remove any existing bounds.
        // * If the cursor is positioned this call will return EINVAL which is unintiuitive.
        // * action=clear removes _both_ bounds and action=set checks any existing bounds for
        //   soundness so it makes more sense to start with a clean slate.
        self.reset()?;
        self.set_bound(bounds.start_bound(), false)?;
        self.set_bound(bounds.end_bound(), true)
    }

    fn set_bound(&mut self, bound: Bound<&K::Ref<'_>>, upper: bool) -> Result<()> {
        let (key, config_str) = match bound {
            Bound::Unbounded => return Ok(()),
            Bound::Included(k) => {
                if upper {
                    (k, c"bound=upper")
                } else {
                    (k, c"bound=lower")
                }
            }
            Bound::Excluded(k) => {
                if upper {
                    (k, c"bound=upper,inclusive=false")
                } else {
                    (k, c"bound=lower,inclusive=false")
                }
            }
        };
        K::pack(*key, &mut self.key_buf)?;
        unsafe {
            wt_call!(void self.inner.0, set_key, &Item::from(self.key_buf.as_slice()).0)?;
            wt_call!(self.inner.0, bound, config_str.as_ptr())
        }
    }

    /// Reset the cursor to an unpositioned state.
    pub fn reset(&mut self) -> Result<()> {
        unsafe { wt_call!(self.inner.0, reset) }
    }

    /// Return the current key. This assumes that we have just positioned the cursor
    /// and WT_NOTFOUND will not be returned.
    #[inline]
    fn key(&self) -> Result<K::Ref<'_>> {
        let mut k = Item::default();
        unsafe { wt_call!(self.inner.0, get_key, &mut k.0).and_then(|()| K::unpack(k.into())) }
    }

    /// Return the current raw value. This assumes that we have just positioned the cursor
    /// and WT_NOTFOUND will not be returned.
    #[inline]
    fn value(&self) -> Result<V::Ref<'_>> {
        let mut v = Item::default();
        unsafe { wt_call!(self.inner.0, get_value, &mut v.0).and_then(|()| V::unpack(v.into())) }
    }

    /// Return the current key and value. This assumes that we have just positioned the cursor
    /// and WT_NOTFOUND will not be returned.
    #[inline]
    fn key_and_value(&self) -> Result<(K::Ref<'_>, V::Ref<'_>)> {
        let mut k = Item::default();
        let mut v = Item::default();
        unsafe {
            if self.supports_get_raw_kv {
                wt_call!(self.inner.0, get_raw_key_value, &mut k.0, &mut v.0)
            } else {
                wt_call!(self.inner.0, get_key, &mut k.0)
                    .and_then(|()| wt_call!(self.inner.0, get_value, &mut v.0))
            }?
        };
        K::unpack(k.into()).and_then(|k| V::unpack(v.into()).map(|v| (k, v)))
    }
}

impl<'a, K: Formatted, V: Formatted> Iterator for TypedCursor<'a, K, V> {
    type Item = Result<(K, V)>;

    /// Advance and return the next record.
    ///
    /// If this cursor is unpositioned, returns values from the start of the collection or any bound.
    fn next(&mut self) -> Option<Self::Item> {
        unsafe { self.next_unsafe() }.map(|r| r.map(|(k, v)| (k.into(), v.into())))
    }
}

pub struct TypedCursorGuard<'a, K, V>(ManuallyDrop<TypedCursor<'a, K, V>>);

impl<'a, K, V> TypedCursorGuard<'a, K, V> {
    pub fn new(cursor: TypedCursor<'a, K, V>) -> Self {
        Self(ManuallyDrop::new(cursor))
    }
}

impl<'a, K, V> Deref for TypedCursorGuard<'a, K, V> {
    type Target = TypedCursor<'a, K, V>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, K, V> DerefMut for TypedCursorGuard<'a, K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, K, V> Drop for TypedCursorGuard<'a, K, V> {
    fn drop(&mut self) {
        // Safety: we never intend to allow RecordCursorGuard to drop the value.
        let cursor = unsafe { ManuallyDrop::take(&mut self.0) };
        cursor.session.return_cursor(cursor.inner);
    }
}

impl<'a, K: Formatted, V: Formatted> Iterator for TypedCursorGuard<'a, K, V> {
    type Item = Result<(K, V)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}
