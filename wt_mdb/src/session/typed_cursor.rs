use std::{
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Bound, Deref, DerefMut, RangeBounds},
};

use rustix::io::Errno;

use crate::{
    map_not_found,
    session::{
        format::{Formatted, MaxLenFormatWriter, PackedFormatReader, PackedFormatWriter},
        InnerCursor, Item,
    },
    wt_call, Error, Result,
};

use super::Session;

/// Pack a value that needs non-trivial formatting into buf.
fn pack_non_trivial<F: Formatted>(value: &F::Ref<'_>, buf: &mut Vec<u8>) -> Result<()> {
    let max_len = if let Some(n) = F::FORMAT.max_len() {
        n
    } else {
        let mut writer = MaxLenFormatWriter::new(F::FORMAT);
        F::pack(&mut writer, value)?;
        writer.close()?
    };

    buf.resize(max_len, 0);
    let mut writer = PackedFormatWriter::new(F::FORMAT, buf.as_mut_slice())?;
    F::pack(&mut writer, value)?;
    let len = writer.close()?;
    buf.truncate(len);
    Ok(())
}

/// Format a variable using the passed formatter into buf (if necessary).
///
/// This is an end run around lifetime issues as Formatted::pack_trivial and the buffer will have
/// different lifetimes.
macro_rules! format_to_buf {
    ($var:ident, $formatter:ident, $buf:expr) => {{
        if let Some(packed) = $formatter::pack_trivial(&$var) {
            Ok(packed)
        } else {
            pack_non_trivial::<$formatter>(&$var, &mut $buf).map(|()| $buf.as_slice())
        }
    }};
}
pub struct TypedCursor<'a, K, V> {
    inner: InnerCursor,
    session: &'a Session,
    // TODO: consider using SmallVec for buffers. This will make the cursor larger but will require
    // zero allocations, particularly for small keys.
    key_buf: Vec<u8>,
    value_buf: Vec<u8>,
    _km: PhantomData<&'a K>,
    _vm: PhantomData<&'a V>,
}

impl<'a, K: Formatted, V: Formatted> TypedCursor<'a, K, V> {
    pub(super) fn new(inner: InnerCursor, session: &'a Session) -> Result<Self> {
        if inner.key_format() == K::FORMAT && inner.value_format() == V::FORMAT {
            Ok(Self {
                inner,
                session,
                key_buf: vec![],
                value_buf: vec![],
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
        map_not_found(
            unsafe { wt_call!(self.inner.0, next) }
                .and_then(|()| self.key())
                .and_then(|k| self.value().map(|v| (k, v))),
        )
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

    /// Set the bounds this cursor. This affects almost all positioning operations, so for instance
    /// a `seek_exact()` with a key out of bounds might yield `None`.
    ///
    /// Cursor bounds are removed by `reset()`.
    pub fn set_bounds<'b>(&mut self, bounds: impl RangeBounds<K::Ref<'b>>) -> Result<()> {
        self.set_bound(bounds.start_bound(), false)?;
        self.set_bound(bounds.end_bound(), true)
    }

    pub fn set_bound(&mut self, bound: Bound<&K::Ref<'_>>, upper: bool) -> Result<()> {
        let bound: Bound<&[u8]> = match bound {
            Bound::Included(k) => Bound::Included(format_to_buf!(k, K, self.key_buf)?),
            Bound::Excluded(k) => Bound::Excluded(format_to_buf!(k, K, self.key_buf)?),
            Bound::Unbounded => Bound::Unbounded,
        };
        let (key, config_str) = match bound {
            Bound::Included(key) => (
                Some(key),
                if upper {
                    c"bound=upper,action=set"
                } else {
                    c"bound=lower,action=set"
                },
            ),
            Bound::Excluded(key) => (
                Some(key),
                if upper {
                    c"bound=upper,action=set,inclusive=false"
                } else {
                    c"bound=lower,action=set,inclusive=false"
                },
            ),
            Bound::Unbounded => (
                None,
                if upper {
                    c"bound=upper,action=clear"
                } else {
                    c"bound=lower,action=clear"
                },
            ),
        };
        if let Some(k) = key.map(Item::from) {
            unsafe { wt_call!(void self.inner.0, set_key, &k) }?;
        }
        unsafe { wt_call!(self.inner.0, bound, config_str.as_ptr()) }
    }

    /// Reset the cursor to an unpositioned state.
    pub fn reset(&mut self) -> Result<()> {
        unsafe { wt_call!(self.inner.0, reset) }
    }

    /// Return the current key. This assumes that we have just positioned the cursor
    /// and WT_NOTFOUND will not be returned.
    fn key(&self) -> Result<K::Ref<'_>> {
        let mut k = Item::default();
        unsafe {
            wt_call!(self.inner.0, get_key, &mut k.0).and_then(|()| Self::unpack::<K>(k.into()))
        }
    }

    /// Return the current raw value. This assumes that we have just positioned the cursor
    /// and WT_NOTFOUND will not be returned.
    fn value(&self) -> Result<V::Ref<'_>> {
        let mut v = Item::default();
        unsafe {
            wt_call!(self.inner.0, get_value, &mut v.0).and_then(|()| Self::unpack::<V>(v.into()))
        }
    }

    fn unpack<'b, F: Formatted>(packed: &'b [u8]) -> Result<F::Ref<'b>> {
        if let Some(unpacked) = F::unpack_trivial(packed) {
            return Ok(unpacked);
        }

        let mut reader = PackedFormatReader::new(F::FORMAT, packed)?;
        F::unpack(&mut reader)
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
