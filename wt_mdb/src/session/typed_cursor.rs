use std::{
    borrow::Cow,
    ffi::CStr,
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Bound, Deref, DerefMut, RangeBounds},
};

use rustix::io::Errno;

use crate::{
    map_not_found,
    session::{
        format::{
            Formatter, FormatterRef, MaxLenFormatWriter, PackedFormatReader, PackedFormatWriter,
        },
        raw_cursor::RawCursor,
    },
    Error, Result,
};

use super::Session;

pub struct TypedCursor<'a, K, V> {
    raw: RawCursor<'a>,
    _km: PhantomData<&'a K>,
    _vm: PhantomData<&'a V>,
}

impl<'a, K: Formatter, V: Formatter> TypedCursor<'a, K, V> {
    pub fn new(raw: RawCursor<'a>) -> Result<Self> {
        if raw.key_format() == K::FORMAT && raw.value_format() == V::FORMAT {
            Ok(Self {
                raw,
                _km: PhantomData,
                _vm: PhantomData,
            })
        } else {
            Err(Error::Errno(Errno::INVAL))
        }
    }

    pub fn session(&self) -> &Session {
        self.raw.session()
    }

    /// Returns the name of the table.
    pub fn table_name(&self) -> &CStr {
        self.raw.table_name()
    }

    /// Set the contents of `record` in the collection.
    pub fn set(&mut self, key: &K::Ref<'_>, value: &V::Ref<'_>) -> Result<()> {
        let key = Self::pack_cow::<K>(key)?;
        let value = Self::pack_cow::<V>(value)?;
        self.raw.set(key.as_ref(), value.as_ref())
    }

    /// Remove a record by `key`.
    ///
    /// This may return a `WiredTigerError::NotFound` if the key does not exist in the collection.
    pub fn remove(&mut self, key: &K::Ref<'_>) -> Result<()> {
        let key = Self::pack_cow::<K>(key)?;
        self.raw.remove(key.as_ref())
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
            unsafe { self.raw.next_unsafe() }
                .and_then(|(k, v)| Self::unpack::<K>(k).map(|k| (k, v)))
                .and_then(|(k, v)| Self::unpack::<V>(v).map(|v| (k, v))),
        )
    }

    /// Seek to the for `key` and return any associated `RecordView` if present.
    ///
    /// # Safety
    /// If the cursor's parent session returns this record during a transaction and that transaction
    /// is rolled back, we cannot guarantee that view value data is safe to access. Use
    /// `seek_exact()` to ensure safe access at the cost of a copy of the record value.
    pub unsafe fn seek_exact_unsafe(&mut self, key: &K::Ref<'_>) -> Option<Result<V::Ref<'_>>> {
        let key = match Self::pack_cow::<K>(key) {
            Ok(k) => k,
            Err(e) => return Some(Err(e)),
        };
        map_not_found(
            unsafe { self.raw.seek_exact_unsafe(key.as_ref()) }.and_then(|v| Self::unpack::<V>(v)),
        )
    }

    /// Seek to the for `key` and return any associated `Record` if present.
    pub fn seek_exact(&mut self, key: &K::Ref<'_>) -> Option<Result<V::Owned>> {
        unsafe { self.seek_exact_unsafe(key) }.map(|r| r.map(|v| v.to_formatter_owned()))
    }

    /// Return the largest key in the collection or `None` if the collection is empty.
    pub fn largest_key(&mut self) -> Option<Result<K::Owned>> {
        todo!()
    }

    /// Set the bounds this cursor. This affects almost all positioning operations, so for instance
    /// a `seek_exact()` with a key out of bounds might yield `None`.
    ///
    /// Cursor bounds are removed by `reset()`.
    pub fn set_bounds<'b>(&mut self, bounds: impl RangeBounds<K::Ref<'b>>) -> Result<()> {
        self.set_bound(bounds.start_bound(), false)?;
        self.set_bound(bounds.end_bound(), true)
    }

    fn set_bound<'b>(&mut self, bound: Bound<&K::Ref<'b>>, upper: bool) -> Result<()> {
        let bound = match bound {
            Bound::Included(k) => Bound::Included(Self::pack_cow::<K>(k)?),
            Bound::Excluded(k) => Bound::Excluded(Self::pack_cow::<K>(k)?),
            Bound::Unbounded => Bound::Unbounded,
        };
        self.raw
            .set_bound(bound.as_ref().map(|k| k.as_ref()), upper)
    }

    /// Reset the cursor to an unpositioned state.
    pub fn reset(&mut self) -> Result<()> {
        self.raw.reset()
    }

    // TODO: this may cause us to create a new buffer on every single pack call which sucks.
    // Consider implementing some sort of cow guard and freelisting these on the cursor or session.
    // Also consider using Smallvec here to avoid allocation for small cases (like i64).
    fn pack_cow<'b, F: Formatter>(value: &F::Ref<'b>) -> Result<Cow<'b, [u8]>> {
        if let Some(f) = F::pack_trivial(value) {
            return Ok(f.into());
        }

        let max_len = if let Some(n) = F::FORMAT.max_len() {
            n
        } else {
            let mut writer = MaxLenFormatWriter::new(F::FORMAT);
            F::pack(&mut writer, value)?;
            writer.close()?
        };

        let mut buf = vec![0u8; max_len];
        let mut writer = PackedFormatWriter::new(F::FORMAT, buf.as_mut_slice())?;
        F::pack(&mut writer, value)?;
        let len = writer.close()?;
        buf.truncate(len);
        Ok(buf.into())
    }

    fn unpack<'b, F: Formatter>(packed: &'b [u8]) -> Result<F::Ref<'b>> {
        if let Some(unpacked) = F::unpack_trivial(packed) {
            return Ok(unpacked);
        }

        let mut reader = PackedFormatReader::new(F::FORMAT, packed)?;
        F::unpack(&mut reader)
    }
}

impl<'a, K: Formatter, V: Formatter> Iterator for TypedCursor<'a, K, V> {
    type Item = Result<(K::Owned, V::Owned)>;

    /// Advance and return the next record.
    ///
    /// If this cursor is unpositioned, returns values from the start of the collection or any bound.
    // XXX I can't set a lifetime on &mut self without breaking the trait so i'm fucked.
    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

pub struct TypedCursorGuard<'a, K, V>(ManuallyDrop<TypedCursor<'a, K, V>>);

impl<'a, K, V> TypedCursorGuard<'a, K, V> {
    pub(super) fn new(cursor: TypedCursor<'a, K, V>) -> Self {
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
        let cursor = unsafe { ManuallyDrop::take(&mut self.0) }.raw;
        cursor.return_to_session();
    }
}
