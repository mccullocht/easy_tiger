use std::{
    ffi::CStr,
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut, RangeBounds},
};

use rustix::io::Errno;

use crate::{
    map_not_found,
    session::{
        format::{Formatter, MaxLenFormatWriter, PackedFormatReader, PackedFormatWriter},
        raw_cursor::RawCursor,
    },
    Error, Result,
};

use super::Session;

pub struct TypedCursor<'a, 'b, K: Formatter<'b>, V: Formatter<'b>>
where
    Self: 'b,
    'a: 'b,
{
    // XXX tuple struct
    raw: RawCursor<'a>,
    key_buf: Vec<u8>,
    val_buf: Vec<u8>,
    _km: PhantomData<&'b K>,
    _vm: PhantomData<&'b V>,
}

impl<'a, 'b, K: Formatter<'b>, V: Formatter<'b>> TypedCursor<'a, 'b, K, V>
where
    Self: 'b,
    'a: 'b,
{
    pub fn new(raw: RawCursor<'a>) -> Result<Self> {
        if raw.key_format() == K::FORMAT && raw.value_format() == V::FORMAT {
            Ok(Self {
                raw,
                key_buf: vec![],
                val_buf: vec![],
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
    pub fn set(&'b mut self, key: &K, value: &V) -> Result<()> {
        let key = Self::pack(key, &mut self.key_buf)?;
        let value = Self::pack(value, &mut self.val_buf)?;
        self.raw.set(key, value)
    }

    /// Remove a record by `key`.
    ///
    /// This may return a `WiredTigerError::NotFound` if the key does not exist in the collection.
    pub fn remove(&'b mut self, key: &K) -> Result<()> {
        let key = Self::pack(key, &mut self.key_buf)?;
        self.raw.remove(key)
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
    pub unsafe fn next_unsafe(&'b mut self) -> Option<Result<(K, V)>> {
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
    pub unsafe fn seek_exact_unsafe(&'b mut self, key: &K) -> Option<Result<V>> {
        let key = match Self::pack(key, &mut self.key_buf) {
            Ok(k) => k,
            Err(e) => return Some(Err(e)),
        };
        map_not_found(unsafe { self.raw.seek_exact_unsafe(key) }.and_then(|v| Self::unpack(v)))
    }

    /// Seek to the for `key` and return any associated `Record` if present.
    pub fn seek_exact(&'b mut self, key: &K) -> Option<Result<V::FormatterOwned>> {
        unsafe { self.seek_exact_unsafe(key) }.map(|r| r.map(|v| v.to_formatter_owned()))
    }

    /// Return the largest key in the collection or `None` if the collection is empty.
    pub fn largest_key(&mut self) -> Option<Result<K::FormatterOwned>> {
        todo!()
    }

    /// Set the bounds this cursor. This affects almost all positioning operations, so for instance
    /// a `seek_exact()` with a key out of bounds might yield `None`.
    ///
    /// Cursor bounds are removed by `reset()`.
    pub fn set_bounds(&mut self, bounds: impl RangeBounds<K>) -> Result<()> {
        todo!()
    }

    /// Reset the cursor to an unpositioned state.
    pub fn reset(&mut self) -> Result<()> {
        self.raw.reset()
    }

    fn pack<F: Formatter<'b>>(value: &F, buf: &'b mut Vec<u8>) -> Result<&'b [u8]> {
        if let Some(formatted) = value.format_pack_trivial() {
            return Ok(formatted);
        }

        let max_len = if let Some(n) = F::FORMAT.max_len() {
            n
        } else {
            let mut writer = MaxLenFormatWriter::new(F::FORMAT);
            value.format_pack(&mut writer)?;
            writer.close()?
        };

        buf.resize(max_len, 0);
        let mut writer = PackedFormatWriter::new(F::FORMAT, buf.as_mut_slice())?;
        value.format_pack(&mut writer)?;
        let len = writer.close()?;
        buf.truncate(len);
        Ok(buf.as_mut_slice())
    }

    fn unpack<F: Formatter<'b>>(packed: &'b [u8]) -> Result<F> {
        if let Some(unpacked) = F::format_unpack_trivial(packed) {
            return Ok(unpacked);
        }

        let mut reader = PackedFormatReader::new(F::FORMAT, packed)?;
        F::format_unpack(&mut reader)
    }

    fn unpack_owned<F: Formatter<'b>>(packed: &'b [u8]) -> Result<F::FormatterOwned> {
        Self::unpack(packed).map(|f: F| f.to_formatter_owned())
    }
}

impl<'a, 'b, K: Formatter<'b>, V: Formatter<'b>> Iterator for TypedCursor<'a, 'b, K, V>
where
    Self: 'b,
    'a: 'b,
{
    type Item = Result<(K::FormatterOwned, V::FormatterOwned)>;

    /// Advance and return the next record.
    ///
    /// If this cursor is unpositioned, returns values from the start of the collection or any bound.
    // XXX I can't set a lifetime on &mut self without breaking the trait so i'm fucked.
    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

pub struct TypedCursorGuard<'a, 'b, K: Formatter<'b>, V: Formatter<'b>>(
    ManuallyDrop<TypedCursor<'a, 'b, K, V>>,
);

impl<'a, 'b, K: Formatter<'b>, V: Formatter<'b>> TypedCursorGuard<'a, 'b, K, V> {
    pub(super) fn new(cursor: TypedCursor<'a, 'b, K, V>) -> Self {
        Self(ManuallyDrop::new(cursor))
    }
}

impl<'a, 'b, K: Formatter<'b>, V: Formatter<'b>> Deref for TypedCursorGuard<'a, 'b, K, V> {
    type Target = TypedCursor<'a, 'b, K, V>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, 'b, K: Formatter<'b>, V: Formatter<'b>> DerefMut for TypedCursorGuard<'a, 'b, K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, 'b, K: Formatter<'b>, V: Formatter<'b>> Drop for TypedCursorGuard<'a, 'b, K, V> {
    fn drop(&mut self) {
        // Safety: we never intend to allow RecordCursorGuard to drop the value.
        let cursor = unsafe { ManuallyDrop::take(&mut self.0) }.raw;
        cursor.return_to_session();
    }
}
