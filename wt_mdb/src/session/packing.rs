//! Wrap WT routines to pack and unpack values store in a table.
#![allow(dead_code)] // XXX
use std::{
    ffi::{c_char, c_void, CStr},
    marker::PhantomData,
};

use crate::{make_result, session::Item, Error, Result};
use rustix::io::Errno;
use wt_sys::{
    wiredtiger_pack_int, wiredtiger_pack_item, wiredtiger_pack_start, wiredtiger_pack_str,
    wiredtiger_pack_uint, wiredtiger_unpack_int, wiredtiger_unpack_item, wiredtiger_unpack_start,
    wiredtiger_unpack_str, wiredtiger_unpack_uint, WT_PACK_STREAM,
};

/// Classification of the format to help optimize packing and unpacking.
// XXX during packing/set:
// * Variable results in packing to size estimate, allocate a vector, pack to write.
// * Trivial results in packing to a trivial FormatWriter that records a slice.
// * Fixed results in allocation of a vector, pack to write.
// XXX trivial can also be implemented by extending the trait -- trivial pack can return Option<&[u8]>
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum FormatClassification {
    /// Contains multiple columns, at least one of which is variable length.
    Variable,
    /// Contains a single variable-length column and can be trivially packed.
    Trivial,
    /// Contains a bunch of fixed-length fields with a size upper bound.
    Fixed(usize),
}

impl FormatClassification {
    const fn new(format: &CStr) -> Self {
        let bytes = format.to_bytes();
        if bytes.len() == 1 && (bytes[0] == b'S' || bytes[0] == b'u') {
            return Self::Trivial;
        }
        let mut i = 0usize;
        let mut max_len = 0usize;
        while i < bytes.len() {
            match bytes[i] {
                b'b' | b'B' | b'h' | b'H' | b'i' | b'I' | b'l' | b'L' => max_len += 5,
                b'q' | b'Q' | b'r' => max_len += 9,
                _ => return Self::Variable,
            };
            i += 1;
        }
        Self::Fixed(max_len)
    }
}

/// A validated WiredTiger format string.
///
/// This only supports a subset of of the documented WT types, in particular it does not support
/// strings of a fixed length.
// XXX extend to allow sizes. i think this can be made to work.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FormatString(&'static CStr, FormatClassification);

type FormatStringIter = std::iter::Copied<std::slice::Iter<'static, u8>>;

impl FormatString {
    /// Create a new format string.
    ///
    /// *Panics* it the format does not pass validation.
    pub const fn new(format: &'static CStr) -> Self {
        let bytes = format.to_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            match bytes[i] {
                // XXX figure out what is going on with 'U'. Is it for 'u' when there are multiple
                // columns in the table?
                b'b' | b'B' | b'h' | b'H' | b'i' | b'I' | b'l' | b'L' | b'q' | b'Q' | b'r'
                | b'S' | b'u' => i += 1,
                _ => panic!("invalid column type"),
            }
        }
        Self(format, FormatClassification::new(format))
    }

    /// Return the number of columns in the format.
    // TODO: if we support fixed length fields then push this information into format classification.
    pub const fn column_len(&self) -> usize {
        self.0.to_bytes().len()
    }

    fn iter(&self) -> FormatStringIter {
        self.0.to_bytes().iter().copied()
    }
}

impl PartialEq<CStr> for FormatString {
    fn eq(&self, other: &CStr) -> bool {
        self.0 == other
    }
}

/// A single column value suitable for packing.
// XXX consider replacing this with a trait ColumnValuePrimitive.
#[derive(Debug, Clone, Copy)]
pub enum ColumnValue<'b> {
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    CStr(&'b CStr),
    Item(&'b [u8]),
}

impl<'b> ColumnValue<'b> {
    /// Maximum encoded length of any particular column value.
    pub const fn max_len(&self) -> usize {
        match self {
            // NB: for whatever reason the constant for this (WT_INTPACK32_MAXSIZE) is not being
            // produced by bindgen. This is a bit pessimistic.
            Self::I8(_)
            | Self::U8(_)
            | Self::I16(_)
            | Self::U16(_)
            | Self::I32(_)
            | Self::U32(_) => 5,
            // NB: for whatever reason the constant for this (WT_INTPACK64_MAXSIZE) is not being
            // produced by bindgen.
            Self::I64(_) | Self::U64(_) => 9,
            Self::CStr(s) => s.to_bytes_with_nul().len(),
            Self::Item(b) => b.len() + 5,
        }
    }

    fn format_match(&self, format: u8) -> bool {
        match self {
            Self::I8(_) => format == b'b',
            Self::U8(_) => format == b'B',
            Self::I16(_) => format == b'h',
            Self::U16(_) => format == b'H',
            Self::I32(_) => format == b'i' || format == b'l',
            Self::U32(_) => format == b'I' || format == b'L',
            Self::I64(_) => format == b'q',
            Self::U64(_) => format == b'Q' || format == b'r',
            Self::CStr(_) => format == b'S',
            Self::Item(_) => format == b'u',
        }
    }
}

// XXX dunno if it makes sense to perform TryFrom conversion here -- could do From and panic?
macro_rules! column_value_conversion {
    ($src:ty, $var:ident) => {
        impl From<$src> for ColumnValue<'_> {
            fn from(value: $src) -> Self {
                Self::$var(value)
            }
        }

        impl TryFrom<ColumnValue<'_>> for $src {
            type Error = Error;

            fn try_from(value: ColumnValue<'_>) -> Result<Self> {
                match value {
                    ColumnValue::$var(v) => Ok(v),
                    _ => Err(Error::Errno(Errno::INVAL)),
                }
            }
        }
    };
    (l $src:ty, $var:ident) => {
        impl<'b> From<&'b $src> for ColumnValue<'b> {
            fn from(value: &'b $src) -> Self {
                Self::$var(value)
            }
        }

        impl<'b> TryFrom<ColumnValue<'b>> for &'b $src {
            type Error = Error;

            fn try_from(value: ColumnValue<'b>) -> Result<Self> {
                match value {
                    ColumnValue::$var(v) => Ok(v),
                    _ => Err(Error::Errno(Errno::INVAL)),
                }
            }
        }
    };
}

column_value_conversion!(i8, I8);
column_value_conversion!(i16, I16);
column_value_conversion!(i32, I32);
column_value_conversion!(i64, I64);
column_value_conversion!(u8, U8);
column_value_conversion!(u16, U16);
column_value_conversion!(u32, U32);
column_value_conversion!(u64, U64);
column_value_conversion!(l CStr, CStr);
column_value_conversion!(l[u8], Item);

pub enum PackedElement<'b> {
    Signed(i64),
    Unsigned(u64),
    Str(&'b CStr),
    Item(&'b [u8]),
}

impl<'b> From<ColumnValue<'b>> for PackedElement<'b> {
    fn from(value: ColumnValue<'b>) -> Self {
        match value {
            ColumnValue::I8(v) => Self::Signed(v.into()),
            ColumnValue::I16(v) => Self::Signed(v.into()),
            ColumnValue::I32(v) => Self::Signed(v.into()),
            ColumnValue::I64(v) => Self::Signed(v),
            ColumnValue::U8(v) => Self::Unsigned(v.into()),
            ColumnValue::U16(v) => Self::Unsigned(v.into()),
            ColumnValue::U32(v) => Self::Unsigned(v.into()),
            ColumnValue::U64(v) => Self::Unsigned(v),
            ColumnValue::CStr(s) => Self::Str(s),
            ColumnValue::Item(b) => Self::Item(b),
        }
    }
}

/// Used by formatted objects to pack their data.
pub trait FormatWriter {
    /// Add a single value to the writer.
    ///
    /// This may fail if the passed value does not match the expect format.
    // TODO: add an unchecked version for use in derive macros.
    fn pack<'a>(&mut self, v: impl Into<ColumnValue<'a>>) -> Result<()>;
}

/// Pack a stream of values described by a format into a byte array for using in WT.
pub struct PackFormatWriter<'b> {
    format: FormatString,
    format_it: FormatStringIter,
    stream: *mut WT_PACK_STREAM,
    buffer: PhantomData<&'b [u8]>,
}

impl<'b> PackFormatWriter<'b> {
    pub fn new(format: FormatString, buffer: &'b mut [u8]) -> Result<Self> {
        let format_it = format.iter();
        let mut stream = std::ptr::null_mut();
        make_result(
            unsafe {
                wiredtiger_pack_start(
                    std::ptr::null_mut(),
                    format.0.as_ptr(),
                    buffer.as_ptr() as *mut c_void,
                    buffer.len(),
                    &mut stream,
                )
            },
            (),
        )
        .map(|()| Self {
            format,
            format_it,
            stream,
            buffer: PhantomData,
        })
    }

    pub fn close(mut self) -> Result<usize> {
        if self.format_it.next().is_some() {
            return Err(Error::Errno(Errno::INVAL));
        }
        let mut len = 0;
        let result = make_result(
            unsafe { wt_sys::wiredtiger_pack_close(self.stream, &mut len) },
            (),
        )
        .map(|()| len);
        self.stream = std::ptr::null_mut();
        result
    }
}

impl FormatWriter for PackFormatWriter<'_> {
    fn pack<'a>(&mut self, v: impl Into<ColumnValue<'a>>) -> Result<()> {
        let cv: ColumnValue<'_> = v.into();
        let f = self.format_it.next().ok_or(Error::Errno(Errno::INVAL))?;
        if !cv.format_match(f) {
            return Err(Error::Errno(Errno::INVAL));
        }
        make_result(
            unsafe {
                match PackedElement::from(cv) {
                    PackedElement::Signed(v) => wiredtiger_pack_int(self.stream, v),
                    PackedElement::Unsigned(v) => wiredtiger_pack_uint(self.stream, v),
                    PackedElement::Str(v) => wiredtiger_pack_str(self.stream, v.as_ptr()),
                    PackedElement::Item(v) => {
                        wiredtiger_pack_item(self.stream, &mut Item::from(v).0)
                    }
                }
            },
            (),
        )
    }
}

impl Drop for PackFormatWriter<'_> {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            let mut used = 0;
            unsafe { wt_sys::wiredtiger_pack_close(self.stream, &mut used) };
        }
    }
}

/// Pack a stream of values according to a format string and estimate an upper bound of the number
/// of bytes required to encode the value.
pub struct MaxLenFormatWriter {
    format: FormatString,
    it: std::slice::Iter<'static, u8>,
    max_len: usize,
}

impl MaxLenFormatWriter {
    pub fn new(format: FormatString) -> Self {
        Self {
            format,
            it: format.0.to_bytes().iter(),
            max_len: 0usize,
        }
    }

    /// Close the stream and return a maximum expected length.
    ///
    /// Returns an error if the pack stream does not match `format.columns_len()`.
    pub fn close(mut self) -> Result<usize> {
        if self.it.next().is_none() {
            Ok(self.max_len)
        } else {
            Err(Error::Errno(Errno::INVAL))
        }
    }
}

impl FormatWriter for MaxLenFormatWriter {
    fn pack<'a>(&mut self, v: impl Into<ColumnValue<'a>>) -> Result<()> {
        let v: ColumnValue<'a> = v.into();
        if v.format_match(*self.it.next().ok_or(Error::Errno(Errno::INVAL))?) {
            Ok(())
        } else {
            Err(Error::Errno(Errno::INVAL))
        }
    }
}

// XXX I want a trivial writer but it's going to have to do lifetime casting to work correctly, and
// is still somewhat unsound as I'm assuming the value will live longer than it might.

pub struct UnpackStream<'b> {
    format: FormatString,
    format_it: FormatStringIter,
    stream: *mut WT_PACK_STREAM,
    buffer: PhantomData<&'b [u8]>,
}

impl<'b> UnpackStream<'b> {
    pub fn new(format: FormatString, buffer: &'b [u8]) -> Result<Self> {
        let format_it = format.iter();
        let mut stream = std::ptr::null_mut();
        make_result(
            unsafe {
                wiredtiger_unpack_start(
                    std::ptr::null_mut(),
                    format.0.as_ptr(),
                    buffer.as_ptr() as *mut c_void,
                    buffer.len(),
                    &mut stream,
                )
            },
            (),
        )
        .map(|()| Self {
            format,
            format_it,
            stream,
            buffer: PhantomData,
        })
    }

    pub fn unpack<V: TryFrom<ColumnValue<'b>, Error = Error>>(&mut self) -> Result<V> {
        let f = self.format_it.next().ok_or(Error::not_found_error())?;
        let cv = match f {
            b'b' => ColumnValue::I8(self.unpack_int()? as i8),
            b'h' => ColumnValue::I16(self.unpack_int()? as i16),
            b'i' | b'l' => ColumnValue::I32(self.unpack_int()? as i32),
            b'q' => ColumnValue::I64(self.unpack_int()?),
            b'B' => ColumnValue::U8(self.unpack_uint()? as u8),
            b'H' => ColumnValue::U16(self.unpack_uint()? as u16),
            b'I' | b'L' => ColumnValue::U32(self.unpack_uint()? as u32),
            b'Q' | b'r' => ColumnValue::U64(self.unpack_uint()?),
            b'S' => ColumnValue::CStr(self.unpack_str()?),
            b'u' => ColumnValue::Item(self.unpack_item()?),
            _ => unreachable!("unexpected type from validated FormatString"),
        };
        V::try_from(cv)
    }

    fn unpack_int(&mut self) -> Result<i64> {
        let mut v = 0;
        make_result(unsafe { wiredtiger_unpack_int(self.stream, &mut v) }, ()).map(|()| v)
    }

    fn unpack_uint(&mut self) -> Result<u64> {
        let mut v = 0;
        make_result(unsafe { wiredtiger_unpack_uint(self.stream, &mut v) }, ()).map(|()| v)
    }

    fn unpack_str(&mut self) -> Result<&'b CStr> {
        // NB: this only works because we don't allow length-delimited strings, which are not null
        // terminated if they reach length.
        let mut p: *const c_char = std::ptr::null_mut();
        make_result(unsafe { wiredtiger_unpack_str(self.stream, &mut p) }, ())
            .map(|()| unsafe { CStr::from_ptr(p) })
    }

    fn unpack_item(&mut self) -> Result<&'b [u8]> {
        let mut item = Item::default();
        make_result(
            unsafe { wiredtiger_unpack_item(self.stream, &mut item.0) },
            (),
        )
        .map(|()| item.into())
    }

    pub fn close(mut self) -> Result<()> {
        let mut len = 0;
        let result = make_result(
            unsafe { wt_sys::wiredtiger_pack_close(self.stream, &mut len) },
            (),
        );
        self.stream = std::ptr::null_mut();
        result
    }
}

impl Drop for UnpackStream<'_> {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            let mut len = 0;
            unsafe { wt_sys::wiredtiger_pack_close(self.stream, &mut len) };
        }
    }
}

pub trait Packed<'b> {
    /// The format of this packed value.
    ///
    /// This is used to validate that a cursor key or value matches the expected format.
    const FORMAT: FormatString;

    // XXX pack estimate size with a stream object. only used for variable size formats.
    // XXX pack into a buffer. uses a small vec. cow-ish return type. trivial formats just return a ref.

    // XXX Packed trait ought to be able to predict the size (from &self) and write to a &mut [u8].
    // Avoid exposing the stream it is crazy town. Should return the size if serializing into a
    // buffer but we ought to be able to serialize to a vector.
    // XXX there also need to be a fast path for packing 'u' where we do _nothing_.
    fn format_pack(&self, writer: &mut impl FormatWriter) -> Result<()>;
    // XXX again: avoid the stream (accept &[u8] or Item) and fast path for 'u' (should be trivial).
    fn unpack(packed: &'b [u8]) -> Result<Self>
    where
        Self: Sized;
}

impl<'b> Packed<'b> for i64 {
    const FORMAT: FormatString = FormatString::new(c"q");

    fn format_pack(&self, writer: &mut impl FormatWriter) -> Result<()> {
        writer.pack(*self)
    }

    fn unpack(packed: &'b [u8]) -> Result<Self> {
        let mut stream = UnpackStream::new(Self::FORMAT, packed)?;
        stream.unpack()
    }
}

impl<'b> Packed<'b> for u64 {
    const FORMAT: FormatString = FormatString::new(c"Q");

    fn format_pack(&self, writer: &mut impl FormatWriter) -> Result<()> {
        writer.pack(*self)
    }

    fn unpack(packed: &'b [u8]) -> Result<Self>
    where
        Self: Sized,
    {
        let mut stream = UnpackStream::new(Self::FORMAT, packed)?;
        stream.unpack()
    }
}

impl<'b> Packed<'b> for &'b [u8] {
    const FORMAT: FormatString = FormatString::new(c"u");

    fn format_pack(&self, writer: &mut impl FormatWriter) -> Result<()> {
        writer.pack(*self)
    }

    fn unpack(packed: &'b [u8]) -> Result<Self>
    where
        Self: Sized,
    {
        // NB: for a packed byte array with no other fields this is a valid transform.
        Ok(packed)
    }
}

impl<'b> Packed<'b> for &'b CStr {
    const FORMAT: FormatString = FormatString::new(c"S");

    fn format_pack(&self, writer: &mut impl FormatWriter) -> Result<()> {
        writer.pack(*self)
    }

    fn unpack(packed: &'b [u8]) -> Result<Self>
    where
        Self: Sized,
    {
        // XXX is this a good idea?
        Ok(unsafe { CStr::from_bytes_with_nul_unchecked(packed) })
    }
}

pub struct StatValue<'b> {
    pub description: &'static CStr,
    pub value_str: &'b CStr,
    pub value: i64,
}

impl<'b> Packed<'b> for StatValue<'b> {
    const FORMAT: FormatString = FormatString::new(c"SSq");

    fn format_pack(&self, writer: &mut impl FormatWriter) -> Result<()> {
        writer.pack(self.description)?;
        writer.pack(self.value_str)?;
        writer.pack(self.value)
    }

    fn unpack(packed: &'b [u8]) -> Result<Self>
    where
        Self: Sized,
    {
        let mut stream = UnpackStream::new(Self::FORMAT, packed)?;
        let description = {
            let d: &CStr = stream.unpack()?;
            // Safety: description strings in wt metadata cursors are statically defined.
            unsafe { CStr::from_ptr::<'static>(d.as_ptr()) }
        };
        let value_str = stream.unpack()?;
        let value = stream.unpack()?;
        Ok(Self {
            description,
            value_str,
            value,
        })
    }
}
