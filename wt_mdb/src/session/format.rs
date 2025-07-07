//! Wrap WT routines to pack and unpack values store in a table.

#![allow(dead_code)] // XXX remove me.

use std::{
    ffi::{c_char, c_void, CStr, CString},
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
// XXX figure out visibility
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum FormatClassification {
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
    pub const fn column_len(&self) -> usize {
        self.0.to_bytes().len()
    }

    pub(crate) fn max_len(&self) -> Option<usize> {
        if let FormatClassification::Fixed(n) = self.1 {
            Some(n)
        } else {
            None
        }
    }

    fn iter(&self) -> FormatStringIter {
        self.0.to_bytes().iter().copied()
    }
}

impl PartialEq<&CStr> for FormatString {
    fn eq(&self, other: &&CStr) -> bool {
        self.0 == *other
    }
}

impl PartialEq<FormatString> for &CStr {
    fn eq(&self, other: &FormatString) -> bool {
        self == &other.0
    }
}

/// Primitive types that can be formatted into column values in WiredTiger.
// TODO: seal this trait. All primitive should be defined in the crate.
// XXX so packing is not my problem, the lifetime is ugly but not a real problem.
// XXX I'm struggling mightily with unpacking because I cannot assign a lifetime to return Self.
pub trait ColumnValue<'b> {
    /// Maximum encoded length of this value.
    fn max_len(&self) -> usize;

    /// Returns true if the format type matches this type.
    fn format_match(format: u8) -> bool;

    /// Transform this into an element for packing.
    fn to_packed(&self) -> PackedElement<'b>;

    /// Transform packed back into this primitive value type.
    ///
    /// Returns `None` on type mismatch or overflow.
    fn from_packed(packed: PackedElement<'b>) -> Option<Self>
    where
        Self: Sized;
}

macro_rules! define_column_value_primitive {
    ($p:ty, $max_len:expr, $match:literal, $packed:ty) => {
        impl<'b> ColumnValue<'b> for $p {
            fn max_len(&self) -> usize {
                $max_len
            }

            fn format_match(format: u8) -> bool {
                $match.to_bytes().contains(&format)
            }

            fn to_packed(&self) -> PackedElement<'b> {
                PackedElement::from(<$packed>::from(*self))
            }

            fn from_packed(packed: PackedElement<'b>) -> Option<Self> {
                <$p>::try_from(<$packed>::try_from(packed).ok()?).ok()
            }
        }
    };
}

define_column_value_primitive!(i8, 5, c"b", i64);
define_column_value_primitive!(i16, 5, c"h", i64);
define_column_value_primitive!(i32, 5, c"il", i64);
define_column_value_primitive!(i64, 9, c"q", i64);
define_column_value_primitive!(u8, 5, c"B", u64);
define_column_value_primitive!(u16, 5, c"H", u64);
define_column_value_primitive!(u32, 5, c"IL", u64);
define_column_value_primitive!(u64, 9, c"Qr", u64);

impl<'b> ColumnValue<'b> for &'b CStr {
    fn max_len(&self) -> usize {
        self.to_bytes_with_nul().len()
    }

    fn format_match(format: u8) -> bool {
        format == b'S'
    }

    fn to_packed(&self) -> PackedElement<'b> {
        PackedElement::from(*self)
    }

    fn from_packed(packed: PackedElement<'b>) -> Option<Self>
    where
        Self: Sized,
    {
        Self::try_from(packed).ok()
    }
}

impl<'b> ColumnValue<'b> for &'b [u8] {
    fn max_len(&self) -> usize {
        self.len() + 5
    }

    fn format_match(format: u8) -> bool {
        format == b'u'
    }

    fn to_packed(&self) -> PackedElement<'b> {
        PackedElement::from(*self)
    }

    fn from_packed(packed: PackedElement<'b>) -> Option<Self>
    where
        Self: Sized,
    {
        Self::try_from(packed).ok()
    }
}

trait CV2 {
    type Unpacked<'a>: Sized;

    fn from_packed<'a>(packed: PackedElement<'a>) -> Option<Self::Unpacked<'a>>;
}

impl CV2 for &[u8] {
    type Unpacked<'a> = &'a [u8];

    fn from_packed<'a>(packed: PackedElement<'a>) -> Option<Self::Unpacked<'a>> {
        match packed {
            PackedElement::Item(v) => Some(v),
            _ => None,
        }
    }
}

/// A single element in the format stream.
pub enum PackedElement<'b> {
    Signed(i64),
    Unsigned(u64),
    CStr(&'b CStr),
    Item(&'b [u8]),
}

pub struct TypeMismatchError;

macro_rules! packed_element_convert {
    ($($var:ident = $other:ty),*) => {
        $(
            impl<'b> From<$other> for PackedElement<'b> {
                fn from(value: $other) -> PackedElement<'b> {
                    Self::$var(value)
                }
            }

            impl<'b> TryFrom<PackedElement<'b>> for $other {
                type Error = TypeMismatchError;

                fn try_from(value: PackedElement<'b>) -> std::result::Result<$other, Self::Error> {
                    if let PackedElement::$var(v) = value {
                        Ok(v)
                    } else {
                        Err(TypeMismatchError)
                    }
                }
            }
        )*
    };
}
packed_element_convert!(Signed = i64, Unsigned = u64, CStr = &'b CStr, Item = &'b [u8]);

/// Used by formatted objects to pack their data.
pub trait FormatWriter {
    /// Add a single value to the writer.
    ///
    /// This may fail if the passed value does not match the expect format.
    fn pack<'a, V: ColumnValue<'a>>(&mut self, v: V) -> Result<()>;

    // TODO: add an unchecked pack for use in derive macros.
}

/// Pack a stream of values described by a format into a byte array for using in WT.
pub struct PackedFormatWriter<'b> {
    format: FormatString,
    format_it: FormatStringIter,
    stream: *mut WT_PACK_STREAM,
    buffer: PhantomData<&'b [u8]>,
}

impl<'b> PackedFormatWriter<'b> {
    /// Create a writer that will pack a series of values described by `format` into `buffer`.
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

    /// Close this writer and return the number of bytes written to the input buffer.
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

impl FormatWriter for PackedFormatWriter<'_> {
    fn pack<'a, V: ColumnValue<'a>>(&mut self, v: V) -> Result<()> {
        let f = self.format_it.next().ok_or(Error::Errno(Errno::INVAL))?;
        if !V::format_match(f) {
            return Err(Error::Errno(Errno::INVAL));
        }
        make_result(
            unsafe {
                match v.to_packed() {
                    PackedElement::Signed(v) => wiredtiger_pack_int(self.stream, v),
                    PackedElement::Unsigned(v) => wiredtiger_pack_uint(self.stream, v),
                    PackedElement::CStr(v) => wiredtiger_pack_str(self.stream, v.as_ptr()),
                    PackedElement::Item(v) => {
                        wiredtiger_pack_item(self.stream, &mut Item::from(v).0)
                    }
                }
            },
            (),
        )
    }
}

impl Drop for PackedFormatWriter<'_> {
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
    /// Create a new writer to record an upper bound length for a sequences of values described by
    /// `format`.
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
    fn pack<'a, V: ColumnValue<'a>>(&mut self, v: V) -> Result<()> {
        if V::format_match(*self.it.next().ok_or(Error::Errno(Errno::INVAL))?) {
            self.max_len += v.max_len();
            Ok(())
        } else {
            Err(Error::Errno(Errno::INVAL))
        }
    }
}

pub struct PackedFormatReader<'b> {
    format: FormatString,
    format_it: FormatStringIter,
    stream: *mut WT_PACK_STREAM,
    buffer: PhantomData<&'b [u8]>,
}

impl<'b> PackedFormatReader<'b> {
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

    pub fn unpack<V: ColumnValue<'b>>(&mut self) -> Result<V> {
        let f = self.format_it.next().ok_or(Error::not_found_error())?;
        if !V::format_match(f) {
            return Err(Error::Errno(Errno::INVAL));
        }
        // OK at this point I need to go from format => packed value.
        let element: PackedElement<'b> = match f {
            b'b' | b'h' | b'i' | b'l' | b'q' => {
                let mut v = 0;
                make_result(unsafe { wiredtiger_unpack_int(self.stream, &mut v) }, ())?;
                v.into()
            }
            b'B' | b'H' | b'I' | b'L' | b'Q' | b'r' => {
                let mut v = 0;
                make_result(unsafe { wiredtiger_unpack_uint(self.stream, &mut v) }, ())?;
                v.into()
            }
            b'S' => {
                // NB: this only works because we don't allow length-delimited strings, which are not null
                // terminated if they reach length.
                let mut p: *const c_char = std::ptr::null_mut();
                make_result(unsafe { wiredtiger_unpack_str(self.stream, &mut p) }, ())?;
                unsafe { CStr::from_ptr(p) }.into()
            }
            b'u' => {
                let mut item = Item::default();
                make_result(
                    unsafe { wiredtiger_unpack_item(self.stream, &mut item.0) },
                    (),
                )?;
                <&'b [u8]>::from(item).into()
            }
            _ => unreachable!("unexpected type from validated FormatString"),
        };
        V::from_packed(element).ok_or(Error::Errno(Errno::INVAL))
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

impl Drop for PackedFormatReader<'_> {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            let mut len = 0;
            unsafe { wt_sys::wiredtiger_pack_close(self.stream, &mut len) };
        }
    }
}

/// Something that can be formatted/packed into a key or value in WiredTiger.
pub trait Formatter {
    /// The format of this packed value.
    ///
    /// This is used to validate that a cursor key or value matches the expected format.
    const FORMAT: FormatString;

    type Ref<'a>: FormatterRef<'a, Self::Owned>;
    type Owned: FormatterOwned;

    /// Format the contents of this object into `writer`.
    fn pack(writer: &mut impl FormatWriter, value: &Self::Ref<'_>) -> Result<()>;
    /// Unpack formatted data into a new object.
    fn unpack<'b>(reader: &mut PackedFormatReader<'b>) -> Result<Self::Ref<'b>>;

    /// Return a "packed" byte array for trivially packed formats, to avoid a copy.
    /// Most implementations can simply return `None`.
    #[allow(unused)]
    fn pack_trivial<'b>(value: &Self::Ref<'b>) -> Option<&'b [u8]> {
        None
    }

    /// Return an "unpacked" byte array for trivially packed formats, to avoid a copy.
    /// Most implementations can simply return `None`.
    #[allow(unused)]
    fn unpack_trivial<'b>(packed: &'b [u8]) -> Option<Self::Ref<'b>> {
        None
    }
}

/// A reference to a value unpacked by the formatter.
pub trait FormatterRef<'a, Owned>: Sized {
    /// Get an owned copy of the ref value.
    fn to_formatter_owned(&self) -> Owned;
}

/// An owned value unpacked by the formatter.
pub trait FormatterOwned: Sized {
    type Ref<'a>: FormatterRef<'a, Self>
    where
        Self: 'a;

    /// Get a reference to this owned value.
    fn to_formatter_ref<'a>(&'a self) -> Self::Ref<'a>;
}

macro_rules! define_primitive_formatter {
    ($name: ident, $primitive:ty, $format:literal) => {
        impl<'a> FormatterRef<'a, $primitive> for $primitive {
            fn to_formatter_owned(&self) -> $primitive {
                *self
            }
        }

        impl FormatterOwned for $primitive {
            type Ref<'a> = $primitive;

            fn to_formatter_ref<'a>(&'a self) -> Self::Ref<'a> {
                *self
            }
        }

        #[derive(Debug, Copy, Clone)]
        pub struct $name;

        impl Formatter for $name {
            const FORMAT: FormatString = FormatString::new($format);

            type Ref<'a> = $primitive;
            type Owned = $primitive;

            fn pack(writer: &mut impl FormatWriter, value: &Self::Ref<'_>) -> Result<()> {
                writer.pack(*value)
            }

            fn unpack<'b>(reader: &mut PackedFormatReader<'b>) -> Result<Self::Ref<'b>> {
                reader.unpack()
            }
        }
    };
}

define_primitive_formatter!(I8Formatter, i8, c"b");
define_primitive_formatter!(I16Formatter, i16, c"h");
define_primitive_formatter!(I32Formatter, i32, c"i");
define_primitive_formatter!(I64Formatter, i64, c"q");
define_primitive_formatter!(U8Formatter, u8, c"B");
define_primitive_formatter!(U16Formatter, u16, c"H");
define_primitive_formatter!(U32Formatter, u32, c"I");
define_primitive_formatter!(U64Formatter, u64, c"Q");

impl FormatterOwned for Vec<u8> {
    type Ref<'a> = &'a [u8];
    fn to_formatter_ref<'a>(&'a self) -> Self::Ref<'a> {
        self.as_slice()
    }
}

impl<'a> FormatterRef<'a, Vec<u8>> for &'a [u8] {
    fn to_formatter_owned(&self) -> Vec<u8> {
        self.to_vec()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ByteSliceFormatter;

impl Formatter for ByteSliceFormatter {
    const FORMAT: FormatString = FormatString::new(c"u");

    type Ref<'a> = &'a [u8];
    type Owned = Vec<u8>;

    fn pack(writer: &mut impl FormatWriter, value: &Self::Ref<'_>) -> Result<()> {
        writer.pack(*value)
    }

    fn unpack<'b>(reader: &mut PackedFormatReader<'b>) -> Result<Self::Ref<'b>> {
        reader.unpack()
    }

    fn pack_trivial<'b>(value: &Self::Ref<'b>) -> Option<&'b [u8]> {
        Some(*value)
    }

    fn unpack_trivial<'b>(packed: &'b [u8]) -> Option<Self::Ref<'b>> {
        Some(packed)
    }
}

impl FormatterOwned for CString {
    type Ref<'a> = &'a CStr;
    fn to_formatter_ref<'a>(&'a self) -> Self::Ref<'a> {
        self.as_c_str()
    }
}

impl<'a> FormatterRef<'a, CString> for &'a CStr {
    fn to_formatter_owned(&self) -> CString {
        CString::from(*self)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct CStringFormatter;

impl Formatter for CStringFormatter {
    const FORMAT: FormatString = FormatString::new(c"S");

    type Ref<'a> = &'a CStr;
    type Owned = CString;

    fn pack(writer: &mut impl FormatWriter, value: &Self::Ref<'_>) -> Result<()> {
        writer.pack(*value)
    }

    fn unpack<'b>(reader: &mut PackedFormatReader<'b>) -> Result<Self::Ref<'b>> {
        reader.unpack()
    }

    fn pack_trivial<'b>(value: &Self::Ref<'b>) -> Option<&'b [u8]> {
        Some(value.to_bytes_with_nul())
    }

    fn unpack_trivial<'b>(packed: &'b [u8]) -> Option<Self::Ref<'b>> {
        CStr::from_bytes_with_nul(packed).ok()
    }
}

// XXX this belongs with a typed cursor for stats.
#[derive(Debug, Clone)]
pub struct StatValue {
    pub description: &'static CStr,
    pub value_str: CString,
    pub value: i64,
}

// XXX this belongs with a typed cursor for stats.
#[derive(Debug, Copy, Clone)]
pub struct StatValueRef<'b> {
    pub description: &'static CStr,
    pub value_str: &'b CStr,
    pub value: i64,
}

impl<'a> FormatterRef<'a, StatValue> for StatValueRef<'a> {
    fn to_formatter_owned(&self) -> StatValue {
        StatValue {
            description: self.description,
            value_str: self.value_str.into(),
            value: self.value,
        }
    }
}

impl FormatterOwned for StatValue {
    type Ref<'a> = StatValueRef<'a>;

    fn to_formatter_ref<'a>(&'a self) -> Self::Ref<'a> {
        StatValueRef {
            description: self.description,
            value_str: self.value_str.as_c_str(),
            value: self.value,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct StatValueFormatter;

impl Formatter for StatValueFormatter {
    const FORMAT: FormatString = FormatString::new(c"SSq");

    type Ref<'a> = StatValueRef<'a>;
    type Owned = StatValue;

    fn pack(writer: &mut impl FormatWriter, value: &Self::Ref<'_>) -> Result<()> {
        writer.pack(value.description)?;
        writer.pack(value.value_str)?;
        writer.pack(value.value)
    }

    fn unpack<'b>(reader: &mut PackedFormatReader<'b>) -> Result<Self::Ref<'b>> {
        let description = {
            let d: &CStr = reader.unpack()?;
            // Safety: description strings in wt metadata cursors are statically defined.
            unsafe { CStr::from_ptr::<'static>(d.as_ptr()) }
        };
        let value_str = reader.unpack()?;
        let value = reader.unpack()?;
        Ok(StatValueRef {
            description,
            value_str,
            value,
        })
    }
}
