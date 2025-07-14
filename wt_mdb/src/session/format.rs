//! Wrap WT routines to pack and unpack values store in a table.

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
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
                // TODO figure out what is going on with 'U'. Is it for 'u' when there are multiple
                // columns in the table?
                b'b' | b'B' | b'h' | b'H' | b'i' | b'I' | b'l' | b'L' | b'q' | b'Q' | b'r'
                | b'S' | b'u' => i += 1,
                _ => panic!("invalid column type"),
            }
        }
        Self(format, FormatClassification::new(format))
    }

    pub fn format_str(&self) -> &'static str {
        self.0.to_str().expect("valid format strings are ASCII")
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
pub trait ColumnValue<'b>:
    Into<PackedElement<'b>> + TryFrom<PackedElement<'b>, Error = TypeMismatchError>
{
    /// Maximum encoded length of this value.
    fn max_len(&self) -> usize;

    /// Returns true if the format type matches this type.
    fn format_match(format: u8) -> bool;
}

macro_rules! define_column_value_conversions {
    ($primitive:ty, $elemvar:ident) => {
        impl<'b> From<$primitive> for PackedElement<'b> {
            fn from(value: $primitive) -> Self {
                PackedElement::$elemvar(value.into())
            }
        }

        impl<'b> TryFrom<PackedElement<'b>> for $primitive {
            type Error = TypeMismatchError;

            fn try_from(v: PackedElement<'b>) -> std::result::Result<$primitive, Self::Error> {
                if let PackedElement::$elemvar(v) = v {
                    v.try_into().map_err(|_| TypeMismatchError)
                } else {
                    Err(TypeMismatchError)
                }
            }
        }
    };
}

macro_rules! define_column_value_primitive {
    ($p:ty, $max_len:expr, $format_match:literal, $elemvar:ident) => {
        impl<'b> ColumnValue<'b> for $p {
            fn max_len(&self) -> usize {
                $max_len
            }

            fn format_match(format: u8) -> bool {
                $format_match.to_bytes().contains(&format)
            }
        }

        define_column_value_conversions!($p, $elemvar);
    };
}

define_column_value_primitive!(i8, 5, c"b", Signed);
define_column_value_primitive!(i16, 5, c"h", Signed);
define_column_value_primitive!(i32, 5, c"il", Signed);
define_column_value_primitive!(i64, 9, c"q", Signed);
define_column_value_primitive!(u8, 5, c"B", Unsigned);
define_column_value_primitive!(u16, 5, c"H", Unsigned);
define_column_value_primitive!(u32, 5, c"IL", Unsigned);
define_column_value_primitive!(u64, 9, c"Qr", Unsigned);

impl<'b> ColumnValue<'b> for &'b CStr {
    fn max_len(&self) -> usize {
        self.to_bytes_with_nul().len()
    }

    fn format_match(format: u8) -> bool {
        format == b'S'
    }
}

define_column_value_conversions!(&'b CStr, CStr);

impl<'b> ColumnValue<'b> for &'b [u8] {
    fn max_len(&self) -> usize {
        self.len() + 5
    }

    fn format_match(format: u8) -> bool {
        format == b'u'
    }
}

define_column_value_conversions!(&'b [u8], Item);

/// A single element in the format stream.
pub enum PackedElement<'b> {
    Signed(i64),
    Unsigned(u64),
    CStr(&'b CStr),
    Item(&'b [u8]),
}

pub struct TypeMismatchError;

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
                match v.into() {
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
    it: std::slice::Iter<'static, u8>,
    max_len: usize,
}

impl MaxLenFormatWriter {
    /// Create a new writer to record an upper bound length for a sequences of values described by
    /// `format`.
    pub fn new(format: FormatString) -> Self {
        Self {
            it: format.0.to_bytes().iter(),
            max_len: 0usize,
        }
    }

    /// Close the stream and return a maximum expected length.
    ///
    /// Returns an error if the pack stream does not match `format.columns_len()`.
    pub fn close(mut self) -> Result<usize> {
        self.it
            .next()
            .map(|_| Err(Error::Errno(Errno::INVAL)))
            .unwrap_or(Ok(self.max_len))
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
        V::try_from(element).map_err(|_| Error::Errno(Errno::INVAL))
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
pub trait Formatted: Sized {
    /// The format of this packed value.
    ///
    /// This is used to validate that a cursor key or value matches the expected format.
    const FORMAT: FormatString;

    /// A reference to this this type. This may be a partial references with some unpacked
    /// primitives (ints, etc) but also fields referencing strings or bytes.
    type Ref<'a>: Into<Self>;

    /// Obtain a reference type.
    // NB: this cannot simply be another trait, unfortunately:
    // * Borrow doesn't work for formatted structs that contain some references.
    // * AsRef doesn't work for primitive integer types.
    // * From/Into doesn't work for CString or Vec<u8>
    fn to_formatted_ref(&self) -> Self::Ref<'_>;

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

macro_rules! define_primitive_formatter {
    ($primitive:ty, $format:literal) => {
        define_primitive_formatter!(
            $primitive,
            $primitive,
            into,
            $format,
            default_trivial_pack,
            default_trivial_unpack
        );
    };
    ($owned:ty, $ref:ty, $as_ref:ident, $format:literal, $pack_trivial:ident, $unpack_trivial:ident) => {
        impl Formatted for $owned {
            const FORMAT: FormatString = FormatString::new($format);

            type Ref<'a> = $ref;

            fn to_formatted_ref(&self) -> Self::Ref<'_> {
                (*self).$as_ref()
            }

            fn pack(writer: &mut impl FormatWriter, value: &Self::Ref<'_>) -> Result<()> {
                writer.pack(*value)
            }

            fn unpack<'b>(reader: &mut PackedFormatReader<'b>) -> Result<Self::Ref<'b>> {
                reader.unpack()
            }

            fn pack_trivial<'b>(value: &Self::Ref<'b>) -> Option<&'b [u8]> {
                $pack_trivial(value)
            }

            fn unpack_trivial<'b>(packed: &'b [u8]) -> Option<Self::Ref<'b>> {
                $unpack_trivial(packed)
            }
        }
    };
}

define_primitive_formatter!(i8, c"b");
define_primitive_formatter!(i16, c"h");
define_primitive_formatter!(i32, c"i");
define_primitive_formatter!(i64, c"q");
define_primitive_formatter!(u8, c"B");
define_primitive_formatter!(u16, c"H");
define_primitive_formatter!(u32, c"I");
define_primitive_formatter!(u64, c"Q");
define_primitive_formatter!(Vec<u8>, &'a [u8], as_ref, c"u", Some, Some);
define_primitive_formatter!(
    CString,
    &'a CStr,
    as_ref,
    c"S",
    pack_trivial_cstr,
    unpack_trivial_cstr
);

#[inline(always)]
fn default_trivial_pack<'b, T>(_t: &T) -> Option<&'b [u8]> {
    None
}

#[inline(always)]
fn default_trivial_unpack<T>(_p: &[u8]) -> Option<T> {
    None
}

#[inline(always)]
fn pack_trivial_cstr(value: &CStr) -> Option<&[u8]> {
    Some(value.to_bytes_with_nul())
}

#[inline(always)]
fn unpack_trivial_cstr(packed: &[u8]) -> Option<&CStr> {
    CStr::from_bytes_with_nul(packed).ok()
}
