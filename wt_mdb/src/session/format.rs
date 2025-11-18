//! Wrap WT routines to pack and unpack values store in a table.

use std::ffi::{c_char, c_void, CStr, CString};

use crate::{make_result, Error, Result};

/// A validated WiredTiger format string.
///
/// This only supports a subset of of the documented WT types, in particular it does not support
/// strings of a fixed length.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct FormatString(&'static CStr);

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
        Self(format)
    }

    pub fn format_str(&self) -> &'static str {
        self.0.to_str().expect("valid format strings are ASCII")
    }

    /// Return the number of columns in the format.
    pub const fn column_len(&self) -> usize {
        self.0.to_bytes().len()
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

/// Something that can be formatted/packed into a key or value in WiredTiger.
///
/// For performance reasons you should prefer a "u" format and manual packing of your values from
/// byte strings. WiredTiger's struct packing is very expensive when it cannot be inlined.
pub trait Formatted: Sized {
    /// The format of this packed value.
    ///
    /// This is used to validate that a cursor key or value matches the expected format.
    const FORMAT: FormatString;

    /// A reference to this this type. This may be a partial references with some unpacked
    /// primitives (ints, etc) but also fields referencing strings or bytes.
    type Ref<'a>: Into<Self> + Copy;

    /// Obtain a reference type.
    // NB: this cannot simply be another trait, unfortunately:
    // * Borrow doesn't work for formatted structs that contain some references.
    // * AsRef doesn't work for primitive integer types.
    // * From/Into doesn't work for CString or Vec<u8>
    fn to_formatted_ref(&self) -> Self::Ref<'_>;

    /// Pack a reference into a byte array.
    fn pack(value: Self::Ref<'_>, packed: &mut Vec<u8>) -> Result<()>;
    /// Unpack a byte array into a reference.
    fn unpack<'b>(packed: &'b [u8]) -> Result<Self::Ref<'b>>;
}

macro_rules! define_primitive_formatter {
    ($owned:ty, $format:literal) => {
        impl Formatted for $owned {
            const FORMAT: FormatString = FormatString::new($format);

            type Ref<'a> = $owned;

            fn to_formatted_ref(&self) -> Self::Ref<'_> {
                (*self).into()
            }

            fn pack(value: Self::Ref<'_>, packed: &mut Vec<u8>) -> Result<()> {
                pack1::<Self>(value, packed)
            }

            fn unpack<'b>(packed: &'b [u8]) -> Result<Self::Ref<'b>> {
                unpack1::<Self>(packed)
            }
        }

        impl FormattedPrimitive for $owned {
            type Raw = $owned;
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

impl Formatted for Vec<u8> {
    const FORMAT: FormatString = FormatString::new(c"u");

    type Ref<'a> = &'a [u8];

    fn to_formatted_ref(&self) -> Self::Ref<'_> {
        (*self).as_ref()
    }

    fn pack(value: Self::Ref<'_>, packed: &mut Vec<u8>) -> Result<()> {
        packed.clear();
        packed.extend_from_slice(value);
        Ok(())
    }

    fn unpack<'b>(packed: &'b [u8]) -> Result<Self::Ref<'b>> {
        Ok(packed)
    }
}

impl Formatted for CString {
    const FORMAT: FormatString = FormatString::new(c"S");

    type Ref<'a> = &'a CStr;

    fn to_formatted_ref(&self) -> Self::Ref<'_> {
        (*self).as_ref()
    }

    fn pack(value: Self::Ref<'_>, packed: &mut Vec<u8>) -> Result<()> {
        packed.clear();
        packed.extend_from_slice(value.to_bytes_with_nul());
        Ok(())
    }

    fn unpack<'b>(packed: &'b [u8]) -> Result<Self::Ref<'b>> {
        CStr::from_bytes_with_nul(packed)
            .map_err(|_| Error::WiredTiger(crate::WiredTigerError::Generic))
    }
}

/// A formatted type that can be packed into a single WT column.
///
/// This trait is sealed and should only map directly to a single entry in a format string.
pub trait FormattedPrimitive: Formatted {
    type Raw: Sized
        + Copy
        + Default
        + std::fmt::Debug
        + for<'a> From<Self::Ref<'a>>
        + for<'a> Into<Self::Ref<'a>>;
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(transparent)]
pub struct RawCStr(*const c_char);

impl From<RawCStr> for &CStr {
    fn from(value: RawCStr) -> Self {
        unsafe { CStr::from_ptr(value.0) }
    }
}

impl From<&CStr> for RawCStr {
    fn from(value: &CStr) -> Self {
        Self(value.as_ptr())
    }
}

impl FormattedPrimitive for CString {
    type Raw = RawCStr;
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(transparent)]
pub struct RawItem(super::Item);

impl From<RawItem> for &[u8] {
    fn from(value: RawItem) -> Self {
        <&[u8]>::from(value.0)
    }
}

impl From<&[u8]> for RawItem {
    fn from(value: &[u8]) -> Self {
        Self(value.into())
    }
}

impl FormattedPrimitive for Vec<u8> {
    type Raw = RawItem;
}

pub fn pack1<A: FormattedPrimitive>(a: A::Ref<'_>, packed: &mut Vec<u8>) -> Result<()> {
    let mut len = 0usize;
    let raw = A::Raw::from(a);
    make_result(
        unsafe {
            wt_sys::wiredtiger_struct_size(
                std::ptr::null_mut(),
                &mut len,
                A::FORMAT.0.as_ptr(),
                raw,
            )
        },
        (),
    )?;
    packed.resize(len, 0);
    make_result(
        unsafe {
            wt_sys::wiredtiger_struct_pack(
                std::ptr::null_mut(),
                packed.as_mut_ptr() as *mut c_void,
                len,
                A::FORMAT.0.as_ptr(),
                raw,
            )
        },
        (),
    )
}

pub fn pack2<A: FormattedPrimitive, B: FormattedPrimitive>(
    format: FormatString,
    a: A::Ref<'_>,
    b: B::Ref<'_>,
    packed: &mut Vec<u8>,
) -> Result<()> {
    let mut len = 0usize;
    let raw_a = A::Raw::from(a);
    let raw_b = B::Raw::from(b);
    make_result(
        unsafe {
            wt_sys::wiredtiger_struct_size(
                std::ptr::null_mut(),
                &mut len,
                format.0.as_ptr(),
                raw_a,
                raw_b,
            )
        },
        (),
    )?;
    packed.resize(len, 0);
    make_result(
        unsafe {
            wt_sys::wiredtiger_struct_pack(
                std::ptr::null_mut(),
                packed.as_mut_ptr() as *mut c_void,
                len,
                format.0.as_ptr(),
                raw_a,
                raw_b,
            )
        },
        (),
    )
}

pub fn pack3<A: FormattedPrimitive, B: FormattedPrimitive, C: FormattedPrimitive>(
    format: FormatString,
    a: A::Ref<'_>,
    b: B::Ref<'_>,
    c: C::Ref<'_>,
    packed: &mut Vec<u8>,
) -> Result<()> {
    let mut len = 0usize;
    let raw_a = A::Raw::from(a);
    let raw_b = B::Raw::from(b);
    let raw_c = C::Raw::from(c);
    make_result(
        unsafe {
            wt_sys::wiredtiger_struct_size(
                std::ptr::null_mut(),
                &mut len,
                format.0.as_ptr(),
                raw_a,
                raw_b,
                raw_c,
            )
        },
        (),
    )?;
    packed.resize(len, 0);
    make_result(
        unsafe {
            wt_sys::wiredtiger_struct_pack(
                std::ptr::null_mut(),
                packed.as_mut_ptr() as *mut c_void,
                len,
                format.0.as_ptr(),
                raw_a,
                raw_b,
                raw_c,
            )
        },
        (),
    )
}

pub fn unpack1<'a, A: FormattedPrimitive>(packed: &'a [u8]) -> Result<A::Ref<'a>> {
    let mut raw = A::Raw::default();
    make_result(
        unsafe {
            wt_sys::wiredtiger_struct_unpack(
                std::ptr::null_mut(),
                packed.as_ptr() as *const c_void,
                packed.len(),
                A::FORMAT.0.as_ptr(),
                &mut raw,
            )
        },
        (),
    )?;
    Ok(raw.into())
}

pub fn unpack2<'a, A: FormattedPrimitive, B: FormattedPrimitive>(
    format: FormatString,
    packed: &'a [u8],
) -> Result<(A::Ref<'a>, B::Ref<'a>)> {
    let mut raw_a = A::Raw::default();
    let mut raw_b = B::Raw::default();
    make_result(
        unsafe {
            wt_sys::wiredtiger_struct_unpack(
                std::ptr::null_mut(),
                packed.as_ptr() as *const c_void,
                packed.len(),
                format.0.as_ptr(),
                &mut raw_a,
                &mut raw_b,
            )
        },
        (),
    )?;
    Ok((raw_a.into(), raw_b.into()))
}

pub fn unpack3<'a, A: FormattedPrimitive, B: FormattedPrimitive, C: FormattedPrimitive>(
    format: FormatString,
    packed: &'a [u8],
) -> Result<(A::Ref<'a>, B::Ref<'a>, C::Ref<'a>)> {
    let mut raw_a = A::Raw::default();
    let mut raw_b = B::Raw::default();
    let mut raw_c = C::Raw::default();
    make_result(
        unsafe {
            wt_sys::wiredtiger_struct_unpack(
                std::ptr::null_mut(),
                packed.as_ptr() as *const c_void,
                packed.len(),
                format.0.as_ptr(),
                &mut raw_a,
                &mut raw_b,
                &mut raw_c,
            )
        },
        (),
    )?;
    Ok((raw_a.into(), raw_b.into(), raw_c.into()))
}
