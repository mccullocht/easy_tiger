//! Bindings for the WiredTiger C library that align with MongoDB usage.
//!
//! To use, open a `Connection` to an on-disk database. `Session`s are used to group
//! operations (including in transactions) and create `RecordCursor`s used to read
//! and write table data.
//!
//! Unlike the general-purpose WiredTiger library, this library only allows tables
//! that are keyed by `i64` with byte array payloads.
mod connection;
mod record_cursor;
mod session;

use wt_sys::wiredtiger_strerror;

use std::ffi::{CStr, CString};
use std::io::ErrorKind;
use std::num::NonZero;
use std::ptr::NonNull;

/// WiredTiger specific error codes.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(i32)]
pub enum WiredTigerError {
    Rollback = wt_sys::WT_ROLLBACK,
    DuplicateKey = wt_sys::WT_DUPLICATE_KEY,
    Generic = wt_sys::WT_ERROR,
    NotFound = wt_sys::WT_NOTFOUND,
    Panic = wt_sys::WT_PANIC,
    RunRecovery = wt_sys::WT_RUN_RECOVERY,
    CacheFull = wt_sys::WT_CACHE_FULL,
    PrepareConflict = wt_sys::WT_PREPARE_CONFLICT,
    TrySalvage = wt_sys::WT_TRY_SALVAGE,
}

impl WiredTigerError {
    pub fn to_c_str(self) -> &'static CStr {
        unsafe { CStr::from_ptr(wiredtiger_strerror(self as i32)) }
    }

    fn try_from_code(value: i32) -> Option<WiredTigerError> {
        match value {
            wt_sys::WT_ROLLBACK => Some(WiredTigerError::Rollback),
            wt_sys::WT_DUPLICATE_KEY => Some(WiredTigerError::DuplicateKey),
            wt_sys::WT_ERROR => Some(WiredTigerError::Generic),
            wt_sys::WT_NOTFOUND => Some(WiredTigerError::NotFound),
            wt_sys::WT_PANIC => Some(WiredTigerError::Panic),
            wt_sys::WT_RUN_RECOVERY => Some(WiredTigerError::RunRecovery),
            wt_sys::WT_CACHE_FULL => Some(WiredTigerError::CacheFull),
            wt_sys::WT_PREPARE_CONFLICT => Some(WiredTigerError::PrepareConflict),
            wt_sys::WT_TRY_SALVAGE => Some(WiredTigerError::TrySalvage),
            _ => None,
        }
    }
}

impl std::fmt::Display for WiredTigerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_c_str().to_string_lossy())
    }
}

impl From<WiredTigerError> for ErrorKind {
    fn from(value: WiredTigerError) -> Self {
        match value {
            WiredTigerError::Rollback
            | WiredTigerError::Panic
            | WiredTigerError::RunRecovery
            | WiredTigerError::PrepareConflict
            | WiredTigerError::TrySalvage
            | WiredTigerError::Generic => ErrorKind::Other,
            WiredTigerError::DuplicateKey => ErrorKind::AlreadyExists,
            WiredTigerError::NotFound => ErrorKind::NotFound,
            WiredTigerError::CacheFull => ErrorKind::OutOfMemory,
        }
    }
}

/// An Error, either WiredTiger-specific or POSIX.
// TODO: real posix mapping and an unknown type.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Error {
    WiredTiger(WiredTigerError),
    Posix(i32),
}

impl Error {
    fn generic_error() -> Self {
        Error::WiredTiger(WiredTigerError::Generic)
    }
}

impl From<NonZero<i32>> for Error {
    fn from(value: NonZero<i32>) -> Self {
        WiredTigerError::try_from_code(value.get())
            .map(Error::WiredTiger)
            .unwrap_or(Error::Posix(value.get()))
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WiredTiger(w) => write!(f, "WT {}", w),
            Self::Posix(p) => write!(f, "POSIX {}", p),
        }
    }
}

impl std::error::Error for Error {}

impl From<Error> for std::io::Error {
    fn from(value: Error) -> Self {
        let kind = match &value {
            Error::WiredTiger(wt) => (*wt).into(),
            Error::Posix(_) => ErrorKind::Other,
        };
        std::io::Error::new(kind, value)
    }
}

pub use connection::{Connection, ConnectionOptions, ConnectionOptionsBuilder};
pub use record_cursor::{Record, RecordCursor, RecordView};
pub use session::{
    BeginTransactionOptions, BeginTransactionOptionsBuilder, CommitTransactionOptions,
    CommitTransactionOptionsBuilder, CreateOptions, CreateOptionsBuilder, DropOptions,
    DropOptionsBuilder, RollbackTransactionOptions, RollbackTransactionOptionsBuilder, Session,
};
pub type Result<T> = std::result::Result<T, Error>;

fn make_result<T>(code: i32, value: T) -> Result<T> {
    NonZero::<i32>::new(code)
        .map(|c| Err(Error::from(c)))
        .unwrap_or(Ok(value))
}

fn wrap_ptr_create<T>(code: i32, ptr: *mut T) -> Result<NonNull<T>> {
    let p = make_result(code, ptr)?;
    NonNull::new(p).ok_or(Error::generic_error())
}

fn make_table_uri(table_name: &str) -> CString {
    CString::new(format!("table:{}", table_name)).expect("no nulls in table_name")
}

#[cfg(test)]
mod test {
    use std::io::ErrorKind;

    use crate::{
        connection::{Connection, ConnectionOptions, ConnectionOptionsBuilder},
        record_cursor::{Record, RecordView},
        Error, WiredTigerError,
    };

    fn conn_options() -> Option<ConnectionOptions> {
        Some(ConnectionOptionsBuilder::default().create().into())
    }

    #[test]
    fn insert_and_iterate() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();
        session.create_record_table("test", None).unwrap();
        let mut cursor = session.open_record_cursor("test").unwrap();
        assert_eq!(cursor.set(&RecordView::new(11, b"bar")), Ok(()));
        assert_eq!(cursor.set(&RecordView::new(7, b"foo")), Ok(()));
        assert_eq!(cursor.next(), Some(Ok(Record::new(7, b"foo"))));
        assert_eq!(cursor.next(), Some(Ok(Record::new(11, b"bar"))));
        assert_eq!(cursor.next(), None);
    }

    #[test]
    fn insert_and_search() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();
        session.create_record_table("test", None).unwrap();
        let mut cursor = session.open_record_cursor("test").unwrap();
        let value: &[u8] = b"bar";
        assert_eq!(cursor.set(&RecordView::new(7, value)), Ok(()));
        assert_eq!(cursor.set(&Record::new(11, value)), Ok(()));
        assert_eq!(cursor.seek_exact(7), Some(Ok(Record::new(7, value))));
        assert_eq!(cursor.seek_exact(11), Some(Ok(Record::new(11, value))));
    }

    #[test]
    fn insert_and_remove() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();
        session.create_record_table("test", None).unwrap();
        let mut cursor = session.open_record_cursor("test").unwrap();
        assert_eq!(cursor.set(&RecordView::new(11, b"bar")), Ok(()));
        assert_eq!(cursor.set(&RecordView::new(7, b"foo")), Ok(()));
        assert_eq!(cursor.remove(7), Ok(()));
        assert_eq!(cursor.next(), Some(Ok(Record::new(11, b"bar"))));
        assert_eq!(cursor.next(), None);
        assert_eq!(
            cursor.remove(13),
            Err(Error::WiredTiger(WiredTigerError::NotFound))
        );
    }

    #[test]
    fn largest_key() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();
        session.create_record_table("test", None).unwrap();
        let mut cursor = session.open_record_cursor("test").unwrap();
        assert_eq!(cursor.largest_key(), None);
        assert_eq!(cursor.set(&RecordView::new(-1, b"bar")), Ok(()));
        assert_eq!(cursor.largest_key(), Some(Ok(-1)));
        assert_eq!(cursor.set(&RecordView::new(7, b"foo")), Ok(()));
        assert_eq!(cursor.largest_key(), Some(Ok(7)));
    }

    #[test]
    fn transaction_commit() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();
        session.create_record_table("test", None).unwrap();
        let read_session = conn.open_session().unwrap();
        let mut read_cursor = read_session.open_record_cursor("test").unwrap();

        let mut cursor = session.open_record_cursor("test").unwrap();
        assert_eq!(session.begin_transaction(None), Ok(()));
        assert_eq!(cursor.set(&RecordView::new(1, b"foo")), Ok(()));
        assert_eq!(cursor.set(&RecordView::new(2, b"bar")), Ok(()));
        assert_eq!(cursor.next(), Some(Ok(Record::new(1, b"foo"))));
        assert_eq!(read_cursor.next(), None);
        assert_eq!(session.commit_transaction(None), Ok(()));
        assert_eq!(read_cursor.next(), Some(Ok(Record::new(1, b"foo"))));
    }

    #[test]
    fn transaction_rollback() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();
        session.create_record_table("test", None).unwrap();

        let mut cursor = session.open_record_cursor("test").unwrap();
        assert_eq!(session.begin_transaction(None), Ok(()));
        assert_eq!(cursor.set(&RecordView::new(1, b"foo")), Ok(()));
        assert_eq!(cursor.next(), Some(Ok(Record::new(1, b"foo"))));
        assert_eq!(session.rollback_transaction(None), Ok(()));
        assert_eq!(cursor.next(), None);
    }

    #[test]
    fn bulk_load() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();

        // Create Vec<Record>, bulk_load() into session, compare cursors.
        let records = vec![
            Record::new(7, b"foo"),
            Record::new(11, b"bar"),
            Record::new(19, b"quux"),
        ];
        assert_eq!(
            session.bulk_load("test", None, records.clone().into_iter()),
            Ok(())
        );

        let cursor = session.open_record_cursor("test").unwrap();
        for (expected, actual) in records.iter().zip(cursor) {
            assert_eq!(Ok(expected), actual.as_ref());
        }
    }

    #[test]
    fn bulk_load_existing_table() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();

        // Bulk load will happily load into an empty table, so to get it to fail we insert a record.
        assert_eq!(session.create_record_table("test", None), Ok(()));
        let mut cursor = session.open_record_cursor("test").unwrap();
        assert_eq!(cursor.set(&RecordView::new(1, b"bar")), Ok(()));
        assert_eq!(
            session.bulk_load("test", None, [Record::new(7, b"foo")].into_iter()),
            Err(Error::Posix(16)) // EBUSY
        );
    }

    #[test]
    fn bulk_load_out_of_order() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();

        assert_eq!(
            session.bulk_load(
                "test",
                None,
                [Record::new(11, b"bar"), Record::new(7, b"foo")].into_iter()
            ),
            Err(Error::Posix(22)) // EINVAL
        );
    }

    #[test]
    fn io_error() {
        let err = Error::WiredTiger(WiredTigerError::NotFound);
        assert_eq!(
            std::io::Error::from(err).to_string(),
            std::io::Error::new(ErrorKind::NotFound, err).to_string()
        );
    }
}
