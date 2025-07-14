//! Bindings for the WiredTiger C library that align with MongoDB usage.
//!
//! To use, open a `Connection` to an on-disk database. `Session`s are used to group
//! operations (including in transactions) and create `RecordCursor`s used to read
//! and write table data.
//!
//! Unlike the general-purpose WiredTiger library, this library only allows tables
//! that are keyed by `i64` with byte array payloads.
pub mod config;
mod connection;
pub mod options;
mod session;

use rustix::io::Errno;
use wt_sys::wiredtiger_strerror;

use std::ffi::CStr;
use std::io;
use std::io::ErrorKind;
use std::num::NonZero;

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
// TODO: use rustix::io::Errno instead of Posix
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Error {
    WiredTiger(WiredTigerError),
    Errno(Errno),
}

impl Error {
    fn generic_error() -> Self {
        Error::WiredTiger(WiredTigerError::Generic)
    }

    /// Return a WiredTiger `NotFound` error.
    pub fn not_found_error() -> Self {
        Error::WiredTiger(WiredTigerError::NotFound)
    }
}

impl From<NonZero<i32>> for Error {
    fn from(value: NonZero<i32>) -> Self {
        WiredTigerError::try_from_code(value.get())
            .map(Error::WiredTiger)
            .unwrap_or_else(|| Error::Errno(Errno::from_raw_os_error(value.get())))
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WiredTiger(w) => write!(f, "WT {w}"),
            Self::Errno(p) => write!(f, "errno {p}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<Error> for std::io::Error {
    fn from(value: Error) -> Self {
        match &value {
            Error::WiredTiger(wt) => io::Error::new((*wt).into(), value),
            Error::Errno(errno) => io::Error::from(*errno),
        }
    }
}

pub use connection::Connection;
pub use session::{
    FormatString, Formatted, IndexCursor, IndexCursorGuard, RecordCursor, RecordCursorGuard,
    Session, StatCursor, TypedCursor, TypedCursorGuard,
};
pub type Result<T> = std::result::Result<T, Error>;

fn make_result<T>(code: i32, value: T) -> Result<T> {
    NonZero::<i32>::new(code)
        .map(|c| Err(Error::from(c)))
        .unwrap_or(Ok(value))
}

fn map_not_found<T>(r: Result<T>) -> Option<Result<T>> {
    if r.as_ref().is_err_and(|e| *e == Error::not_found_error()) {
        None
    } else {
        Some(r)
    }
}

/// Call a `$func` on the `NonNull` WiredTiger object `$ptr` optionally with some `$args`.
/// Usually `$func` is expected to return an integer code which will be coerced into `wt_mdb::Result<()>`;
/// start the macro with `void` if `$func` returns void.
/// This may panic if any of the function pointers is `None`; this invariant is guaranteed by WT.
macro_rules! wt_call {
    ($ptr:expr, $func:ident) => {
        crate::make_result($ptr.as_ref().$func.expect("function pointer must be non-null")($ptr.as_ptr()), ())
    };
    ($ptr:expr, $func:ident, $( $args:expr ),* ) => {
        crate::make_result($ptr.as_ref().$func.expect("function pointer must be non-null")($ptr.as_ptr(), $($args), *), ())
    };
    (void $ptr:expr, $func:ident) => {
        {$ptr.as_ref().$func.expect("function pointer must be non-null")($ptr.as_ptr()); Ok(())}
    };
    (void $ptr:expr, $func:ident, $( $args:expr ),* ) => {
        {$ptr.as_ref().$func.expect("function pointer must be non-null")($ptr.as_ptr(), $($args), *); Ok::<(), crate::Error>(())}
    };
}

use wt_call;

#[cfg(test)]
mod test {
    use std::io::ErrorKind;

    use rustix::io::Errno;

    use crate::{
        connection::Connection,
        options::{
            ConnectionOptions, ConnectionOptionsBuilder, CreateOptions, CreateOptionsBuilder,
            Statistics,
        },
        Error, WiredTigerError,
    };

    fn conn_options() -> Option<ConnectionOptions> {
        Some(ConnectionOptionsBuilder::default().create().into())
    }

    fn index_table_options() -> Option<CreateOptions> {
        Some(
            CreateOptionsBuilder::default()
                .key_format::<Vec<u8>>()
                .value_format::<Vec<u8>>()
                .into(),
        )
    }

    #[test]
    fn record_insert_and_iterate() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();
        session.create_table("test", None).unwrap();
        let mut cursor = session.open_record_cursor("test").unwrap();
        assert_eq!(cursor.set(11, b"bar"), Ok(()));
        assert_eq!(cursor.set(7, b"foo"), Ok(()));
        assert_eq!(cursor.next(), Some(Ok((7, b"foo".to_vec()))));
        assert_eq!(cursor.next(), Some(Ok((11, b"bar".to_vec()))));
        assert_eq!(cursor.next(), None);
    }

    #[test]
    fn index_insert_and_iterate() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();
        session.create_table("test", index_table_options()).unwrap();
        let mut cursor = session.open_index_cursor("test").unwrap();
        assert_eq!(cursor.set(&b"b".as_slice(), &b"bar".as_slice()), Ok(()));
        assert_eq!(cursor.set(&b"a".as_slice(), &b"foo".as_slice()), Ok(()));
        assert_eq!(cursor.next(), Some(Ok((b"a".to_vec(), b"foo".to_vec()))));
        assert_eq!(cursor.next(), Some(Ok((b"b".to_vec(), b"bar".to_vec()))));
        assert_eq!(cursor.next(), None);
    }

    #[test]
    fn record_insert_and_search() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();
        session.create_table("test", None).unwrap();
        let mut cursor = session.open_record_cursor("test").unwrap();
        let value: &[u8] = b"bar";
        assert_eq!(cursor.set(7, value), Ok(()));
        assert_eq!(cursor.set(11, value), Ok(()));
        assert_eq!(cursor.seek_exact(7), Some(Ok(value.to_vec())));
        assert_eq!(cursor.seek_exact(11), Some(Ok(value.to_vec())));
    }

    #[test]
    fn index_insert_and_search() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();
        session.create_table("test", index_table_options()).unwrap();
        let mut cursor = session.open_index_cursor("test").unwrap();
        let value = b"bar".to_vec();
        assert_eq!(cursor.set(&b"a".as_slice(), &value.as_slice()), Ok(()));
        assert_eq!(cursor.set(&b"b".as_slice(), &value.as_slice()), Ok(()));
        assert_eq!(cursor.seek_exact(&b"a".as_slice()), Some(Ok(value.clone())));
        assert_eq!(cursor.seek_exact(&b"b".as_slice()), Some(Ok(value.clone())));
    }

    #[test]
    fn record_insert_and_remove() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();
        session.create_table("test", None).unwrap();
        let mut cursor = session.open_record_cursor("test").unwrap();
        assert_eq!(cursor.set(11, b"bar"), Ok(()));
        assert_eq!(cursor.set(7, b"foo"), Ok(()));
        assert_eq!(cursor.remove(7), Ok(()));
        assert_eq!(cursor.next(), Some(Ok((11, b"bar".to_vec()))));
        assert_eq!(cursor.next(), None);
        assert_eq!(
            cursor.remove(13),
            Err(Error::WiredTiger(WiredTigerError::NotFound))
        );
    }

    #[test]
    fn index_insert_and_remove() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();
        session.create_table("test", index_table_options()).unwrap();
        let mut cursor = session.open_index_cursor("test").unwrap();
        assert_eq!(cursor.set(&b"b".as_slice(), &b"bar".as_slice()), Ok(()));
        assert_eq!(cursor.set(&b"a".as_slice(), &b"foo".as_slice()), Ok(()));
        assert_eq!(cursor.remove(&b"a".as_slice()), Ok(()));
        assert_eq!(cursor.next(), Some(Ok((b"b".to_vec(), b"bar".to_vec()))));
        assert_eq!(cursor.next(), None);
        assert_eq!(
            cursor.remove(&b"c".as_slice()),
            Err(Error::WiredTiger(WiredTigerError::NotFound))
        );
    }

    #[test]
    fn record_largest_key() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();
        session.create_table("test", None).unwrap();
        let mut cursor = session.open_record_cursor("test").unwrap();
        assert_eq!(cursor.largest_key(), None);
        assert_eq!(cursor.set(-1, b"bar"), Ok(()));
        assert_eq!(cursor.largest_key(), Some(Ok(-1)));
        assert_eq!(cursor.set(7, b"foo"), Ok(()));
        assert_eq!(cursor.largest_key(), Some(Ok(7)));
    }

    #[test]
    fn index_largest_key() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();
        session.create_table("test", index_table_options()).unwrap();
        let mut cursor = session.open_index_cursor("test").unwrap();
        assert_eq!(cursor.largest_key(), None);
        assert_eq!(cursor.set(&b"a".as_slice(), &b"bar".as_slice()), Ok(()));
        assert_eq!(cursor.largest_key(), Some(Ok(b"a".to_vec())));
        assert_eq!(cursor.set(&b"b".as_slice(), &b"foo".as_slice()), Ok(()));
        assert_eq!(cursor.largest_key(), Some(Ok(b"b".to_vec())));
    }

    #[test]
    fn transaction_commit() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();
        session.create_table("test", None).unwrap();
        let read_session = conn.open_session().unwrap();
        let mut read_cursor = read_session.open_record_cursor("test").unwrap();

        let mut cursor = session.open_record_cursor("test").unwrap();
        assert_eq!(session.begin_transaction(None), Ok(()));
        assert_eq!(cursor.set(1, b"foo"), Ok(()));
        assert_eq!(cursor.set(2, b"bar"), Ok(()));
        assert_eq!(cursor.next(), Some(Ok((1, b"foo".to_vec()))));
        assert_eq!(read_cursor.next(), None);
        assert_eq!(session.commit_transaction(None), Ok(()));
        assert_eq!(read_cursor.next(), Some(Ok((1, b"foo".to_vec()))));
    }

    #[test]
    fn transaction_rollback() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();
        session.create_table("test", None).unwrap();

        let mut cursor = session.open_record_cursor("test").unwrap();
        assert_eq!(session.begin_transaction(None), Ok(()));
        assert_eq!(cursor.set(1, b"foo"), Ok(()));
        assert_eq!(cursor.next(), Some(Ok((1, b"foo".to_vec()))));
        assert_eq!(session.rollback_transaction(None), Ok(()));
        assert_eq!(cursor.next(), None);
    }

    #[test]
    fn cursor_cache() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();
        session.create_table("test", None).unwrap();

        let mut cursor = session.get_record_cursor("test").unwrap();
        assert_eq!(cursor.set(1, b"foo"), Ok(()));
        drop(cursor);

        cursor = session.get_record_cursor("test").unwrap();
        assert_eq!(cursor.next(), Some(Ok((1, b"foo".to_vec()))));
        assert_eq!(cursor.next(), None);
    }

    #[test]
    fn bulk_load() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();

        // Create Vec<Record>, bulk_load() into session, compare cursors.
        let records = vec![
            (7, b"foo".to_vec()),
            (11, b"bar".to_vec()),
            (19, b"quux".to_vec()),
        ];
        assert_eq!(
            session.bulk_load("test", None, records.clone().into_iter()),
            Ok(())
        );

        let cursor = session.open_record_cursor("test").unwrap();
        for (expected, actual) in records.iter().zip(cursor) {
            assert_eq!(Ok((expected.0, expected.1.clone())), actual);
        }
    }

    #[test]
    fn bulk_load_existing_table() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();

        // Bulk load will happily load into an empty table, so to get it to fail we insert a record.
        assert_eq!(session.create_table("test", None), Ok(()));
        let mut cursor = session.open_record_cursor("test").unwrap();
        assert_eq!(cursor.set(1, b"bar"), Ok(()));
        assert_eq!(
            session.bulk_load("test", None, [(7, b"foo".to_vec())].into_iter()),
            Err(Error::Errno(Errno::BUSY))
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
                [(11, b"bar".to_vec()), (7, b"foo".to_vec())].into_iter()
            ),
            Err(Error::Errno(Errno::INVAL))
        );
    }

    #[test]
    fn checkpoint() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();
        session.create_table("test", None).unwrap();
        session.checkpoint().unwrap();
    }

    #[test]
    fn statistics() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(
            tmpdir.path().to_str().unwrap(),
            Some(
                ConnectionOptionsBuilder::default()
                    .create()
                    .statistics(Statistics::Fast)
                    .into(),
            ),
        )
        .unwrap();
        let session = conn.open_session().unwrap();
        session.create_table("test", None).unwrap();
        let mut cursor = session.open_record_cursor("test").unwrap();
        assert_eq!(cursor.set(11, b"bar"), Ok(()));
        assert_eq!(cursor.set(7, b"foo"), Ok(()));

        for stat in session
            .new_stats_cursor(Statistics::Fast, None)
            .expect("new cursor")
        {
            let (_, stat) = stat.unwrap();
            assert!(stat.value >= 0, "{}", stat.description.to_string_lossy());
        }

        for stat in session
            .new_stats_cursor(Statistics::Fast, Some("test"))
            .unwrap()
        {
            let (_, stat) = stat.unwrap();
            assert!(stat.value >= 0, "{}", stat.description.to_string_lossy());
        }

        assert!(
            session
                .new_stats_cursor(Statistics::Fast, None)
                .unwrap()
                .seek_exact(wt_sys::WT_STAT_CONN_READ_IO as i32)
                .unwrap()
                .unwrap()
                .value
                > 0
        );
    }

    #[test]
    fn metadata_cursors() {
        let tmpdir = tempfile::tempdir().unwrap();
        let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
        let session = conn.open_session().unwrap();
        session.create_table("test", None).unwrap();
        let cursor = session.open_metadata_cursor().unwrap();
        assert_eq!(
            cursor
                .map(|e| e.map(|(k, _)| k.into_string().expect("key into string")))
                .collect::<Result<Vec<_>, crate::Error>>()
                .expect("collect keys"),
            vec![
                "metadata:",
                "colgroup:test",
                "file:WiredTigerHS.wt",
                "file:test.wt",
                "table:test"
            ],
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
