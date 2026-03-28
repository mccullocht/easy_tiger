use std::{ffi::CString, sync::Arc};

use thread_local::ThreadLocal;

use crate::{
    session::{
        table_uri, BeginTransactionOptions, CommitTransactionOptions, Formatted,
        MetadataCursorGuard, QueryTransactionTimestampType, RollbackTransactionOptions, Session,
        SetTransactionTimestampType, StatValue, METADATA_URI,
    },
    Connection, IndexCursorGuard, RecordCursorGuard, Result, StatCursor, Statistics,
    TypedCursorGuard,
};

/// A new Transaction over a WiredTiger database.
///
/// If this transaction is not committed or rolled back explicitly it will be rolled back when
/// it is dropped.
pub struct Transaction {
    session: Session,
    open: bool, // true if the transaction is open.
}

impl Transaction {
    /// Create a new transaction over `connection`.
    ///
    /// This automatically opens a session and begins a transaction on that session.
    pub(crate) fn new(
        connection: &Arc<Connection>,
        options: Option<BeginTransactionOptions>,
    ) -> Result<Self> {
        let session = connection.open_session()?;
        session.begin_transaction(options)?;
        Ok(Self {
            session,
            open: true,
        })
    }

    /// Return a reference to the underlying [`Connection`].
    pub fn connection(&self) -> &Arc<Connection> {
        self.session.connection()
    }

    /// Open a cursor over `table_name` with formatted keys and values.
    pub fn open_cursor<K: Formatted, V: Formatted>(
        &self,
        table_name: &str,
    ) -> Result<TypedCursorGuard<'_, K, V>> {
        let table_uri = table_uri(table_name);
        self.session.get_or_create_typed_cursor_uri(&table_uri)
    }

    /// Open a cursor over `table_name` with `i64` keys and byte-array values.
    pub fn open_record_cursor(&self, table_name: &str) -> Result<RecordCursorGuard<'_>> {
        let table_uri = table_uri(table_name);
        self.session
            .get_or_create_typed_cursor_uri::<i64, Vec<u8>>(&table_uri)
    }

    /// Open a cursor over `table_name` with byte array keys and values.
    pub fn open_index_cursor(&self, table_name: &str) -> Result<IndexCursorGuard<'_>> {
        let table_uri = table_uri(table_name);
        self.session
            .get_or_create_typed_cursor_uri::<Vec<u8>, Vec<u8>>(&table_uri)
    }

    /// Open a metadata cursor.
    pub fn open_metadata_cursor(&self) -> Result<MetadataCursorGuard<'_>> {
        self.session.get_or_create_typed_cursor_uri(METADATA_URI)
    }

    /// Return a new cursor that provides statistics keyed by WT constants.
    ///
    /// Accepts a level which may be no higher than the connection level, and optionally a table
    /// name to observe the stats of.
    pub fn open_stats_cursor(
        &self,
        level: Statistics,
        table: Option<&str>,
    ) -> Result<StatCursor<'_>> {
        let table_stats_uri = table.map(|t| {
            CString::new(format!("statistics:table:{t}")).expect("no nulls in table name")
        });
        let uri = table_stats_uri.as_deref().unwrap_or(c"statistics:");
        let options = level
            .to_config_string_clause()
            .map(|s| CString::new(s.into_bytes()).expect("no nulls"));
        self.session
            .new_typed_cursor_uri::<i32, StatValue>(uri, options.as_deref())
    }

    /// Set the transaction timestamp for the current transaction.
    ///
    /// This should be called on a session with an active transaction. This may be called in the
    /// during a transaction to use a different commit timestamp for different parts of the
    /// transaction. For read timestamps this may only be called once per transaction, including
    /// any read timestamp set when the transaction begins.
    ///
    /// Timestamps are bound in part by global state; consult WiredTiger documentation for more
    /// information.
    pub fn set_transaction_timestamp(
        &self,
        txn_type: SetTransactionTimestampType,
        timestamp: u64,
    ) -> Result<()> {
        self.session.set_transaction_timestamp(txn_type, timestamp)
    }

    /// Query the session for transaction timestamp state. Callers may select one of the types of
    /// timestamps to query. Returns a 64-bit unsigned timestamp.
    pub fn query_transaction_timestamp(
        &self,
        timestamp: QueryTransactionTimestampType,
    ) -> Result<u64> {
        self.session.query_transaction_timestamp(timestamp)
    }

    /// Commit this transaction with any specified options.
    ///
    /// This may result in a WT_ROLLBACK error if there was a conflict with another transaction, in
    /// which case the caller should retry.
    pub fn commit(mut self, options: Option<CommitTransactionOptions>) -> Result<()> {
        self.open = false;
        self.session.commit_transaction(options)
    }

    /// Rollback this transaction with any specified options.
    pub fn rollback(mut self, options: Option<RollbackTransactionOptions>) -> Result<()> {
        self.open = false;
        self.session.rollback_transaction(options)
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        if self.open {
            // We ignore the result of rollback here, since we can't do anything about it.
            let _ = self.session.rollback_transaction(None);
        }

        // TODO: pool sessions in the connection.
    }
}

/// A transaction that can be shared across threads for concurrent reads.
///
/// WiredTiger's underlying session and cursor infrastructure is not thread safe so it can be
/// challenging to perform concurrent reads. This struct provides a wrapper that provides consistent
/// reads across multiple threads by propagating the read timestamp from your write transaction to
/// thread-local read transactions.
///
/// These transactions are functionally read-only as they cannot be committed and will be rolled
/// back on drop, but attempts to write may succed and will be visible in the transaction itself.
///
/// *SAFETY*: this struct may defeat WiredTiger's conflict detection for write transactions. Reads
/// on values that are modified in another transaction that commits first may succeed even though
/// in a single-threaded context they would have been rolled back.
// TODO: utilize read-only cursor support and only provide read-only transactions from this struct.
pub struct ConcurrentReadTransaction {
    conn: Arc<Connection>,
    ts: u64,
    txns: ThreadLocal<Transaction>,
}

impl ConcurrentReadTransaction {
    /// Create a new concurrent read transaction with the provided timestamp.
    pub fn new(conn: &Arc<Connection>, ts: u64) -> Self {
        Self {
            conn: Arc::clone(conn),
            ts,
            txns: ThreadLocal::new(),
        }
    }

    pub fn from_write_transaction(txn: &Transaction) -> Result<Self> {
        Ok(Self::new(
            txn.connection(),
            txn.query_transaction_timestamp(QueryTransactionTimestampType::Read)?,
        ))
    }

    /// Get a transaction for the current thread.
    pub fn get(&self) -> Result<&Transaction> {
        self.txns.get_or_try(|| {
            Transaction::new(
                &self.conn,
                Some(BeginTransactionOptions::with_read_timestamp(self.ts)),
            )
        })
    }
}
