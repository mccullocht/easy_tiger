use std::{
    ffi::CString,
    ptr::{self, NonNull},
};

use wt_sys::{WT_CURSOR, WT_SESSION};

use crate::{
    connection::Connection, make_result, record_cursor::RecordCursor, wrap_ptr_create, Result,
};

/// An options builder for creating a table, column group, index, or file in WiredTiger.
#[derive(Default)]
pub struct CreateOptionsBuilder;

/// Options when creating a table, column group, index, or file in WiredTiger.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct CreateOptions {
    rep: CString,
}

impl Default for CreateOptions {
    fn default() -> Self {
        CreateOptions {
            rep: CString::from(c"key_format=q,value_format=u"),
        }
    }
}

impl From<CreateOptionsBuilder> for CreateOptions {
    fn from(_value: CreateOptionsBuilder) -> Self {
        Self::default()
    }
}

/// Options builder when beginning a WiredTiger transaction.
#[derive(Default)]
pub struct BeginTransactionOptionsBuilder {
    name: Option<String>,
}

impl BeginTransactionOptionsBuilder {
    /// Set the name of the transaction, for debugging purposes.
    pub fn set_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_owned());
        self
    }
}

/// Options when beginning a new transaction.
#[derive(Debug, Hash, Eq, PartialEq, Clone, Default)]
pub struct BeginTransactionOptions {
    rep: CString,
}

impl From<BeginTransactionOptionsBuilder> for BeginTransactionOptions {
    fn from(value: BeginTransactionOptionsBuilder) -> Self {
        let rep = if let Some(name) = value.name {
            CString::new(format!("name={}", name)).expect("no nulls")
        } else {
            CString::default()
        };
        Self { rep }
    }
}

/// Options build for commit_transaction() operations.
#[derive(Default)]
pub struct CommitTransactionOptionsBuilder {
    operation_timeout_ms: Option<u32>,
}

impl CommitTransactionOptionsBuilder {
    /// When set to a non-zero value acts a requested time limit for the operations in ms.
    /// This is not a guarantee -- the operation may still take longer than the timeout.
    /// If the limit is reached the operation may be rolled back.
    pub fn operation_timeout_ms(mut self, timeout: Option<u32>) -> Self {
        self.operation_timeout_ms = timeout;
        self
    }
}

/// Options for commit_transaction() operations.
#[derive(Debug, Hash, Eq, PartialEq, Clone, Default)]
pub struct CommitTransactionOptions {
    rep: CString,
}

impl From<CommitTransactionOptionsBuilder> for CommitTransactionOptions {
    fn from(value: CommitTransactionOptionsBuilder) -> Self {
        let rep = if let Some(timeout) = value.operation_timeout_ms {
            CString::new(format!("operation_timeout_ms={}", timeout)).expect("no nulls")
        } else {
            CString::default()
        };
        Self { rep }
    }
}

/// Options build for rollback_transaction() operations.
#[derive(Default)]
pub struct RollbackTransactionOptionsBuilder {
    operation_timeout_ms: Option<u32>,
}

impl RollbackTransactionOptionsBuilder {
    /// When set to a non-zero value acts a requested time limit for the operations in ms.
    /// This is not a guarantee -- the operation may still take longer than the timeout.
    /// If the limit is reached the operation may be rolled back.
    pub fn operation_timeout_ms(mut self, timeout: Option<u32>) -> Self {
        self.operation_timeout_ms = timeout;
        self
    }
}

/// Options for rollback_transaction() operations.
#[derive(Debug, Hash, Eq, PartialEq, Clone, Default)]
pub struct RollbackTransactionOptions {
    rep: CString,
}

impl From<RollbackTransactionOptionsBuilder> for RollbackTransactionOptions {
    fn from(value: RollbackTransactionOptionsBuilder) -> Self {
        let rep = if let Some(timeout) = value.operation_timeout_ms {
            CString::new(format!("operation_timeout_ms={}", timeout)).expect("no nulls")
        } else {
            CString::default()
        };
        Self { rep }
    }
}

/// A WiredTiger session.
///
/// Sessions are used to create cursors to view and mutate data and manage
/// transaction state.
// TODO: not sure if the model of this is quite right w.r.t. thread safety.
pub struct Session<'a> {
    session: NonNull<WT_SESSION>,
    connection: &'a Connection,
}

impl<'a> Session<'a> {
    pub(crate) fn new(session: NonNull<WT_SESSION>, connection: &'a Connection) -> Self {
        Self {
            session,
            connection,
        }
    }

    pub fn connection(&self) -> &Connection {
        self.connection
    }

    /// Create a new table, column group, index, or file.
    // TODO: restrict api to table creation only.
    pub fn create_record_table(
        &self,
        table_name: &str,
        config: Option<CreateOptions>,
    ) -> Result<()> {
        let name = CString::new(format!("table:{}", table_name)).expect("no nulls");
        unsafe {
            make_result(
                (self.session.as_ref().create.unwrap())(
                    self.session.as_ptr(),
                    name.as_ptr(),
                    config.unwrap_or_default().rep.as_ptr(),
                ),
                (),
            )
        }
    }

    // TODO: restrict api to table access only.
    pub fn open_record_cursor(&self, table_name: &str) -> Result<RecordCursor> {
        let name = CString::new(format!("table:{}", table_name)).expect("no nulls");
        let mut cursorp: *mut WT_CURSOR = ptr::null_mut();
        let result: i32;
        unsafe {
            result = (self.session.as_ref().open_cursor.unwrap())(
                self.session.as_ptr(),
                name.as_ptr(),
                ptr::null_mut(),
                ptr::null(),
                &mut cursorp,
            );
        }
        wrap_ptr_create(result, cursorp).map(|cursor| RecordCursor::new(cursor, self))
    }

    /// Starts a transaction in this session.
    ///
    /// The transaction remains active until `commit_transaction` or `rollback_transaction` are called.
    /// Operations performed on cursors already opened or opened before the transaction ends happen
    /// within the context of this transaction.
    ///
    /// This may not be called on a session with an active transaction or an error will be returned (EINVAL)
    /// but otherwise behavior is unspecified.
    pub fn begin_transaction(&self, options: Option<&BeginTransactionOptions>) -> Result<()> {
        let optionsp = options.map(|o| o.rep.as_ptr()).unwrap_or(ptr::null());
        unsafe {
            make_result(
                self.session.as_ref().begin_transaction.unwrap()(self.session.as_ptr(), optionsp),
                (),
            )
        }
    }

    /// Commit the current transaction.
    ///
    /// If this method returns an error, the transaction was rolled back and all cursors associated with the session
    /// are reset.
    ///
    /// A transaction must be in progress when this method is called or an error will be returned (EINVAL) but behavior
    /// is otherwise unspecified.
    pub fn commit_transaction(&self, options: Option<&CommitTransactionOptions>) -> Result<()> {
        let optionsp = options.map(|o| o.rep.as_ptr()).unwrap_or(ptr::null());
        unsafe {
            make_result(
                self.session.as_ref().commit_transaction.unwrap()(self.session.as_ptr(), optionsp),
                (),
            )
        }
    }

    /// Rollback the current transaction.
    ///
    /// All cursors associated with the session are reset.
    ///
    /// A transaction must be in progress when this method is called or an error will be returned (EINVAL) but behavior
    /// is otherwise unspecified.
    pub fn rollback_transaction(&self, options: Option<&RollbackTransactionOptions>) -> Result<()> {
        let optionsp = options.map(|o| o.rep.as_ptr()).unwrap_or(ptr::null());
        unsafe {
            make_result(
                self.session.as_ref().rollback_transaction.unwrap()(
                    self.session.as_ptr(),
                    optionsp,
                ),
                (),
            )
        }
    }

    /// Close this session.
    pub fn close(mut self) -> Result<()> {
        self.close_internal()
    }

    fn close_internal(&mut self) -> Result<()> {
        make_result(
            unsafe {
                self.session.as_ref().close.unwrap()(self.session.as_ptr(), std::ptr::null())
            },
            (),
        )
    }
}

impl<'a> Drop for Session<'a> {
    fn drop(&mut self) {
        let _ = self.close_internal();
    }
}
