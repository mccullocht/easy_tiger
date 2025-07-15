//! WiredTiger implementations of graph access abstractions.
//!
//! None of these abstractions will begin or commit transactions, for performance reasons it
//! is recommeded that callers begin a transaction before performing their search or mutation
//! and commit or rollback the transaction when they are done.

use std::{borrow::Cow, ffi::CString, io, sync::Arc};

use rustix::io::Errno;
use wt_mdb::{
    config::{ConfigItem, ConfigParser},
    options::{CreateOptions, DropOptions},
    Connection, Error, RecordCursorGuard, Result, Session, WiredTigerError,
};

use crate::graph::{
    Graph, GraphConfig, GraphVectorIndexReader, GraphVertex, NavVectorStore, RawVectorStore,
};

/// Key in the graph table containing the entry point.
pub const ENTRY_POINT_KEY: i64 = -1;
/// Key in the graph table containing configuration.
pub const CONFIG_KEY: i64 = -2;

fn read_app_metadata_internal(session: &Session, table_name: &str) -> Result<String> {
    let mut cursor = session.open_metadata_cursor()?;
    let metadata = cursor
        .seek_exact(&CString::new([b"table:", table_name.as_bytes()].concat()).expect("no nulls"))
        .ok_or(Error::not_found_error())??;
    let mut parser = ConfigParser::new(metadata.to_bytes())?;
    if let ConfigItem::Struct(app_metadata) = parser
        .get("app_metadata")
        .ok_or(Error::not_found_error())??
    {
        Ok(app_metadata.to_owned())
    } else {
        Err(Error::Errno(Errno::INVAL))
    }
}

/// Read the `app_metadata` config field associated with the named table.
///
/// Returns `None` if table or app_metadata config field could not be found.
pub fn read_app_metadata(session: &Session, table_name: &str) -> Option<Result<String>> {
    match read_app_metadata_internal(session, table_name) {
        Ok(m) => Some(Ok(m)),
        Err(e) if e == Error::not_found_error() => None,
        Err(e) => Some(Err(e)),
    }
}

/// Implementation of GraphVertex that reads from an encoded value in a WiredTiger record table.
// TODO: perhaps instead of returning this wrapper we should just return an iterator?
pub struct CursorGraphVertex<'a>(&'a [u8]);

impl<'a> CursorGraphVertex<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self(data)
    }
}

impl GraphVertex for CursorGraphVertex<'_> {
    type EdgeIterator<'c>
        = Leb128EdgeIterator<'c>
    where
        Self: 'c;

    fn edges(&self) -> Self::EdgeIterator<'_> {
        Leb128EdgeIterator {
            data: self.0,
            prev: 0,
        }
    }
}

/// Iterator over edge data in a graph backed by WiredTiger.
/// Create by calling `CursorGraphNode.edges()`.
pub struct Leb128EdgeIterator<'a> {
    data: &'a [u8],
    prev: i64,
}

impl Iterator for Leb128EdgeIterator<'_> {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        let delta = leb128::read::signed(&mut self.data).ok()?;
        self.prev += delta;
        Some(self.prev)
    }
}

/// Implementation of `Graph` that reads from a WiredTiger `RecordCursor`.
pub struct CursorGraph<'a>(RecordCursorGuard<'a>);

impl<'a> CursorGraph<'a> {
    pub fn new(cursor: RecordCursorGuard<'a>) -> Self {
        Self(cursor)
    }

    pub(crate) fn set_entry_point(&mut self, entry_point: i64) -> Result<()> {
        self.0.set(ENTRY_POINT_KEY, &entry_point.to_le_bytes())
    }

    pub(crate) fn set(&mut self, vertex_id: i64, edges: impl Into<Vec<i64>>) -> Result<()> {
        self.0.set(vertex_id, &encode_graph_vertex(edges.into()))
    }

    pub(crate) fn remove(&mut self, vertex_id: i64) -> Result<()> {
        self.0.remove(vertex_id).or_else(|e| {
            if e == Error::not_found_error() {
                Ok(())
            } else {
                Err(e)
            }
        })
    }
}

impl Graph for CursorGraph<'_> {
    type Vertex<'c>
        = CursorGraphVertex<'c>
    where
        Self: 'c;

    fn entry_point(&mut self) -> Option<Result<i64>> {
        let result = unsafe { self.0.seek_exact_unsafe(ENTRY_POINT_KEY)? };
        Some(result.map(|r| i64::from_le_bytes(r.try_into().expect("8 bytes"))))
    }

    fn get_vertex(&mut self, vertex_id: i64) -> Option<Result<Self::Vertex<'_>>> {
        let r = unsafe { self.0.seek_exact_unsafe(vertex_id)? };
        Some(r.map(CursorGraphVertex::new))
    }
}

/// Implementation of NavVectorStore that reads from a WiredTiger `RecordCursor`.
pub struct CursorVectorStore<'a>(RecordCursorGuard<'a>);

impl<'a> CursorVectorStore<'a> {
    pub fn new(cursor: RecordCursorGuard<'a>) -> Self {
        Self(cursor)
    }

    pub(crate) fn set(&mut self, vertex_id: i64, vector: impl AsRef<[u8]>) -> Result<()> {
        self.0.set(vertex_id, vector.as_ref())
    }

    pub(crate) fn remove(&mut self, vertex_id: i64) -> Result<()> {
        self.0.remove(vertex_id).or_else(|e| {
            if e == Error::not_found_error() {
                Ok(())
            } else {
                Err(e)
            }
        })
    }

    fn get(&mut self, vertex_id: i64) -> Option<Result<Cow<'_, [u8]>>> {
        Some(unsafe { self.0.seek_exact_unsafe(vertex_id)? }.map(|v| v.into()))
    }
}

impl RawVectorStore for CursorVectorStore<'_> {
    fn get_raw_vector(&mut self, vertex_id: i64) -> Option<Result<Cow<'_, [u8]>>> {
        self.get(vertex_id)
    }
}

impl NavVectorStore for CursorVectorStore<'_> {
    fn get_nav_vector(&mut self, vertex_id: i64) -> Option<Result<Cow<'_, [u8]>>> {
        self.get(vertex_id)
    }
}

/// Immutable features of a WiredTiger graph vector index. These can be read from the db and
/// stored in a catalog for convenient access at runtime.
#[derive(Clone)]
pub struct TableGraphVectorIndex {
    graph_table_name: String,
    raw_table_name: String,
    nav_table_name: String,
    config: GraphConfig,
}

impl TableGraphVectorIndex {
    /// Create a new `TableGraphVectorIndex` from the relevant db tables, extracting
    /// immutable graph metadata that can be used across operations.
    pub fn from_db(connection: &Arc<Connection>, table_basename: &str) -> io::Result<Self> {
        let session = connection.open_session()?;
        let [graph_table_name, raw_table_name, nav_table_name] =
            Self::generate_table_names(table_basename);
        let mut cursor = session.open_record_cursor(&graph_table_name)?;
        let config_json = unsafe { cursor.seek_exact_unsafe(CONFIG_KEY) }
            .unwrap_or(Err(Error::WiredTiger(WiredTigerError::NotFound)))?;
        let config: GraphConfig = serde_json::from_slice(config_json)?;
        Ok(Self {
            graph_table_name,
            raw_table_name,
            nav_table_name,
            config,
        })
    }

    /// Create a new `TableGraphVectorIndex` for table initialization, providing
    /// graph metadata up front.
    pub fn from_init(config: GraphConfig, index_name: &str) -> io::Result<Self> {
        let [graph_table_name, raw_table_name, nav_table_name] =
            Self::generate_table_names(index_name);
        Ok(Self {
            graph_table_name,
            raw_table_name,
            nav_table_name,
            config,
        })
    }

    /// Create necessary tables for the index and write index metadata.
    pub fn init_index(
        connection: &Arc<Connection>,
        table_options: Option<CreateOptions>,
        config: GraphConfig,
        index_name: &str,
    ) -> io::Result<Self> {
        let index = Self::from_init(config, index_name)?;
        let session = connection.open_session()?;
        session.create_table(&index.graph_table_name, table_options.clone())?;
        session.create_table(&index.raw_table_name, table_options.clone())?;
        session.create_table(&index.nav_table_name, table_options)?;
        let mut cursor = session.open_record_cursor(&index.graph_table_name)?;
        cursor.set(CONFIG_KEY, &serde_json::to_vec(&index.config)?)?;
        Ok(index)
    }

    /// Drop all tables for `index_name`.
    pub fn drop_tables(
        session: &Session,
        index_name: &str,
        options: &Option<DropOptions>,
    ) -> Result<()> {
        for table_name in Self::generate_table_names(index_name) {
            session.drop_table(&table_name, options.clone())?;
        }
        Ok(())
    }

    /// Generate the names of the tables used for `index_name`.
    pub fn generate_table_names(index_name: &str) -> [String; 3] {
        [
            format!("{index_name}.graph"),
            format!("{index_name}.raw_vectors"),
            format!("{index_name}.nav_vectors"),
        ]
    }

    /// Return `GraphMetadata` for this index.
    pub fn config(&self) -> &GraphConfig {
        &self.config
    }

    /// Return the name of the table containing the graph.
    pub fn graph_table_name(&self) -> &str {
        &self.graph_table_name
    }

    /// Return the name of the table containing raw vectors.
    pub fn raw_table_name(&self) -> &str {
        &self.raw_table_name
    }

    /// Return the name of the table containing the navigational vectors.
    pub fn nav_table_name(&self) -> &str {
        &self.nav_table_name
    }
}

/// A `GraphVectorIndexReader` implementation that operates entirely on a WiredTiger graph.
pub struct SessionGraphVectorIndexReader {
    index: Arc<TableGraphVectorIndex>,
    session: Session,
}

impl SessionGraphVectorIndexReader {
    /// Create a new `TableGraphVectorIndex` given a named index and a session to access that data.
    pub fn new(index: Arc<TableGraphVectorIndex>, session: Session) -> Self {
        Self { index, session }
    }

    /// Return a reference to the underlying `Session`.
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Return a reference to the index in use.
    pub fn index(&self) -> &TableGraphVectorIndex {
        &self.index
    }

    /// Unwrap into the inner `Session`.
    pub fn into_session(self) -> Session {
        self.session
    }
}

impl GraphVectorIndexReader for SessionGraphVectorIndexReader {
    type Graph<'a>
        = CursorGraph<'a>
    where
        Self: 'a;
    type RawVectorStore<'a>
        = CursorVectorStore<'a>
    where
        Self: 'a;
    type NavVectorStore<'a>
        = CursorVectorStore<'a>
    where
        Self: 'a;

    fn config(&self) -> &GraphConfig {
        &self.index.config
    }

    fn graph(&self) -> Result<Self::Graph<'_>> {
        Ok(CursorGraph::new(
            self.session
                .get_record_cursor(self.index.graph_table_name())?,
        ))
    }

    fn raw_vectors(&self) -> Result<Self::RawVectorStore<'_>> {
        Ok(CursorVectorStore::new(
            self.session
                .get_record_cursor(self.index.raw_table_name())?,
        ))
    }

    fn nav_vectors(&self) -> Result<Self::NavVectorStore<'_>> {
        Ok(CursorVectorStore::new(
            self.session
                .get_record_cursor(self.index.nav_table_name())?,
        ))
    }
}

/// Encode the contents of a graph vertex to use as a WiredTiger table value.
///
/// Vector ought to be provided if using [crate::graph::GraphLayout::VectorInGraph],
/// in which case the vector is
pub fn encode_graph_vertex(mut edges: Vec<i64>) -> Vec<u8> {
    // A 64-bit value may occupy up to 10 bytes when leb128 encoded so reserve enough space for that.
    // There is unfortunately no constant for this in the leb128 crate.
    let mut out: Vec<u8> = Vec::with_capacity(edges.len() * 10);
    edges.sort();
    for (prev, next) in std::iter::once(&0).chain(edges.iter()).zip(edges.iter()) {
        leb128::write::signed(&mut out, *next - *prev).unwrap();
    }

    out
}

/// Encode the contents of a raw vector to use as a WiredTiger table value.
// TODO: replace this with a more comprehensive vector table offering.
pub fn encode_raw_vector(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|d| d.to_le_bytes()).collect()
}
