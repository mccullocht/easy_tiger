//! WiredTiger implementations of graph access abstractions.
//!
//! None of these abstractions will begin or commit transactions, for performance reasons it
//! is recommeded that callers begin a transaction before performing their search or mutation
//! and commit or rollback the transaction when they are done.

use std::{borrow::Cow, io, ops::Deref, sync::Arc};

use wt_mdb::{
    options::CreateOptions, Connection, Error, Record, RecordCursorGuard, RecordView, Result,
    Session, WiredTigerError,
};

use crate::graph::{
    Graph, GraphConfig, GraphLayout, GraphVectorIndexReader, GraphVertex, NavVectorStore,
    RawVector, RawVectorStore,
};

/// Key in the graph table containing the entry point.
pub const ENTRY_POINT_KEY: i64 = -1;
/// Key in the graph table containing configuration.
pub const CONFIG_KEY: i64 = -2;

/// Implementation of GraphVertex that reads from an encoded value in a WiredTiger record table.
pub struct CursorGraphVertex<'a> {
    data: Cow<'a, [u8]>,
    raw_vector: Option<RawVector<'a>>,
    edges_start: usize,
}

impl<'a> CursorGraphVertex<'a> {
    fn new(config: &GraphConfig, data: Cow<'a, [u8]>) -> Self {
        let raw_vector = if config.layout == GraphLayout::RawVectorInGraph {
            Some(RawVector::from_cow_partial(
                data.clone(),
                config.dimensions.get(),
            ))
        } else {
            None
        };
        let edges_start = raw_vector.as_ref().map(|v| v.bytes_len()).unwrap_or(0);
        Self {
            data,
            raw_vector,
            edges_start,
        }
    }
}

impl GraphVertex for CursorGraphVertex<'_> {
    type EdgeIterator<'c>
        = Leb128EdgeIterator<'c>
    where
        Self: 'c;

    fn vector(&self) -> Option<Cow<'_, [f32]>> {
        self.raw_vector.as_ref().map(|v| Cow::from(v.deref()))
    }

    fn edges(&self) -> Self::EdgeIterator<'_> {
        Leb128EdgeIterator {
            data: &self.data.as_ref()[self.edges_start..],
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
pub struct CursorGraph<'a> {
    config: GraphConfig,
    cursor: RecordCursorGuard<'a>,
}

impl<'a> CursorGraph<'a> {
    pub fn new(config: GraphConfig, cursor: RecordCursorGuard<'a>) -> Self {
        Self { config, cursor }
    }

    pub(crate) fn set_entry_point(&mut self, entry_point: i64) -> Result<()> {
        self.cursor.set(&RecordView::new(
            ENTRY_POINT_KEY,
            &entry_point.to_le_bytes(),
        ))
    }

    pub(crate) fn set(&mut self, vertex_id: i64, encoded_graph_node: &[u8]) -> Result<()> {
        self.cursor
            .set(&RecordView::new(vertex_id, encoded_graph_node))
    }

    pub(crate) fn remove(&mut self, vertex_id: i64) -> Result<()> {
        self.cursor.remove(vertex_id).or_else(|e| {
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
        let result = unsafe { self.cursor.seek_exact_unsafe(ENTRY_POINT_KEY)? };
        Some(result.map(|r| i64::from_le_bytes(r.value().try_into().unwrap())))
    }

    fn get_vertex(&mut self, vertex_id: i64) -> Option<Result<Self::Vertex<'_>>> {
        let r =
            unsafe { self.cursor.seek_exact_unsafe(vertex_id)? }.map(RecordView::into_inner_value);
        Some(r.map(|r| CursorGraphVertex::new(&self.config, r)))
    }
}

impl RawVectorStore for CursorGraph<'_> {
    fn get_raw_vector(&mut self, vertex_id: i64) -> Option<Result<RawVector<'_>>> {
        let r =
            unsafe { self.cursor.seek_exact_unsafe(vertex_id)? }.map(RecordView::into_inner_value);
        Some(r.map(|r| RawVector::from_cow_partial(r, self.config.dimensions.get())))
    }
}

/// Implementation of NavVectorStore that reads from a WiredTiger `RecordCursor`.
pub struct CursorNavVectorStore<'a> {
    cursor: RecordCursorGuard<'a>,
}

impl<'a> CursorNavVectorStore<'a> {
    pub fn new(cursor: RecordCursorGuard<'a>) -> Self {
        Self { cursor }
    }

    pub(crate) fn set(&mut self, vertex_id: i64, vector: Cow<'_, [u8]>) -> Result<()> {
        self.cursor.set(&RecordView::new(vertex_id, vector))
    }

    pub(crate) fn remove(&mut self, vertex_id: i64) -> Result<()> {
        self.cursor.remove(vertex_id).or_else(|e| {
            if e == Error::not_found_error() {
                Ok(())
            } else {
                Err(e)
            }
        })
    }
}

impl NavVectorStore for CursorNavVectorStore<'_> {
    fn get_nav_vector(&mut self, vertex_id: i64) -> Option<Result<Cow<'_, [u8]>>> {
        Some(unsafe { self.cursor.seek_exact_unsafe(vertex_id)? }.map(RecordView::into_inner_value))
    }
}

/// Immutable features of a WiredTiger graph vector index. These can be read from the db and
/// stored in a catalog for convenient access at runtime.
pub struct TableGraphVectorIndex {
    graph_table_name: String,
    raw_table_name: Option<String>,
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
        let config: GraphConfig = serde_json::from_slice(config_json.value())?;
        let raw_table_name = Self::compute_raw_table_name(config.layout, raw_table_name);
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
        let raw_table_name = Self::compute_raw_table_name(config.layout, raw_table_name);
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
        if let Some(raw_table_name) = index.raw_table_name.as_ref() {
            session.create_table(raw_table_name, table_options.clone())?;
        }
        session.create_table(&index.nav_table_name, table_options)?;
        let mut cursor = session.open_record_cursor(&index.graph_table_name)?;
        cursor.set(&Record::new(CONFIG_KEY, serde_json::to_vec(&index.config)?))?;
        Ok(index)
    }

    /// Generate the names of the tables used for `index_name`.
    pub fn generate_table_names(index_name: &str) -> [String; 3] {
        [
            format!("{}.graph", index_name),
            format!("{}.raw_vectors", index_name),
            format!("{}.nav_vectors", index_name),
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
        self.raw_table_name
            .as_ref()
            .unwrap_or(&self.graph_table_name)
    }

    /// Return the name of the table containing the navigational vectors.
    pub fn nav_table_name(&self) -> &str {
        &self.nav_table_name
    }

    fn compute_raw_table_name(layout: GraphLayout, raw_table_name: String) -> Option<String> {
        match layout {
            GraphLayout::RawVectorInGraph => None,
            GraphLayout::Split => Some(raw_table_name),
        }
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
        = CursorGraph<'a>
    where
        Self: 'a;
    type NavVectorStore<'a>
        = CursorNavVectorStore<'a>
    where
        Self: 'a;

    fn config(&self) -> &GraphConfig {
        &self.index.config
    }

    fn graph(&self) -> Result<Self::Graph<'_>> {
        Ok(CursorGraph::new(
            self.index.config,
            self.session
                .get_record_cursor(self.index.graph_table_name())?,
        ))
    }

    fn raw_vectors(&self) -> Result<Self::RawVectorStore<'_>> {
        Ok(CursorGraph::new(
            self.index.config,
            self.session
                .get_record_cursor(self.index.raw_table_name())?,
        ))
    }

    fn nav_vectors(&self) -> Result<Self::NavVectorStore<'_>> {
        Ok(CursorNavVectorStore::new(
            self.session
                .get_record_cursor(self.index.nav_table_name())?,
        ))
    }
}

/// Encode the contents of a graph vertex to use as a WiredTiger table value.
///
/// Vector ought to be provided if using [crate::graph::GraphLayout::VectorInGraph],
/// in which case the vector is
pub fn encode_graph_vertex(edges: Vec<i64>, vector: Option<&[f32]>) -> Vec<u8> {
    encode_graph_vertex_internal(edges, vector)
}

pub(crate) fn encode_graph_vertex_internal<'a>(
    mut edges: Vec<i64>,
    into_vector: Option<impl Into<VectorRep<'a>>>,
) -> Vec<u8> {
    let vector = into_vector.map(|v| v.into());
    // A 64-bit value may occupy up to 10 bytes when leb128 encoded so reserve enough space for that.
    // There is unfortunately no constant for this in the leb128 crate.
    let mut out: Vec<u8> =
        Vec::with_capacity(vector.as_ref().map(|v| v.byte_len()).unwrap_or(0) + edges.len() * 10);
    if let Some(v) = vector {
        v.append_bytes(&mut out)
    }

    edges.sort();
    for (prev, next) in std::iter::once(&0).chain(edges.iter()).zip(edges.iter()) {
        leb128::write::signed(&mut out, *next - *prev).unwrap();
    }

    out
}

/// Encode the contents of a raw vector to use as a WiredTiger table value.
pub fn encode_raw_vector(vector: &[f32]) -> Vec<u8> {
    VectorRep::from(vector).to_byte_vec()
}

pub(crate) enum VectorRep<'a> {
    Float(&'a [f32]),
    Bytes(&'a [u8]),
}

impl VectorRep<'_> {
    fn byte_len(&self) -> usize {
        match *self {
            Self::Float(f) => std::mem::size_of_val(f),
            Self::Bytes(b) => b.len(),
        }
    }

    fn append_bytes(&self, vec: &mut Vec<u8>) {
        match *self {
            Self::Float(f) => {
                for d in f {
                    vec.extend_from_slice(&d.to_le_bytes());
                }
            }
            Self::Bytes(b) => vec.extend_from_slice(b),
        }
    }

    fn to_byte_vec(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.byte_len());
        self.append_bytes(&mut bytes);
        bytes
    }
}

impl<'a> From<&'a [f32]> for VectorRep<'a> {
    fn from(value: &'a [f32]) -> Self {
        VectorRep::Float(value)
    }
}

impl<'a> From<&'a [u8]> for VectorRep<'a> {
    fn from(value: &'a [u8]) -> Self {
        VectorRep::Bytes(value)
    }
}
