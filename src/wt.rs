//! WiredTiger implementations of graph access abstractions.
//!
//! None of these abstractions will begin or commit transactions, for performance reasons it
//! is recommeded that callers begin a transaction before performing their search or mutation
//! and commit or rollback the transaction when they are done.

use std::{borrow::Cow, io, sync::Arc};

use wt_mdb::{
    options::CreateOptions, Connection, Error, Record, RecordCursorGuard, RecordView, Result,
    Session, WiredTigerError,
};

use crate::{
    graph::{Graph, GraphMetadata, GraphVectorIndexReader, GraphVertex, NavVectorStore},
    worker_pool::WorkerPool,
};

// TODO: drop WiredTiger from most of these names, it feels redundant and verbose.

/// Key in the graph table containing the entry point.
pub const ENTRY_POINT_KEY: i64 = -1;
/// Key in the graph table containing metadata.
pub const METADATA_KEY: i64 = -2;

/// Implementation of NavVectorStore that reads from a WiredTiger `RecordCursor`.
pub struct WiredTigerNavVectorStore<'a> {
    cursor: RecordCursorGuard<'a>,
}

impl<'a> WiredTigerNavVectorStore<'a> {
    pub fn new(cursor: RecordCursorGuard<'a>) -> Self {
        Self { cursor }
    }

    pub(crate) fn set(&mut self, vertex_id: i64, vector: Cow<'_, [u8]>) -> Result<()> {
        self.cursor.set(&RecordView::new(vertex_id, vector))
    }

    pub(crate) fn remove(&mut self, vertex_id: i64) -> Result<()> {
        self.cursor.remove(vertex_id)
    }
}

impl NavVectorStore for WiredTigerNavVectorStore<'_> {
    fn get(&mut self, vertex_id: i64) -> Option<Result<Cow<'_, [u8]>>> {
        Some(unsafe { self.cursor.seek_exact_unsafe(vertex_id)? }.map(RecordView::into_inner_value))
    }
}

/// Implementation of GraphVertex that reads from an encoded value in a WiredTiger record table.
pub struct WiredTigerGraphVertex<'a> {
    // split point between vector data and edge data.
    split: usize,
    data: Cow<'a, [u8]>,
}

impl<'a> WiredTigerGraphVertex<'a> {
    fn new(metadata: &GraphMetadata, data: Cow<'a, [u8]>) -> Self {
        Self {
            split: metadata.dimensions.get() * std::mem::size_of::<f32>(),
            data,
        }
    }

    pub(crate) fn vector_bytes(&self) -> &[u8] {
        &self.data[..self.split]
    }

    // Vector f32 data is stored little endian so we can get away with aliasing. Slice requires
    // that the pointer be properly aligned, which WiredTiger does not guarantee.
    fn maybe_alias_vector_data(&self) -> Option<&[f32]> {
        #[cfg(target_endian = "little")]
        {
            // WiredTiger does not guarantee that the returned memory will be aligned, a
            // Try to align it and if that fails, copy the data.
            let (prefix, vector, _) =
                unsafe { self.data.as_ref()[0..self.split].align_to::<f32>() };
            if prefix.is_empty() {
                return Some(vector);
            }
        }
        None
    }
}

impl GraphVertex for WiredTigerGraphVertex<'_> {
    type EdgeIterator<'c>
        = WiredTigerEdgeIterator<'c>
    where
        Self: 'c;

    fn vector(&self) -> Cow<'_, [f32]> {
        self.maybe_alias_vector_data()
            .map(|v| v.into())
            .unwrap_or_else(|| {
                self.data.as_ref()[0..self.split]
                    .chunks(std::mem::size_of::<f32>())
                    .map(|b| f32::from_le_bytes(b.try_into().expect("array of 4 conversion")))
                    .collect::<Vec<_>>()
                    .into()
            })
    }

    fn edges(&self) -> Self::EdgeIterator<'_> {
        WiredTigerEdgeIterator {
            data: &self.data.as_ref()[self.split..],
            prev: 0,
        }
    }
}

/// Iterator over edge data in a graph backed by WiredTiger.
/// Create by calling `WiredTigerGraphNode.edges()`.
pub struct WiredTigerEdgeIterator<'a> {
    data: &'a [u8],
    prev: i64,
}

impl Iterator for WiredTigerEdgeIterator<'_> {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        let delta = leb128::read::signed(&mut self.data).ok()?;
        self.prev += delta;
        Some(self.prev)
    }
}

/// Implementation of `Graph` that reads from a WiredTiger `RecordCursor`.
pub struct WiredTigerGraph<'a> {
    metadata: GraphMetadata,
    cursor: RecordCursorGuard<'a>,
}

impl<'a> WiredTigerGraph<'a> {
    pub fn new(metadata: GraphMetadata, cursor: RecordCursorGuard<'a>) -> Self {
        Self { metadata, cursor }
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
        self.cursor.remove(vertex_id)
    }
}

impl Graph for WiredTigerGraph<'_> {
    type Vertex<'c>
        = WiredTigerGraphVertex<'c>
    where
        Self: 'c;

    fn entry_point(&mut self) -> Option<Result<i64>> {
        let result = unsafe { self.cursor.seek_exact_unsafe(ENTRY_POINT_KEY)? };
        Some(result.map(|r| i64::from_le_bytes(r.value().try_into().unwrap())))
    }

    fn get(&mut self, vertex_id: i64) -> Option<Result<Self::Vertex<'_>>> {
        let r =
            unsafe { self.cursor.seek_exact_unsafe(vertex_id)? }.map(RecordView::into_inner_value);
        Some(r.map(|r| WiredTigerGraphVertex::new(&self.metadata, r)))
    }
}

/// Immutable features of a WiredTiger graph vector index. These can be read from the db and
/// stored in a catalog for convenient access at runtime.
pub struct WiredTigerGraphVectorIndex {
    graph_table_name: String,
    nav_table_name: String,
    metadata: GraphMetadata,
}

impl WiredTigerGraphVectorIndex {
    /// Create a new `WiredTigerGraphVectorIndex` from the relevant db tables, extracting
    /// immutable graph metadata that can be used across operations.
    pub fn from_db(connection: &Arc<Connection>, table_basename: &str) -> io::Result<Self> {
        let session = connection.open_session()?;
        let (graph_table_name, nav_table_name) = Self::generate_table_names(table_basename);
        let mut cursor = session.open_record_cursor(&graph_table_name)?;
        let metadata_json = unsafe { cursor.seek_exact_unsafe(METADATA_KEY) }
            .unwrap_or(Err(Error::WiredTiger(WiredTigerError::NotFound)))?;
        let metadata = serde_json::from_slice(metadata_json.value())?;
        Ok(Self {
            graph_table_name,
            nav_table_name,
            metadata,
        })
    }

    /// Create a new `WiredTigerGraphVectorIndex` for table initialization, providing
    /// graph metadata up front.
    pub fn from_init(metadata: GraphMetadata, table_basename: &str) -> io::Result<Self> {
        let (graph_table_name, nav_table_name) = Self::generate_table_names(table_basename);
        Ok(Self {
            graph_table_name,
            nav_table_name,
            metadata,
        })
    }

    /// Create necessary tables for the index and write index metadata.
    pub fn init_index(
        &self,
        connection: &Arc<Connection>,
        table_options: Option<CreateOptions>,
    ) -> io::Result<()> {
        let session = connection.open_session()?;
        session.create_record_table(&self.graph_table_name, table_options.clone())?;
        session.create_record_table(&self.nav_table_name, table_options)?;
        let mut cursor = session.open_record_cursor(&self.graph_table_name)?;
        Ok(cursor.set(&Record::new(
            METADATA_KEY,
            serde_json::to_vec(&self.metadata)?,
        ))?)
    }

    /// Return `GraphMetadata` for this index.
    pub fn metadata(&self) -> &GraphMetadata {
        &self.metadata
    }

    /// Return the name of the table containing the graph.
    pub fn graph_table_name(&self) -> &str {
        &self.graph_table_name
    }

    /// Return the name of the table containing the navigational vectors.
    pub fn nav_table_name(&self) -> &str {
        &self.nav_table_name
    }

    fn generate_table_names(table_basename: &str) -> (String, String) {
        (
            format!("{}.graph", table_basename),
            format!("{}.nav_vectors", table_basename),
        )
    }
}

/// A `GraphVectorIndexReader` implementation that operates entirely on a WiredTiger graph.
pub struct WiredTigerGraphVectorIndexReader {
    index: Arc<WiredTigerGraphVectorIndex>,
    session: Session,
    worker_pool: Option<WorkerPool>,
}

impl WiredTigerGraphVectorIndexReader {
    /// Create a new `WiredTigerGraphVectorIndex` given a named index and a session to access that data.
    /// Optionally provide a `WorkerPool` for use with parallel `lookup()`.
    pub fn new(
        index: Arc<WiredTigerGraphVectorIndex>,
        session: Session,
        worker_pool: Option<WorkerPool>,
    ) -> Self {
        Self {
            index,
            session,
            worker_pool,
        }
    }

    /// Return a reference to the underlying `Session`.
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Return a reference to the index in use.
    pub fn index(&self) -> &WiredTigerGraphVectorIndex {
        &self.index
    }

    /// Unwrap into the inner `Session`.
    pub fn into_session(self) -> Session {
        self.session
    }
}

impl GraphVectorIndexReader for WiredTigerGraphVectorIndexReader {
    type Graph<'a>
        = WiredTigerGraph<'a>
    where
        Self: 'a;
    type NavVectorStore<'a>
        = WiredTigerNavVectorStore<'a>
    where
        Self: 'a;

    fn metadata(&self) -> &GraphMetadata {
        &self.index.metadata
    }

    fn graph(&self) -> Result<Self::Graph<'_>> {
        Ok(WiredTigerGraph::new(
            self.index.metadata,
            self.session
                .get_record_cursor(&self.index.graph_table_name)?,
        ))
    }

    fn nav_vectors(&self) -> Result<Self::NavVectorStore<'_>> {
        Ok(WiredTigerNavVectorStore::new(
            self.session.get_record_cursor(&self.index.nav_table_name)?,
        ))
    }

    fn parallel_lookup(&self) -> bool {
        self.worker_pool.is_some()
    }

    fn lookup<D>(&self, vertex_id: i64, done: D)
    where
        D: FnOnce(Option<Result<WiredTigerGraphVertex<'_>>>) + Send + Sync + 'static,
    {
        if let Some(workers) = self.worker_pool.as_ref() {
            let index = self.index.clone();
            workers.execute(move |session| {
                let mut cursor = match session.get_record_cursor(index.graph_table_name()) {
                    Ok(cursor) => cursor,
                    Err(e) => {
                        done(Some(Err(e)));
                        return;
                    }
                };
                done(
                    unsafe { cursor.seek_exact_unsafe(vertex_id) }.map(|result| {
                        result.map(|r| {
                            WiredTigerGraphVertex::new(index.metadata(), r.into_inner_value())
                        })
                    }),
                );
            });
        } else {
            match self.graph() {
                Ok(mut graph) => done(graph.get(vertex_id)),
                Err(e) => done(Some(Err(e))),
            }
        }
    }
}

/// Encode the contents of a graph node as a value that can be set in the WiredTiger table.
pub fn encode_graph_node(vector: &[f32], edges: Vec<i64>) -> Vec<u8> {
    encode_graph_node_internal(vector, edges)
}

pub(crate) fn encode_graph_node_internal<'a>(
    into_vector: impl Into<VectorRep<'a>>,
    mut edges: Vec<i64>,
) -> Vec<u8> {
    let vector = into_vector.into();
    // A 64-bit value may occupy up to 10 bytes when leb128 encoded so reserve enough space for that.
    // There is unfortunately no constant for this in the leb128 crate.
    let mut out: Vec<u8> = Vec::with_capacity(vector.byte_len() + edges.len() * 10);
    vector.append_bytes(&mut out);

    edges.sort();
    for (prev, next) in std::iter::once(&0).chain(edges.iter()).zip(edges.iter()) {
        leb128::write::signed(&mut out, *next - *prev).unwrap();
    }

    out
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
