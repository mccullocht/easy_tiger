use std::{borrow::Cow, io, num::NonZero, sync::Arc};

use wt_mdb::{Connection, Error, RecordCursor, RecordView, Result, Session, WiredTigerError};

use crate::{
    graph::{Graph, GraphMetadata, GraphNode, GraphVectorIndexReader, NavVectorStore},
    scoring::{DotProductScorer, F32VectorScorer, HammingScorer},
};

// TODO: drop WiredTiger from most of these names, it feels redundant and verbose.

/// Key in the graph table containing the entry point.
pub const ENTRY_POINT_KEY: i64 = -1;
/// Key in the graph table containing metadata.
pub const METADATA_KEY: i64 = -2;

/// Parameters to to open and access a WiredTiger graph index.
// XXX not sure if I really need this anymore.
#[derive(Clone)]
pub struct WiredTigerIndexParams {
    /// Connection to WiredTiger database.
    pub connection: Arc<Connection>,
    /// Name of the table containing raw vectors and graph.
    pub graph_table_name: String,
    /// Name of the table containing navigational quantized vectors.
    pub nav_table_name: String,
}

impl WiredTigerIndexParams {
    pub fn new(connection: Arc<Connection>, table_basename: &str) -> Self {
        Self {
            connection,
            graph_table_name: format!("{}.graph", table_basename),
            nav_table_name: format!("{}.nav_vectors", table_basename),
        }
    }
}

/// Implementation of NavVectorStore that reads from a WiredTiger `RecordCursor`.
pub struct WiredTigerNavVectorStore {
    cursor: RecordCursor,
}

impl WiredTigerNavVectorStore {
    pub fn new(cursor: RecordCursor) -> Self {
        Self { cursor }
    }
}

impl NavVectorStore for WiredTigerNavVectorStore {
    fn get(&mut self, node: i64) -> Option<Result<Cow<'_, [u8]>>> {
        Some(unsafe { self.cursor.seek_exact_unsafe(node)? }.map(RecordView::into_inner_value))
    }
}

/// Implementation of GraphNode that reads from an encoded value in a WiredTiger record table.
pub struct WiredTigerGraphNode<'a> {
    dimensions: NonZero<usize>,
    data: Cow<'a, [u8]>,
}

impl<'a> WiredTigerGraphNode<'a> {
    pub fn new(metadata: &GraphMetadata, data: Cow<'a, [u8]>) -> Self {
        Self {
            dimensions: metadata.dimensions,
            data,
        }
    }

    // Vector f32 data is stored little endian so we can get away with aliasing. Slice requires
    // that the pointer be properly aligned, which WiredTiger does not guarantee.
    fn maybe_alias_vector_data(&self) -> Option<&[f32]> {
        #[cfg(target_endian = "little")]
        {
            // WiredTiger does not guarantee that the returned memory will be aligned, a
            // Try to align it and if that fails, copy the data.
            let (prefix, vector, _) = unsafe { self.data.as_ref().align_to::<f32>() };
            if prefix.is_empty() {
                return Some(&vector[..self.dimensions.get()]);
            }
        }
        None
    }
}

impl<'a> GraphNode for WiredTigerGraphNode<'a> {
    type EdgeIterator<'c> = WiredTigerEdgeIterator<'c> where Self: 'c;

    fn vector(&self) -> Cow<'_, [f32]> {
        self.maybe_alias_vector_data()
            .map(|v| v.into())
            .unwrap_or_else(|| {
                let mut out = vec![0.0f32; self.dimensions.get()];
                for (i, o) in self
                    .data
                    .as_ref()
                    .chunks(std::mem::size_of::<f32>())
                    .zip(out.iter_mut())
                {
                    *o = f32::from_le_bytes(i.try_into().expect("array of 4 conversion."));
                }
                out.into()
            })
    }

    fn edges(&self) -> Self::EdgeIterator<'_> {
        WiredTigerEdgeIterator {
            data: &self.data.as_ref()[(self.dimensions.get() * std::mem::size_of::<f32>())..],
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

impl<'a> Iterator for WiredTigerEdgeIterator<'a> {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        let delta = leb128::read::signed(&mut self.data).ok()?;
        self.prev += delta;
        Some(self.prev)
    }
}

/// Implementation of `Graph` that reads from a WiredTiger `RecordCursor`.
pub struct WiredTigerGraph {
    metadata: GraphMetadata,
    cursor: RecordCursor,
}

impl WiredTigerGraph {
    pub fn new(metadata: GraphMetadata, cursor: RecordCursor) -> Self {
        Self { metadata, cursor }
    }
}

impl Graph for WiredTigerGraph {
    type Node<'c> = WiredTigerGraphNode<'c> where Self: 'c;

    fn entry_point(&mut self) -> Option<i64> {
        // TODO: handle errors better here. This probably requires changing the trait signature.
        let result = unsafe { self.cursor.seek_exact_unsafe(ENTRY_POINT_KEY)? };
        Some(
            result
                .map(|r| i64::from_le_bytes(r.value().try_into().unwrap()))
                .unwrap_or(0),
        )
    }

    fn get(&mut self, node: i64) -> Option<Result<Self::Node<'_>>> {
        let r = unsafe { self.cursor.seek_exact_unsafe(node)? }.map(RecordView::into_inner_value);
        Some(r.map(|r| WiredTigerGraphNode::new(&self.metadata, r)))
    }
}

/// Read graph index metadata from the named graph table.
// TODO: replace this call
// TODO: better story around caching this data and session/cursor management in general.
pub fn read_graph_metadata(
    connection: Arc<Connection>,
    graph_table_name: &str,
) -> io::Result<GraphMetadata> {
    let mut session = connection.open_session()?;
    let mut cursor = session.open_record_cursor(graph_table_name)?;
    let metadata_json = unsafe { cursor.seek_exact_unsafe(METADATA_KEY) }
        .unwrap_or(Err(Error::WiredTiger(WiredTigerError::NotFound)))?;
    serde_json::from_slice(metadata_json.value()).map_err(io::Error::from)
}

/// Immutable features of a WiredTiger graph vector index. These can be read from the db and
/// stored in a catalog for convenient access at runtime.
pub struct WiredTigerGraphVectorIndex {
    index_params: WiredTigerIndexParams,
    metadata: GraphMetadata,
}

impl WiredTigerGraphVectorIndex {
    /// Create a new `WiredTigerGraphVectorIndex` from `index_params`, caching immutable
    /// graph metadata.
    pub fn new(index_params: WiredTigerIndexParams) -> io::Result<Self> {
        let mut session = index_params.connection.open_session()?;
        let mut cursor = session.open_record_cursor(&index_params.graph_table_name)?;
        let metadata_json = unsafe { cursor.seek_exact_unsafe(METADATA_KEY) }
            .unwrap_or(Err(Error::WiredTiger(WiredTigerError::NotFound)))?;
        let metadata = serde_json::from_slice(metadata_json.value())?;
        Ok(Self {
            index_params,
            metadata,
        })
    }

    /// Return `GraphMetadata` for this index.
    pub fn metadata(&self) -> &GraphMetadata {
        &self.metadata
    }
}

/// A `GraphVectorIndexReader` implementation that operates entirely on a WiredTiger graph.
pub struct WiredTigerGraphVectorIndexReader {
    index: Arc<WiredTigerGraphVectorIndex>,
    session: Session,
}

impl WiredTigerGraphVectorIndexReader {
    /// Create a new `WiredTigerGraphVectorIndex` given a named index and a session to access that data.
    pub fn new(index: Arc<WiredTigerGraphVectorIndex>, session: Session) -> Self {
        Self { index, session }
    }

    /// Unwrap into the inner Session.
    pub fn into_session(self) -> Session {
        self.session
    }
}

impl GraphVectorIndexReader for WiredTigerGraphVectorIndexReader {
    type Graph = WiredTigerGraph;
    type NavVectorStore = WiredTigerNavVectorStore;

    fn scorer(&self) -> Box<dyn F32VectorScorer> {
        Box::new(DotProductScorer)
    }

    fn graph(&mut self) -> Result<Self::Graph> {
        Ok(WiredTigerGraph::new(
            self.index.metadata,
            self.session
                .open_record_cursor(&self.index.index_params.graph_table_name)?,
        ))
    }

    fn nav_scorer(&self) -> Box<dyn crate::scoring::QuantizedVectorScorer> {
        Box::new(HammingScorer)
    }

    fn nav_vectors(&mut self) -> Result<Self::NavVectorStore> {
        Ok(WiredTigerNavVectorStore::new(
            self.session
                .open_record_cursor(&self.index.index_params.nav_table_name)?,
        ))
    }
}

/// Encode the contents of a graph node as a value that can be set in the WiredTiger table.
pub fn encode_graph_node(vector: &[f32], mut edges: Vec<i64>) -> Vec<u8> {
    // A 64-bit value may occupy up to 10 bytes when leb128 encoded so reserve enough space for that.
    // There is unfortunately no constant for this in the leb128 crate.
    let mut out = Vec::with_capacity(vector.len() * std::mem::size_of::<f64>() + edges.len() * 10);
    for d in vector.iter() {
        out.extend_from_slice(&d.to_le_bytes());
    }
    edges.sort();
    let mut last = 0;
    for e in edges {
        leb128::write::signed(&mut out, e - last).unwrap();
        last = e;
    }
    out
}
