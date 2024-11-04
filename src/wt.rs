use std::{borrow::Cow, num::NonZero};

use wt_mdb::{RecordCursor, RecordView, Result};

use crate::graph::{Graph, GraphNode, NavVectorStore};

/// Metadata about graph shape and construction.
#[derive(Copy, Clone, Debug)]
pub struct GraphMetadata {
    pub dimensions: NonZero<usize>,
}

/// Implementation of NavVectorStore that reads from a WiredTiger `RecordCursor``.
pub struct WiredTigerNavVectorStore<'a> {
    cursor: RecordCursor<'a>,
}

impl<'a> WiredTigerNavVectorStore<'a> {
    pub fn new(cursor: RecordCursor<'a>) -> Self {
        Self { cursor }
    }
}

impl<'a> NavVectorStore for WiredTigerNavVectorStore<'a> {
    fn get(&mut self, node: i64) -> Option<Result<Cow<'_, [u8]>>> {
        Some(unsafe { self.cursor.seek_exact_unsafe(node)? }.map(RecordView::into_inner_value))
    }
}

/// Implementation of GraphNode that reads from an encoded value in a WiredTiger record table.
pub struct WiredTigerGraphNode<'a> {
    dimensions: NonZero<usize>, // XXX this should be a reference to GraphMetadata.
    data: Cow<'a, [u8]>,
}

impl<'a> WiredTigerGraphNode<'a> {
    pub fn new(dimensions: NonZero<usize>, data: Cow<'a, [u8]>) -> Self {
        Self { dimensions, data }
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
                return Some(vector);
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
            data: &self.data.as_ref()[..(self.dimensions.get() * std::mem::size_of::<f32>())],
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
pub struct WiredTigerGraph<'a> {
    metadata: GraphMetadata,
    cursor: RecordCursor<'a>,
}

impl<'a> WiredTigerGraph<'a> {
    pub fn new(metadata: GraphMetadata, cursor: RecordCursor<'a>) -> Self {
        Self { metadata, cursor }
    }
}

impl<'a> Graph for WiredTigerGraph<'a> {
    type Node<'c> = WiredTigerGraphNode<'c> where Self: 'c;

    fn get(&mut self, node: i64) -> Option<Result<Self::Node<'_>>> {
        let r = unsafe { self.cursor.seek_exact_unsafe(node)? }.map(RecordView::into_inner_value);
        Some(r.map(|r| WiredTigerGraphNode::new(self.metadata.dimensions, r)))
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
