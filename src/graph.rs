use std::borrow::Cow;

use wt_mdb::Result;

/// A node in the Vamana graph.
pub trait GraphNode {
    type EdgeIterator<'a>: Iterator<Item = i64>
    where
        Self: 'a;

    /// Access the raw float vector.
    fn vector(&self) -> Cow<'_, [f32]>;

    /// Access the edges of the graph. These may be returned in an arbitrary order.
    fn edges(&self) -> Self::EdgeIterator<'_>;
}

/// A Vamana graph.
pub trait Graph {
    type Node<'c>: GraphNode
    where
        Self: 'c;

    /// Get the contents of a single node.
    // XXX this design is weird to allow this to be cursor backed.
    fn get(&mut self, node: i64) -> Option<Result<Self::Node<'_>>>;
}

/// Vector store for vectors used to navigate the graph.
pub trait NavVectorStore {
    /// Get the navigation vector for a single node.
    // XXX this design is weird to allow this to be cursor backed.
    fn get(&mut self, node: i64) -> Option<Result<Cow<'_, [u8]>>>;
}
