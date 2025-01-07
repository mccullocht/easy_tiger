// DO NOT MERGE: this should not be public.
pub mod kmeans;
// DO NOT MERGE: this should not be public.
pub mod tree;

// TODO: WiredTiger bits.
// After we've trained the tree we need to enumerate tree nodes and generate keys & values.
// * keys always begin with an identifier for the node in the kmeans tree.
// * parent nodes have an entry per centroid and the key ends with child node + {parent,leaf} type.
// * leaf nodes have an entry per indexed vector and the key ends with collection record identifer.
// * values can contain raw vector, quantized vector, or both.
