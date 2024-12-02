use core::f64;
use std::collections::HashMap;

use crate::{
    graph::{Graph, GraphVertex},
    scoring::F32VectorScorer,
    Neighbor,
};
use wt_mdb::{Error, Result};

pub struct CrudGraph<R> {
    reader: R,
    vertex_cache: HashMap<i64, Option<CrudGraphVertex>>,
}

// XXX I haven't thought much about how I'm going to interact with this.
// * One appears in CrudGraph for each node participating in a transaction
// * Might want a notion of is-mutated for write back. Would always mutate on insert,
//   but update and delete might not.
// * Need abstractions to iterate and mutate edges
//   - insert will add scored edges
//   - Unscored edges can be used when adding backlinks on insert
//   - Unscored edges can be used in re-linking on delete.
//   - when I insert unscored edges
//   - scored edges are needed pruning.
//
// XXX is this the right way of doing this, vs simply mutating in-place on the graph?
// feels like I'm building a shitty ORM
struct CrudGraphVertex {
    vector: Vec<f32>,
    edges: Vec<Neighbor>,
    // if set, one or more edges has a score of NaN.
    any_unscored: bool,
    // if set, this vertex is dirty and should be written back.
    dirty: bool,
}

impl CrudGraphVertex {
    /// Create a new vertex with a vector and a set of scored edges.
    fn new(vector: Vec<f32>, edges: Vec<Neighbor>) -> Self {
        let any_unscored = edges.iter().any(|n| n.score().is_nan());
        Self {
            vector,
            edges,
            any_unscored,
            dirty: false,
        }
    }

    /// Return edges in an arbitrary order.
    fn edges(&self) -> impl Iterator<Item = i64> + ExactSizeIterator + '_ {
        self.edges.iter().map(|n| n.vertex())
    }

    /// Insert an edge to `vertex_id`. Return the number of edges on the vertex.
    fn insert_edge(&mut self, vertex_id: i64) -> usize {
        if self
            .edges
            .iter()
            .find(|n| n.vertex() == vertex_id)
            .is_none()
        {
            self.edges.push(Neighbor::new(vertex_id, f64::NAN));
            self.any_unscored = true;
            self.dirty = true;
        }
        self.edges.len()
    }

    /// Remove an edge to `vertex_id`. Return the number of edges on the vertex.
    fn remove_edge(&mut self, vertex_id: i64) -> usize {
        let len = self.edges.len();
        self.edges.retain(|n| n.vertex() != vertex_id);
        if len != self.edges.len() {
            self.dirty = true;
        }
        self.edges.len()
    }

    fn prune_edges<S: Iterator<Item = usize>>(
        &mut self,
        graph: &mut impl Graph,
        scorer: &dyn F32VectorScorer,
        prune: impl FnOnce(&[Neighbor]) -> S,
    ) -> Result<Vec<i64>> {
        if self.any_unscored {
            for n in self.edges.iter_mut().filter(|n| n.score().is_nan()) {
                let vertex = graph
                    .get(n.vertex())
                    .unwrap_or(Err(Error::not_found_error()))?;
                n.score = scorer.score(&self.vector, &vertex.vector());
            }
            self.any_unscored = false;
        }
        self.edges.sort();

        // Prune and move kept edges to the beginning of the edge array.
        let mut kept = 0usize;
        for (i, j) in prune(&self.edges).enumerate() {
            self.edges.swap(i, j);
            kept += 1;
        }

        // Extract the list of vertices that were pruned off and truncate the edge list.
        let (_, pruned) = self.edges.split_at(kept);
        let pruned_vertices = pruned.iter().map(Neighbor::vertex).collect();
        self.edges.truncate(kept);
        self.dirty = true;

        Ok(pruned_vertices)
    }
}

/// Convert a `GraphVertex` into an unscored vertex.
impl<V: GraphVertex> From<V> for CrudGraphVertex {
    fn from(value: V) -> Self {
        Self {
            vector: value.vector().to_vec(),
            edges: value.edges().map(|e| Neighbor::new(e, f64::NAN)).collect(),
            any_unscored: true,
            dirty: false,
        }
    }
}
