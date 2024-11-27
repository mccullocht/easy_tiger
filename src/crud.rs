use std::collections::HashMap;

use crate::{
    graph::{Graph, GraphVertex},
    scoring::F32VectorScorer,
    Neighbor,
};
use wt_mdb::{Error, Result};

pub struct CrudGraph<R> {
    reader: R,
    vertex_cache: HashMap<i64, CrudGraphVertex>,
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
// for rep:
// * edges are always Neighbor
// * unscored edges are NaN
// * I record if any unscored edges have been inserted
struct CrudGraphVertex {
    vector: Vec<f32>,
    edges: Vec<Neighbor>,
    // if set, one or more edges has a score of NaN.
    any_unscored: bool,
    // if set, this vertex is dirty and should be written back.
    dirty: bool,
    // XXX should I have deleted here?
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

    fn sorted_neighbors<G: Graph>(
        &mut self,
        graph: &mut G,
        scorer: &dyn F32VectorScorer,
    ) -> Result<&[Neighbor]> {
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
        Ok(&self.edges)
    }

    // insert_edge(&mut self, vertex_id: i64) -> usize;
    // remove_edge(&mut self, vertex_id: i64) -> usize;
    // prune_edges(&mut self, keep: impl Iterator<Item=i64>) -> Vec<i64>;
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
