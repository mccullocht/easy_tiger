use std::num::NonZero;

use crate::{quantization::binary_quantize, scoring::VectorScorer, Neighbor};

struct TestVector {
    vector: Vec<f32>,
    nav_vector: Vec<u8>,
    edges: Vec<i64>,
}

pub struct TestGraph {
    rep: Vec<TestVector>,
}

impl TestGraph {
    pub fn new<S, T, V>(max_edges: NonZero<usize>, scorer: S, iter: T) -> Self
    where
        S: VectorScorer<Elem = f32>,
        T: IntoIterator<Item = V>,
        V: Into<Vec<f32>>,
    {
        let rep = iter
            .into_iter()
            .map(|x| {
                let mut v = x.into();
                S::normalize(&mut v);
                let b = binary_quantize(&v);
                TestVector {
                    vector: v,
                    nav_vector: b,
                    edges: Vec::new(),
                }
            })
            .collect();
        TestGraph { rep }
    }

    fn compute_edges<S>(&self, index: usize, max_edges: NonZero<usize>, scorer: &S) -> Vec<i64>
    where
        S: VectorScorer<Elem = f32>,
    {
        let q = &self.rep[index].vector;
        let mut scored = self
            .rep
            .iter()
            .enumerate()
            .filter_map(|(i, n)| {
                if i != index {
                    Some(Neighbor::new(i as i64, S::score(q, &n.vector)))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        scored.sort();
        if scored.is_empty() {
            return vec![];
        }

        let mut edges = Vec::with_capacity(std::cmp::min(scored.len(), max_edges.get()));
        edges.push(scored[0]);
        // RNG prune: eliminate any neighbors that are closer to a selected edge than the vertex.
        for n in scored.iter().skip(1) {
            if edges.len() == max_edges.get() {
                break;
            }

            for p in edges.iter() {}
        }
        todo!()
    }
}
