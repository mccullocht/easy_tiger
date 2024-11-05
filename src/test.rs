use std::num::NonZero;

use crate::{quantization::binary_quantize, Neighbor};

struct TestVector {
    vector: Vec<f32>,
    nav_vector: Vec<u8>,
    edges: Vec<i64>,
}

pub struct TestGraph {
    rep: Vec<TestVector>,
}

impl TestGraph {
    pub fn new<T, V>(max_edges: NonZero<usize>, iter: T) -> Self
    where
        T: IntoIterator<Item = V>,
        V: Into<Vec<f32>>,
    {
        let rep = iter
            .into_iter()
            .map(|x| {
                let v = x.into();
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

    fn compute_edges(&self, index: usize, max_edges: NonZero<usize>) -> Vec<i64> {
        let q = &self.rep[index].vector;
        let mut scored = self
            .rep
            .iter()
            .enumerate()
            .filter_map(|(i, n)| {
                if i != index {
                    Some(Neighbor::new(
                        i as i64,
                        simsimd::SpatialSimilarity::dot(q, &n.vector).unwrap(),
                    ))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        scored.sort();
        if (scored.is_empty()) {
            return vec![];
        }

        let mut edges = Vec::with_capacity(std::cmp::min(scored.len(), max_edges.get()));
        edges.push(scored[0]);
        // score against all previous vectors in the list
        for (i, n) in scored.iter().enumerate().skip(1) {}
        todo!()
    }
}
