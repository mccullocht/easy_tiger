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
    fn new<T, V>(max_edges: NonZero<usize>, iter: T) -> Self
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

    fn search(&self, index: usize) -> Vec<i64> {
        let q = &self.rep[index].vector;
        let neighbors = self
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
        todo!()
    }
}
