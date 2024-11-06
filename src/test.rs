use std::{borrow::Cow, num::NonZero, rc::Rc};

use wt_mdb::Result;

use crate::{
    graph::{Graph, GraphNode, NavVectorStore},
    quantization::binary_quantize,
    scoring::VectorScorer,
    Neighbor,
};

#[derive(Debug)]
struct TestVector {
    vector: Vec<f32>,
    nav_vector: Vec<u8>,
    edges: Vec<i64>,
}

#[derive(Clone, Debug)]
pub struct TestVectorData(Rc<Vec<TestVector>>);

impl TestVectorData {
    pub fn new<S, T, V>(max_edges: NonZero<usize>, scorer: S, iter: T) -> Self
    where
        S: VectorScorer<Elem = f32>,
        T: IntoIterator<Item = V>,
        V: Into<Vec<f32>>,
    {
        let mut rep = iter
            .into_iter()
            .map(|x| {
                let mut v = x.into();
                scorer.normalize(&mut v);
                let b = binary_quantize(&v);
                TestVector {
                    vector: v,
                    nav_vector: b,
                    edges: Vec::new(),
                }
            })
            .collect::<Vec<_>>();

        for i in 0..rep.len() {
            rep[i].edges = Self::compute_edges(&rep, i, max_edges, &scorer);
        }
        TestVectorData(Rc::new(rep))
    }

    fn compute_edges<S>(
        graph: &[TestVector],
        index: usize,
        max_edges: NonZero<usize>,
        scorer: &S,
    ) -> Vec<i64>
    where
        S: VectorScorer<Elem = f32>,
    {
        let q = &graph[index].vector;
        let mut scored = graph
            .iter()
            .enumerate()
            .filter_map(|(i, n)| {
                if i != index {
                    Some(Neighbor::new(i as i64, scorer.score(q, &n.vector)))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        scored.sort();
        if scored.is_empty() {
            return vec![];
        }

        let mut selected = Vec::with_capacity(std::cmp::min(scored.len(), max_edges.get()));
        selected.push(scored[0]);
        // RNG prune: select edges that are closer to the vertex than they are to any of the other
        // nodes we've already selected an edge to.
        for n in scored.iter().skip(1) {
            if selected.len() == max_edges.get() {
                break;
            }

            let q = &graph[n.node() as usize].vector;
            if !selected
                .iter()
                .any(|p| scorer.score(q, &graph[p.node() as usize].vector) > n.score())
            {
                selected.push(*n);
            }
        }
        selected.into_iter().map(|n| n.node()).collect()
    }
}

#[derive(Debug)]
pub struct TestGraph(TestVectorData);

impl From<TestVectorData> for TestGraph {
    fn from(value: TestVectorData) -> Self {
        TestGraph(value.clone())
    }
}

impl Graph for TestGraph {
    type Node<'c> = TestGraphNode<'c>;

    fn entry_point(&self) -> Option<i64> {
        if self.0 .0.is_empty() {
            None
        } else {
            Some(0)
        }
    }

    fn get(&mut self, node: i64) -> Option<Result<Self::Node<'_>>> {
        if node < 0 || node as usize >= self.0 .0.len() {
            None
        } else {
            Some(Ok(TestGraphNode(&self.0 .0[node as usize])))
        }
    }
}

pub struct TestGraphNode<'a>(&'a TestVector);

impl<'a> GraphNode for TestGraphNode<'a> {
    type EdgeIterator<'c> = std::iter::Copied<std::slice::Iter<'c, i64>> where Self: 'c;

    fn vector(&self) -> Cow<'_, [f32]> {
        Cow::from(&self.0.vector)
    }

    fn edges(&self) -> Self::EdgeIterator<'_> {
        self.0.edges.iter().copied()
    }
}

#[derive(Debug)]
pub struct TestNavVectorStore(TestVectorData);

impl From<TestVectorData> for TestNavVectorStore {
    fn from(value: TestVectorData) -> Self {
        TestNavVectorStore(value)
    }
}

impl NavVectorStore for TestNavVectorStore {
    fn get(&mut self, node: i64) -> Option<Result<Cow<'_, [u8]>>> {
        if node < 0 || node as usize >= self.0 .0.len() {
            None
        } else {
            Some(Ok(Cow::from(&self.0 .0[node as usize].nav_vector)))
        }
    }
}
