use std::{borrow::Cow, num::NonZero, usize};

use wt_mdb::Result;

use crate::{
    graph::{
        Graph, GraphMetadata, GraphNode, GraphSearchParams, GraphVectorIndexReader, NavVectorStore,
    },
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

#[derive(Debug)]
pub struct TestGraphVectorIndex {
    data: Vec<TestVector>,
    metadata: GraphMetadata,
}

impl TestGraphVectorIndex {
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
        let metadata = GraphMetadata {
            dimensions: NonZero::new(rep.first().map(|v| v.vector.len()).unwrap_or(1)).unwrap(),
            max_edges: max_edges,
            index_search_params: GraphSearchParams {
                beam_width: NonZero::new(usize::MAX).unwrap(),
                num_rerank: usize::MAX,
            },
        };
        Self {
            data: rep,
            metadata,
        }
    }

    pub fn reader(&self) -> TestGraphVectorIndexReader {
        TestGraphVectorIndexReader(self)
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
pub struct TestGraphVectorIndexReader<'a>(&'a TestGraphVectorIndex);

impl<'a> GraphVectorIndexReader for TestGraphVectorIndexReader<'a> {
    type Graph = TestGraph<'a>;
    type NavVectorStore = TestNavVectorStore<'a>;

    fn metadata(&self) -> &GraphMetadata {
        &self.0.metadata
    }

    fn graph(&mut self) -> Result<Self::Graph> {
        Ok(TestGraph(self.0))
    }

    fn nav_vectors(&mut self) -> Result<Self::NavVectorStore> {
        Ok(TestNavVectorStore(self.0))
    }
}

#[derive(Debug)]
pub struct TestGraph<'a>(&'a TestGraphVectorIndex);

impl<'a> Graph for TestGraph<'a> {
    type Node<'c> = TestGraphNode<'c> where Self: 'c;

    fn entry_point(&mut self) -> Option<i64> {
        if self.0.data.is_empty() {
            None
        } else {
            Some(0)
        }
    }

    fn get(&mut self, node: i64) -> Option<Result<Self::Node<'_>>> {
        if node < 0 || node as usize >= self.0.data.len() {
            None
        } else {
            Some(Ok(TestGraphNode(&self.0.data[node as usize])))
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
pub struct TestNavVectorStore<'a>(&'a TestGraphVectorIndex);

impl<'a> NavVectorStore for TestNavVectorStore<'a> {
    fn get(&mut self, node: i64) -> Option<Result<Cow<'_, [u8]>>> {
        if node < 0 || node as usize >= self.0.data.len() {
            None
        } else {
            Some(Ok(Cow::from(&self.0.data[node as usize].nav_vector)))
        }
    }
}
