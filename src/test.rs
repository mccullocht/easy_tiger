use std::{borrow::Cow, num::NonZero, usize};

use wt_mdb::Result;

use crate::{
    graph::{
        Graph, GraphMetadata, GraphSearchParams, GraphVectorIndexReader, GraphVertex,
        NavVectorStore,
    },
    quantization::binary_quantize,
    scoring::F32VectorScorer,
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
        S: F32VectorScorer,
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
        S: F32VectorScorer,
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

            let q = &graph[n.vertex() as usize].vector;
            if !selected
                .iter()
                .any(|p| scorer.score(q, &graph[p.vertex() as usize].vector) > n.score())
            {
                selected.push(*n);
            }
        }
        selected.into_iter().map(|n| n.vertex()).collect()
    }
}

#[derive(Debug)]
pub struct TestGraphVectorIndexReader<'a>(&'a TestGraphVectorIndex);

impl<'a> GraphVectorIndexReader for TestGraphVectorIndexReader<'a> {
    type Graph<'b> = TestGraph<'b> where Self: 'b;
    type NavVectorStore<'b> = TestNavVectorStore<'b> where Self: 'b;

    fn metadata(&self) -> &GraphMetadata {
        &self.0.metadata
    }

    fn graph(&self) -> Result<Self::Graph<'_>> {
        Ok(TestGraph(self.0))
    }

    fn nav_vectors(&self) -> Result<Self::NavVectorStore<'_>> {
        Ok(TestNavVectorStore(self.0))
    }
}

#[derive(Debug)]
pub struct TestGraph<'a>(&'a TestGraphVectorIndex);

impl<'a> Graph for TestGraph<'a> {
    type Vertex<'c> = TestGraphVertex<'c> where Self: 'c;

    fn entry_point(&mut self) -> Option<Result<i64>> {
        if self.0.data.is_empty() {
            None
        } else {
            Some(Ok(0))
        }
    }

    fn get(&mut self, vertex_id: i64) -> Option<Result<Self::Vertex<'_>>> {
        if vertex_id < 0 || vertex_id as usize >= self.0.data.len() {
            None
        } else {
            Some(Ok(TestGraphVertex(&self.0.data[vertex_id as usize])))
        }
    }
}

pub struct TestGraphVertex<'a>(&'a TestVector);

impl<'a> GraphVertex for TestGraphVertex<'a> {
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
    fn get(&mut self, vertex_id: i64) -> Option<Result<Cow<'_, [u8]>>> {
        if vertex_id < 0 || vertex_id as usize >= self.0.data.len() {
            None
        } else {
            Some(Ok(Cow::from(&self.0.data[vertex_id as usize].nav_vector)))
        }
    }
}
