//! Tools to bulk load an index.
//!
//! For a brand new index we can take a set of vectors and build the graph in memory,
//! then bulk load it into WiredTiger. This is much faster than incrementally building
//! the graph through a series of transactions as there are fewer, simpler abstractions
//! around vector access and graph edge state.
//!
//! Caveats:
//! * Only `numpy` little-endian formatted `f32` vectors are accepted.
//! * Row keys are assigned densely beginning at zero.
use core::f64;
use std::{
    borrow::Cow,
    num::NonZero,
    sync::{
        atomic::{self, AtomicI64},
        Arc,
    },
};

use memmap2::{Mmap, MmapMut};
use rand::prelude::*;
use rayon::prelude::*;
use wt_mdb::{Connection, Record, Result, Session};

use crate::{
    graph::{prune_edges, Graph, GraphConfig, GraphVectorIndexReader, GraphVertex, NavVectorStore},
    input::{DerefVectorStore, VectorStore},
    quantization::{binary_quantize, binary_quantized_bytes},
    scoring::F32VectorScorer,
    search::GraphSearcher,
    wt::{
        encode_graph_node, CursorNavVectorStore, TableGraphVectorIndex, CONFIG_KEY, ENTRY_POINT_KEY,
    },
    Neighbor,
};

/// Stats for the built graph.
#[derive(Debug)]
pub struct GraphStats {
    /// Number of vertices in the graph.
    pub vertices: usize,
    /// Total number of out edges across all nodes.
    /// The graph is undirected so the actual edge count would be half of this.
    pub edges: usize,
    /// Number of vertices that have no out edges.
    /// The graph is undirected so these nodes also should not have in edges.
    pub unconnected: usize,
}

// TODO: rather than relying on users calling these methods in the right order, instead
// consume/perform/generate, e.g.:
//
// pub fn QuantizedVectorLoader::new(...) -> Self
// pub fn QuantizedVectorLoader::load(self, progress_cb) -> Result<BulkGraphBuilder>
// pub fn BulkGraphBuilder::build(self, progress_cb) -> Result<BulkGraphPruner>
// pub fn BulkGraphPruner::load(self, progress_cb) -> Result<()>
//
// if you do this the graph build can be locked, but then transform into a representation that
// is better for single-threaded access for cleanup.

// TODO: tests for bulk load builder, mostly the built graph.

/// Builds a Vamana graph for a bulk load.
pub struct BulkLoadBuilder<D> {
    connection: Arc<Connection>,
    index: TableGraphVectorIndex,
    limit: usize,

    vectors: DerefVectorStore<f32, D>,
    centroid: Vec<f32>,

    memory_quantized_vectors: bool,
    quantized_vectors: Option<DerefVectorStore<u8, Mmap>>,

    graph: Vec<Vec<Neighbor>>,
    entry_vertex: AtomicI64,
    scorer: Box<dyn F32VectorScorer>,
}

impl<D> BulkLoadBuilder<D>
where
    D: Send + Sync,
{
    /// Create a new bulk graph builder with the passed vector set and configuration.
    /// `limit` limits the number of vectors processed to less than the full set.
    pub fn new(
        connection: Arc<Connection>,
        index: TableGraphVectorIndex,
        vectors: DerefVectorStore<f32, D>,
        memory_quantized_vectors: bool,
        limit: usize,
    ) -> Self {
        let scorer = index.config().new_scorer();
        Self {
            connection,
            index,
            limit,
            vectors,
            centroid: Vec::new(),
            memory_quantized_vectors,
            quantized_vectors: None,
            graph: Vec::new(),
            entry_vertex: AtomicI64::new(-1),
            scorer,
        }
    }

    /// Load binary quantized vector data into the nav vectors table.
    pub fn load_nav_vectors<P: Fn()>(&mut self, progress: P) -> Result<()> {
        let session = self.connection.open_session()?;
        let dim = self.index.config().dimensions.get();
        let mut sum = vec![0.0; dim];
        let mut quantized_vectors = if self.memory_quantized_vectors {
            Some(MmapMut::map_anon(binary_quantized_bytes(dim) * self.vectors.len()).unwrap())
        } else {
            None
        };
        session.bulk_load(
            self.index.nav_table_name(),
            None,
            self.vectors
                .iter()
                .enumerate()
                .take(self.limit)
                .map(|(i, v)| {
                    progress();
                    for (i, o) in v.iter().zip(sum.iter_mut()) {
                        *o += *i as f64;
                    }
                    let quantized = binary_quantize(v);
                    if let Some(q) = quantized_vectors.as_mut() {
                        let start = i * quantized.len();
                        q[start..(start + quantized.len())].copy_from_slice(&quantized);
                    }
                    Record::new(i as i64, quantized)
                }),
        )?;

        self.quantized_vectors = quantized_vectors.map(|m| {
            DerefVectorStore::new(
                m.make_read_only().unwrap(),
                NonZero::new(binary_quantized_bytes(dim)).unwrap(),
            )
            .unwrap()
        });
        self.centroid = self
            .scorer
            .normalize_vector(
                sum.into_iter()
                    .map(|s| (s / self.limit as f64) as f32)
                    .collect::<Vec<_>>()
                    .into(),
            )
            .into_owned();
        Ok(())
    }

    pub fn init_graph<P>(&mut self, progress: P) -> Result<()>
    where
        P: Fn() + Send + Sync,
    {
        let mut r = StdRng::seed_from_u64(0xbeabadd00d);
        self.graph = (0..self.limit)
            .into_iter()
            .map(|vertex| {
                (0..self.index.config().max_edges.get())
                    .into_iter()
                    .map(|_| Neighbor::new(r.gen_range(0..self.limit as i64), 0.0))
                    .filter(|x| x.vertex() != vertex as i64)
                    .collect::<Vec<_>>()
            })
            .collect();
        let ep = (0..self.limit)
            .into_par_iter()
            .map(|i| {
                let ep_neighbor = Neighbor::new(
                    i as i64,
                    self.scorer.score(&self.vectors[i], &self.centroid),
                );
                progress();
                ep_neighbor
            })
            .max()
            .unwrap();
        self.entry_vertex
            .store(ep.vertex(), atomic::Ordering::SeqCst);
        Ok(())
    }

    /// Insert all vectors from the passed vector store into the graph.
    ///
    /// This operation uses rayon to parallelize large parts of the graph build.
    pub fn insert_all<P>(&mut self, progress: P) -> Result<()>
    where
        P: Fn() + Send + Sync,
    {
        // XXX at scale I am being murdered by copying vectors. this might actually be a result of
        // page faulting. this could be fixed by allowing the searcher to return the vectors and
        // prune_edges to accept the parallel vectors.
        self.graph = (0..self.limit)
            .into_par_iter()
            .map_init(
                || {
                    (
                        self.connection.open_session().unwrap(),
                        GraphSearcher::new(self.index.config().index_search_params),
                    )
                },
                |(session, searcher), vertex| {
                    session.begin_transaction(None)?;
                    let mut reader = BulkLoadGraphVectorIndexReader(self, session);
                    let new_edges = self.search_for_insert(vertex, searcher, &mut reader)?;
                    session.rollback_transaction(None)?;
                    progress();
                    Ok::<_, wt_mdb::Error>(new_edges)
                },
            )
            .collect::<Result<Vec<_>>>()?;
        Ok(())
    }

    /// Cleanup the graph.
    ///
    /// This may prune edges and/or ensure graph connectivity.
    pub fn cleanup<P>(&mut self, progress: P) -> Result<()>
    where
        P: Fn(),
    {
        let max_edges = self.index.config().max_edges;
        let mut graph: Vec<Vec<Neighbor>> = Vec::with_capacity(self.limit);
        graph.resize_with(self.limit, || Vec::with_capacity(max_edges.get() * 2));
        for (vertex, edges) in self.graph.iter().enumerate() {
            for edge in edges.iter() {
                // Skip if vertex already has this edge. Insertion below may do this as it ensures
                // reciprocal edges.
                if graph[vertex]
                    .iter()
                    .find(|n| n.vertex() == edge.vertex())
                    .is_some()
                {
                    continue;
                }

                if self.insert_edge(vertex, *edge, &mut graph)? {
                    self.insert_edge(
                        edge.vertex() as usize,
                        Neighbor::new(vertex as i64, edge.score()),
                        &mut graph,
                    )?;
                }
            }
        }

        // XXX this is quite slow. another approach would be to prune in parallel and emit the
        // dropped edges into another data structure. we'd then drop edges until every vertex meets
        // the max_edges invariant.
        for vertex in 0..self.limit {
            if graph[vertex].len() <= max_edges.get() {
                progress();
                continue;
            }

            let split = prune_edges(
                &mut graph[vertex],
                max_edges,
                &mut BulkLoadBuilderGraph(self),
                self.scorer.as_ref(),
            )?;
            let dropped = graph[vertex].split_off(split);
            for edge in dropped {
                graph[edge.vertex() as usize].retain(|n| n.vertex() != vertex as i64);
            }
            progress();
        }
        self.graph = graph;
        Ok(())
    }

    /// Bulk load the graph table with raw vectors and graph edges.
    ///
    /// Returns statistics about the generated graph.
    pub fn load_graph<P>(&self, progress: P) -> Result<GraphStats>
    where
        P: Fn(),
    {
        let mut stats = GraphStats {
            vertices: 0,
            edges: 0,
            unconnected: 0,
        };
        let config_rows = vec![
            Record::new(
                CONFIG_KEY,
                serde_json::to_vec(&self.index.config()).unwrap(),
            ),
            Record::new(
                ENTRY_POINT_KEY,
                self.entry_vertex
                    .load(atomic::Ordering::Relaxed)
                    .to_le_bytes()
                    .to_vec(),
            ),
        ];
        let session = self.connection.open_session()?;
        session.bulk_load(
            self.index.graph_table_name(),
            None,
            config_rows.into_iter().chain(
                self.vectors
                    .iter()
                    .zip(self.graph.iter())
                    .enumerate()
                    .take(self.limit)
                    .map(|(i, (v, vertex))| {
                        progress();
                        stats.vertices += 1;
                        stats.edges += vertex.len();
                        if vertex.is_empty() {
                            stats.unconnected += 1;
                        }
                        Record::new(
                            i as i64,
                            encode_graph_node(
                                &self.scorer.normalize_vector(v.into()),
                                vertex.iter().map(|n| n.vertex()).collect(),
                            ),
                        )
                    }),
            ),
        )?;
        Ok(stats)
    }

    fn search_for_insert(
        &self,
        vertex_id: usize,
        searcher: &mut GraphSearcher,
        reader: &mut BulkLoadGraphVectorIndexReader<'_, D>,
    ) -> Result<Vec<Neighbor>> {
        let mut graph = BulkLoadBuilderGraph(self);
        let mut candidates = searcher.search_for_insert(vertex_id as i64, reader)?;
        let split = prune_edges(
            &mut candidates,
            self.index.config().max_edges,
            &mut graph,
            self.scorer.as_ref(),
        )?;
        candidates.truncate(split);
        Ok(candidates)
    }

    fn insert_edge(
        &self,
        vertex: usize,
        edge: Neighbor,
        graph: &mut Vec<Vec<Neighbor>>,
    ) -> Result<bool> {
        let inserted_edge = if graph[vertex].len() == graph[vertex].capacity() {
            if edge < *graph[vertex].last().unwrap() {
                let dropped = graph[vertex].pop().unwrap();
                graph[dropped.vertex() as usize].retain(|n| n.vertex() != vertex as i64);
                true
            } else {
                false
            }
        } else {
            graph[vertex].push(edge);
            if graph[vertex].len() == graph[vertex].capacity() {
                graph[vertex].sort();
            }
            true
        };
        Ok(inserted_edge)
    }

    fn get_vector(&self, index: usize) -> Cow<'_, [f32]> {
        self.scorer.normalize_vector(self.vectors[index].into())
    }
}

struct BulkLoadGraphVectorIndexReader<'a, D: Send>(&'a BulkLoadBuilder<D>, &'a Session);

impl<D: Send + Sync> GraphVectorIndexReader for BulkLoadGraphVectorIndexReader<'_, D> {
    type Graph<'b>
        = BulkLoadBuilderGraph<'b, D>
    where
        Self: 'b;
    type NavVectorStore<'b>
        = BulkLoadNavVectorStore<'b>
    where
        Self: 'b;

    fn config(&self) -> &GraphConfig {
        self.0.index.config()
    }

    fn graph(&self) -> Result<Self::Graph<'_>> {
        Ok(BulkLoadBuilderGraph(self.0))
    }

    fn nav_vectors(&self) -> Result<Self::NavVectorStore<'_>> {
        if let Some(s) = self.0.quantized_vectors.as_ref() {
            Ok(BulkLoadNavVectorStore::Memory(s))
        } else {
            Ok(BulkLoadNavVectorStore::Cursor(CursorNavVectorStore::new(
                self.1.get_record_cursor(self.0.index.nav_table_name())?,
            )))
        }
    }
}

struct BulkLoadBuilderGraph<'a, D: Send>(&'a BulkLoadBuilder<D>);

impl<D: Send + Sync> Graph for BulkLoadBuilderGraph<'_, D> {
    type Vertex<'c>
        = BulkLoadGraphVertex<'c, D>
    where
        Self: 'c;

    fn entry_point(&mut self) -> Option<Result<i64>> {
        let vertex = self.0.entry_vertex.load(atomic::Ordering::Relaxed);
        if vertex >= 0 {
            Some(Ok(vertex))
        } else {
            None
        }
    }

    fn get(&mut self, vertex_id: i64) -> Option<Result<Self::Vertex<'_>>> {
        Some(Ok(BulkLoadGraphVertex {
            builder: self.0,
            vertex_id,
        }))
    }
}

struct BulkLoadGraphVertex<'a, D> {
    builder: &'a BulkLoadBuilder<D>,
    vertex_id: i64,
}

impl<D: Send + Sync> GraphVertex for BulkLoadGraphVertex<'_, D> {
    type EdgeIterator<'c>
        = BulkNodeEdgesIterator<'c>
    where
        Self: 'c;

    fn vector(&self) -> Cow<'_, [f32]> {
        self.builder.get_vector(self.vertex_id as usize)
    }

    fn edges(&self) -> Self::EdgeIterator<'_> {
        BulkNodeEdgesIterator::new(self.builder.graph[self.vertex_id as usize].iter())
    }
}

struct BulkNodeEdgesIterator<'a>(std::slice::Iter<'a, Neighbor>);

impl<'a> BulkNodeEdgesIterator<'a> {
    fn new(it: std::slice::Iter<'a, Neighbor>) -> Self {
        Self(it)
    }
}

impl Iterator for BulkNodeEdgesIterator<'_> {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|n| n.vertex())
    }
}

enum BulkLoadNavVectorStore<'a> {
    Cursor(CursorNavVectorStore<'a>),
    Memory(&'a DerefVectorStore<u8, memmap2::Mmap>),
}

impl NavVectorStore for BulkLoadNavVectorStore<'_> {
    fn get(&mut self, vertex_id: i64) -> Option<Result<Cow<'_, [u8]>>> {
        match self {
            Self::Cursor(c) => c.get(vertex_id),
            Self::Memory(m) => {
                if vertex_id >= 0 && (vertex_id as usize) < m.len() {
                    Some(Ok(m[vertex_id as usize].into()))
                } else {
                    None
                }
            }
        }
    }
}
