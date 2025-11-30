//! Tools for mutating WiredTiger backed vector indices.
use std::sync::Arc;

use crate::vamana::{
    graph::{
        prune_edges, EdgeSetDistanceComputer, Graph, GraphLayout, GraphVectorIndex,
        GraphVectorStore, GraphVertex,
    },
    search::GraphSearcher,
    wt::{SessionGraphVectorIndexReader, TableGraphVectorIndex, ENTRY_POINT_KEY},
};
use crate::Neighbor;
use vectors::{F32VectorCoder, VectorDistance};
use wt_mdb::{Error, Result, Session};

/// Perform mutations on the vector index.
///
/// This accepts a [wt_mdb::Session] that is used to mutate the index. Callers should begin a
/// transaction before creating [IndexMutator] and commit the transaction when they are done
/// mutating.
pub struct IndexMutator {
    reader: SessionGraphVectorIndexReader,
    searcher: GraphSearcher,

    nav_coder: Box<dyn F32VectorCoder>,
    rerank_coder: Option<Box<dyn F32VectorCoder>>,
}

impl IndexMutator {
    /// Create a new mutator operating on `index` and wrapping `session`.
    ///
    /// Callers should typically begin a transaction on the session _before_ creating this object,
    /// then [Self::into_session()] this struct and commit the transaction when done.
    pub fn new(index: Arc<TableGraphVectorIndex>, session: Session) -> Self {
        let searcher = GraphSearcher::new(index.config().index_search_params);
        let nav_coder = index.nav_table().new_coder();
        let rerank_coder = index.rerank_table().map(|t| t.new_coder());
        Self {
            reader: SessionGraphVectorIndexReader::new(index, session),
            searcher,
            nav_coder,
            rerank_coder,
        }
    }

    /// Obtain the inner [wt_mdb::Session].
    pub fn into_session(self) -> Session {
        self.reader.into_session()
    }

    /// Obtain a reference to the underlying [wt_mdb::Session]
    pub fn session(&self) -> &Session {
        self.reader.session()
    }

    /// Insert a vertex for `vector`. Returns the assigned id.
    pub fn insert(&mut self, vector: &[f32]) -> Result<i64> {
        // A freshly initialized table might will have the metadata key but no entry point.
        let vertex_id = self
            .reader
            .session()
            .get_record_cursor(self.reader.index().graph_table_name())?
            .largest_key()
            .unwrap_or(Ok(-1))
            .map(|i| std::cmp::max(i, -1) + 1)?;
        self.insert_internal(vertex_id, vector).map(|_| vertex_id)
    }

    fn insert_internal(&mut self, vertex_id: i64, vector: &[f32]) -> Result<()> {
        // TODO: make this an error instead of panicking.
        assert_eq!(self.reader.config().dimensions.get(), vector.len());

        let mut candidate_edges = self.searcher.search(vector, &mut self.reader)?;
        let mut graph = self.reader.graph()?;
        if candidate_edges.is_empty() {
            // Proceed through the rest of the function so that the inserts happen.
            // This is mostly as a noop because there are no edges.
            graph.set_entry_point(vertex_id)?;
        }
        let edge_set_distance_computer =
            EdgeSetDistanceComputer::new(&self.reader, &candidate_edges)?;
        let selected_len = prune_edges(
            &mut candidate_edges,
            self.reader.config().max_edges,
            edge_set_distance_computer,
        );
        candidate_edges.truncate(selected_len);

        self.reader.graph()?.set(
            vertex_id,
            candidate_edges
                .iter()
                .map(|n| n.vertex())
                .collect::<Vec<_>>(),
        )?;
        self.reader
            .nav_vectors()?
            .set(vertex_id, self.nav_coder.encode(vector))?;
        if let Some((vectors, coder)) = self.reader.rerank_vectors().zip(self.rerank_coder.as_ref())
        {
            vectors?.set(vertex_id, coder.encode(vector))?;
        }

        let mut vectors = self.reader.high_fidelity_vectors()?;
        let distance_fn = vectors.new_distance_function();
        let mut pruned_edges = vec![];
        for src_vertex_id in candidate_edges.into_iter().map(|n| n.vertex()) {
            self.insert_edge(
                &mut graph,
                &mut vectors,
                distance_fn.as_ref(),
                src_vertex_id,
                vertex_id,
                &mut pruned_edges,
            )?;
        }

        for (src_vertex_id, dst_vertex_id) in pruned_edges {
            self.remove_edge(&mut graph, src_vertex_id, dst_vertex_id)?;
        }

        Ok(())
    }

    /// returns true if the edge already exists or was inserted successfully.
    fn insert_edge(
        &self,
        graph: &mut impl Graph,
        vectors: &mut impl GraphVectorStore,
        distance_fn: &dyn VectorDistance,
        src_vertex_id: i64,
        dst_vertex_id: i64,
        pruned_edges: &mut Vec<(i64, i64)>,
    ) -> Result<bool> {
        let vertex = graph
            .get_vertex(src_vertex_id)
            .unwrap_or(Err(Error::not_found_error()))?;
        let mut edges = vertex.edges().collect::<Vec<_>>();
        if edges.contains(&dst_vertex_id) {
            return Ok(true); // edge already exists.
        }
        edges.push(dst_vertex_id);
        edges.sort_unstable();
        let inserted = if edges.len() <= self.reader.config().max_edges.get() {
            true
        } else {
            let src_vector = vectors
                .get(src_vertex_id)
                .expect("row exists")
                .map(|v| v.to_vec())?;
            let mut neighbors = edges
                .iter()
                .map(|e| {
                    vectors
                        .get(*e)
                        .unwrap_or(Err(Error::not_found_error()))
                        .map(|dst| Neighbor::new(*e, distance_fn.distance(&src_vector, dst)))
                })
                .collect::<Result<Vec<Neighbor>>>()?;
            neighbors.sort();
            let edge_set_distance_computer =
                EdgeSetDistanceComputer::new(&self.reader, &neighbors)?;
            let selected_len = prune_edges(
                &mut neighbors,
                self.reader.config().max_edges,
                edge_set_distance_computer,
            );
            // Ensure the graph is undirected by removing links from pruned edges back to this node.
            for v in neighbors.iter().skip(selected_len).map(Neighbor::vertex) {
                pruned_edges.push((v, src_vertex_id))
            }
            edges.clear();
            edges.extend(neighbors.iter().take(selected_len).map(Neighbor::vertex));
            edges.contains(&dst_vertex_id)
        };
        self.set_graph_edges(src_vertex_id, edges)
            .map(|()| inserted)
    }

    fn remove_edge(
        &self,
        graph: &mut impl Graph,
        src_vertex_id: i64,
        dst_vertex_id: i64,
    ) -> Result<()> {
        let vertex = graph
            .get_vertex(src_vertex_id)
            .unwrap_or(Err(Error::not_found_error()))?;
        let edges = vertex.edges().filter(|v| *v != dst_vertex_id).collect();
        self.set_graph_edges(src_vertex_id, edges)
    }

    /// Delete `vertex_id`, removing both the vertex and any incoming edges.
    pub fn delete(&mut self, vertex_id: i64) -> Result<()> {
        let mut graph = self.reader.graph()?;
        let mut vectors = self.reader.high_fidelity_vectors()?;
        let distance_fn = vectors.new_distance_function();

        let (vector, edges) = graph
            .get_vertex(vertex_id)
            .unwrap_or(Err(Error::not_found_error()))
            .map(|v| {
                vectors
                    .get(vertex_id)
                    .expect("row exists")
                    .map(|vec| (vec.to_vec(), v.edges().collect::<Vec<_>>()))
            })??;

        // TODO: unified graph index writer trait to handles removal and other mutations.
        graph.remove(vertex_id)?;
        self.reader.nav_vectors()?.remove(vertex_id)?;
        if let Some(vectors) = self.reader.rerank_vectors() {
            vectors?.remove(vertex_id)?;
        }
        for e in edges.iter() {
            self.remove_edge(&mut graph, *e, vertex_id)?;
        }

        // Cache information about each vertex linked to vertex_id.
        // Remove any links back to vertex_id.
        let vertex_data = edges
            .into_iter()
            .map(|e| {
                graph
                    .get_vertex(e)
                    .unwrap_or(Err(Error::not_found_error()))
                    .map(|v| {
                        let vector = vectors.get(e).expect("row exists").map(|rv| rv.to_vec());
                        vector.map(|rv| {
                            (
                                e,
                                rv,
                                v.edges().filter(|d| *d != vertex_id).collect::<Vec<_>>(),
                            )
                        })
                    })
            })
            .collect::<Result<Result<Vec<_>>>>()??;

        // Create links between edges of the deleted node if needed.
        self.cross_link_peer_vertices(
            &mut graph,
            &mut vectors,
            &vertex_data,
            distance_fn.as_ref(),
        )?;

        // Oh no, we've deleted the entry point! Find the closes point amongst the
        // edges of this node to use as a new one. So long as at least one vector is in
        // the index it will be safe to unwrap() entry_point() here.
        if graph.entry_point().unwrap()? == vertex_id {
            let mut neighbors = vertex_data
                .iter()
                .map(|(id, vec, _)| Neighbor::new(*id, distance_fn.distance(&vector, vec)))
                .collect::<Vec<_>>();
            neighbors.sort();
            if let Some(ep_neighbor) = neighbors.first() {
                graph.set_entry_point(ep_neighbor.vertex())?
            } else {
                graph.remove(ENTRY_POINT_KEY)?
            }
        }

        Ok(())
    }

    fn cross_link_peer_vertices(
        &self,
        graph: &mut impl Graph,
        vectors: &mut impl GraphVectorStore,
        vertex_data: &[(i64, Vec<u8>, Vec<i64>)],
        distance_fn: &dyn VectorDistance,
    ) -> Result<()> {
        // Compute the distance between each pair of edges and insert symmetrical links.
        let mut edge_scores = vec![vec![]; vertex_data.len()];
        for (i, (src_vertex_id, src_vector, _)) in vertex_data.iter().enumerate() {
            for (j, (dst_vertex_id, dst_vector, _)) in vertex_data.iter().enumerate().skip(i + 1) {
                let dist = distance_fn.distance(src_vector, dst_vector);
                edge_scores[i].push(Neighbor::new(*dst_vertex_id, dist));
                edge_scores[j].push(Neighbor::new(*src_vertex_id, dist));
            }
        }

        // Take the list of scored edges and truncate to 50% of max_edges, then filter out all of
        // the edges that already exist in the graph based on vertex_data. The rest we will attempt
        // to insert symmetrically to maintain an undirected graph.
        let relink_edges = self.reader.config().max_edges.get().max(2) / 2;
        for (current_edges, scored_edges) in vertex_data
            .iter()
            .map(|(_, _, e)| e)
            .zip(edge_scores.iter_mut())
        {
            scored_edges.sort_unstable();
            scored_edges.truncate(relink_edges);
            scored_edges.retain(|n| !current_edges.contains(&n.vertex()));
        }

        let mut pruned_edges = vec![];
        for (src_vertex_id, mut edge_scores) in
            vertex_data.iter().map(|v| v.0).zip(edge_scores.into_iter())
        {
            edge_scores.sort_unstable();
            for dst_vertex_id in edge_scores
                .into_iter()
                .take(relink_edges)
                .map(|n| n.vertex())
            {
                // Insert edge symmetrically. If the first edge insertion succeeds and the symmetric
                // edge insertion fails, then remove the edge to ensure the graph is undirected.
                if self.insert_edge(
                    graph,
                    vectors,
                    distance_fn,
                    src_vertex_id,
                    dst_vertex_id,
                    &mut pruned_edges,
                )? && !self.insert_edge(
                    graph,
                    vectors,
                    distance_fn,
                    dst_vertex_id,
                    src_vertex_id,
                    &mut pruned_edges,
                )? {
                    self.remove_edge(graph, src_vertex_id, dst_vertex_id)?;
                }
            }

            // Remove all back edges removed by the insertions above.
            for (src_vertex_id, dst_vertex_id) in pruned_edges.drain(..) {
                self.remove_edge(graph, src_vertex_id, dst_vertex_id)?;
            }
        }

        Ok(())
    }

    /// Update the contents of `vertex_id` with `vector`.
    pub fn update(&mut self, vertex_id: i64, vector: &[f32]) -> Result<()> {
        // TODO: a non-trivial implementation might perform the search like during insert
        // and skip edge updates if the edge set is identical or nearly identical.
        self.delete(vertex_id)?;
        self.insert_internal(vertex_id, vector)
    }

    fn set_graph_edges(&self, vertex_id: i64, edges: Vec<i64>) -> Result<()> {
        match self.reader.config().layout {
            GraphLayout::Split => self.reader.graph()?.set(vertex_id, edges),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{num::NonZero, sync::Arc};

    use vectors::{F32VectorCoding, VectorSimilarity};
    use wt_mdb::{options::ConnectionOptionsBuilder, Connection, Result};

    use crate::vamana::{
        graph::{
            Graph, GraphConfig, GraphLayout, GraphSearchParams, GraphVectorIndex, GraphVertex,
        },
        search::GraphSearcher,
        wt::{SessionGraphVectorIndexReader, TableGraphVectorIndex},
    };

    use super::IndexMutator;

    struct Fixture {
        index: Arc<TableGraphVectorIndex>,
        conn: Arc<Connection>,
        _dir: tempfile::TempDir,
    }

    impl Fixture {
        fn search_params() -> GraphSearchParams {
            GraphSearchParams {
                beam_width: NonZero::new(16).unwrap(),
                num_rerank: 16,
            }
        }

        fn new_reader(&self) -> SessionGraphVectorIndexReader {
            SessionGraphVectorIndexReader::new(
                self.index.clone(),
                self.conn.open_session().unwrap(),
            )
        }

        fn new_mutator(&self) -> IndexMutator {
            IndexMutator::new(self.index.clone(), self.conn.open_session().unwrap())
        }

        fn insert_many(&self, vectors: &[[f32; 2]]) -> Result<Vec<i64>> {
            let mut mutator = self.new_mutator();
            vectors
                .iter()
                .map(|v| mutator.insert(v.as_ref()))
                .collect::<Result<Vec<_>>>()
        }

        fn search(&self, query: &[f32]) -> Result<Vec<i64>> {
            let mut searcher = GraphSearcher::new(Self::search_params());
            let mut reader = self.new_reader();
            searcher
                .search(query, &mut reader)
                .map(|neighbors| neighbors.into_iter().map(|n| n.vertex()).collect())
        }
    }

    impl Default for Fixture {
        fn default() -> Self {
            let dir = tempfile::TempDir::new().unwrap();
            let conn = Connection::open(
                dir.path().to_str().unwrap(),
                Some(ConnectionOptionsBuilder::default().create().into()),
            )
            .unwrap();
            let index = Arc::new(
                TableGraphVectorIndex::init_index(
                    &conn,
                    GraphConfig {
                        dimensions: NonZero::new(2).unwrap(),
                        similarity: VectorSimilarity::Euclidean,
                        nav_format: F32VectorCoding::BinaryQuantized,
                        rerank_format: Some(F32VectorCoding::F32),
                        layout: GraphLayout::Split,
                        max_edges: NonZero::new(4).unwrap(),
                        index_search_params: Self::search_params(),
                    },
                    "test",
                )
                .unwrap(),
            );
            Self {
                _dir: dir,
                conn,
                index,
            }
        }
    }

    #[test]
    fn empty_graph() -> Result<()> {
        let fixture = Fixture::default();

        let mut reader = fixture.new_reader();
        assert_eq!(reader.graph()?.entry_point(), None);
        let mut searcher = GraphSearcher::new(Fixture::search_params());
        assert_eq!(searcher.search(&[0.5, -0.5], &mut reader), Ok(vec![]));
        Ok(())
    }

    #[test]
    fn insert_one() -> Result<()> {
        let fixture = Fixture::default();

        let mut mutator = fixture.new_mutator();
        let id = mutator.insert(&[0.0, 0.0])?;
        assert_eq!(id, 0);
        assert_eq!(fixture.new_reader().graph()?.entry_point(), Some(Ok(id)));
        assert_eq!(fixture.search(&[1.0, 1.0]), Ok(vec![id]));
        Ok(())
    }

    #[test]
    fn insert_two() -> Result<()> {
        let fixture = Fixture::default();

        fixture.insert_many(&[[0.0, 0.0], [0.5, 0.5]])?;
        assert_eq!(fixture.search(&[1.0, 1.0]), Ok(vec![1, 0]));
        Ok(())
    }

    // Insert enough vectors that we have to prune the edge list for the entry point.
    #[test]
    fn insert_to_prune() -> Result<()> {
        let fixture = Fixture::default();

        let vertex_ids = fixture.insert_many(&[
            [0.0, 0.0],
            [0.1, 0.1],
            [-0.1, 0.1],
            [0.1, -0.1],
            [-0.2, -0.2],
            [-0.1, -0.1],
        ])?;

        let reader = fixture.new_reader();
        let mut graph = reader.graph()?;
        let vertex = graph.get_vertex(vertex_ids[0]).unwrap()?;
        assert_eq!(vertex.edges().collect::<Vec<_>>(), &[1, 2, 3, 5]);
        assert_eq!(fixture.search(&[0.0, 0.0]), Ok(vec![0, 1, 2, 3, 5, 4]));

        Ok(())
    }

    #[test]
    fn delete_one() -> Result<()> {
        let fixture = Fixture::default();

        let vertex_ids = fixture.insert_many(&[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])?;
        fixture.new_mutator().delete(vertex_ids[1])?;
        assert_eq!(
            fixture.search(&[0.0, 0.0])?,
            vertex_ids
                .iter()
                .copied()
                .filter(|i| *i != vertex_ids[1])
                .collect::<Vec<_>>()
        );

        Ok(())
    }

    // Delete an edge
    #[test]
    fn delete_relink() -> Result<()> {
        let fixture = Fixture::default();

        let vertex_ids = fixture.insert_many(&[
            [0.0, 0.0],
            [0.1, 0.1],
            [-0.1, 0.1],
            [0.1, -0.1],
            [-0.2, -0.2],
            [-0.1, -0.1],
        ])?;

        let reader = fixture.new_reader();
        let mut graph = reader.graph()?;
        let vertex = graph.get_vertex(vertex_ids[0]).unwrap()?;
        assert_eq!(vertex.edges().collect::<Vec<_>>(), &[1, 2, 3, 5]);
        assert_eq!(fixture.search(&[0.0, 0.0]), Ok(vec![0, 1, 2, 3, 5, 4]));

        fixture.new_mutator().delete(1)?;
        let reader = fixture.new_reader();
        let mut graph = reader.graph()?;
        let vertex = graph.get_vertex(vertex_ids[0]).unwrap()?;
        assert_eq!(vertex.edges().collect::<Vec<_>>(), &[2, 3, 5]);
        assert_eq!(fixture.search(&[0.0, 0.0]), Ok(vec![0, 2, 3, 5, 4]));

        Ok(())
    }

    #[test]
    fn delete_entry_point() -> Result<()> {
        let fixture = Fixture::default();

        let mut mutator = fixture.new_mutator();
        let entry_id = mutator.insert(&[0.0, 0.0])?;
        let next_entry_id = mutator.insert(&[0.5, 0.5])?;
        mutator.insert(&[1.0, 1.0])?;

        mutator.delete(entry_id)?;
        assert_eq!(
            fixture.new_reader().graph()?.entry_point(),
            Some(Ok(next_entry_id))
        );
        assert_eq!(fixture.search(&[0.0, 0.0])?, vec![1, 2]);

        Ok(())
    }

    #[test]
    fn delete_only_point() -> Result<()> {
        let fixture = Fixture::default();

        let mut mutator = fixture.new_mutator();
        let id = mutator.insert(&[0.0, 0.0])?;
        assert_eq!(fixture.search(&[0.0, 0.0])?, vec![id]);
        mutator.delete(id)?;
        assert_eq!(fixture.search(&[0.0, 0.0])?, Vec::<i64>::new());

        Ok(())
    }

    #[test]
    fn update() -> Result<()> {
        let fixture = Fixture::default();

        let vertex_ids = fixture.insert_many(&[
            [0.0, 0.0],
            [0.1, 0.1],
            [-0.1, 0.1],
            [0.1, -0.1],
            [-0.2, -0.2],
            [-0.1, -0.1],
        ])?;
        assert_eq!(fixture.search(&[0.0, 0.0]), Ok(vec![0, 1, 2, 3, 5, 4]));
        fixture.new_mutator().update(vertex_ids[0], &[1.0, 1.0])?;
        assert_eq!(fixture.search(&[0.0, 0.0]), Ok(vec![1, 2, 3, 5, 4, 0]));

        Ok(())
    }
}
