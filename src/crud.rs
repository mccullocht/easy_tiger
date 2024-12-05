//! Tools for mutating WiredTiger backed vector indices.
use std::sync::Arc;

use crate::{
    graph::{prune_edges, Graph, GraphVectorIndexReader, GraphVertex},
    quantization::binary_quantize,
    scoring::F32VectorScorer,
    search::GraphSearcher,
    wt::{
        encode_graph_node, encode_graph_node_internal, WiredTigerGraph, WiredTigerGraphVectorIndex,
        WiredTigerGraphVectorIndexReader, ENTRY_POINT_KEY,
    },
    Neighbor,
};
use wt_mdb::{Error, Result, Session};

/// Perform mutations on the vector index.
///
/// This accepts a [wt_mdb::Session] that is used to mutate the index. Callers should begin a
/// transaction before creating [IndexMutator] and commit the transaction when they are done
/// mutating.
pub struct IndexMutator {
    reader: WiredTigerGraphVectorIndexReader,
    searcher: GraphSearcher,
}

impl IndexMutator {
    /// Create a new mutator operating on `index` and wrapping `session`.
    ///
    /// Callers should typically begin a transaction on the session _before_ creating this object,
    /// then [Self::into_session()] this struct and commit the transaction when done.
    pub fn new(index: Arc<WiredTigerGraphVectorIndex>, session: Session) -> Self {
        let searcher = GraphSearcher::new(index.metadata().index_search_params);
        Self {
            reader: WiredTigerGraphVectorIndexReader::new(index, session, None),
            searcher,
        }
    }

    /// Obtain the inner [wt_mdb::Session].
    pub fn into_session(self) -> Session {
        self.reader.into_session()
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
        assert_eq!(self.reader.metadata().dimensions.get(), vector.len());

        let scorer = self.reader.metadata().new_scorer();
        // TODO: normalize input vector.
        let mut candidate_edges = self.searcher.search(vector, &mut self.reader)?;
        let mut graph = self.reader.graph()?;
        if candidate_edges.is_empty() {
            // Proceed through the rest of the function so that the inserts happen.
            // This is mostly as a noop because there are no edges.
            graph.set_entry_point(vertex_id)?;
        }
        let selected_len = prune_edges(
            &mut candidate_edges,
            self.reader.metadata().max_edges,
            &mut graph,
            scorer.as_ref(),
        )?;
        candidate_edges.truncate(selected_len);

        graph.set(
            vertex_id,
            &encode_graph_node_internal(
                vector,
                candidate_edges.iter().map(|n| n.vertex()).collect(),
            ),
        )?;
        self.reader
            .nav_vectors()?
            .set(vertex_id, binary_quantize(vector).into())?;

        let mut pruned_edges = vec![];
        for src_vertex_id in candidate_edges.into_iter().map(|n| n.vertex()) {
            self.insert_edge(
                &mut graph,
                scorer.as_ref(),
                src_vertex_id,
                vertex_id,
                &mut pruned_edges,
            )?;
        }

        // TODO: group and bulk delete if there are common src_vertex_id.
        for (src_vertex_id, dst_vertex_id) in pruned_edges {
            let vertex = graph
                .get(src_vertex_id)
                .unwrap_or(Err(Error::not_found_error()))?;
            let edges = vertex.edges().filter(|v| *v != dst_vertex_id).collect();
            let encoded = encode_graph_node_internal(vertex.vector_bytes(), edges);
            graph.set(src_vertex_id, &encoded)?;
        }

        Ok(())
    }

    fn insert_edge(
        &self,
        graph: &mut WiredTigerGraph<'_>,
        scorer: &dyn F32VectorScorer,
        src_vertex_id: i64,
        dst_vertex_id: i64,
        pruned_edges: &mut Vec<(i64, i64)>,
    ) -> Result<()> {
        let vertex = graph
            .get(src_vertex_id)
            .unwrap_or(Err(Error::not_found_error()))?;
        let mut edges = std::iter::once(dst_vertex_id)
            .chain(vertex.edges().filter(|v| *v != dst_vertex_id))
            .collect::<Vec<_>>();
        let encoded = if edges.len() >= self.reader.metadata().max_edges.get() {
            let src_vector = vertex.vector().to_vec();
            let mut neighbors = edges
                .iter()
                .map(|e| {
                    graph
                        .get(*e)
                        .unwrap_or(Err(Error::not_found_error()))
                        .map(|dst| Neighbor::new(*e, scorer.score(&src_vector, &dst.vector())))
                })
                .collect::<Result<Vec<Neighbor>>>()?;
            neighbors.sort();
            let selected_len = prune_edges(
                &mut neighbors,
                self.reader.metadata().max_edges,
                graph,
                scorer,
            )?;
            // Ensure the graph is undirected by removing links from pruned edges back to this node.
            for v in neighbors.iter().skip(selected_len).map(Neighbor::vertex) {
                pruned_edges.push((v, src_vertex_id))
            }
            edges.clear();
            edges.extend(neighbors.iter().take(selected_len).map(Neighbor::vertex));

            encode_graph_node(&src_vector, edges)
        } else {
            encode_graph_node_internal(vertex.vector_bytes(), edges)
        };
        graph.set(src_vertex_id, &encoded)
    }

    /// Delete `vertex_id`, removing both the vertex and any incoming edges.
    pub fn delete(&mut self, vertex_id: i64) -> Result<()> {
        let scorer = self.reader.metadata().new_scorer();
        let mut graph = self.reader.graph()?;

        let (vector, edges) = graph
            .get(vertex_id)
            .unwrap_or(Err(Error::not_found_error()))
            .map(|v| (v.vector().to_vec(), v.edges().collect::<Vec<_>>()))?;

        graph.remove(vertex_id)?;
        self.reader.nav_vectors()?.remove(vertex_id)?;

        // Cache information about each vertex linked to vertex_id.
        // Remove any links back to vertex_id.
        let mut vertex_data = edges
            .into_iter()
            .map(|e| {
                graph
                    .get(e)
                    .unwrap_or(Err(Error::not_found_error()))
                    .map(|v| {
                        (
                            e,
                            v.vector().to_vec(),
                            v.edges().filter(|d| *d != vertex_id).collect::<Vec<_>>(),
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        // Create links between edges of the deleted node if needed.
        self.cross_link_peer_vertices(&mut vertex_data, scorer.as_ref())?;

        // Oh no, we've deleted the entry point! Find the closes point amongst the
        // edges of this node to use as a new one. So long as at least one vector is in
        // the index it will be safe to unwrap() entry_point() here.
        if graph.entry_point().unwrap()? == vertex_id {
            let mut neighbors = vertex_data
                .iter()
                .map(|(id, vec, _)| Neighbor::new(*id, scorer.score(&vector, vec)))
                .collect::<Vec<_>>();
            neighbors.sort();
            if let Some(ep_neighbor) = neighbors.first() {
                graph.set(ENTRY_POINT_KEY, &ep_neighbor.vertex().to_le_bytes())?
            } else {
                graph.remove(ENTRY_POINT_KEY)?
            }
        }

        // Write all the mutated nodes back to WT.
        for (vertex_id, vector, edges) in vertex_data {
            graph.set(vertex_id, &encode_graph_node(&vector, edges))?;
        }

        Ok(())
    }

    fn cross_link_peer_vertices(
        &self,
        vertex_data: &mut [(i64, Vec<f32>, Vec<i64>)],
        scorer: &dyn F32VectorScorer,
    ) -> Result<()> {
        // Score all pairs of vectors among the passed vertices.
        let mut candidate_links = vertex_data
            .iter()
            .enumerate()
            .flat_map(|(src, (_, src_vector, src_edges))| {
                vertex_data
                    .iter()
                    .enumerate()
                    .skip(src + 1)
                    .filter_map(|(dst, (dst_vertex, dst_vector, _))| {
                        if !src_edges.contains(dst_vertex) {
                            Some((src, dst, scorer.score(src_vector, dst_vector)))
                        } else {
                            None
                        }
                    })
                    // TODO: this feels unnecessary but it doesn't compile without it.
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // Sort candidate links in descending order by score, then apply all that fit --
        // that is inserting the edge into both vertices does not exceed max_edges.
        candidate_links.sort_by(|a, b| {
            a.2.total_cmp(&b.2)
                .reverse()
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.1.cmp(&b.1))
        });
        let max_edges = self.reader.metadata().max_edges.get();
        for (src, dst, _) in candidate_links {
            if vertex_data[src].2.len() < max_edges && vertex_data[dst].2.len() < max_edges {
                let src_id = vertex_data[src].0;
                let dst_id = vertex_data[dst].0;
                vertex_data[src].2.push(dst_id);
                vertex_data[dst].2.push(src_id);
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
}

#[cfg(test)]
mod tests {
    use std::{num::NonZero, sync::Arc};

    use wt_mdb::{options::ConnectionOptionsBuilder, Connection, Result};

    use crate::{
        graph::{Graph, GraphMetadata, GraphSearchParams, GraphVectorIndexReader, GraphVertex},
        scoring::VectorSimilarity,
        search::GraphSearcher,
        wt::{WiredTigerGraphVectorIndex, WiredTigerGraphVectorIndexReader},
    };

    use super::IndexMutator;

    struct Fixture {
        index: Arc<WiredTigerGraphVectorIndex>,
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

        fn new_reader(&self) -> WiredTigerGraphVectorIndexReader {
            WiredTigerGraphVectorIndexReader::new(
                self.index.clone(),
                self.conn.open_session().unwrap(),
                None,
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
                WiredTigerGraphVectorIndex::from_init(
                    GraphMetadata {
                        dimensions: NonZero::new(2).unwrap(),
                        similarity: VectorSimilarity::Euclidean,
                        max_edges: NonZero::new(4).unwrap(),
                        index_search_params: Self::search_params(),
                    },
                    "test",
                )
                .unwrap(),
            );
            index.init_index(&conn, None).unwrap();
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
        let vertex = graph.get(vertex_ids[0]).unwrap()?;
        assert_eq!(vertex.edges().collect::<Vec<_>>(), &[1, 5]);
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
        let vertex = graph.get(vertex_ids[0]).unwrap()?;
        assert_eq!(vertex.edges().collect::<Vec<_>>(), &[1, 5]);
        assert_eq!(fixture.search(&[0.0, 0.0]), Ok(vec![0, 1, 2, 3, 5, 4]));

        fixture.new_mutator().delete(1)?;
        let reader = fixture.new_reader();
        let mut graph = reader.graph()?;
        let vertex = graph.get(vertex_ids[0]).unwrap()?;
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
