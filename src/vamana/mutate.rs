//! Tools for mutating a vamana graph vector index.
use crate::vamana::{
    prune_edges, search::GraphSearcher, EdgeSetDistanceComputer, Graph, GraphVectorIndex,
    GraphVectorStore,
};
use crate::Neighbor;
use vectors::VectorDistance;
use wt_mdb::{Error, Result};

/// Insert a vertex for `vector` and return the id assigned to the vector.
pub fn insert_vector(vector: &[f32], index: &impl GraphVectorIndex) -> Result<i64> {
    let vertex_id = index.graph()?.next_available_vertex_id()?;
    insert_internal(vertex_id, vector, index).map(|_| vertex_id)
}

/// Delete `vertex_id` from the graph index.
///
/// May return a non found error if `vertex_id` is not present in the index.
pub fn delete_vector(vertex_id: i64, index: &impl GraphVectorIndex) -> Result<()> {
    let mut graph = index.graph()?;
    let mut vectors = index.high_fidelity_vectors()?;
    let distance_fn = vectors.new_distance_function();

    let edges = graph.remove_vertex(vertex_id)?;
    let vector = vectors
        .get(vertex_id)
        .expect("row exists")
        .map(|v| v.to_vec())?;
    index.nav_vectors()?.remove(vertex_id)?;
    if let Some(vectors) = index.rerank_vectors() {
        vectors?.remove(vertex_id)?;
    }
    for e in edges.iter() {
        remove_edge_directed(&mut graph, *e, vertex_id)?;
    }

    // Cache information about each vertex linked to vertex_id.
    // Remove any links back to vertex_id.
    let vertex_data = edges
        .into_iter()
        .map(|e| {
            graph
                .edges(e)
                .unwrap_or(Err(Error::not_found_error()))
                .map(|edges| {
                    let vector = vectors.get(e).expect("row exists").map(|rv| rv.to_vec());
                    vector.map(|rv| (e, rv, edges.filter(|d| *d != vertex_id).collect::<Vec<_>>()))
                })
        })
        .collect::<Result<Result<Vec<_>>>>()??;

    // Create links between edges of the deleted node if needed.
    cross_link_peer_vertices(
        index,
        &mut graph,
        &mut vectors,
        &vertex_data,
        distance_fn.as_ref(),
    )?;

    // Oh no, we've deleted the entry point! Find the closest point amongst the edges of this node
    // to use as a new entry point.
    if graph
        .entry_point()
        .expect("there was at least one vertex")?
        == vertex_id
    {
        let mut neighbors = vertex_data
            .iter()
            .map(|(id, vec, _)| Neighbor::new(*id, distance_fn.distance(&vector, vec)))
            .collect::<Vec<_>>();
        neighbors.sort_unstable();
        if let Some(ep_neighbor) = neighbors.first() {
            graph.set_entry_point(ep_neighbor.vertex())?
        } else {
            graph.remove_entry_point()?
        }
    }

    Ok(())
}

/// Upsert vector with the externally assigned `vertex_id`.
pub fn upsert_vector(vertex_id: i64, vector: &[f32], index: &impl GraphVectorIndex) -> Result<()> {
    let mut graph = index.graph()?;
    if graph.edges(vertex_id).is_some() {
        delete_vector(vertex_id, index)?;
    }
    insert_internal(vertex_id, vector, index)
}

/// Insert `vector` at `vertex_id` into the index.
///
/// In addition to inserting the vector in the store this method will also choose edges for the new
/// vertex, insert back edges to maintain the undirected property of the graph, and potentially
/// prune out edges in backlink nodes to maintain the max_edges limit.
///
/// This method assumes that `vertex_id` does not already exist.
fn insert_internal(vertex_id: i64, vector: &[f32], index: &impl GraphVectorIndex) -> Result<()> {
    // TODO: make this an error instead of panicking.
    assert_eq!(index.config().dimensions.get(), vector.len());

    let mut searcher = GraphSearcher::new(index.config().index_search_params);
    let mut candidate_edges = searcher.search(vector, index)?;
    let mut graph = index.graph()?;
    if candidate_edges.is_empty() {
        graph.set_entry_point(vertex_id)?;
    }

    let edge_set_distance_computer = EdgeSetDistanceComputer::new(index, &candidate_edges)?;
    let selected_len = prune_edges(
        &mut candidate_edges,
        &index.config().pruning,
        edge_set_distance_computer,
    );
    candidate_edges.truncate(selected_len);

    graph.set_edges(
        vertex_id,
        candidate_edges
            .iter()
            .map(|n| n.vertex())
            .collect::<Vec<_>>(),
    )?;
    let mut nav_vectors = index.nav_vectors()?;
    nav_vectors.set(vertex_id, nav_vectors.new_coder().encode(vector))?;
    if let Some(vectors) = index.rerank_vectors() {
        let mut vectors = vectors?;
        vectors.set(vertex_id, vectors.new_coder().encode(vector))?;
    }

    let mut vectors = index.high_fidelity_vectors()?;
    let distance_fn = vectors.new_distance_function();
    let mut pruned_edges = vec![];
    for src_vertex_id in candidate_edges.into_iter().map(|n| n.vertex()) {
        let edges = insert_edge_directed(
            index,
            &mut graph,
            &mut vectors,
            distance_fn.as_ref(),
            src_vertex_id,
            vertex_id,
            &mut pruned_edges,
        )?;
        graph.set_edges(src_vertex_id, edges)?;

        for (src_vertex_id, dst_vertex_id) in pruned_edges.drain(..) {
            remove_edge_directed(&mut graph, src_vertex_id, dst_vertex_id)?;
        }
    }

    Ok(())
}

/// Attempt to insert a directed edge from `src_vertex_id` to `dst_vertex_id` and return the set of
/// edges for `src_vertex_id` after the insertion attempt.
///
/// If the edge already exists in the graph then no change is made. If inserting the edges would
/// exceed max_edges then the edges are pruned according to policy. This process may result in the
/// inserted edges being dropped. Pruning will also fill `pruned_edges` with any back edges that
/// need to be removed to maintain an undirected graph.
#[allow(clippy::too_many_arguments)]
fn insert_edge_directed(
    index: &impl GraphVectorIndex,
    graph: &mut impl Graph,
    vectors: &mut impl GraphVectorStore,
    distance_fn: &dyn VectorDistance,
    src_vertex_id: i64,
    dst_vertex_id: i64,
    pruned_edges: &mut Vec<(i64, i64)>,
) -> Result<Vec<i64>> {
    let mut edges = graph
        .edges(src_vertex_id)
        .unwrap_or(Err(Error::not_found_error()))?
        .collect::<Vec<_>>();
    if edges.contains(&dst_vertex_id) {
        return Ok(edges); // edge already exists.
    }
    edges.push(dst_vertex_id);
    if edges.len() <= index.config().pruning.max_edges.get() {
        return Ok(edges);
    }

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
    let edge_set_distance_computer = EdgeSetDistanceComputer::new(index, &neighbors)?;
    let selected_len = prune_edges(
        &mut neighbors,
        &index.config().pruning,
        edge_set_distance_computer,
    );
    // Ensure the graph is undirected by removing links from pruned edges back to this node.
    for v in neighbors.iter().skip(selected_len).map(Neighbor::vertex) {
        pruned_edges.push((v, src_vertex_id))
    }
    edges.clear();
    edges.extend(neighbors.iter().take(selected_len).map(Neighbor::vertex));
    Ok(edges)
}

/// Reads edges for `src_vertex_id`, removes `dst_vertex_id`, and writes back to the graph.
fn remove_edge_directed(
    graph: &mut impl Graph,
    src_vertex_id: i64,
    dst_vertex_id: i64,
) -> Result<()> {
    let edges = graph
        .edges(src_vertex_id)
        .unwrap_or(Err(Error::not_found_error()))?
        .filter(|v| *v != dst_vertex_id)
        .collect::<Vec<_>>();
    graph.set_edges(src_vertex_id, edges)
}

/// Cross link vertices from a deleted vertex.
///
/// vertex_data is a list of (vertex_id, vector, edges) for each vertex that was linked to the
/// deleted vertex. This method will score each pair of edges and re-insert the top edges that are
/// not already present in the graph to maintain connectivity.
fn cross_link_peer_vertices(
    index: &impl GraphVectorIndex,
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
    let relink_edges = index.config().pruning.max_edges.get().max(2) / 2;
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
    for (src_vertex_id, dst_vertex_id) in vertex_data
        .iter()
        .zip(edge_scores.into_iter())
        .flat_map(|(v, e)| std::iter::repeat(v.0).zip(e.into_iter().map(|n| n.vertex())))
    {
        // Insert edge symmetrically to maintain an undirected graph.
        let src_edges = insert_edge_directed(
            index,
            graph,
            vectors,
            distance_fn,
            src_vertex_id,
            dst_vertex_id,
            &mut pruned_edges,
        )?;
        let dst_edges = insert_edge_directed(
            index,
            graph,
            vectors,
            distance_fn,
            dst_vertex_id,
            src_vertex_id,
            &mut pruned_edges,
        )?;

        // If the edge was not inserted in both directions, then do not commit any of the
        // changes that were made here.
        if !src_edges.contains(&dst_vertex_id) || !dst_edges.contains(&src_vertex_id) {
            pruned_edges.clear();
            continue;
        }

        // Apply the changes to src and dst vertexes and remove any pruned edges.
        graph.set_edges(src_vertex_id, src_edges)?;
        graph.set_edges(dst_vertex_id, dst_edges)?;
        for (src_vertex_id, dst_vertex_id) in pruned_edges.drain(..) {
            remove_edge_directed(graph, src_vertex_id, dst_vertex_id)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{num::NonZero, sync::Arc};

    use vectors::{F32VectorCoding, VectorSimilarity};
    use wt_mdb::{Connection, Result};

    use crate::vamana::{
        mutate::{delete_vector, insert_vector, upsert_vector},
        search::GraphSearcher,
        wt::{SessionGraphVectorIndex, TableGraphVectorIndex},
        EdgePruningConfig, Graph, GraphConfig, GraphSearchParams, GraphVectorIndex,
    };

    struct Fixture {
        index: Arc<TableGraphVectorIndex>,
        conn: Arc<Connection>,
        wt_index: SessionGraphVectorIndex,
        _dir: tempfile::TempDir,
    }

    impl Fixture {
        fn search_params() -> GraphSearchParams {
            GraphSearchParams {
                beam_width: NonZero::new(16).unwrap(),
                num_rerank: 16,
                patience: None,
            }
        }

        fn new_reader(&self) -> SessionGraphVectorIndex {
            SessionGraphVectorIndex::new(self.index.clone(), self.conn.open_session().unwrap())
        }

        fn insert_many(&self, vectors: &[[f32; 2]]) -> Result<Vec<i64>> {
            vectors
                .iter()
                .map(|v| insert_vector(v.as_ref(), &self.wt_index))
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
                Some(
                    wt_mdb::connection::OptionsBuilder::default()
                        .create()
                        .into(),
                ),
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
                        pruning: EdgePruningConfig::new(NonZero::new(4).unwrap()),
                        index_search_params: Self::search_params(),
                    },
                    "test",
                )
                .unwrap(),
            );
            let wt_index =
                SessionGraphVectorIndex::new(index.clone(), conn.open_session().unwrap());
            Self {
                _dir: dir,
                conn,
                index,
                wt_index,
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

        let id = insert_vector(&[0.0, 0.0], &fixture.wt_index)?;
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

        let reader = &fixture.wt_index;
        let mut graph = reader.graph()?;
        assert_eq!(
            graph.edges(vertex_ids[0]).unwrap()?.collect::<Vec<_>>(),
            &[1, 2, 3, 5]
        );
        assert_eq!(fixture.search(&[0.0, 0.0]), Ok(vec![0, 1, 2, 3, 5, 4]));

        Ok(())
    }

    #[test]
    fn delete_one() -> Result<()> {
        let fixture = Fixture::default();

        let vertex_ids = fixture.insert_many(&[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])?;
        delete_vector(vertex_ids[1], &fixture.wt_index)?;
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
        assert_eq!(
            graph.edges(vertex_ids[0]).unwrap()?.collect::<Vec<_>>(),
            &[1, 2, 3, 5]
        );
        assert_eq!(fixture.search(&[0.0, 0.0]), Ok(vec![0, 1, 2, 3, 5, 4]));

        delete_vector(1, &fixture.wt_index)?;
        let reader = fixture.new_reader();
        let mut graph = reader.graph()?;
        assert_eq!(
            graph.edges(vertex_ids[0]).unwrap()?.collect::<Vec<_>>(),
            &[2, 3, 5]
        );
        assert_eq!(fixture.search(&[0.0, 0.0]), Ok(vec![0, 2, 3, 5, 4]));

        Ok(())
    }

    #[test]
    fn delete_entry_point() -> Result<()> {
        let fixture = Fixture::default();

        let entry_id = insert_vector(&[0.0, 0.0], &fixture.wt_index)?;
        let next_entry_id = insert_vector(&[0.5, 0.5], &fixture.wt_index)?;
        insert_vector(&[1.0, 1.0], &fixture.wt_index)?;

        delete_vector(entry_id, &fixture.wt_index)?;
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

        let id = insert_vector(&[0.0, 0.0], &fixture.wt_index)?;
        assert_eq!(fixture.search(&[0.0, 0.0])?, vec![id]);
        delete_vector(id, &fixture.wt_index)?;
        assert_eq!(fixture.search(&[0.0, 0.0])?, Vec::<i64>::new());

        Ok(())
    }

    #[test]
    fn upsert() -> Result<()> {
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
        upsert_vector(vertex_ids[0], &[1.0, 1.0], &fixture.wt_index)?;
        assert_eq!(fixture.search(&[0.0, 0.0]), Ok(vec![1, 2, 3, 5, 4, 0]));

        Ok(())
    }
}
