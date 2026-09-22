//! Tools for mutating a vamana graph vector index.
use crate::Neighbor;
use crate::vamana::select_pruned_edges;
use crate::vamana::{
    EdgeSetDistanceComputer, EdgeType, Graph, GraphVectorIndex, GraphVectorStore, prune_edges,
    search::{GraphSearcher, Options as GraphSearchOptions},
};
use std::collections::hash_map::Entry::Vacant;
use std::collections::{HashMap, hash_map::Entry};
use std::num::NonZero;
use wt_mdb::{Error, Result};

/// Insert a vertex for `vector` and return the id assigned to the vector.
pub fn insert_vector(vector: &[f32], index: &impl GraphVectorIndex) -> Result<i64> {
    insert_vector_with_options(vector, index, GraphSearchOptions::default())
}

/// Insert a vertex for `vector` and return the id assigned to the vector.
///
/// `options` is used for the graph search that selects candidate edges for the new vertex; in
/// particular a filter may be used to exclude specific vertex ids from being selected as edges
/// (their edges are still traversed during the search, they are just never linked to).
pub fn insert_vector_with_options<F: FnMut(i64) -> bool>(
    vector: &[f32],
    index: &impl GraphVectorIndex,
    options: GraphSearchOptions<F>,
) -> Result<i64> {
    let vertex_id = index.graph()?.next_available_vertex_id()?;
    insert_internal(vertex_id, vector, index, options).map(|_| vertex_id)
}

/// Delete `vertex_id` from the graph index.
///
/// May return a non found error if `vertex_id` is not present in the index.
pub fn delete_vector(vertex_id: i64, index: &impl GraphVectorIndex) -> Result<()> {
    delete_internal(vertex_id, index)
}

/// Upsert vector with the externally assigned `vertex_id`.
pub fn upsert_vector(vertex_id: i64, vector: &[f32], index: &impl GraphVectorIndex) -> Result<()> {
    upsert_vector_with_options(vertex_id, vector, index, GraphSearchOptions::default())
}

/// Upsert vector with the externally assigned `vertex_id`.
///
/// `options` is used for the graph search that selects candidate edges for the (re-)inserted
/// vertex; in particular a filter may be used to exclude specific vertex ids from being selected
/// as edges (their edges are still traversed during the search, they are just never linked to).
pub fn upsert_vector_with_options<F: FnMut(i64) -> bool>(
    vertex_id: i64,
    vector: &[f32],
    index: &impl GraphVectorIndex,
    options: GraphSearchOptions<F>,
) -> Result<()> {
    let mut graph = index.graph()?;
    if graph.edges(vertex_id).is_some() {
        delete_vector(vertex_id, index)?;
    }
    insert_internal(vertex_id, vector, index, options)
}

/// Repair the edges for a vertex.
///
/// Search using the highest fidelity vector available for vertex_id and add the resulting edges
/// back to vertex_id along with any back edges.
pub fn repair(vertex_id: i64, index: &impl GraphVectorIndex) -> Result<()> {
    let mut graph = index.graph()?;
    if graph.edges(vertex_id).is_none() {
        return Ok(());
    }

    let mut vectors = index.high_fidelity_vectors()?;
    let query = vectors
        .new_coder()
        .decode(vectors.get(vertex_id).expect("vertex_id found")?);

    let mut searcher = GraphSearcher::new(index.config().index_search_params);
    let (candidates, _) = searcher.search_with_options(
        &query,
        GraphSearchOptions::default().return_seen(true),
        index,
    )?;

    let mut vertex_buf = VertexBuffer::new(index)?;
    for e in candidates.into_iter().map(|n| n.vertex()) {
        vertex_buf.insert_edge_directed(vertex_id, e)?;
        vertex_buf.insert_edge_directed(e, vertex_id)?;
    }

    vertex_buf.flush()
}

struct VertexBuffer<'a, I: GraphVectorIndex> {
    index: &'a I,
    graph: I::Graph<'a>,
    vectors: I::VectorStore<'a>,
    cache: HashMap<i64, Vec<i64>>,
    buffer_limit: usize,
}

impl<'a, I: GraphVectorIndex> VertexBuffer<'a, I> {
    pub fn new(index: &'a I) -> Result<Self> {
        let graph = index.graph()?;
        let vectors = index.high_fidelity_vectors()?;
        let buffer_limit = index.config().pruning.max_edges.get() * 2;
        Ok(VertexBuffer {
            index,
            graph,
            vectors,
            cache: HashMap::new(),
            buffer_limit,
        })
    }

    pub fn index(&self) -> &'a I {
        self.index
    }

    /// Insert a new edge from src to dst. Call with arguments reversed to get a back edge.
    ///
    /// May fail if there was an error reading `src` edges.
    pub fn insert_edge_directed(&mut self, src: i64, dst: i64) -> Result<()> {
        let limit = self.buffer_limit;
        let edges = self.read_edges(src)?;
        if edges.contains(&dst) {
            return Ok(());
        }
        edges.push(dst);
        if edges.len() < limit {
            Ok(())
        } else {
            self.prune_edges(src)
        }
    }

    /// Remove the edge from src to dst, if present.
    ///
    /// May fail if there was an error reading `src` edges.
    pub fn remove_edge_directed(&mut self, src: i64, dst: i64) -> Result<()> {
        let edges = self.read_edges(src)?;
        if let Some(i) = edges.iter().position(|&v| v == dst) {
            edges.swap_remove(i);
        }
        Ok(())
    }

    /// Returns true if an edge from src to dst already exists.
    pub fn edge_exists(&mut self, src: i64, dst: i64) -> Result<bool> {
        let edges = self.read_edges(src)?;
        Ok(edges.contains(&dst))
    }

    /// For a buffered vertex, returns true if the edge count is max_edges or more.
    pub fn is_saturated(&self, vertex: i64) -> Option<bool> {
        self.cache
            .get(&vertex)
            .map(|e| e.len() >= self.index.config().pruning.max_edges.get())
    }

    /// Prune any buffered vertexes that have more edges than policy allows.
    pub fn prune_buffered(&mut self) -> Result<()> {
        let to_prune = self
            .cache
            .iter()
            .filter(|(_, e)| e.len() > self.index.config().pruning.max_edges.get())
            .map(|(&v, _)| v)
            .collect::<Vec<_>>();
        for vertex in to_prune {
            self.prune_edges(vertex)?;
        }
        Ok(())
    }

    /// Flush buffered vertexes back into the graph.
    ///
    /// Vertexes may buffer more edges than policy allows; such vertexes will be pruned before they
    /// are written back.
    pub fn flush(mut self) -> Result<()> {
        self.prune_buffered()?;

        // Flush the edges back into the graph.
        for (vertex, edges) in self.cache {
            self.graph.set_edges(vertex, edges)?;
        }
        Ok(())
    }

    fn read_edges(&mut self, vertex: i64) -> Result<&mut Vec<i64>> {
        if let Vacant(e) = self.cache.entry(vertex) {
            let edges = self
                .graph
                .edges(vertex)
                .transpose()?
                .map(|it| it.collect::<Vec<_>>())
                .unwrap_or_default();
            e.insert(edges);
        }
        Ok(self
            .cache
            .get_mut(&vertex)
            .expect("entry was just inserted"))
    }

    fn prune_edges(&mut self, vertex: i64) -> Result<()> {
        let edges = self.read_edges(vertex)?.to_vec();
        let vertex_vector = self
            .vectors
            .get(vertex)
            .unwrap_or(Err(Error::not_found_error()))?
            .to_vec();
        let (neighbors, computer) = EdgeSetDistanceComputer::from_directed_edges(
            &vertex_vector,
            &mut self.vectors,
            edges.as_slice(),
        )?;
        let selected = select_pruned_edges(&neighbors, &self.index.config().pruning, computer);
        if self.index.config().edge_type == EdgeType::Undirected {
            let mut sit = selected.iter().copied().peekable();
            for i in (0..neighbors.len()).filter(|i| sit.next_if_eq(i).is_none()) {
                self.remove_edge_directed(neighbors[i].vertex(), vertex)?;
            }
        }
        let edges = self.read_edges(vertex)?;
        edges.clear();
        for v in selected {
            edges.push(neighbors[v].vertex());
        }
        Ok(())
    }
}

/// Insert `vector` at `vertex_id` into the index.
///
/// In addition to inserting the vector in the store this method will also choose edges for the new
/// vertex, insert back edges to maintain the undirected property of the graph, and potentially
/// prune out edges in backlink nodes to maintain the max_edges limit.
///
/// This method assumes that `vertex_id` does not already exist.
fn insert_internal<F: FnMut(i64) -> bool>(
    vertex_id: i64,
    vector: &[f32],
    index: &impl GraphVectorIndex,
    options: GraphSearchOptions<F>,
) -> Result<()> {
    // TODO: make this an error instead of panicking.
    assert_eq!(index.config().dimensions.get(), vector.len());

    // The graph search prepares the query itself, so hand it the raw vector. For encoding we need
    // the centered form that matches every stored vector.
    let prepared: &[f32] =
        &vectors::prepare_vector(vector, None, false, index.config().centroid.as_deref());

    let options = options.return_seen(true);
    let mut searcher = GraphSearcher::new(index.config().index_search_params);
    let (mut candidate_edges, _) = searcher.search_with_options(vector, options, index)?;
    let mut graph = index.graph()?;
    if candidate_edges.is_empty() {
        graph.set_entry_point(vertex_id)?;
    }

    let mut pruning_config = index.config().pruning;
    if index.config().edge_type == EdgeType::Undirected && !candidate_edges.is_empty() {
        // For undirected graphs prune for alpha-RNG but otherwise keep everything. Back edges for
        // anything we insert may result in pruning of `vertex_id` and we would like to saturate
        // the graph as best we can.
        pruning_config.max_edges = NonZero::new(candidate_edges.len()).unwrap();
    }
    let edge_set_distance_computer = EdgeSetDistanceComputer::new(index, &candidate_edges)?;
    let selected_len = prune_edges(
        &mut candidate_edges,
        &pruning_config,
        edge_set_distance_computer,
    );
    candidate_edges.truncate(selected_len);

    let mut nav_vectors = index.nav_vectors()?;
    nav_vectors.set(vertex_id, nav_vectors.new_coder().encode(prepared))?;
    if let Some(vectors) = index.rerank_vectors() {
        let mut vectors = vectors?;
        vectors.set(vertex_id, vectors.new_coder().encode(prepared))?;
    }

    let mut vertex_buf = VertexBuffer::new(index)?;
    // Ensure the edge row for vertex_id is written even when there are no candidate edges; callers
    // rely on the row existing (search treats a missing edge row as not found).
    vertex_buf.read_edges(vertex_id)?;
    for e in candidate_edges {
        vertex_buf.insert_edge_directed(vertex_id, e.vertex())?;
        vertex_buf.insert_edge_directed(e.vertex(), vertex_id)?;
        // Undirected graphs may have back edges from existing vertexes pruned. If the inserted
        // vertex is saturated, prune buffered edges and we may continue inserting if the vertex
        // is no longer saturated.
        if vertex_buf.is_saturated(vertex_id).unwrap_or(false) {
            vertex_buf.prune_buffered()?;
            if vertex_buf.is_saturated(vertex_id).unwrap_or(false) {
                break;
            }
        }
    }

    vertex_buf.flush()
}

pub fn delete_internal(vertex_id: i64, index: &impl GraphVectorIndex) -> Result<()> {
    let mut graph = index.graph()?;
    let mut vectors = index.high_fidelity_vectors()?;

    let ep = graph.entry_point().transpose()?;
    if ep == Some(vertex_id) {
        // The entry point is being deleted: search for the nearest neighbor to vertex_id's highest
        // fidelity vector, excluding vertex_id itself, and promote the best result to be the new
        // entry point. The search is seeded with vertex_id's neighbors because the entry point is
        // filtered out of traversal and cannot seed the search itself. If there are no results then
        // the graph is empty; remove the entry point.
        let seeds = graph
            .edges(vertex_id)
            .transpose()?
            .map(|e| e.collect::<Vec<_>>())
            .unwrap_or_default();
        let encoded = vectors.get(vertex_id).expect("row exists")?.to_vec();
        let query = vectors.new_coder().decode(&encoded);
        let mut searcher = GraphSearcher::new(index.config().index_search_params);
        let options = GraphSearchOptions::with_filter(|id| id != vertex_id).with_seeds(seeds);
        let (results, _) = searcher.search_with_options(&query, options, index)?;
        match results.first() {
            Some(neighbor) => graph.set_entry_point(neighbor.vertex())?,
            None => graph.remove_entry_point()?,
        }
    }

    let edges = graph.remove_vertex(vertex_id)?;
    index.nav_vectors()?.remove(vertex_id)?;
    if let Some(vectors) = index.rerank_vectors() {
        vectors?.remove(vertex_id)?;
    }

    let mut vertex_buf = VertexBuffer::new(index)?;
    match index.config().edge_type {
        EdgeType::Undirected => delete_vector_undirected(vertex_id, edges, &mut vertex_buf),
        EdgeType::Directed => delete_vector_directed(vertex_id, edges, &mut vertex_buf),
    }?;
    vertex_buf.flush()
}

fn delete_vector_undirected<I: GraphVectorIndex>(
    vertex_id: i64,
    edges: Vec<i64>,
    vertex_buf: &mut VertexBuffer<'_, I>,
) -> Result<()> {
    for &e in edges.iter() {
        vertex_buf.remove_edge_directed(e, vertex_id)?;
    }

    // Insert all possible pairings of the edges from the deleted vertex.
    for (i, &src) in edges.iter().enumerate() {
        for &dst in edges.iter().skip(i + 1) {
            if !vertex_buf.edge_exists(src, dst)? {
                vertex_buf.insert_edge_directed(src, dst)?;
                vertex_buf.insert_edge_directed(dst, src)?;
            }
        }
    }

    Ok(())
}

/// Delete a vector in a directed graph.
///
/// This utilizes Inplace Delete (Algorithm 6) from https://www.vldb.org/pvldb/vol18/p5166-upreti.pdf
fn delete_vector_directed<I: GraphVectorIndex>(
    vertex_id: i64,
    edges: Vec<i64>,
    vertex_buf: &mut VertexBuffer<'_, I>,
) -> Result<()> {
    let mut graph = vertex_buf.index().graph()?;
    let mut vectors = vertex_buf.index().high_fidelity_vectors()?;
    let distance_fn = vectors.new_distance_function();

    // Build a map of vertices that reference vertex_id, searching within 2 hops.
    // Track remaining edges (after removing vertex_id) to know which vertices need replacement edges.
    let mut seen_vertexes: HashMap<i64, Vec<i64>> = HashMap::new();
    for &v in edges.iter() {
        let vedges = graph
            .edges(v)
            .transpose()?
            .map(|e| e.collect::<Vec<_>>())
            .unwrap_or_default();
        // Remove the edge to vertex_id and track remaining edges.
        vertex_buf.remove_edge_directed(v, vertex_id)?;
        // Always visit 2-hop neighbors (even when v has no remaining edges) so that vertices
        // reachable only through v are discovered and can get their own vertex_id edge removed.
        if !vedges.is_empty() {
            seen_vertexes.entry(v).or_default();
        }
        for &vv in vedges.iter() {
            if let Entry::Vacant(entry) = seen_vertexes.entry(vv) {
                let vvedges = graph
                    .edges(vv)
                    .transpose()?
                    .map(|e| e.collect::<Vec<_>>())
                    .unwrap_or_default();
                // Remove the edge to vertex_id and track remaining edges.
                vertex_buf.remove_edge_directed(vv, vertex_id)?;
                if !vvedges.is_empty() {
                    entry.insert(vvedges);
                }
            }
        }
    }

    // Each vertex that I removed an edge from may get replacement edges.
    // Fetch these vectors since they will be used repeatedly.
    let mut replacement_candidates = Vec::with_capacity(edges.len());
    for e in edges {
        if let Some(v) = vectors.get(e).transpose()? {
            replacement_candidates.push((e, v.to_vec()));
        }
    }

    // For each vertex that we removed vertex_id from, score all of the replacement candidates
    // and insert the top candidates via VertexBuffer.
    let mut replacements = Vec::with_capacity(replacement_candidates.len());
    for (id, remaining_edges) in seen_vertexes.iter_mut() {
        replacements.clear();

        let cvector = vectors.get(*id).expect("row exists")?.to_vec();
        for (rid, rv) in replacement_candidates.iter() {
            // Skip anything that exists already in the edge set.
            if !remaining_edges.contains(rid) {
                replacements.push(Neighbor::new(*rid, distance_fn.distance(&cvector, rv)));
            }
        }

        if replacements.is_empty() {
            continue;
        }

        if replacements.len() > 4 {
            replacements.select_nth_unstable(3);
        }
        for c in replacements.iter().take(4).map(|n| n.vertex()) {
            vertex_buf.insert_edge_directed(*id, c)?;
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
        EdgePruningConfig, EdgeType, Graph, GraphConfig, GraphSearchParams, GraphVectorIndex,
        mutate::{delete_vector, insert_vector, upsert_vector, upsert_vector_with_options},
        search::{GraphSearcher, Options as GraphSearchOptions},
        wt::{TableGraphVectorIndex, TransactionGraphVectorIndex},
    };

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
                patience: None,
            }
        }

        fn new_txn_index(&self) -> TransactionGraphVectorIndex {
            TransactionGraphVectorIndex::new(
                self.index.clone(),
                self.conn.begin_transaction(None).unwrap(),
            )
        }

        fn insert_many(&self, vectors: &[[f32; 2]]) -> Result<Vec<i64>> {
            let index = self.new_txn_index();
            vectors
                .iter()
                .map(|v| insert_vector(v.as_ref(), &index))
                .collect::<Result<Vec<_>>>()
                .and_then(|ids| index.commit(None).map(|_| ids))
        }

        fn search(&self, query: &[f32]) -> Result<Vec<i64>> {
            let mut searcher = GraphSearcher::new(Self::search_params());
            let reader = self.new_txn_index();
            searcher
                .search(query, &reader)
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
                        centroid: None,
                        edge_type: EdgeType::Undirected,
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

        let reader = fixture.new_txn_index();
        assert_eq!(reader.graph()?.entry_point(), None);
        let mut searcher = GraphSearcher::new(Fixture::search_params());
        assert_eq!(searcher.search(&[0.5, -0.5], &reader), Ok(vec![]));
        Ok(())
    }

    #[test]
    fn insert_one() -> Result<()> {
        let fixture = Fixture::default();

        let index = fixture.new_txn_index();
        let id = insert_vector(&[0.0, 0.0], &index)?;
        index.commit(None)?;

        assert_eq!(id, 0);
        assert_eq!(fixture.new_txn_index().graph()?.entry_point(), Some(Ok(id)));
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

        let reader = fixture.new_txn_index();
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
        let txn_index = fixture.new_txn_index();
        delete_vector(vertex_ids[1], &txn_index)?;
        txn_index.commit(None)?;

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

        let reader = fixture.new_txn_index();
        let mut graph = reader.graph()?;
        assert_eq!(
            graph.edges(vertex_ids[0]).unwrap()?.collect::<Vec<_>>(),
            &[1, 2, 3, 5]
        );
        assert_eq!(fixture.search(&[0.0, 0.0]), Ok(vec![0, 1, 2, 3, 5, 4]));

        let txn_index = fixture.new_txn_index();
        delete_vector(1, &txn_index)?;
        txn_index.commit(None)?;

        let reader = fixture.new_txn_index();
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

        let txn_index = fixture.new_txn_index();
        let entry_id = insert_vector(&[0.0, 0.0], &txn_index)?;
        let next_entry_id = insert_vector(&[0.5, 0.5], &txn_index)?;
        insert_vector(&[1.0, 1.0], &txn_index)?;

        delete_vector(entry_id, &txn_index)?;
        assert_eq!(txn_index.graph()?.entry_point(), Some(Ok(next_entry_id)));
        txn_index.commit(None)?;

        assert_eq!(fixture.search(&[0.0, 0.0])?, vec![1, 2]);

        Ok(())
    }

    #[test]
    fn delete_only_point() -> Result<()> {
        let fixture = Fixture::default();

        let txn_index = fixture.new_txn_index();
        let id = insert_vector(&[0.0, 0.0], &txn_index)?;
        txn_index.commit(None)?;
        assert_eq!(fixture.search(&[0.0, 0.0])?, vec![id]);

        let txn_index = fixture.new_txn_index();
        delete_vector(id, &txn_index)?;
        txn_index.commit(None)?;
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

        let txn_index = fixture.new_txn_index();
        upsert_vector(vertex_ids[0], &[1.0, 1.0], &txn_index)?;
        txn_index.commit(None)?;

        assert_eq!(fixture.search(&[0.0, 0.0]), Ok(vec![1, 2, 3, 5, 4, 0]));

        Ok(())
    }

    // Verify that a filter passed via upsert_vector_with_options excludes the filtered vertex
    // from selection as an edge, even when it would otherwise be the closest candidate.
    #[test]
    fn upsert_with_options_filter_excludes_vertex() -> Result<()> {
        let fixture = Fixture::default();

        let vertex_ids = fixture.insert_many(&[
            [0.0, 0.0],
            [0.1, 0.1],
            [-0.1, 0.1],
            [0.1, -0.1],
            [-0.2, -0.2],
            [-0.1, -0.1],
        ])?;
        let reader = fixture.new_txn_index();
        let mut graph = reader.graph()?;
        assert!(
            graph
                .edges(vertex_ids[0])
                .unwrap()?
                .collect::<Vec<_>>()
                .contains(&vertex_ids[1])
        );

        let txn_index = fixture.new_txn_index();
        upsert_vector_with_options(
            vertex_ids[0],
            &[0.0, 0.0],
            &txn_index,
            GraphSearchOptions::with_filter(|i: i64| i != vertex_ids[1]),
        )?;
        txn_index.commit(None)?;

        let reader = fixture.new_txn_index();
        let mut graph = reader.graph()?;
        let edges = graph.edges(vertex_ids[0]).unwrap()?.collect::<Vec<_>>();
        assert!(
            !edges.contains(&vertex_ids[1]),
            "filtered vertex should never be selected as an edge, got {edges:?}"
        );
        // The rest of the graph should still be reachable through the other edges.
        assert_eq!(
            fixture
                .search(&[0.0, 0.0])?
                .into_iter()
                .collect::<std::collections::HashSet<_>>(),
            vertex_ids
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
        );

        Ok(())
    }

    struct DirectedFixture {
        index: Arc<TableGraphVectorIndex>,
        conn: Arc<Connection>,
        _dir: tempfile::TempDir,
    }

    impl DirectedFixture {
        fn search_params() -> GraphSearchParams {
            GraphSearchParams {
                beam_width: NonZero::new(16).unwrap(),
                num_rerank: 16,
                patience: None,
            }
        }

        fn new_txn_index(&self) -> TransactionGraphVectorIndex {
            TransactionGraphVectorIndex::new(
                self.index.clone(),
                self.conn.begin_transaction(None).unwrap(),
            )
        }

        fn insert_many(&self, vectors: &[[f32; 2]]) -> Result<Vec<i64>> {
            let index = self.new_txn_index();
            vectors
                .iter()
                .map(|v| insert_vector(v.as_ref(), &index))
                .collect::<Result<Vec<_>>>()
                .and_then(|ids| index.commit(None).map(|_| ids))
        }

        fn search(&self, query: &[f32]) -> Result<Vec<i64>> {
            let mut searcher = GraphSearcher::new(Self::search_params());
            let reader = self.new_txn_index();
            searcher
                .search(query, &reader)
                .map(|neighbors| neighbors.into_iter().map(|n| n.vertex()).collect())
        }
    }

    impl Default for DirectedFixture {
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
                        centroid: None,
                        edge_type: EdgeType::Directed,
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
    fn directed_empty_graph() -> Result<()> {
        let fixture = DirectedFixture::default();

        let reader = fixture.new_txn_index();
        assert_eq!(reader.graph()?.entry_point(), None);
        let mut searcher = GraphSearcher::new(DirectedFixture::search_params());
        assert_eq!(searcher.search(&[0.5, -0.5], &reader), Ok(vec![]));
        Ok(())
    }

    #[test]
    fn directed_insert_one() -> Result<()> {
        let fixture = DirectedFixture::default();

        let index = fixture.new_txn_index();
        let id = insert_vector(&[0.0, 0.0], &index)?;
        index.commit(None)?;

        assert_eq!(id, 0);
        assert_eq!(fixture.new_txn_index().graph()?.entry_point(), Some(Ok(id)));
        assert_eq!(fixture.search(&[1.0, 1.0]), Ok(vec![id]));
        Ok(())
    }

    #[test]
    fn directed_insert_two() -> Result<()> {
        let fixture = DirectedFixture::default();

        fixture.insert_many(&[[0.0, 0.0], [0.5, 0.5]])?;
        assert_eq!(fixture.search(&[1.0, 1.0]), Ok(vec![1, 0]));
        Ok(())
    }

    // Verify that inserting a vertex also adds best-effort back edges to its chosen neighbors.
    #[test]
    fn directed_back_edges_added_on_insert() -> Result<()> {
        let fixture = DirectedFixture::default();

        let ids = fixture.insert_many(&[[0.0, 0.0], [1.0, 1.0]])?;

        let reader = fixture.new_txn_index();
        let mut graph = reader.graph()?;
        let edges_0: Vec<i64> = graph.edges(ids[0]).unwrap()?.collect();
        let edges_1: Vec<i64> = graph.edges(ids[1]).unwrap()?.collect();
        // vertex 1 was inserted second and selects vertex 0 as a forward edge.
        assert!(
            edges_1.contains(&ids[0]),
            "vertex 1 should have a forward edge to vertex 0"
        );
        // vertex 0 should have received a back edge to vertex 1.
        assert!(
            edges_0.contains(&ids[1]),
            "vertex 0 should have received a back edge to vertex 1"
        );
        Ok(())
    }

    #[test]
    fn directed_insert_to_prune() -> Result<()> {
        let fixture = DirectedFixture::default();

        fixture.insert_many(&[
            [0.0, 0.0],
            [0.1, 0.1],
            [-0.1, 0.1],
            [0.1, -0.1],
            [-0.2, -0.2],
            [-0.1, -0.1],
        ])?;

        assert_eq!(fixture.search(&[0.0, 0.0]), Ok(vec![0, 1, 2, 3, 5, 4]));
        Ok(())
    }

    #[test]
    fn directed_delete_one() -> Result<()> {
        let fixture = DirectedFixture::default();

        let vertex_ids = fixture.insert_many(&[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])?;
        let txn_index = fixture.new_txn_index();
        delete_vector(vertex_ids[1], &txn_index)?;
        txn_index.commit(None)?;

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

    // Delete a hub vertex and verify the graph remains fully searchable.
    #[test]
    fn directed_delete_relink() -> Result<()> {
        let fixture = DirectedFixture::default();

        fixture.insert_many(&[
            [0.0, 0.0],
            [0.1, 0.1],
            [-0.1, 0.1],
            [0.1, -0.1],
            [-0.2, -0.2],
            [-0.1, -0.1],
        ])?;
        assert_eq!(fixture.search(&[0.0, 0.0]), Ok(vec![0, 1, 2, 3, 5, 4]));

        let txn_index = fixture.new_txn_index();
        delete_vector(1, &txn_index)?;
        txn_index.commit(None)?;

        assert_eq!(fixture.search(&[0.0, 0.0]), Ok(vec![0, 2, 3, 5, 4]));
        Ok(())
    }

    #[test]
    fn directed_delete_entry_point() -> Result<()> {
        let fixture = DirectedFixture::default();

        let txn_index = fixture.new_txn_index();
        let entry_id = insert_vector(&[0.0, 0.0], &txn_index)?;
        let next_entry_id = insert_vector(&[0.5, 0.5], &txn_index)?;
        insert_vector(&[1.0, 1.0], &txn_index)?;

        delete_vector(entry_id, &txn_index)?;
        assert_eq!(txn_index.graph()?.entry_point(), Some(Ok(next_entry_id)));
        txn_index.commit(None)?;

        assert_eq!(fixture.search(&[0.0, 0.0])?, vec![1, 2]);
        Ok(())
    }

    #[test]
    fn directed_delete_only_point() -> Result<()> {
        let fixture = DirectedFixture::default();

        let txn_index = fixture.new_txn_index();
        let id = insert_vector(&[0.0, 0.0], &txn_index)?;
        txn_index.commit(None)?;
        assert_eq!(fixture.search(&[0.0, 0.0])?, vec![id]);

        let txn_index = fixture.new_txn_index();
        delete_vector(id, &txn_index)?;
        txn_index.commit(None)?;
        assert_eq!(fixture.search(&[0.0, 0.0])?, Vec::<i64>::new());
        Ok(())
    }

    #[test]
    fn directed_upsert() -> Result<()> {
        let fixture = DirectedFixture::default();

        let vertex_ids = fixture.insert_many(&[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])?;
        assert_eq!(fixture.search(&[0.0, 0.0]), Ok(vec![0, 1, 2]));

        let txn_index = fixture.new_txn_index();
        upsert_vector(vertex_ids[0], &[2.0, 2.0], &txn_index)?;
        txn_index.commit(None)?;

        // vertex 0 moved far away; vertex 1 is now nearest to [0.0, 0.0]
        let results = fixture.search(&[0.0, 0.0])?;
        assert_eq!(results[0], vertex_ids[1]);
        Ok(())
    }
}
