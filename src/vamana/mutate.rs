//! Tools for mutating a vamana graph vector index.
use crate::vamana::{
    prune_edges, prune_edges_saturating,
    search::{GraphSearcher, Options as GraphSearchOptions},
    EdgePruningConfig, EdgeSetDistanceComputer, EdgeType, Graph, GraphVectorIndex, GraphVectorStore,
};
use crate::Neighbor;
use std::collections::{hash_map::Entry, HashMap, HashSet, VecDeque};
use vectors::VectorDistance;
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
    match index.config().edge_type {
        EdgeType::Undirected => delete_vector_undirected(vertex_id, index),
        EdgeType::Directed => delete_vector_directed(vertex_id, index),
    }
}

fn delete_vector_undirected(vertex_id: i64, index: &impl GraphVectorIndex) -> Result<()> {
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
                .unwrap_or_else(|| Err(Error::not_found_error()))
                .map(|edges| {
                    let vector = vectors.get(e).expect("row exists").map(|rv| rv.to_vec());
                    vector.map(|rv| (e, rv, edges.filter(|d| *d != vertex_id).collect::<Vec<_>>()))
                })
        })
        .collect::<Result<Result<Vec<_>>>>()??;

    // Repair connectivity around the deleted node by gathering candidates for each of its edges
    // and linking them, following the Wolverine++ candidate collection policy.
    repair_deleted_vertex(
        index,
        &mut graph,
        &mut vectors,
        distance_fn.as_ref(),
        &vector,
        &vertex_data,
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

/// Delete a vector in a directed graph.
///
/// This utilizes Inplace Delete (Algorithm 6) from https://www.vldb.org/pvldb/vol18/p5166-upreti.pdf
fn delete_vector_directed(vertex_id: i64, index: &impl GraphVectorIndex) -> Result<()> {
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

    // A candidate vertex to reprocess. Reprocess if there is an edge to vertex_id, saving all of
    // the other edges, otherwise skip.
    enum Vertex {
        Skip,
        Reprocess(Vec<i64>),
    }

    impl Vertex {
        fn new(delete_vertex_id: i64, mut edges: Vec<i64>) -> Self {
            if let Some(pos) = edges.iter().position(|e| *e == delete_vertex_id) {
                edges.remove(pos);
                Self::Reprocess(edges)
            } else {
                Self::Skip
            }
        }
    }

    // Cast a net looking for vertexes that reference vertex_id, searching within 2 hops.
    let mut seen_vertexes: HashMap<i64, Vertex> = HashMap::new();
    for v in edges.iter() {
        let Some(vedges) = graph.edges(*v).transpose()?.map(|e| e.collect::<Vec<_>>()) else {
            continue;
        };
        seen_vertexes
            .entry(*v)
            .or_insert_with(|| Vertex::new(vertex_id, vedges.clone()));
        for vv in vedges.iter() {
            if let Entry::Vacant(entry) = seen_vertexes.entry(*vv) {
                let Some(vvedges) = graph.edges(*vv).transpose()?.map(|e| e.collect::<Vec<_>>())
                else {
                    continue;
                };
                entry.insert(Vertex::new(vertex_id, vvedges));
            }
        }
    }

    // Each vertex that I removed an edge from may get replacement edges
    // Fetch these vectors since they will be used repeatedly.
    let mut replacement_candidates = Vec::with_capacity(edges.len());
    for e in edges {
        if let Some(v) = vectors.get(e).transpose()? {
            replacement_candidates.push((e, v.to_vec()));
        }
    }

    // For each vertex that we removed vertex_id from, score all of the replacement candidates
    // and select some of them into the edge set, pruning if needed.
    let mut replacements = Vec::with_capacity(replacement_candidates.len());
    for (id, vertex) in seen_vertexes.iter_mut() {
        let Vertex::Reprocess(edges) = vertex else {
            continue;
        };
        edges.sort_unstable();
        replacements.clear();

        let cvector = vectors.get(*id).expect("row exists")?.to_vec();
        for (rid, rv) in replacement_candidates.iter() {
            // Skip anything that exists already in the edge set.
            if !edges.contains(rid) {
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
            if let Err(i) = edges.binary_search(&c) {
                edges.insert(i, c);
            }
        }

        // If there are now too many edges, rehydrate the edge list and prune.
        if edges.len() > index.config().pruning.max_edges.get() {
            let (neighbors, keep) =
                rehydrate_and_prune_directed(&cvector, &mut vectors, edges, &index.config().pruning)?;
            edges.clear();
            edges.extend(neighbors.iter().take(keep).map(Neighbor::vertex));
        }

        graph.set_edges(*id, edges.to_vec())?;
    }

    // Oh no, we've deleted the entry point! Find the closest point amongst the edges of this node
    // to use as a new entry point.
    // TODO: consider all of seen_vertexes instead since they pointed to the removed vertex.
    if graph
        .entry_point()
        .expect("there was at least one vertex")?
        == vertex_id
    {
        let mut neighbors = replacement_candidates
            .iter()
            .map(|(id, vec)| Neighbor::new(*id, distance_fn.distance(&vector, vec)))
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

    let mut searcher = GraphSearcher::new(index.config().index_search_params);
    let mut candidate_edges = searcher.search_with_options(vector, options, index)?;
    let mut graph = index.graph()?;
    if candidate_edges.is_empty() {
        graph.set_entry_point(vertex_id)?;
    }

    let edge_set_distance_computer = EdgeSetDistanceComputer::new(index, &candidate_edges)?;
    let selected_len = prune_edges_saturating(
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
    let mut pruned_edges = vec![];
    for src_vertex_id in candidate_edges.into_iter().map(|n| n.vertex()) {
        let edges = insert_edge_directed(
            index,
            &mut graph,
            &mut vectors,
            src_vertex_id,
            vertex_id,
            &mut pruned_edges,
        )?;
        graph.set_edges(src_vertex_id, edges)?;

        if index.config().edge_type == EdgeType::Undirected {
            for (src_vertex_id, dst_vertex_id) in pruned_edges.drain(..) {
                remove_edge_directed(&mut graph, src_vertex_id, dst_vertex_id)?;
            }
        }
    }

    Ok(())
}

/// Rehydrates `edges` as neighbors with distances from `vertex_vector`, skipping any dangling
/// edges, and prunes if needed. Returns `(neighbors, keep)` where `neighbors[..keep]` are the
/// selected edges in ascending distance order and `neighbors[keep..]` are the pruned ones.
fn rehydrate_and_prune_directed(
    vertex_vector: &[u8],
    vectors: &mut impl GraphVectorStore,
    edges: &[i64],
    config: &EdgePruningConfig,
) -> Result<(Vec<Neighbor>, usize)> {
    let (mut neighbors, computer) =
        EdgeSetDistanceComputer::from_directed_edges(vertex_vector, vectors, edges)?;
    let keep = if neighbors.len() > config.max_edges.get() {
        prune_edges(&mut neighbors, config, computer)
    } else {
        neighbors.len()
    };
    Ok((neighbors, keep))
}

/// Attempt to insert a directed edge from `src_vertex_id` to `dst_vertex_id` and return the set of
/// edges for `src_vertex_id` after the insertion attempt.
///
/// If the edge already exists in the graph then no change is made. If inserting the edges would
/// exceed max_edges then the edges are pruned according to policy. This process may result in the
/// inserted edges being dropped. Pruning will also fill `pruned_edges` with any back edges that
/// need to be removed to maintain an undirected graph.
fn insert_edge_directed(
    index: &impl GraphVectorIndex,
    graph: &mut impl Graph,
    vectors: &mut impl GraphVectorStore,
    src_vertex_id: i64,
    dst_vertex_id: i64,
    pruned_edges: &mut Vec<(i64, i64)>,
) -> Result<Vec<i64>> {
    let mut edges = graph
        .edges(src_vertex_id)
        .unwrap_or_else(|| Err(Error::not_found_error()))?
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
    let (neighbors, selected_len) =
        rehydrate_and_prune_directed(&src_vector, vectors, &edges, &index.config().pruning)?;
    for v in neighbors.iter().skip(selected_len).map(Neighbor::vertex) {
        pruned_edges.push((v, src_vertex_id));
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
        .unwrap_or_else(|| Err(Error::not_found_error()))?
        .filter(|v| *v != dst_vertex_id)
        .collect::<Vec<_>>();
    graph.set_edges(src_vertex_id, edges)
}

/// Repair connectivity around a deleted vertex using the Wolverine++ candidate collection policy
/// from https://www.vldb.org/pvldb/vol18/p2268-zheng.pdf.
///
/// `peers` is a list of `(vertex_id, vector, edges)` for each out-neighbor of the deleted vertex
/// `p` (whose vector is `deleted_vector`), with edges back to `p` already removed. For each such
/// out-neighbor `p_out`, we gather a set of high quality candidate vertices and attempt to connect
/// each of them to `p_out`. Candidates are seeded from the other out-neighbors of `p` that are
/// closer to `p_out` than `p` was, then expanded through the graph, keeping only nodes that fall in
/// the candidate region `CR(p, p_out)`.
fn repair_deleted_vertex(
    index: &impl GraphVectorIndex,
    graph: &mut impl Graph,
    vectors: &mut impl GraphVectorStore,
    distance_fn: &dyn VectorDistance,
    deleted_vector: &[u8],
    peers: &[(i64, Vec<u8>, Vec<i64>)],
) -> Result<()> {
    // Maximum candidate set size C_s. Pruning bounds the degree at max_edges regardless.
    let max_candidates = index.config().pruning.max_edges.get();

    // Cache of vertex vectors fetched during expansion, seeded with the peer vectors.
    let mut vector_cache: HashMap<i64, Vec<u8>> =
        peers.iter().map(|(id, v, _)| (*id, v.clone())).collect();

    for (pout_id, pout_vec, pout_edges) in peers.iter() {
        let d_p_pout = distance_fn.distance(deleted_vector, pout_vec);

        // Vertices already chosen to connect to p_out, or excluded from selection (p_out itself
        // and its existing edges).
        let mut selected: HashSet<i64> = HashSet::new();
        selected.insert(*pout_id);
        selected.extend(pout_edges.iter().copied());

        // Candidate vertices to connect to p_out.
        let mut candidates: Vec<i64> = Vec::new();

        // Seed with the other out-neighbors of p that are closer to p_out than p is. Conditions are
        // relaxed here (candidate region not required) since few peers fall in the region.
        for (cid, cvec, _) in peers.iter() {
            if selected.contains(cid) {
                continue;
            }
            if distance_fn.distance(cvec, pout_vec) < d_p_pout {
                selected.insert(*cid);
                candidates.push(*cid);
            }
        }

        // Expand over the two-hop neighborhood of p. The expansion frontier is seeded with all
        // one-hop neighbors of p so that the full Hop2(p) candidate pool is examined, not just the
        // relaxed seeds above; nodes admitted from the two-hop region are themselves expanded until
        // the candidate set reaches C_s. CR(p, p_out) does the filtering.
        let mut expansion_queue: VecDeque<i64> = peers.iter().map(|(id, _, _)| *id).collect();
        let mut expanded: HashSet<i64> = HashSet::new();
        while let Some(cid) = expansion_queue.pop_front() {
            if candidates.len() >= max_candidates {
                break;
            }
            if !expanded.insert(cid) {
                continue;
            }
            let Some(neighbors) = graph.edges(cid).transpose()?.map(|e| e.collect::<Vec<_>>())
            else {
                continue;
            };
            for n in neighbors {
                if candidates.len() >= max_candidates {
                    break;
                }
                if selected.contains(&n) {
                    continue;
                }
                let nvec = match vector_cache.get(&n) {
                    Some(v) => v.clone(),
                    None => {
                        let Some(v) = vectors.get(n).transpose()? else {
                            continue;
                        };
                        let v = v.to_vec();
                        vector_cache.insert(n, v.clone());
                        v
                    }
                };
                if in_candidate_region(distance_fn, deleted_vector, pout_vec, &nvec, d_p_pout) {
                    selected.insert(n);
                    candidates.push(n);
                    expansion_queue.push_back(n);
                }
            }
        }

        // Connect each gathered candidate to p_out. Pruning maintains the degree bound.
        for cid in candidates {
            insert_undirected_edge(index, graph, vectors, cid, *pout_id)?;
        }
    }

    Ok(())
}

/// Whether candidate `c` lies in the Wolverine++ candidate region `CR(p, p_out)`:
///   `d(c, p) > d(p, p_out) ∧ d(c, p_out) < d(p, p_out) ∧ d²(c,p_out) + d²(p,p_out) > d²(c,p)`
///
/// The first condition keeps candidates far enough from the deleted vertex that they are minimally
/// affected by its removal, the second ensures the new edge is short enough to repair the search
/// path, and the third (a law-of-cosines angle test) prefers candidates positioned so a single edge
/// repairs as many disrupted paths as possible.
fn in_candidate_region(
    distance_fn: &dyn VectorDistance,
    p: &[u8],
    pout: &[u8],
    c: &[u8],
    d_p_pout: f64,
) -> bool {
    let d_c_p = distance_fn.distance(c, p);
    let d_c_pout = distance_fn.distance(c, pout);
    d_c_p > d_p_pout
        && d_c_pout < d_p_pout
        && d_c_pout * d_c_pout + d_p_pout * d_p_pout > d_c_p * d_c_p
}

/// Attempt to insert an undirected edge between `a` and `b`, pruning as needed. If pruning drops the
/// edge in either direction then no changes are committed.
fn insert_undirected_edge(
    index: &impl GraphVectorIndex,
    graph: &mut impl Graph,
    vectors: &mut impl GraphVectorStore,
    a: i64,
    b: i64,
) -> Result<()> {
    let mut pruned_edges = vec![];
    let a_edges = insert_edge_directed(index, graph, vectors, a, b, &mut pruned_edges)?;
    let b_edges = insert_edge_directed(index, graph, vectors, b, a, &mut pruned_edges)?;

    // If the edge was not inserted in both directions, do not commit any of the changes.
    if !a_edges.contains(&b) || !b_edges.contains(&a) {
        return Ok(());
    }

    graph.set_edges(a, a_edges)?;
    graph.set_edges(b, b_edges)?;
    for (src, dst) in pruned_edges {
        remove_edge_directed(graph, src, dst)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{num::NonZero, sync::Arc};

    use vectors::{F32VectorCoding, VectorSimilarity};
    use wt_mdb::{Connection, Result};

    use crate::vamana::{
        mutate::{delete_vector, insert_vector, upsert_vector, upsert_vector_with_options},
        search::{GraphSearcher, Options as GraphSearchOptions},
        wt::{TableGraphVectorIndex, TransactionGraphVectorIndex},
        EdgePruningConfig, EdgeType, Graph, GraphConfig, GraphSearchParams, GraphVectorIndex,
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

        // Deleting the collinear middle vertex leaves vertex_ids[2] in a sparse region with an
        // empty candidate region, so Wolverine++ does not repair the path to it.
        assert_eq!(fixture.search(&[0.0, 0.0])?, vec![vertex_ids[0]]);

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

        // Deleting the central hub (vertex 0) leaves its former peers in sparse regions with empty
        // candidate regions, so Wolverine++ does not cross-link them; only the reinserted vertex 0
        // and its nearest peer remain reachable from the origin.
        assert_eq!(fixture.search(&[0.0, 0.0]), Ok(vec![1, 0]));

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
        assert!(graph
            .edges(vertex_ids[0])
            .unwrap()?
            .collect::<Vec<_>>()
            .contains(&vertex_ids[1]));

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
        let edges = graph
            .edges(vertex_ids[0])
            .unwrap()?
            .collect::<Vec<_>>();
        assert!(
            !edges.contains(&vertex_ids[1]),
            "filtered vertex should never be selected as an edge, got {edges:?}"
        );
        // Deleting the central hub (vertex 0) during upsert leaves its former peers in sparse
        // regions that Wolverine++ does not repair, so only the reinserted vertex 0 remains
        // reachable from the origin.
        assert_eq!(
            fixture.search(&[0.0, 0.0])?.into_iter().collect::<std::collections::HashSet<_>>(),
            std::collections::HashSet::from([vertex_ids[0]])
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
