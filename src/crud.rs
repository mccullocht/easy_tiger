use std::{collections::BTreeSet, num::NonZero};

use crate::{
    graph::{Graph, GraphVectorIndexReader, GraphVertex},
    quantization::binary_quantize,
    scoring::F32VectorScorer,
    search::GraphSearcher,
    wt::{
        encode_graph_node, encode_graph_node_internal, WiredTigerGraph,
        WiredTigerGraphVectorIndexReader,
    },
    Neighbor,
};
use wt_mdb::{Error, Result};

pub struct CrudGraph {
    reader: WiredTigerGraphVectorIndexReader,
    searcher: GraphSearcher,
}

// XXX I think I need a queue for pruning backlinks: (src, dst) and use a priority
// queue to order it so that I avoid re-reading the same vertex.
impl CrudGraph {
    /// Insert a vertex for `vector`. Returns the assigned id.
    pub fn insert(&mut self, vector: &[f32]) -> Result<i64> {
        let vertex_id = self.assign_vertex_id()?;

        let scorer = self.reader.metadata().new_scorer();
        // XXX does this normalize the vector? I can't remember.
        let mut candidate_edges = self.searcher.search(vector, &mut self.reader)?;
        let mut graph = self.reader.graph()?;
        if candidate_edges.is_empty() {
            // XXX update the entry point. it's fine to fall through and do the rest of this.
            todo!()
        }
        let selected_len = Self::prune(
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

        // XXX should group if there are common src vertices.
        for (src_vertex_id, dst_vertex_id) in pruned_edges {
            self.delete_edge(&mut graph, src_vertex_id, dst_vertex_id)?;
        }

        Ok(vertex_id)
    }

    // XXX integrate this into the wt graph impl?
    fn assign_vertex_id(&mut self) -> Result<i64> {
        self.reader
            .session()
            .get_record_cursor(self.reader.index().graph_table_name())?
            .largest_key()
            .unwrap_or(Err(Error::not_found_error()))
            .map(|i| i + 1)
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
            let selected_len = Self::prune(
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

    fn delete_edge(
        &self,
        graph: &mut WiredTigerGraph<'_>,
        src_vertex_id: i64,
        dst_vertex_id: i64,
    ) -> Result<()> {
        let vertex = graph
            .get(src_vertex_id)
            .unwrap_or(Err(Error::not_found_error()))?;
        let edges = vertex.edges().filter(|v| *v != dst_vertex_id).collect();
        let encoded = encode_graph_node_internal(vertex.vector_bytes(), edges);
        graph.set(src_vertex_id, &encoded)
    }

    pub fn delete(&mut self, vertex_id: i64) -> Result<()> {
        let scorer = self.reader.metadata().new_scorer();
        let mut graph = self.reader.graph()?;

        let edges = graph
            .get(vertex_id)
            .unwrap_or(Err(Error::not_found_error()))?
            .edges()
            .collect::<Vec<_>>();

        graph.remove(vertex_id)?;
        self.reader.nav_vectors()?.remove(vertex_id)?;

        // Cache information about each vertex linked to vertex_id.
        // Remove any links back to vertex_id.
        let mut vertices = edges
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

        // Score all pairs of vectors among the edges of vertex_id that are not already connected.
        let mut candidate_links = vertices
            .iter()
            .enumerate()
            .flat_map(|(src, (_, src_vector, src_edges))| {
                vertices
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
            if vertices[src].2.len() < max_edges && vertices[dst].2.len() < max_edges {
                let src_id = vertices[src].0;
                let dst_id = vertices[dst].0;
                vertices[src].2.push(dst_id);
                vertices[dst].2.push(src_id);
            }
        }

        // Write all the mutated nodes back to WT.
        for (vertex_id, vector, edges) in vertices {
            graph.set(vertex_id, &encode_graph_node(&vector, edges))?;
        }

        Ok(())
    }

    pub fn update(&mut self, _vertex_id: i64, _vector: &[f32]) -> Result<()> {
        todo!()
    }

    // XXX share this with the bulk loader implementation.
    fn prune(
        edges: &mut [Neighbor],
        max_edges: NonZero<usize>,
        graph: &mut impl Graph,
        scorer: &dyn F32VectorScorer,
    ) -> Result<usize> {
        if edges.is_empty() {
            return Ok(0);
        }

        // TODO: replace with a fixed length bitset
        let mut selected = BTreeSet::new();
        selected.insert(0); // we always keep the first node.
        for alpha in [1.0, 1.2] {
            for (i, e) in edges.iter().enumerate().skip(1) {
                if selected.contains(&i) {
                    continue;
                }

                // TODO: fix error handling so we can reuse this elsewhere.
                let e_vec = graph
                    .get(e.vertex())
                    .unwrap_or(Err(Error::not_found_error()))?
                    .vector()
                    .into_owned();
                for p in selected.iter().take_while(|j| **j < i).map(|j| edges[*j]) {
                    let p_node = graph
                        .get(p.vertex())
                        .unwrap_or(Err(Error::not_found_error()))?;
                    if scorer.score(&e_vec, &p_node.vector()) > e.score * alpha {
                        selected.insert(i);
                        break;
                    }
                }

                if selected.len() >= max_edges.get() {
                    break;
                }
            }

            if selected.len() >= max_edges.get() {
                break;
            }
        }

        // Partition edges into selected and unselected.
        for (i, j) in selected.iter().enumerate() {
            edges.swap(i, *j);
        }

        Ok(selected.len())
    }
}
