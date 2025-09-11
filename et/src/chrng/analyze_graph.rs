//! Analyze the contents of a clustered hierarchical relative neighbor graph.

use std::{collections::HashSet, io, sync::Arc, u32};

use clap::Args;
use easy_tiger::{chrng::ClusterKey, vamana::wt::TableGraphVectorIndex};
use wt_mdb::Connection;

#[derive(Args)]
pub struct AnalyzeGraphArgs {}

pub fn analyze_graph(
    connection: Arc<Connection>,
    index_name: &str,
    _: AnalyzeGraphArgs,
) -> io::Result<()> {
    let session = connection.open_session()?;
    let tail_index = TableGraphVectorIndex::from_db(&connection, &format!("{index_name}.tail"))?;

    let mut num_vertexes = 0usize;
    let mut vertex_outbound_cluster_sum = 0usize;

    let mut current_cluster = u32::MAX;
    let mut vertex_outbound_clusters = HashSet::new();
    let mut cluster_links = HashSet::new();
    for r in session.open_index_cursor(tail_index.graph_table_name())? {
        let (raw_key, raw_value) = r?;

        let key = ClusterKey::try_from(raw_key.as_slice()).unwrap();
        if key.cluster_id != current_cluster {
            current_cluster = key.cluster_id;
        }
        num_vertexes += 1;

        let mut cursor = io::Cursor::new(raw_value);
        let mut edge_cluster_id = None;
        let mut last_key = ClusterKey {
            cluster_id: 0,
            vector_id: 0,
        };
        let mut edge_clusters: Vec<u32> = vec![];
        let mut edges = vec![];
        while let Some(cluster_delta) = leb128::read::unsigned(&mut cursor).ok() {
            let vector_id_delta =
                leb128::read::signed(&mut cursor).expect("vector id follows cluster id");
            if cluster_delta == 0 {
                last_key.vector_id += vector_id_delta;
            } else {
                last_key.cluster_id += cluster_delta as u32;
                last_key.vector_id = vector_id_delta;
            }
            edges.push(last_key);
            let cluster_id = *edge_clusters.last().unwrap_or(&0) + cluster_delta as u32;
            edge_clusters.push(cluster_id);
            // XXX If the first outlink is to cluster 0 this won't be correct.
            if cluster_delta != 0 || edge_cluster_id.is_none() {
                let cluster_id = edge_cluster_id.unwrap_or(0) + cluster_delta as u32;
                edge_cluster_id = Some(cluster_id);
                if cluster_id != current_cluster {
                    cluster_links.insert((current_cluster, cluster_id));
                    vertex_outbound_clusters.insert(cluster_id);
                }
            }
        }

        vertex_outbound_cluster_sum += vertex_outbound_clusters.len();
        vertex_outbound_clusters.clear();
    }

    let mut cluster_outbound_count = vec![0usize; current_cluster as usize + 1];
    for (src, _) in cluster_links {
        cluster_outbound_count[src as usize] += 1;
    }
    let (cluster_link_sum, cluster_link_count) =
        cluster_outbound_count.into_iter().fold((0, 0), |state, c| {
            if c > 0 {
                (state.0 + c, state.1 + 1usize)
            } else {
                state
            }
        });

    println!("Vertexes: {num_vertexes}");
    println!("Clusters: {cluster_link_count}");
    println!(
        "Avg outbound links per vertex: {:.3}",
        vertex_outbound_cluster_sum as f64 / num_vertexes as f64
    );
    println!(
        "Avg outbound links per cluster: {:.3}",
        cluster_link_sum as f64 / cluster_link_count as f64
    );

    Ok(())
}
