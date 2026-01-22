use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::PathBuf,
    sync::Arc,
};

use clap::Args;
use easy_tiger::{
    spann::{centroid_stats::CentroidStats, PostingKey, TableIndex},
    vamana::{wt::SessionGraphVectorIndex, GraphVectorIndex, GraphVectorStore},
};
use rayon::prelude::*;
use vectors::F32VectorCoding;
use wt_mdb::Connection;

#[derive(Args)]
pub struct ExtractIndexArgs {
    /// Prefix for the output files.
    #[arg(long)]
    prefix: PathBuf,
}

pub fn extract_index(
    connection: Arc<Connection>,
    index_name: &str,
    args: ExtractIndexArgs,
) -> io::Result<()> {
    let index = TableIndex::from_db(&connection, index_name)?;
    let session = connection.open_session()?;
    let stats = CentroidStats::from_index_stats(&session, &index)?;
    let centroid_count = stats.centroid_count();
    let centroid_ids = stats
        .assignment_counts_iter()
        .map(|(id, _)| id as u32)
        .collect::<Vec<u32>>();

    println!("Extracting {} centroids...", centroid_ids.len());

    let head_index = SessionGraphVectorIndex::new(index.head_config().clone(), session);
    let head_coder = head_index.high_fidelity_vectors()?.new_coder();
    let posting_coder = index.new_posting_coder();
    let out_coder = F32VectorCoding::F32.new_coder(head_index.index().config().similarity);

    let dim = head_index.index().config().dimensions.get();
    centroid_ids
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, centroid_id)| {
            let head_index = SessionGraphVectorIndex::new(
                index.head_config().clone(),
                connection.open_session()?,
            );
            let mut head_vectors = head_index.high_fidelity_vectors()?;

            let centroid_vec = head_vectors.get(centroid_id as i64).unwrap()?;
            let filename = format!(
                "{}-{:06}-of-{:06}.fvecs",
                args.prefix.to_string_lossy(),
                i,
                centroid_count
            );
            let mut file = BufWriter::with_capacity(4 << 20, File::create(filename)?);

            file.write_all(&out_coder.encode(&head_coder.decode(&centroid_vec)))?;

            let mut postings_cursor = head_index
                .session()
                .get_or_create_typed_cursor::<PostingKey, Vec<u8>>(index.postings_table_name())?;
            postings_cursor.set_bounds(PostingKey::centroid_range(centroid_id))?;
            let mut decode_buf = vec![0.0f32; dim];
            let mut encode_buf = vec![0u8; out_coder.byte_len(dim)];
            while let Some(r) = unsafe { postings_cursor.next_unsafe() } {
                let (_, tail_bytes) = r?;
                posting_coder.decode_to(&tail_bytes, &mut decode_buf);
                out_coder.encode_to(&decode_buf, &mut encode_buf);
                file.write_all(&encode_buf)?;
            }

            Ok::<(), io::Error>(())
        })?;

    Ok(())
}
