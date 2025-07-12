use std::{
    fs::File,
    io::{self, BufWriter, Write},
    num::NonZero,
    path::PathBuf,
    sync::Arc,
};

use clap::Args;
use easy_tiger::{
    input::{DerefVectorStore, VectorStore},
    wt::TableGraphVectorIndex,
    Neighbor,
};
use memmap2::Mmap;
use rayon::prelude::*;
use wt_mdb::Connection;

use crate::ui::progress_bar;

#[derive(Args)]
pub struct ExhaustiveSearchArgs {
    /// Path to numpy formatted little-endian float vectors.
    #[arg(short, long)]
    query_vectors: PathBuf,
    /// Path buf to numpy u32 formatted neighbors file to write.
    /// This should include one row of length neighbors_len for each vector in query_vectors.
    #[arg(long)]
    neighbors: PathBuf,
    /// Number of neighbors for each query in the neighbors file.
    #[arg(long, default_value_t = NonZero::new(100).unwrap())]
    neighbors_len: NonZero<usize>,
    /// Maximum number of records to search for query vectors.
    #[arg(long)]
    record_limit: Option<usize>,
}

pub fn exhaustive_search(
    connection: Arc<Connection>,
    index_name: &str,
    args: ExhaustiveSearchArgs,
) -> io::Result<()> {
    let index = TableGraphVectorIndex::from_db(&connection, index_name)?;
    let query_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.query_vectors)?)? },
        index.config().dimensions,
    )?;

    let mut results = Vec::with_capacity(query_vectors.len());
    let k = args.neighbors_len.get();
    results.resize_with(query_vectors.len(), || Vec::with_capacity(k * 2));
    let distance_fn = index.config().new_distance_function();

    let session = connection.open_session()?;
    let mut cursor = session.open_record_cursor(index.raw_table_name())?;
    let mut limit = std::cmp::max(cursor.largest_key().unwrap().unwrap() + 1, 0) as usize;
    limit = limit.min(args.record_limit.unwrap_or(usize::MAX));
    cursor.set_bounds(0..)?;
    let progress = progress_bar(limit, "");
    for record_result in cursor.take(limit) {
        let record = record_result?;
        results.par_iter_mut().enumerate().for_each(|(i, r)| {
            let n = Neighbor::new(
                record.key(),
                distance_fn.distance(bytemuck::cast_slice(&query_vectors[i]), record.value()),
            );
            if r.len() <= k || n < r[k] {
                r.push(n);
                if r.len() == r.capacity() {
                    r.select_nth_unstable(k);
                    r.truncate(k);
                }
            }
        });
        progress.inc(1);
    }

    let mut writer = BufWriter::new(File::create(args.neighbors)?);
    for mut neighbors in results.into_iter() {
        neighbors.sort_unstable();
        for n in neighbors.into_iter().take(k) {
            writer.write_all(&(n.vertex() as u32).to_le_bytes())?;
        }
    }

    Ok(())
}
