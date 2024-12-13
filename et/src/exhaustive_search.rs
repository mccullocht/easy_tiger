use std::{
    collections::BinaryHeap,
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
use rayon::iter::{IntoParallelIterator, ParallelIterator};
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
    #[arg(long, default_value = "100")]
    neighbors_len: NonZero<usize>,
}

pub fn exhaustive_search(
    connection: Arc<Connection>,
    index_name: &str,
    args: ExhaustiveSearchArgs,
) -> io::Result<()> {
    let index = TableGraphVectorIndex::from_db(&connection, index_name)?;
    let query_vectors = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.query_vectors)?)? },
        index.config().dimensions,
    )?;

    let mut results = Vec::with_capacity(query_vectors.len());
    results.resize_with(query_vectors.len(), || {
        BinaryHeap::with_capacity(args.neighbors_len.get())
    });
    let scorer = index.config().new_scorer();

    let session = connection.open_session()?;
    let mut cursor = session.open_record_cursor(index.graph_table_name())?;
    let limit = std::cmp::max(cursor.largest_key().unwrap().unwrap() + 1, 0);
    cursor.seek_exact(-1).unwrap()?;
    let mut index_vector = vec![0.0f32; index.config().dimensions.get()];
    let progress = progress_bar(limit as usize, None);
    for record_result in cursor {
        let record = record_result?;
        for (i, o) in record
            .value()
            .chunks(std::mem::size_of::<f32>())
            .zip(index_vector.iter_mut())
        {
            *o = f32::from_le_bytes(i.try_into().expect("array of 4 conversion."));
        }

        let similarities = (0..query_vectors.len())
            .into_par_iter()
            .map(|i| scorer.score(&index_vector, &query_vectors[i]))
            .collect::<Vec<_>>();
        for (i, s) in similarities.into_iter().enumerate() {
            let n = Neighbor::new(record.key(), s);
            if results[i].len() < results[i].capacity() {
                results[i].push(n);
            } else {
                let mut peek = results[i].peek_mut().unwrap();
                if n <= *peek {
                    *peek = n;
                }
            }
        }
        progress.inc(1);
    }

    let mut writer = BufWriter::new(File::create(args.neighbors)?);
    for neighbor_heap in results.into_iter() {
        for n in neighbor_heap.into_sorted_vec() {
            writer.write_all(&(n.vertex() as u32).to_le_bytes())?;
        }
    }

    Ok(())
}
