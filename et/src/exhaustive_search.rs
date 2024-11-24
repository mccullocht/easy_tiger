use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    fs::File,
    io::{self, BufWriter, Write},
    num::NonZero,
    path::PathBuf,
    sync::Arc,
};

use clap::Args;
use easy_tiger::{input::NumpyF32VectorStore, wt::WiredTigerGraphVectorIndex, Neighbor};
use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};
use memmap2::Mmap;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use wt_mdb::Connection;

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
    index: WiredTigerGraphVectorIndex,
    args: ExhaustiveSearchArgs,
) -> io::Result<()> {
    let query_vectors = NumpyF32VectorStore::new(
        unsafe { Mmap::map(&File::open(args.query_vectors)?)? },
        index.metadata().dimensions,
    )?;

    let mut results = Vec::with_capacity(query_vectors.len());
    results.resize_with(query_vectors.len(), || {
        BinaryHeap::with_capacity(args.neighbors_len.get())
    });
    let scorer = index.metadata().new_scorer();

    let session = connection.open_session()?;
    let mut cursor = session.open_record_cursor(index.graph_table_name())?;
    let limit = cursor.largest_key().unwrap().unwrap() + 1;
    cursor.seek_exact(-1).unwrap()?;
    let mut index_vector = vec![0.0f32; index.metadata().dimensions.get()];
    let progress = ProgressBar::new(limit as u64)
        .with_style(
            ProgressStyle::default_bar()
                .template("{wide_bar} {pos}/{len} ETA: {eta_precise} Elapsed: {elapsed_precise}")
                .unwrap(),
        )
        .with_finish(ProgressFinish::AndLeave);
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
                results[i].push(Reverse(n));
            } else {
                let mut peek = results[i].peek_mut().unwrap();
                if n <= peek.0 {
                    peek.0 = n;
                }
            }
        }
        progress.inc(1);
    }

    let mut writer = BufWriter::new(File::create(args.neighbors)?);
    for neighbor_heap in results {
        let mut neighbors = neighbor_heap.into_iter().map(|rn| rn.0).collect::<Vec<_>>();
        neighbors.sort();
        for n in neighbors {
            writer.write_all(&(n.vertex() as u32).to_le_bytes())?;
        }
    }

    Ok(())
}
