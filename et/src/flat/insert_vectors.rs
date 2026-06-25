use std::{fs::File, io, num::NonZero, path::PathBuf, sync::Arc};

use clap::Args;
use easy_tiger::{
    flat,
    input::{DerefVectorStore, VectorStore},
};
use wt_mdb::Connection;

use crate::ui::progress_bar;

#[derive(Args)]
pub struct InsertVectorsArgs {
    /// Path to the input vectors to insert (little-endian f32).
    #[arg(short, long)]
    f32_vectors: PathBuf,
    /// Index of the first vector to insert.
    #[arg(long, default_value_t = 0)]
    start: usize,
    /// Number of vectors to insert. If unset, inserts all vectors from --start to end of file.
    #[arg(short, long)]
    count: Option<NonZero<usize>>,
    /// Number of vectors to insert in each transaction batch.
    #[arg(long, default_value_t = NonZero::new(256).unwrap())]
    batch_size: NonZero<usize>,
}

pub fn insert_vectors(
    connection: Arc<Connection>,
    index_name: &str,
    args: InsertVectorsArgs,
) -> io::Result<()> {
    let config = flat::open_config(&connection, index_name)?;
    let table_name = flat::table_name(index_name);

    let f32_vectors = DerefVectorStore::new(
        unsafe { memmap2::Mmap::map(&File::open(args.f32_vectors)?)? },
        config.dimensions,
    )?;
    f32_vectors.data().advise(memmap2::Advice::Sequential)?;

    if args.start > f32_vectors.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "--start {} exceeds vector file length {}",
                args.start,
                f32_vectors.len()
            ),
        ));
    }
    let count = args
        .count
        .map(|c| c.get())
        .unwrap_or(f32_vectors.len() - args.start);
    let end = args.start + count;
    if end > f32_vectors.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "requested range {}..{} exceeds vector file length {}",
                args.start,
                end,
                f32_vectors.len()
            ),
        ));
    }

    let coder = config.format.coder(config.similarity, None);
    let batch_size = args.batch_size.get();
    let progress = progress_bar(count, "inserting vectors");

    for batch_start in (args.start..end).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(end);
        let txn = connection.begin_transaction(None)?;
        {
            let mut cursor = txn.open_record_cursor(&table_name)?;
            for i in batch_start..batch_end {
                let vector: &[f32] = &f32_vectors[i];
                let encoded = coder.encode(vector);
                cursor.set(i as i64, encoded.as_slice())?;
            }
        }
        txn.commit(None)?;
        progress.inc((batch_end - batch_start) as u64);
    }

    progress.finish();
    Ok(())
}
