use std::{fs::File, io, num::NonZero, ops::Range, path::PathBuf, sync::Arc};

use clap::Args;
use easy_tiger::{
    input::{DerefVectorStore, VectorStore},
    vamana::{
        mutate::insert_vector,
        wt::{SessionGraphVectorIndex, TableGraphVectorIndex},
    },
};
use indicatif::ProgressIterator;
use wt_mdb::{Connection, Result};

use crate::ui::progress_bar;

#[derive(Args)]
pub struct InsertArgs {
    /// Path to the numpy formatted vectors to insert.
    #[arg(short, long)]
    vectors: PathBuf,
    /// Index of the first vector to insert.
    #[arg(long, default_value = "0")]
    offset: usize,
    /// Number of vectors to insert from the input.
    ///
    /// This value is bound by offset and the number of input vectors.
    #[arg(long)]
    limit: Option<NonZero<usize>>,
}

fn insert_all<'a>(
    wt_index: &SessionGraphVectorIndex,
    vectors: impl ExactSizeIterator<Item = &'a [f32]>,
) -> Result<Vec<Range<i64>>> {
    let mut keys: Vec<Range<i64>> = vec![];
    // I could probably write this as a fold but it seems annoying.
    let progress = progress_bar(vectors.len(), "");
    for vector in vectors.progress_with(progress) {
        wt_index.session().begin_transaction(None)?;
        let key = insert_vector(vector, wt_index)?;
        if let Some(r) = keys.last_mut() {
            if r.end == key {
                r.end = key + 1;
            } else {
                keys.push(key..(key + 1))
            }
        } else {
            keys.push(key..(key + 1))
        }
        wt_index.session().commit_transaction(None)?;
    }
    Ok(keys)
}

pub fn insert(connection: Arc<Connection>, index_name: &str, args: InsertArgs) -> io::Result<()> {
    let index = Arc::new(TableGraphVectorIndex::from_db(&connection, index_name)?);
    let vectors = DerefVectorStore::<f32, _>::new(
        unsafe { memmap2::Mmap::map(&File::open(args.vectors)?)? },
        index.config().dimensions,
    )?;

    let wt_index = SessionGraphVectorIndex::new(index, connection.open_session()?);
    match insert_all(
        &wt_index,
        vectors
            .iter()
            .skip(args.offset)
            .take(args.limit.map(NonZero::get).unwrap_or(usize::MAX)),
    ) {
        Ok(keys) => {
            println!("Inserted {keys:?}");
        }
        Err(e) => {
            // TODO: custom error that tells you how far we got.
            println!("Insert failed with error {e}");
        }
    }

    Ok(())
}
