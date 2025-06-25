use std::{fs::File, io, num::NonZero, path::PathBuf, sync::Arc};

use clap::Args;
use easy_tiger::{
    distance::VectorSimilarity,
    input::{DerefVectorStore, VectorStore},
    vector_table::{Metadata, Representation, VectorTable},
};
use indicatif::ProgressIterator;
use memmap2::Mmap;
use wt_mdb::Connection;

use crate::ui::progress_bar;

#[derive(Args)]
pub struct VectorTableLoadArgs {
    /// Path to numpy formatted little-endian float vectors.
    #[arg(short, long)]
    input_vectors: PathBuf,
    /// Number of input dimensions.
    #[arg(short, long)]
    dimensions: NonZero<usize>,
    /// Vector similarity function to use for the vectors in this table.
    /// This may affect vector encoding depending on representation.
    #[arg(short, long)]
    similarity: VectorSimilarity,
    /// Vector representation on disk.
    #[arg(short, long)]
    representation: Representation,

    /// If true, drop the table before loading.
    #[arg(long, default_value_t = false)]
    drop_table: bool,
    /// If set, upload the first limit vectors.
    #[arg(long)]
    limit: Option<usize>,
}

pub fn vector_table_load(
    connection: Arc<Connection>,
    index_name: &str,
    args: VectorTableLoadArgs,
) -> io::Result<()> {
    let input_vectors = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.input_vectors)?)? },
        args.dimensions,
    )?;
    let session = connection.open_session()?;
    let metadata = Metadata {
        dimensions: args.dimensions.get(),
        similarity: args.similarity,
        representation: args.representation,
    };
    if args.drop_table {
        session.drop_table(index_name, None)?;
    }
    let limit = args.limit.unwrap_or(input_vectors.len());
    let progress_bar = progress_bar(limit, "vector table load");
    VectorTable::bulk_load(
        &session,
        index_name,
        metadata,
        input_vectors
            .iter()
            .enumerate()
            .take(limit)
            .progress_with(progress_bar)
            .map(|(i, v)| (i as i64, v)),
    )
    .map(|_| ())
}
