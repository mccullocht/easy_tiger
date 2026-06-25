use std::{io, num::NonZero, sync::Arc};

use clap::Args;
use easy_tiger::flat::{self, FlatIndexConfig};
use vectors::{F32VectorCoding, VectorSimilarity};
use wt_mdb::{Connection, connection::DropOptionsBuilder};

#[derive(Args)]
pub struct InitIndexArgs {
    /// Number of dimensions in input vectors.
    #[arg(short, long)]
    dimensions: NonZero<usize>,
    /// Similarity function to use for vector scoring.
    #[arg(short, long, value_enum)]
    similarity: VectorSimilarity,
    /// Encoding format for stored vectors.
    #[arg(short, long)]
    format: F32VectorCoding,
    /// If true, drop any existing table with the same name before creating.
    #[arg(long, default_value_t = false)]
    drop_tables: bool,
}

pub fn init_index(
    connection: Arc<Connection>,
    index_name: &str,
    args: InitIndexArgs,
) -> io::Result<()> {
    if args.drop_tables {
        flat::drop_index(
            &connection,
            index_name,
            Some(DropOptionsBuilder::default().set_force().into()),
        )?;
    }

    let config = FlatIndexConfig {
        dimensions: args.dimensions,
        similarity: args.similarity,
        format: args.format,
    };
    flat::init_index(&connection, index_name, &config)
}
