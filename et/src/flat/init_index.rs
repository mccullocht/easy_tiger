use std::{io, num::NonZero, sync::Arc};

use clap::Args;
use vectors::{F32VectorCoding, VectorSimilarity};
use wt_mdb::{Connection, connection::{CreateOptionsBuilder, DropOptionsBuilder}};

use super::{FlatIndexConfig, flat_table_name};

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
    let table_name = flat_table_name(index_name);
    if args.drop_tables {
        connection
            .drop_table(
                &table_name,
                Some(DropOptionsBuilder::default().set_force().into()),
            )
            .map_err(io::Error::from)?;
    }

    let config = FlatIndexConfig {
        dimensions: args.dimensions,
        similarity: args.similarity,
        format: args.format,
    };
    let metadata = serde_json::to_string(&config).expect("serializable config");
    connection
        .create_table(
            &table_name,
            Some(
                CreateOptionsBuilder::default()
                    .app_metadata(&metadata)
                    .into(),
            ),
        )
        .map_err(io::Error::from)
}
