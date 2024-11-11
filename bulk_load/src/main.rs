use std::{fs::File, io, num::NonZero, path::PathBuf};

use clap::Parser;
use easy_tiger::{
    bulk::BulkLoadBuilder,
    input::NumpyF32VectorStore,
    wt::{GraphMetadata, WiredTigerIndexParams},
};
use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};
use wt_mdb::{Connection, ConnectionOptionsBuilder, DropOptionsBuilder};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the input vectors to bulk ingest.
    #[arg(short, long)]
    f32_vectors: PathBuf,
    /// Number of dimensions in input vectors.
    #[arg(short, long)]
    dimensions: NonZero<usize>,
    /// Limit the number of input vectors. Useful for testing.
    #[arg(short, long)]
    limit: Option<usize>,

    /// Path to the WiredTiger database to upload to.
    #[arg(long)]
    wiredtiger_db_path: PathBuf,
    /// Base name for vector tables.
    #[arg(long)]
    wiredtiger_table_basename: String,
    #[arg(long, default_value = "1024")]
    wiredtiger_cache_size_mb: NonZero<usize>,
    /// If true, create the WiredTiger database if it does not exist.
    #[arg(long, default_value = "true")]
    create_db: bool,
    /// If true, drop any WiredTiger tables with the same name before bulk upload.
    #[arg(long, default_value = "false")]
    drop_tables: bool,
    /// Maximum number of edges for any vertex.
    #[arg(short, long, default_value = "32")]
    max_edges: NonZero<usize>,
}

fn progress_bar(len: usize, message: &'static str) -> ProgressBar {
    ProgressBar::new(len as u64)
        .with_style(
            ProgressStyle::default_bar()
                .template(
                    "{msg} {wide_bar} {pos}/{len} ETA: {eta_precise} Elapsed: {elapsed_precise}",
                )
                .unwrap(),
        )
        .with_message(message)
        .with_finish(ProgressFinish::AndLeave)
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let f32_vectors = NumpyF32VectorStore::new(
        unsafe { memmap2::Mmap::map(&File::open(args.f32_vectors)?)? },
        args.dimensions,
    );

    // TODO: Connection.filename should accept &Path. This will likely be very annoying to plumb to CString.
    let mut connection_options =
        ConnectionOptionsBuilder::default().cache_size_mb(args.wiredtiger_cache_size_mb);
    if args.create_db {
        connection_options = connection_options.create();
    }
    let connection = Connection::open(
        &args.wiredtiger_db_path.to_string_lossy(),
        Some(connection_options.into()),
    )
    .map_err(io::Error::from)?;

    // TODO: max_edges should be configurable.
    let wt_params = WiredTigerIndexParams {
        connection: connection.clone(),
        graph_table_name: format!("{}.graph", args.wiredtiger_table_basename),
        nav_table_name: format!("{}.nav_vectors", args.wiredtiger_table_basename),
    };
    if args.drop_tables {
        let session = connection.open_session().map_err(io::Error::from)?;
        session
            .drop_record_table(
                &wt_params.graph_table_name,
                Some(DropOptionsBuilder::default().set_force().into()),
            )
            .map_err(io::Error::from)?;
        session
            .drop_record_table(
                &wt_params.nav_table_name,
                Some(DropOptionsBuilder::default().set_force().into()),
            )
            .map_err(io::Error::from)?;
    }

    let num_vectors = f32_vectors.len();
    let limit = args.limit.unwrap_or(num_vectors);
    let builder = BulkLoadBuilder::new(
        GraphMetadata {
            dimensions: args.dimensions,
            max_edges: args.max_edges,
        },
        wt_params,
        f32_vectors,
        limit,
    );

    {
        let progress = progress_bar(limit, "load nav vectors");
        builder
            .load_nav_vectors(|| progress.inc(1))
            .map_err(io::Error::from)?;
    }
    {
        let progress = progress_bar(limit, "build graph");
        builder
            .insert_all(|| progress.inc(1))
            .map_err(io::Error::from)?;
    }
    {
        let progress = progress_bar(limit, "cleanup graph");
        builder
            .cleanup(|| progress.inc(1))
            .map_err(io::Error::from)?;
    }
    let stats = {
        let progress = progress_bar(limit, "load graph");
        builder
            .load_graph(|| progress.inc(1))
            .map_err(io::Error::from)?
    };
    println!("Graph stats: {:?}", stats);

    Ok(())
}
