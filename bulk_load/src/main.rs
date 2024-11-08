use std::{fs::File, io, num::NonZero, path::PathBuf};

use clap::Parser;
use easy_tiger::{
    bulk::BulkLoadBuilder,
    input::NumpyF32VectorStore,
    wt::{GraphMetadata, WiredTigerIndexParams},
};
use indicatif::{ProgressBar, ProgressStyle};
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

    /// Path to the WiredTiger database to upload to.
    #[arg(short, long)]
    wiredtiger_db_path: PathBuf,
    /// Base name for vector tables.
    #[arg(short, long)]
    wiredtiger_table_basename: String,
    /// If true, create the WiredTiger database if it does not exist.
    #[arg(short, long, default_value = "true")]
    create: bool, // should be create_db
    /// If true, drop any WiredTiger tables with the same name before bulk upload.
    #[arg(short, long, default_value = "false")]
    drop: bool, // XXX should be drop_tables.
}

fn progress_bar(len: usize, message: &'static str) -> ProgressBar {
    // XXX configure it to leave the progress bar
    ProgressBar::new(len as u64)
        .with_style(
            ProgressStyle::default_bar()
                .template(
                    "{msg} {wide_bar} {pos}/{len} ETA: {eta_precise} Elapsed: {elapsed_precise}",
                )
                .unwrap(),
        )
        .with_message(message)
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let f32_vectors = NumpyF32VectorStore::new(
        unsafe { memmap2::Mmap::map(&File::open(args.f32_vectors)?)? },
        args.dimensions,
    );

    // TODO: wiredtiger_db_path should probably accept a Path
    // TODO: configurable cache size.
    let mut connection_options =
        ConnectionOptionsBuilder::default().cache_size_mb(NonZero::new(4 << 10).unwrap());
    if args.create {
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
    if args.drop {
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
    let builder = BulkLoadBuilder::new(
        GraphMetadata {
            dimensions: args.dimensions,
            max_edges: NonZero::new(32).unwrap(),
        },
        wt_params,
        f32_vectors,
    );

    {
        let progress = progress_bar(num_vectors, "load nav vectors");
        builder
            .load_nav_vectors(|| progress.inc(1))
            .map_err(io::Error::from)?;
    }
    {
        // NB: this step is _very_ slow. there are two things going on:
        // 1) we serialize writes which causes us to leave several threads idle (AFAICT ~50% of threads).
        // 2) searching for quantized vectors in the btree is not as cheap as I'd hoped.
        //
        // (1) is very hard to fix because pruning involves reading the current set of edges and
        // scoring them against one another to do the actual pruning. none of this can be moved out of
        // the lock. This may necessarily involve IO, but maybe it is possible to cache or somehow
        // build the working set required outside of the lock? another possibility is to drop the RNG
        // pruning and stop scoring shit.
        let progress = progress_bar(num_vectors, "build graph");
        builder
            .insert_all(|| progress.inc(1))
            .map_err(io::Error::from)?;
    }
    {
        let progress = progress_bar(num_vectors, "cleanup graph");
        builder
            .insert_all(|| progress.inc(1))
            .map_err(io::Error::from)?;
    }
    {
        let progress = progress_bar(num_vectors, "load graph");
        builder
            .insert_all(|| progress.inc(1))
            .map_err(io::Error::from)?;
    }

    Ok(())
}
