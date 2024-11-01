use std::{fs::File, io, num::NonZero, path::PathBuf};

use clap::Parser;
use easy_tiger::{input::NumpyF32VectorStore, quantization::binary_quantize};
use indicatif::{ProgressBar, ProgressStyle};
use wt_mdb::{Connection, ConnectionOptionsBuilder, DropOptionsBuilder, RecordView};

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
    create: bool,
    /// If true, drop any WiredTiger tables with the same name before bulk upload.
    #[arg(short, long, default_value = "false")]
    drop: bool,
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let f32_vectors = NumpyF32VectorStore::new(
        unsafe { memmap2::Mmap::map(&File::open(args.f32_vectors)?)? },
        args.dimensions,
    );

    // TODO: wiredtiger_db_path should probably accept a Path
    let connection = Connection::open(
        &args.wiredtiger_db_path.to_string_lossy(),
        if args.create {
            Some(ConnectionOptionsBuilder::default().create().into())
        } else {
            None
        },
    )
    .map_err(io::Error::from)?;
    let session = connection.open_session().map_err(io::Error::from)?;

    let quantized_table = format!("{}.quantized", args.wiredtiger_table_basename);
    if args.drop {
        session
            .drop_record_table(
                &quantized_table,
                Some(DropOptionsBuilder::default().set_force().into()),
            )
            .map_err(io::Error::from)?;
    }
    let quantization_progress = ProgressBar::new(f32_vectors.len() as u64)
        .with_style(
            ProgressStyle::default_bar()
                .template(
                    "{msg} {wide_bar} {pos}/{len} ETA: {eta_precise} Elapsed: {elapsed_precise}",
                )
                .unwrap(),
        )
        .with_message("quantize");
    session
        .bulk_load(
            &quantized_table,
            None,
            f32_vectors
                .iter()
                .enumerate()
                .map(|(i, v)| RecordView::new(i as i64, binary_quantize(v)))
                .inspect(|_| quantization_progress.inc(1)),
        )
        .map_err(io::Error::from)
}
