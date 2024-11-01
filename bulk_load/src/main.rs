use std::{fs::File, num::NonZero, path::PathBuf};

use clap::Parser;
use easy_tiger::input::NumpyF32VectorStore;

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
    /// If true, create the wiredtiger database if it does not exist.
    #[arg(short, long, default_value = "true")]
    wiredtiger_create: bool,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let f32_vectors = NumpyF32VectorStore::new(
        unsafe { memmap2::Mmap::map(&File::open(args.f32_vectors)?)? },
        args.dimensions,
    );
    Ok(())
}
