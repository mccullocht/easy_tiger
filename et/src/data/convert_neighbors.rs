use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Write},
    num::NonZero,
    path::PathBuf,
};

use clap::Args;

/// Size in bytes of a single serialized `easy_tiger::Neighbor` (vertex: i64, distance: f64).
const NEIGHBOR_LEN: usize = 16;

#[derive(Args)]
pub struct ConvertNeighborsArgs {
    /// Input file containing raw neighbor rows with no header: `neighbors_len` fixed 16-byte
    /// (vertex: i64, distance: f64) entries per query, back to back.
    #[arg(short, long)]
    input: PathBuf,
    /// Number of neighbor entries per query in the input file.
    #[arg(short = 'n', long, default_value_t = NonZero::new(100).unwrap())]
    neighbors_len: NonZero<usize>,
    /// Output file to write in BigANN format (<count,dim> header followed by the input data
    /// unchanged).
    #[arg(short, long)]
    output: PathBuf,
}

pub fn convert_neighbors(args: ConvertNeighborsArgs) -> io::Result<()> {
    let row_len = args.neighbors_len.get() as u64 * NEIGHBOR_LEN as u64;
    let mut input = BufReader::new(File::open(&args.input)?);
    let file_len = input.get_ref().metadata()?.len();
    if !file_len.is_multiple_of(row_len) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "input file size {file_len} does not divide evenly into rows of {row_len} bytes"
            ),
        ));
    }
    let count = file_len / row_len;

    let mut out = BufWriter::new(File::create(&args.output)?);
    out.write_all(&(count as u32).to_le_bytes())?;
    out.write_all(&(args.neighbors_len.get() as u32).to_le_bytes())?;
    io::copy(&mut input, &mut out)?;

    Ok(())
}
