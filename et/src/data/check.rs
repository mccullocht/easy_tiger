use std::{
    fs::File,
    io::{self, Read},
    path::PathBuf,
};

use clap::Args;

#[derive(Args)]
pub struct CheckArgs {
    /// Input file in BigANN format to inspect.
    #[arg(short, long)]
    input: PathBuf,
}

pub fn check(args: CheckArgs) -> io::Result<()> {
    let mut file = File::open(&args.input)?;
    let file_len = file.metadata()?.len();
    if file_len < 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("input file is only {file_len} bytes, too short for a BigANN header"),
        ));
    }

    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;
    let count = u32::from_le_bytes(header[0..4].try_into().unwrap()) as u64;
    let dimensions = u32::from_le_bytes(header[4..8].try_into().unwrap()) as u64;

    let remaining = file_len - 8;
    let entries = count * dimensions;
    let width = if entries == 0 { 0 } else { remaining / entries };

    println!("count: {count}");
    println!("dimensions: {dimensions}");
    println!("entry width (bytes): {width}");
    if entries == 0 || remaining % entries != 0 {
        eprintln!(
            "warning: remaining file size {remaining} does not divide evenly into {entries} elements"
        );
    }

    Ok(())
}
