use std::{
    fs::File,
    io::{self, BufReader, Read},
    path::PathBuf,
};

use clap::Args;

#[derive(Args)]
pub struct ReadArchiveArgs {
    /// Path to the mongodump archive file.
    #[arg(short, long)]
    archive: PathBuf,

    /// Number of documents to display per collection (default: 5).
    #[arg(short = 'n', long, default_value = "5")]
    count: usize,

    /// Filter by database name (optional).
    #[arg(short, long)]
    database: Option<String>,

    /// Filter by collection name (optional).
    #[arg(short, long)]
    collection: Option<String>,
}

const MAGIC_NUMBER: u32 = 0x8199e26d;
const TERMINATOR: u32 = 0xffffffff;

pub fn read_archive(args: ReadArchiveArgs) -> io::Result<()> {
    let file = File::open(&args.archive)?;
    let mut reader = BufReader::new(file);

    // Read and verify magic number
    let magic = read_u32(&mut reader)?;
    if magic != MAGIC_NUMBER {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Invalid magic number: expected 0x{:08x}, got 0x{:08x}",
                MAGIC_NUMBER, magic
            ),
        ));
    }

    println!("==== MAGIC OK =====");

    // Read header
    let header = read_bson_document(&mut reader)?;
    println!("Archive Header:");
    println!("{}\n", header);

    // Read collection metadata until we hit the terminator
    println!("Collection Metadata:");
    let mut metadata_count = 0;
    loop {
        // Peek ahead to check for terminator
        let mut peek_buf = [0u8; 4];
        reader.read_exact(&mut peek_buf)?;
        let value = u32::from_le_bytes(peek_buf);

        if value == TERMINATOR {
            println!("(Found {} collection metadata entries)\n", metadata_count);
            break;
        }

        // Not a terminator, so it's a BSON document size
        // Read the rest of the document
        let doc_size = value as usize;
        if doc_size < 5 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid BSON document size: {}", doc_size),
            ));
        }

        let mut doc_bytes = vec![0u8; doc_size];
        doc_bytes[0..4].copy_from_slice(&peek_buf);
        reader.read_exact(&mut doc_bytes[4..])?;

        let metadata = bson::Document::from_reader(&mut &doc_bytes[..]).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Failed to parse BSON: {}", e),
            )
        })?;

        println!("{}", metadata);
        metadata_count += 1;
    }

    // Read namespace segments
    println!("Documents:");
    loop {
        match read_namespace_segment(&mut reader, &args) {
            Ok(true) => continue, // More segments to read
            Ok(false) => break,   // EOF reached
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
    }

    Ok(())
}

fn read_namespace_segment(
    reader: &mut BufReader<File>,
    args: &ReadArchiveArgs,
) -> io::Result<bool> {
    // Read namespace header
    let header = read_bson_document(reader)?;

    let db = header.get_str("db").map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Missing 'db' field: {}", e),
        )
    })?;
    let collection = header.get_str("collection").map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Missing 'collection' field: {}", e),
        )
    })?;
    let eof = header.get_bool("EOF").unwrap_or(false);

    // Check if this is an EOF marker
    if eof {
        let terminator = read_u32(reader)?;
        if terminator != TERMINATOR {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Expected terminator after EOF header",
            ));
        }
        return Ok(false);
    }

    // Check if we should filter this collection
    let should_display = args.database.as_ref().map_or(true, |d| d == db)
        && args.collection.as_ref().map_or(true, |c| c == collection);

    if should_display {
        println!("\n--- {}.{} ---", db, collection);
    }

    // Read documents until terminator
    let mut count = 0;
    loop {
        // Peek ahead to check for terminator
        let mut terminator_buf = [0u8; 4];
        reader.read_exact(&mut terminator_buf)?;
        let value = u32::from_le_bytes(terminator_buf);

        if value == TERMINATOR {
            break;
        }

        // Not a terminator, so it's the start of a BSON document
        // Read the rest of the document (first 4 bytes are the size)
        let doc_size = value as usize;
        if doc_size < 5 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid BSON document size: {}", doc_size),
            ));
        }

        let mut doc_bytes = vec![0u8; doc_size];
        doc_bytes[0..4].copy_from_slice(&terminator_buf);
        reader.read_exact(&mut doc_bytes[4..])?;

        if should_display {
            let doc = bson::Document::from_reader(&mut &doc_bytes[..]).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Failed to parse BSON document: {}", e),
                )
            })?;
            println!("{}", doc);
            count += 1;
        }

        if count >= args.count {
            break;
        }
    }

    Ok(true)
}

fn read_u32(reader: &mut BufReader<File>) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_bson_document(reader: &mut BufReader<File>) -> io::Result<bson::Document> {
    // Read the document size (first 4 bytes)
    let size = read_u32(reader)?;

    if size < 5 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid BSON document size: {}", size),
        ));
    }

    // Read the rest of the document
    let mut doc_bytes = vec![0u8; size as usize];
    doc_bytes[0..4].copy_from_slice(&size.to_le_bytes());
    reader.read_exact(&mut doc_bytes[4..])?;

    // Parse BSON document
    bson::Document::from_reader(&mut &doc_bytes[..]).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Failed to parse BSON: {}", e),
        )
    })
}
