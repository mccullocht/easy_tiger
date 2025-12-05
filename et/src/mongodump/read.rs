use std::{
    fs::File,
    io::{self, BufReader},
    path::PathBuf,
};

use clap::Args;

use crate::mongodump::parser::Consumer;

#[derive(Args)]
pub struct ReadArchiveArgs {
    /// Path to the mongodump archive file.
    #[arg(short, long)]
    archive: PathBuf,

    /// Number of documents to display per collection (default: 5).
    #[arg(short = 'n', long, default_value_t = 5)]
    count: usize,

    /// Filter by database name (optional).
    #[arg(short, long)]
    database: Option<String>,

    /// Filter by collection name (optional).
    #[arg(short, long)]
    collection: Option<String>,
}

struct PrintingParser(ReadArchiveArgs);

impl Consumer for PrintingParser {
    fn consume_header(&mut self, header: &bson::RawDocument) -> bson::error::Result<()> {
        println!("Header: {}", bson::Document::try_from(header)?);
        Ok(())
    }

    fn consume_collection_metadata(
        &mut self,
        metadata: &bson::RawDocument,
    ) -> bson::error::Result<()> {
        println!(
            "Collection metadata: {}",
            bson::Document::try_from(metadata)?
        );
        Ok(())
    }

    fn consume_namespace_header(&mut self, header: &bson::RawDocument) -> bson::error::Result<()> {
        println!("Namespace header: {}", bson::Document::try_from(header)?);
        Ok(())
    }

    fn consume_document(
        &mut self,
        db: &str,
        collection: &str,
        doc: &bson::RawDocument,
    ) -> bson::error::Result<bool> {
        if self.0.database.as_ref().map_or(true, |d| d == db)
            && self.0.collection.as_ref().map_or(true, |c| c == collection)
        {
            println!("{}", bson::Document::try_from(doc)?);
            self.0.count -= 1;
        }
        return Ok(self.0.count != 0);
    }
}

pub fn read_archive(args: ReadArchiveArgs) -> io::Result<()> {
    let file = File::open(&args.archive)?;
    let mut reader = BufReader::new(file);

    let mut parser = PrintingParser(args);
    crate::mongodump::parser::parse_archive(&mut reader, &mut parser).unwrap();

    Ok(())
}
