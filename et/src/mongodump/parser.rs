use std::io;

/// A trait for callers to consume a bson document from the archive.
pub trait Consumer {
    #[allow(unused_variables)]
    fn consume_header(&mut self, header: &bson::RawDocument) -> bson::error::Result<()> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn consume_collection_metadata(
        &mut self,
        metadata: &bson::RawDocument,
    ) -> bson::error::Result<()> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn consume_namespace_header(&mut self, header: &bson::RawDocument) -> bson::error::Result<()> {
        Ok(())
    }

    /// Invoked for each document in the archive along with the db and collection it belongs to.
    /// Return false if you want to stop parsing.
    fn consume_document(
        &mut self,
        db: &str,
        collection: &str,
        doc: &bson::RawDocument,
    ) -> bson::error::Result<bool>;
}

const MAGIC_NUMBER: u32 = 0x8199e26d;
const TERMINATOR: u32 = 0xffffffff;

pub fn parse_archive(
    reader: &mut impl std::io::Read,
    consumer: &mut impl Consumer,
) -> bson::error::Result<()> {
    // Read and verify magic number
    let magic = read_u32(reader)?;
    if magic != MAGIC_NUMBER {
        return Err(bson::error::Error::from(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Invalid magic number: expected 0x{:08x}, got 0x{:08x}",
                MAGIC_NUMBER, magic
            ),
        )));
    }

    let mut doc_buf = vec![];
    read_bson_document(reader, &mut doc_buf).unwrap_or(Err(eof_error()))?;
    consumer.consume_header(bson::RawDocument::from_bytes(&doc_buf)?)?;

    while let Some(_coll_meta) = read_bson_document(reader, &mut doc_buf).transpose()? {
        consumer.consume_collection_metadata(bson::RawDocument::from_bytes(&doc_buf)?)?;
    }

    while let Some(_ns_header) = read_bson_document(reader, &mut doc_buf).transpose()? {
        let namespace_header = bson::RawDocument::from_bytes(&doc_buf)?.to_owned();
        consumer.consume_namespace_header(&namespace_header)?;
        if namespace_header.get_bool("EOF").unwrap_or(true) {
            break;
        }

        let db = namespace_header.get_str("db")?;
        let collection = namespace_header.get_str("collection")?;
        while let Some(_doc) = read_bson_document(reader, &mut doc_buf).transpose()? {
            if !consumer.consume_document(
                db,
                collection,
                bson::RawDocument::from_bytes(&doc_buf)?,
            )? {
                break;
            }
        }
    }

    Ok(())
}

fn read_u32(reader: &mut impl std::io::Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_bson_document(
    reader: &mut impl std::io::Read,
    buf: &mut Vec<u8>,
) -> Option<io::Result<()>> {
    let len = read_u32(reader).ok()?;
    match len {
        TERMINATOR => None,
        len if len < 5 => Some(Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid BSON document size: {}", len),
        ))),
        _ => {
            buf.resize(len as usize, 0);
            buf[0..4].copy_from_slice(&len.to_le_bytes());
            reader.read_exact(&mut buf[4..]).map(Some).transpose()
        }
    }
}

fn eof_error() -> io::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, "EOF reached")
}
