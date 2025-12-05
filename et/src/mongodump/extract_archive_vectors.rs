use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Write},
    path::PathBuf,
};

use clap::Args;
use easy_tiger::input::{VecVectorStore, VectorStore};
use rand::{Rng, SeedableRng};

use crate::mongodump::parser::Consumer;

#[derive(Args)]
pub struct ExtractArchiveVectorsArgs {
    /// Path to the mongodump archive file.
    #[arg(short, long)]
    archive: PathBuf,

    /// Path to the train output file.
    #[arg(short, long)]
    train: PathBuf,

    /// Path to the test output file.
    #[arg(short, long)]
    test: PathBuf,

    /// Test set size.
    #[arg(short, long)]
    test_size: usize,

    /// Filter by database name (optional).
    #[arg(short, long)]
    database: Option<String>,

    /// Filter by collection name (optional).
    #[arg(short, long)]
    collection: Option<String>,

    /// Field name containing the vector.
    #[arg(short, long)]
    field: String,

    /// Number of dimensions in the vector.
    #[arg(short, long)]
    dimensions: usize,
}

struct ExtractArchiveVectorsConsumer {
    field: String,
    dimensions: usize,
    db: Option<String>,
    collection: Option<String>,

    train: BufWriter<File>,
    test: VecVectorStore<f32>,
    test_size: usize,

    seen: usize,
    test_sampler: rand_xoshiro::Xoshiro256PlusPlus,
}

impl Consumer for ExtractArchiveVectorsConsumer {
    fn consume_document(
        &mut self,
        db: &str,
        collection: &str,
        doc: &bson::RawDocument,
    ) -> bson::error::Result<bool> {
        if !self.db.as_ref().map_or(true, |d| d == db)
            || !self.collection.as_ref().map_or(true, |c| c == collection)
        {
            return Ok(true);
        }

        let vector_bref = doc.get_binary(&self.field)?;
        if vector_bref.subtype != bson::spec::BinarySubtype::Vector {
            return Ok(true);
        }

        assert_eq!(
            vector_bref.bytes[0], 0x27,
            "Unsupported vector type: {:x}",
            vector_bref.bytes[0]
        );
        assert_eq!(vector_bref.bytes[2..].len(), self.dimensions * 4);
        let vector = vector_bref.bytes[2..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect::<Vec<_>>();
        self.seen += 1;
        if self.test.len() < self.test_size {
            self.test.push(&vector);
            return Ok(true);
        }

        let sample_idx = self.test_sampler.random_range(0..self.seen);
        if sample_idx < self.test_size {
            write_vector(&self.test[sample_idx], &mut self.train)?;
            self.test[sample_idx].copy_from_slice(&vector);
        } else {
            write_vector(&vector, &mut self.train)?;
        }
        Ok(true)
    }
}

fn write_vector(vector: &[f32], writer: &mut impl Write) -> io::Result<()> {
    for d in vector {
        writer.write_all(&d.to_le_bytes())?;
    }
    Ok(())
}

pub fn extract_archive_vectors(args: ExtractArchiveVectorsArgs) -> io::Result<()> {
    let mut archive = BufReader::new(File::open(&args.archive)?);
    let train = BufWriter::new(File::create(&args.train)?);
    let test = VecVectorStore::with_capacity(args.dimensions, args.test_size);
    let test_sampler = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0xbeabad60061ed00d);
    let mut consumer = ExtractArchiveVectorsConsumer {
        field: args.field,
        dimensions: args.dimensions,
        db: args.database,
        collection: args.collection,
        train,
        test,
        test_size: args.test_size,
        seen: 0,
        test_sampler,
    };
    crate::mongodump::parser::parse_archive(&mut archive, &mut consumer).unwrap();

    let mut test = BufWriter::new(File::create(&args.test)?);
    for vector in consumer.test.iter() {
        write_vector(vector, &mut test)?;
    }
    Ok(())
}
