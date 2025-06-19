use std::{
    borrow::Cow,
    io,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use serde::{Deserialize, Serialize};
use simsimd::SpatialSimilarity;
use wt_mdb::{
    config::{ConfigItem, ConfigParser},
    options::CreateOptionsBuilder,
    Error, RecordCursor, RecordView, Result, Session,
};

use crate::distance::VectorSimilarity;

// XXX =============== DISTANCE MOVE BEGIN ===============

/// Score a fixed query provided at object creation against an arbitrary vector.
// XXX this will have all of our existing quantized implementations + f32.
// XXX I can build a trivial edge computer that builds a store of byte vectors a creates
// new QueryDistance for each pruning pass.
pub trait QueryDistance {
    /// Compute the distance between the bound query and vector.
    // XXX it's annoying as fuck but maybe this should return Option<f64>? almost everything
    // has the ability to bail with an error at this point.
    fn distance(&self, vector: &[u8]) -> f64;
}

/// Utility for coercing [f32] and [u8] into a float-vector-comparable shape as cheaply as possible.
#[repr(transparent)]
struct Float32Vector<'a>(Cow<'a, [f32]>);

impl<'a> From<&'a [f32]> for Float32Vector<'a> {
    fn from(value: &'a [f32]) -> Self {
        Self(value.into())
    }
}

impl<'a> From<&'a [u8]> for Float32Vector<'a> {
    fn from(value: &'a [u8]) -> Self {
        #[cfg(target_endian = "little")]
        if let Ok(s) = bytemuck::try_cast_slice::<_, f32>(value) {
            return Self::from(s);
        }
        // Either the platform is big-endian or the data is unaligned.
        Self(
            value
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>()
                .into(),
        )
    }
}

impl<'a> Deref for Float32Vector<'a> {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

struct Float32EuclideanDistance<'a>(Float32Vector<'a>);

impl<'a, V: Into<Float32Vector<'a>>> From<V> for Float32EuclideanDistance<'a> {
    fn from(value: V) -> Self {
        Self(value.into())
    }
}

impl QueryDistance for Float32EuclideanDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = Float32Vector::from(vector);
        SpatialSimilarity::l2sq(&self.0, &vector).expect("same size")
    }
}

struct Float32DotProductDistance<'a>(Float32Vector<'a>);

impl<'a, V: Into<Float32Vector<'a>>> From<V> for Float32DotProductDistance<'a> {
    fn from(value: V) -> Self {
        Self(value.into())
    }
}

impl QueryDistance for Float32DotProductDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = Float32Vector::from(vector);
        SpatialSimilarity::dot(&self.0, &vector).expect("same size")
    }
}

// XXX =============== DISTANCE MOVE END ===============

// XXX =============== QUANTIZATION MOVE BEGIN ===============

/// Encodes a float vector into a byte string.
///
/// Vectors may be encoded for storage, compression, or both.
pub trait VectorEncoder {
    /// Encodes the contents of vector and returns it.
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.encoded_bytes(vector.len()));
        self.encode_into(vector, &mut out);
        out
    }

    /// Encodes the contents of `vector` to `out`.
    fn encode_into(&self, vector: &[f32], out: &mut Vec<u8>);

    /// Returns the number of output bytes required to encode a vector with `dimensions`
    fn encoded_bytes(&self, dimensions: usize) -> usize;
}

struct FloatVectorEncoder;

impl VectorEncoder for FloatVectorEncoder {
    fn encode_into(&self, vector: &[f32], out: &mut Vec<u8>) {
        for d in vector {
            out.extend_from_slice(&d.to_le_bytes());
        }
    }

    fn encoded_bytes(&self, dimensions: usize) -> usize {
        dimensions * 4
    }
}

// XXX =============== QUANTIZATION MOVE END ===============

/// Describes the format used for vectors on disk.
// XXX this should generate a (narrow!) quantizer or vector coder for this purpose
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Representation {
    /// Each dimension is an [`f32`]. This is effectively un-quantized.
    Float32,
    /// Each dimension is mapped to a single bit around the origin and packed
    /// into 8 dimensions per byte.
    Binary,
    /// Each dimension is quantized to an [`i8`]` with the l2 norm stored at the end as an [`f32`].
    I8Naive,
}

impl Representation {
    fn vector_encoder(&self) -> &dyn VectorEncoder {
        match self {
            Self::Float32 => &FloatVectorEncoder,
            _ => unimplemented!(),
        }
    }
}

/// Describe the shape and encoding of vectors stored in a table.
///
/// This is stored as JSON in the `app_metadata` config passed when a table is created.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Metadata {
    /// Number of dimensions.
    pub dimensions: usize,
    /// Similarity function used to compare vectors.
    /// This may impact the on-disk representation of the vectors.
    pub similarity: VectorSimilarity,
    /// On-disk representation of each vector.
    pub representation: Representation,
}

impl Metadata {
    /// Return a new distance scorer bound to `query`.
    fn new_query_scorer<'a>(&self, query: &'a [f32]) -> Box<dyn QueryDistance + 'a> {
        match (self.representation, self.similarity) {
            (Representation::Float32, VectorSimilarity::Euclidean) => {
                Box::new(Float32EuclideanDistance::from(query))
            }
            (Representation::Float32, VectorSimilarity::Dot) => {
                Box::new(Float32DotProductDistance::from(query))
            }
            _ => todo!(),
        }
    }

    /// Return the vector encoder used to marshall vector data for storage.
    // NB: if vector encoders become stateful we could have VectorTable store the encoder.
    fn vector_encoder(&self) -> &dyn VectorEncoder {
        self.representation.vector_encoder()
    }
}

/// A record id keyed table where the values are all vectors.
pub struct VectorTable {
    table_name: String,
    metadata: Metadata,
}

impl VectorTable {
    /// Create an empty table with the given name and configuration.
    pub fn create(session: &Session, table_name: &str, metadata: Metadata) -> io::Result<Self> {
        session.create_table(
            table_name,
            Some(
                CreateOptionsBuilder::default()
                    .table_type(wt_mdb::options::TableType::Record)
                    .app_metadata(&serde_json::to_string(&metadata)?)
                    .into(),
            ),
        )?;
        Ok(Self {
            table_name: table_name.to_string(),
            metadata,
        })
    }

    /// Create a new table with the given name and configuration, then bulk load the
    /// contents of `iter`.
    ///
    /// REQUIRES: iter yields records by increasing key.
    pub fn bulk_load<'a, I>(
        _session: &Session,
        _table_name: &str,
        _metadata: Metadata,
        _iter: impl Iterator<Item = (i64, &'a [f32])>,
    ) -> Result<Self> {
        todo!()
    }

    /// Initialize table information from the database.
    pub fn from_db(session: &Session, table_name: &str) -> io::Result<Self> {
        let mut meta_cursor = session.get_metadata_cursor()?;
        let config = meta_cursor
            .seek_exact(&format!("table:{}", table_name))
            .ok_or(Error::not_found_error())??;
        let mut parser = ConfigParser::new(&config)?;
        if let ConfigItem::Struct(app_metadata) = parser
            .get("app_metadata")
            .ok_or(Error::not_found_error())??
        {
            Ok(Self {
                table_name: table_name.to_string(),
                metadata: serde_json::from_str::<Metadata>(app_metadata)?,
            })
        } else {
            Err(Error::not_found_error().into())
        }
    }

    /// The name of the underlying table.
    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    /// Metadata for this table type.
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Create a new cursor over table in `session`.
    pub fn new_cursor<'a>(self: &Arc<Self>, session: &'a Session) -> Result<VectorTableCursor<'a>> {
        Ok(VectorTableCursor {
            table: Arc::clone(self),
            cursor: session.open_record_cursor(&self.table_name)?,
        })
    }
}

/// A cursor over a vector table.
pub struct VectorTableCursor<'a> {
    table: Arc<VectorTable>,
    cursor: RecordCursor<'a>,
}

impl<'a> VectorTableCursor<'a> {
    /// The table this cursor is acting over.
    ///
    /// This reference may also be used to quasi-clone the cursor using [`VectorTable::new_cursor`].
    pub fn table(&self) -> &Arc<VectorTable> {
        &self.table
    }

    pub fn set_vector(&mut self, key: i64, vector: &[f32]) -> Result<()> {
        // XXX make it falliable.
        assert_eq!(vector.len(), self.table.metadata.dimensions);
        self.cursor.set(&RecordView::new(
            key,
            self.table.metadata.vector_encoder().encode(vector),
        ))
    }

    pub fn into_distance_scorer<'q>(self, query: &'q [f32]) -> VectorTableQueryDistance<'a, 'q> {
        // XXX make it falliable.
        assert_eq!(query.len(), self.table.metadata.dimensions);
        VectorTableQueryDistance::new(self, query)
    }
}

// XXX cursors have a reference to the session so they could be returned instead of having a custom
// guard for each cursor type. would probably need a trait for cursors.

// XXX on the fence about this but we shall see.
impl<'a> Deref for VectorTableCursor<'a> {
    type Target = RecordCursor<'a>;

    fn deref(&self) -> &Self::Target {
        &self.cursor
    }
}

impl<'a> DerefMut for VectorTableCursor<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.cursor
    }
}

/// Score a fixed query provided at object creation against vectors in the table.
pub struct VectorTableQueryDistance<'a, 'q> {
    cursor: VectorTableCursor<'a>,
    scorer: Box<dyn QueryDistance + 'q>,
}

impl<'a, 'q> VectorTableQueryDistance<'a, 'q> {
    fn new(cursor: VectorTableCursor<'a>, query: &'q [f32]) -> Self {
        let scorer = cursor.table.metadata.new_query_scorer(query);
        Self { cursor, scorer }
    }

    /// Compute the distance between the bound query vector and the vector at `record_id`
    pub fn distance(&mut self, record_id: i64) -> Option<Result<f64>> {
        unsafe { self.cursor.seek_exact_unsafe(record_id) }
            .map(|r| r.map(|v| self.scorer.distance(v.value())))
    }

    /// Extract the underlying [VectorTableCursor].
    pub fn into_cursor(self) -> VectorTableCursor<'a> {
        self.cursor
    }
}

// XXX for edge computation:
// * use a cursor to extract all the raw vectors
// * custom implementation for each on-disk vector rep x similarity.
// * just need an efficient implementation, probably per vector rep.
// * impl can manufacture a QueryDistance as needed.

#[cfg(test)]
mod test {
    use std::sync::Arc;
    use tempfile::TempDir;
    use wt_mdb::{
        options::{ConnectionOptions, ConnectionOptionsBuilder},
        Connection, Session,
    };

    use crate::{
        distance::VectorSimilarity,
        vector_table::{Metadata, Representation, VectorTable},
    };

    fn conn_options() -> Option<ConnectionOptions> {
        Some(ConnectionOptionsBuilder::default().create().into())
    }

    // NB: fields are dropped _in the order they appear_.
    // conn/session depend on the tempdir being around.
    #[allow(dead_code)]
    struct Fixture {
        session: Session,
        conn: Arc<Connection>,
        tmpdir: TempDir,
    }

    impl Default for Fixture {
        fn default() -> Self {
            let tmpdir = tempfile::tempdir().unwrap();
            let conn = Connection::open(tmpdir.path().to_str().unwrap(), conn_options()).unwrap();
            let session = conn.open_session().unwrap();
            Self {
                tmpdir,
                conn,
                session,
            }
        }
    }

    #[test]
    fn create_table() {
        let f = Fixture::default();
        let metadata = Metadata {
            dimensions: 2,
            similarity: VectorSimilarity::Euclidean,
            representation: Representation::Float32,
        };
        let create_table = VectorTable::create(&f.session, "test", metadata.clone()).unwrap();
        let read_table = VectorTable::from_db(&f.session, "test").unwrap();
        assert_eq!(create_table.metadata(), read_table.metadata());
    }

    #[test]
    fn mutate_f32_table() {
        let f = Fixture::default();
        let metadata = Metadata {
            dimensions: 2,
            similarity: VectorSimilarity::Euclidean,
            representation: Representation::Float32,
        };
        let table = Arc::new(VectorTable::create(&f.session, "test", metadata.clone()).unwrap());
        let mut cursor = table.new_cursor(&f.session).unwrap();
        cursor.set_vector(1, &[0.0, 1.0]).unwrap();
        cursor.set_vector(2, &[1.0, 0.0]).unwrap();
        let mut scorer = cursor.into_distance_scorer(&[1.0, 1.0]);
        assert_eq!(scorer.distance(0), None);
        assert_eq!(scorer.distance(1), Some(Ok(1.0)));
        assert_eq!(scorer.distance(2), Some(Ok(1.0)));
    }
}
