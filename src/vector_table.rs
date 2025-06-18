use std::{borrow::Cow, ops::Deref, sync::Arc};

use serde::{Deserialize, Serialize};
use simsimd::SpatialSimilarity;
use wt_mdb::{RecordCursor, Result, Session};

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
    pub fn new_query_scorer<'a>(&self, query: &'a [f32]) -> Box<dyn QueryDistance + 'a> {
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

    pub fn new_vector_encoder(&self) -> Box<dyn VectorEncoder> {
        match self.representation {
            Representation::Float32 => Box::new(FloatVectorEncoder),
            _ => todo!(),
        }
    }
}

/// A record id keyed table where the values are all vectors.
pub struct VectorTable {
    _table_name: String,
    _metadata: Metadata,
}

impl VectorTable {
    /// Create an empty table with the given name and configuration.
    pub fn create(_table_name: &str, _metadata: Metadata) -> Result<Self> {
        todo!()
    }

    /// Create a new table with the given name and configuration, then bulk load the
    /// contents of `iter`.
    ///
    /// REQUIRES: iter yields records by increasing key.
    pub fn bulk_load<'a, I>(
        _table_name: &str,
        _metadata: Metadata,
        _iter: impl Iterator<Item = (i64, &'a [f32])>,
    ) -> Result<Self> {
        todo!()
    }

    /// Access a table in the database.
    pub fn from_table(_table_name: &str) -> Result<Self> {
        todo!()
    }

    /// Create a new cursor over table in `session`.
    pub fn new_cursor<'a>(
        self: &Arc<Self>,
        _session: &'a Session,
    ) -> Result<VectorTableCursor<'a>> {
        todo!()
    }
}

pub struct VectorTableCursor<'a> {
    _table: Arc<VectorTable>,
    _cursor: RecordCursor<'a>,
}

// XXX VectorTableCursor can deref to a RecordCursor
// XXX VectorTableCursor can accept f32 input and write it.
// XXX VectorTableCursor can get the raw bytes for a vector (unsafely or unsafely)
// XXX VectorTableCursor can be created from an existing cursor if desired (but not duplicated)

/// Score a fixed query provided at object creation against vectors in the table.
pub struct VectorTableQueryDistance<'a> {
    _cursor: VectorTableCursor<'a>,
    _distance: Box<dyn QueryDistance>,
}

// XXX scoring:
// * each vector type has an in-memory representation and on-disk representation
// * in-memory rep may have different alignments reqs (float vs u8)
// * in-memory rep may need non-trivial transforms from on-disk (little->big endian)
// * in-memory => on-disk is isolated (?) to the ingestion path
// * when query scoring (vec x id) we want query to be in-memory and we will do on-disk -> in-memory transformation
// * edge computation needs to read a bunch of vectors and produce their in-memory rep for scoring.

// XXX for edge computation:
// * wrap VectorTableCursor
// * enum CachedVector { RecordId(i64), Vector(V) }
// problem: i don't know what V is
// i could also have an object-safe trait for edge computation and the implementation could be templated(?)
