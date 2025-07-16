//! Vector handling: formatting/quantization and distance computation.

use crate::distance::VectorSimilarity;

// XXX so we are able to create a formatter that is run on ingress and can be used for both the
// nav vector and raw vector tables, which also makes the raw vector table _optional_.

// XXX i don't want the raw format to dictate the scorer we use because that would force scoring
// to be symmetric and that is not great for us. for a given table we know both the similarity
// and the format so we can create a symmetric implementation but we might need more information
// to create an asymmetric implementation.

pub trait F32VectorFormatter: Send + Sync {
    fn format(&self, vector: &[f32]) -> Vec<u8> {
        let mut out = vec![0; self.byte_len(vector.len())];
        self.format_to(vector, &mut out);
        out
    }

    fn format_to(&self, vector: &[f32], out: &mut [u8]);
    fn byte_len(&self, dimensions: usize) -> usize;
}

// XXX this could include normalization which would be pretty fucking sweet.
pub enum F32VectorFormat {
    /// Little-endian f32 values encoded as bytes.
    Raw,
    /// Single bit (sign bit) per dimension.
    BinaryQuantized,
    /// Normalize and quantize into an i8 value.
    I8NaiveQuantized,
}

pub enum F32QueryVectorDistance {
    Raw(VectorSimilarity),
    I8NaiveQuantized(VectorSimilarity),
    BinaryQuantized,
    AsymmetricBinaryQuantized(usize),
}
