use crate::vectors::{F32VectorCoder, F32VectorCoding, VectorSimilarity};

pub struct VectorCoder {
    dimensions: usize,
    coder: Box<dyn F32VectorCoder>,
}

impl VectorCoder {
    pub fn new(similarity: VectorSimilarity, dimensions: usize) -> Self {
        // Coerce dot => cosine to ensure that the truncated vectors are l2 normalized.
        let similarity = match similarity {
            VectorSimilarity::Dot => VectorSimilarity::Cosine,
            _ => similarity,
        };
        Self {
            dimensions,
            coder: F32VectorCoding::F32.new_coder(similarity),
        }
    }
}

impl F32VectorCoder for VectorCoder {
    fn byte_len(&self, dimensions: usize) -> usize {
        self.coder.byte_len(self.dimensions.min(dimensions))
    }

    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        self.coder
            .encode_to(&vector[..self.dimensions.min(vector.len())], out)
    }

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        self.coder.decode(encoded)
    }
}
