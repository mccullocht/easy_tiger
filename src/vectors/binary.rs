use crate::vectors::F32VectorCoder;

#[derive(Debug, Copy, Clone)]
pub struct BinaryQuantizedVectorCoder;

impl F32VectorCoder for BinaryQuantizedVectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        for (c, o) in vector.chunks(8).zip(out.iter_mut()) {
            *o = c
                .iter()
                .enumerate()
                .filter_map(|(i, d)| if *d > 0.0 { Some(1u8 << i) } else { None })
                .fold(0, |a, b| a | b)
        }
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions.div_ceil(8)
    }
}

// XXX maybe remove this entirely and dump it into a QueryVectorDistance implementation.
#[derive(Debug, Copy, Clone)]
pub struct AsymmetricBinaryQuantizedVectorCoder(usize);

impl AsymmetricBinaryQuantizedVectorCoder {
    /// Create a new quantizer.
    ///
    /// *Panics* if `!(1..=8).contains(&n)`
    pub fn new(n: usize) -> Self {
        assert!((1..=8).contains(&n));
        Self(n)
    }

    /// Produce a packed u8 representation containing only `bit` from each byte of `dword`.
    fn summarize_chunk(mut dword: u64, bit: usize) -> u8 {
        // First, shift and mask so that only `bit` is present as the lowest bit in each byte.
        // Shift bit to the lowest position in each byte of word.
        dword = (dword >> bit) & 0x0101010101010101;

        // Add a magic constant and mask to get a different bit set for each value if the
        // bit for that byte is present in the input word.
        dword = (dword + 0x7f3f1f0f_07030100) & 0x80402010_08040201;

        // Reduce and mask down to a single byte;
        let word = dword | (dword >> 32);
        let hword = word | (word >> 16);
        (hword | (hword >> 8)) as u8
    }
}

impl F32VectorCoder for AsymmetricBinaryQuantizedVectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        if vector.is_empty() || self.0 == 1 {
            BinaryQuantizedVectorCoder.encode_to(vector, out);
        }

        // Scale each dimension to be in [0, 2^n) and produce a trivially quantized vector.
        let (min, max) = vector.iter().fold((f32::MAX, f32::MIN), |(min, max), d| {
            assert!(!d.is_nan());
            (min.min(*d), max.max(*d))
        });
        assert!(min <= max);
        let scale = (max - min) / ((1 << self.0) - 1) as f32;
        let trivial_quantized = vector
            .iter()
            .map(|d| ((*d - min) / scale).round() as u8)
            .collect::<Vec<_>>();

        let doc_bytes = vector.len().div_ceil(8);

        for (i, chunk) in trivial_quantized.chunks(8).enumerate() {
            let word = if chunk.len() == 8 {
                u64::from_le_bytes(chunk.try_into().expect("exactly 8 bytes"))
            } else {
                assert!(chunk.len() <= 8);
                let mut bytes = [0u8; 8];
                bytes[0..chunk.len()].copy_from_slice(chunk);
                u64::from_le_bytes(bytes)
            };
            for bit in 0..self.0 {
                out[doc_bytes * bit + i] = Self::summarize_chunk(word, bit);
            }
        }
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions.div_ceil(8) * self.0
    }
}
