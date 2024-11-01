/// Return the number of output bytes required to binary quantize a vector of `dimensions` length.
pub fn binary_quantized_bytes(dimensions: usize) -> usize {
    (dimensions + 7) / 8
}

/// Binary quantized `vector ` to `out`.
///
/// REQUIRES: out.len() == binary_quantized_bytes(vector.len())
pub fn binary_quantize_to(vector: &[f32], out: &mut [u8]) {
    for (i, d) in vector.iter().enumerate() {
        if *d > 0.0 {
            out[i / 8] |= 1 << (i % 8);
        }
    }
}

/// Return binary quantized form of `vector`.
pub fn binary_quantize(vector: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; binary_quantized_bytes(vector.len())];
    binary_quantize_to(vector, &mut out);
    out
}
