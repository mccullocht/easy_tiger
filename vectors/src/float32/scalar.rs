#[inline]
pub fn dot(a: &[u8], b: &[u8]) -> f64 {
    f32_le_iter(a)
        .zip(f32_le_iter(b))
        .map(|(a, b)| a * b)
        .sum::<f32>()
        .into()
}

#[inline]
pub fn l2sq(a: &[u8], b: &[u8]) -> f64 {
    f32_le_iter(a)
        .zip(f32_le_iter(b))
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum::<f32>()
        .into()
}

pub fn f32_le_iter<'b>(b: &'b [u8]) -> impl ExactSizeIterator<Item = f32> + 'b {
    let (chunks, rem) = b.as_chunks::<{ std::mem::size_of::<f32>() }>();
    debug_assert!(rem.is_empty());
    chunks.iter().map(|c| {
        f32::from_bits(u32::from_le(unsafe {
            // SAFETY: byte array chunk is guaranteed to be the same size as u32/f32.
            std::ptr::read_unaligned(c.as_ptr() as *const u32)
        }))
    })
}
