//! Utilities for packing an unpacking [1,8] bit values.

/// Pack a stream of B bit entries into out.
pub fn pack<const B: usize>(it: impl ExactSizeIterator<Item = u8>, out: &mut [u8]) {
    assert!(
        (it.len() * B).div_ceil(8) <= out.len(),
        "output buffer too small input={} B={} out={}",
        it.len(),
        B,
        out.len()
    );

    let mut buf = 0u16;
    let mut nbuf = 0;
    let mut out_it = out.iter_mut();
    for v in it {
        buf |= (v as u16) << nbuf;
        nbuf += B;

        if nbuf >= 8 {
            *out_it.next().unwrap() = buf as u8;
            buf >>= 8;
            nbuf -= 8;
        }
    }

    // Flush any remaining buffered bits.
    if nbuf > 0 {
        *out_it.next().unwrap() = buf as u8;
    }
}

pub struct UnpackIter<'a, const B: usize> {
    inner: std::slice::Iter<'a, u8>,
    buf: u16,
    nbuf: usize,
}

impl<'a, const B: usize> UnpackIter<'a, B> {
    fn new(encoded: &'a [u8]) -> Self {
        Self {
            inner: encoded.iter(),
            buf: 0,
            nbuf: 0,
        }
    }
}

impl<const B: usize> Iterator for UnpackIter<'_, B> {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if self.nbuf < B {
            let b = *self.inner.next()?;
            self.buf |= (b as u16) << self.nbuf;
            self.nbuf += 8;
        }

        let v = self.buf & ((1 << B) - 1);
        self.buf >>= B;
        self.nbuf -= B;
        Some(v as u8)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = (self.nbuf + self.inner.len() * 8) / B;
        (rem, Some(rem))
    }
}

impl<const B: usize> ExactSizeIterator for UnpackIter<'_, B> {}

/// Unpack a stream of B bit entries from encoded.
pub fn unpack<const B: usize>(encoded: &[u8]) -> impl ExactSizeIterator<Item = u8> {
    UnpackIter::<B>::new(encoded)
}
