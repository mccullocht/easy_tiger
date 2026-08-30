//! Turbo packing format shared by the LVQ, QuIVer, and RaBitQ quantizers.

use std::iter::FusedIterator;

/// The turbo coder requires that all vector data be packed into 16-byte blocks.
pub(crate) const TURBO_BLOCK_SIZE: usize = 16;

/// The number of bytes required to pack `dimensions` with `bits` per entry.
pub(crate) const fn byte_len(dimensions: usize, bits: usize) -> usize {
    (dimensions * bits).div_ceil(8)
}

pub(crate) struct TurboPacker<'a, const B: usize> {
    blocks: &'a mut [[u8; TURBO_BLOCK_SIZE]],
    tail: &'a mut [u8],
    block: usize,
    nbuf: usize,
}

impl<'a, const B: usize> TurboPacker<'a, B> {
    pub(crate) fn new(vector_bytes: &'a mut [u8]) -> Self {
        let (blocks, tail) = vector_bytes.as_chunks_mut::<TURBO_BLOCK_SIZE>();
        Self {
            blocks,
            tail,
            block: 0,
            nbuf: 0,
        }
    }

    pub(crate) fn push(&mut self, q: u8) {
        if self.block < self.blocks.len() {
            let block = &mut self.blocks[self.block];
            let byte = self.nbuf % TURBO_BLOCK_SIZE;
            let shift = self.nbuf / TURBO_BLOCK_SIZE * B;
            block[byte] |= q << shift;
            self.nbuf += 1;
            if self.nbuf == (TURBO_BLOCK_SIZE * 8) / B {
                self.block += 1;
                self.nbuf = 0;
            }
        } else {
            let byte = self.nbuf % self.tail.len();
            let shift = self.nbuf / self.tail.len() * B;
            self.tail[byte] |= q << shift;
            self.nbuf += 1;
            if self.nbuf == self.tail.len() * 8 / B {
                self.block += 1;
                self.nbuf = 0;
            }
        }
    }
}

pub(crate) struct TurboUnpacker<'a, const B: usize> {
    blocks: &'a [[u8; TURBO_BLOCK_SIZE]],
    tail: &'a [u8],
    block: usize,
    pos: usize,
}

impl<'a, const B: usize> TurboUnpacker<'a, B> {
    pub(crate) fn new(vector_bytes: &'a [u8]) -> Self {
        let (blocks, tail) = vector_bytes.as_chunks::<TURBO_BLOCK_SIZE>();
        Self {
            blocks,
            tail,
            block: 0,
            pos: 0,
        }
    }
}

impl<'a, const B: usize> Iterator for TurboUnpacker<'a, B> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.block < self.blocks.len() {
            let block = &self.blocks[self.block];
            let byte = self.pos % TURBO_BLOCK_SIZE;
            let shift = self.pos / TURBO_BLOCK_SIZE * B;
            let v = (block[byte] >> shift) & u8::MAX >> (8 - B);
            self.pos += 1;
            if self.pos == (TURBO_BLOCK_SIZE * 8) / B {
                self.block += 1;
                self.pos = 0;
            }
            Some(v)
        } else if !self.tail.is_empty() && self.block == self.blocks.len() {
            let byte = self.pos % self.tail.len();
            let shift = self.pos / self.tail.len() * B;
            let v = (self.tail[byte] >> shift) & u8::MAX >> (8 - B);
            self.pos += 1;
            if self.pos == self.tail.len() * 8 / B {
                self.block += 1;
                self.pos = 0;
            }
            Some(v)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let total = (self.blocks.len() * TURBO_BLOCK_SIZE * 8) / B + self.tail.len() * 8 / B;
        let next = self.block * TURBO_BLOCK_SIZE + self.pos;
        (total - next, Some(total - next))
    }
}

impl<'a, const B: usize> FusedIterator for TurboUnpacker<'a, B> {}

impl<'a, const B: usize> ExactSizeIterator for TurboUnpacker<'a, B> {}

/// Take a 4 bit encoded input and split it into 4 bitplanes.
///
/// The resulting bitplanes are interleaved at 16 bytes chunks until the tail when they are
/// interleaved in the turbo packing format.
pub(crate) fn bitplane_split4(vector: &[u8]) -> Vec<u8> {
    // 64 bytes contains 128 dims, which is enough to populate 4 128 bit bitplanes.
    let head_len = vector.len() & !63;
    let tail_dim = (vector.len() & 63) * 2;
    let tail_len = tail_dim.div_ceil(8) * 4;
    let len = head_len + tail_len;
    let mut out = vec![0u8; len];
    let (head, tail) = vector.as_chunks::<64>();
    let (ohead, otail) = out.split_at_mut(head.len() * 64);
    let ohead = ohead.as_chunks_mut::<64>().0;
    let nibble_mask = u128::from_ne_bytes([0xf; 16]);
    let bit_mask = u128::from_ne_bytes([1; 16]);
    for (c, o) in head.iter().zip(ohead.iter_mut()) {
        let mut b0 = 0u128;
        let mut b1 = 0u128;
        let mut b2 = 0u128;
        let mut b3 = 0u128;
        for (i, b) in c.as_chunks::<16>().0.iter().enumerate() {
            let b = u128::from_le_bytes(*b);
            let lo = b & nibble_mask;
            let hi = (b >> 4) & nibble_mask;

            b0 |= (lo & bit_mask) << (i * 2);
            b0 |= (hi & bit_mask) << (i * 2 + 1);
            b1 |= ((lo >> 1) & bit_mask) << (i * 2);
            b1 |= ((hi >> 1) & bit_mask) << (i * 2 + 1);
            b2 |= ((lo >> 2) & bit_mask) << (i * 2);
            b2 |= ((hi >> 2) & bit_mask) << (i * 2 + 1);
            b3 |= ((lo >> 3) & bit_mask) << (i * 2);
            b3 |= ((hi >> 3) & bit_mask) << (i * 2 + 1);
        }

        let planes = o.as_chunks_mut::<16>().0;
        planes[0] = b0.to_le_bytes();
        planes[1] = b1.to_le_bytes();
        planes[2] = b2.to_le_bytes();
        planes[3] = b3.to_le_bytes();
    }

    if !tail.is_empty() {
        assert!(otail.len().is_multiple_of(4));
        let mut oiter = otail.chunks_mut(otail.len() / 4);
        let mut b0 = TurboPacker::<1>::new(oiter.next().unwrap());
        let mut b1 = TurboPacker::<1>::new(oiter.next().unwrap());
        let mut b2 = TurboPacker::<1>::new(oiter.next().unwrap());
        let mut b3 = TurboPacker::<1>::new(oiter.next().unwrap());
        for d in TurboUnpacker::<4>::new(tail) {
            b0.push(d & 1);
            b1.push((d >> 1) & 1);
            b2.push((d >> 2) & 1);
            b3.push((d >> 3) & 1);
        }
    }

    out
}

/// Return the number of dimensions that can be packed into a single block.
///
/// So long as `bits` is a power of 2 the returned value will _also_ be a power of 2.
/// This is useful for splitting between the head and tail during vector coding tasks.
pub(crate) const fn block_dim(bits: usize) -> usize {
    (TURBO_BLOCK_SIZE * 8) / bits
}
