//! QuIVer two-bit training free quantization: https://arxiv.org/html/2605.02171v1

#[cfg(target_arch = "aarch64")]
mod aarch64;
mod scalar;

use crate::{F32VectorCoder, QueryVectorDistance, VectorDistance};
use std::borrow::Cow;

/// Encapsulates operations that we may choose to accelerate using platform-specific intrinsics.
trait Kernel: Send + Sync {
    /// Compute `tau` parameter: the mean absolute value of every element in `v`.
    fn tau(v: &[f32]) -> f32;

    /// Compute symmetric distance between two vectors packed using `TurboPacker<2>`.
    fn symmetric_distance(a: &[u8], b: &[u8]) -> i32;

    /// Compute the asymmetric distance between an i8 quantized query vector and a document vector
    /// packed using `TurboPacker<2>` where the magnitude bit represents `weak` or `strong`.
    fn asymmetric_distance(q: &[i8], d: &[u8], weak: i8, strong: i8) -> i32;
}

#[derive(Default, Debug, Copy, Clone)]
struct MeanComputer {
    count: f32,
    mean: f32,
}

impl MeanComputer {
    fn add(&mut self, value: f32) {
        self.count += 1.0;
        self.mean += (value - self.mean) / self.count;
    }
}

struct Header {
    weak: f32,
    strong: f32,
    strong_count: u32,
}

impl Header {
    const LEN: usize = 12;

    fn split(data: &[u8]) -> ([u8; Self::LEN], &[u8]) {
        let (header, vector) = data.split_at(Self::LEN);
        (header.as_chunks::<{ Self::LEN }>().0[0], vector)
    }

    fn split_mut(data: &mut [u8]) -> (&mut [u8; Self::LEN], &mut [u8]) {
        let (header, vector) = data.split_at_mut(Self::LEN);
        (&mut header.as_chunks_mut::<{ Self::LEN }>().0[0], vector)
    }

    fn decode(raw: [u8; Self::LEN]) -> Self {
        let items = raw.as_ref().as_chunks::<4>().0;
        Self {
            weak: f32::from_le_bytes(items[0]),
            strong: f32::from_le_bytes(items[1]),
            strong_count: u32::from_le_bytes(items[2]),
        }
    }

    fn encode(&self, raw: &mut [u8; Self::LEN]) {
        let items = raw.as_mut().as_chunks_mut::<4>().0;
        items[0] = self.weak.to_le_bytes();
        items[1] = self.strong.to_le_bytes();
        items[2] = self.strong_count.to_le_bytes();
    }

    fn split_and_decode(data: &[u8]) -> (Header, &[u8]) {
        let (h, v) = Self::split(data);
        (Self::decode(h), v)
    }
}

/// Coder for QuiVer format.
///
/// The stored encoding contains 2 bits for every dimension packed into bytes.
///
/// The vector begins with additional terms defined by the `Header` struct.
struct Coder<K: Kernel>(K);

impl<K: Kernel> F32VectorCoder for Coder<K> {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let tau = K::tau(vector);
        let mut strong = MeanComputer::default();
        let mut weak = MeanComputer::default();

        let (header_bytes, vector_bytes) = Header::split_mut(out);
        vector_bytes.fill(0);
        let mut packer = super::lvq::packing::TurboPacker::<2>::new(vector_bytes);
        for &v in vector.iter() {
            let q = if v > 0.0 { 2u8 } else { 0u8 }
                | if v.abs() > tau {
                    strong.add(v.abs());
                    1u8
                } else {
                    weak.add(v.abs());
                    0u8
                };
            packer.push(q);
        }
        Header {
            weak: weak.mean,
            strong: strong.mean,
            strong_count: strong.count as u32,
        }
        .encode(header_bytes);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions.div_ceil(4) + Header::LEN
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        let (header_bytes, vector_bytes) = Header::split(encoded);
        let header = Header::decode(header_bytes);
        let decode_table = [-header.weak, -header.strong, header.weak, header.strong];
        for (i, o) in super::lvq::packing::TurboUnpacker::<2>::new(vector_bytes).zip(out.iter_mut())
        {
            *o = decode_table[i as usize];
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        (byte_len - Header::LEN) / 4
    }
}

/// Symmetric distance computation for QuiVer vectors.
///
/// Distance computation ignores the stored tau value. The score most closely resembles an inner
/// product and is normalized as such since the min/max values are [-D*4, +D*4].
struct Distance<K: Kernel>(K);

impl<K: Kernel> VectorDistance for Distance<K> {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let (query_header, query) = Header::split_and_decode(query);
        let (doc_header, doc) = Header::split_and_decode(doc);

        let raw_dist = K::symmetric_distance(query, doc);
        // Use strong count to compute a more accurate denominator for cosine similarity.
        let dim = query.len() as u32 * 4;
        let q_mag = query_header.strong_count * 4 + (dim - query_header.strong_count);
        let d_mag = doc_header.strong_count * 4 + (dim - doc_header.strong_count);
        let norm_factor = ((q_mag as u64 * d_mag as u64) as f64).sqrt();

        // Divide raw distance by norm_factor to get value in [-1,+1], then invert and add to get a
        // distance in [0,1].
        (raw_dist as f64 / norm_factor as f64) * -0.5 + 0.5
    }
}

struct SymmetricalQueryDistance<'q, K: Kernel> {
    dist: Distance<K>,
    query: Cow<'q, [u8]>,
}

impl<'q, K: Kernel> SymmetricalQueryDistance<'q, K> {
    fn new(kernel: K, query: Cow<'q, [u8]>) -> Self {
        Self {
            dist: Distance(kernel),
            query,
        }
    }
}

impl<K: Kernel> QueryVectorDistance for SymmetricalQueryDistance<'_, K> {
    fn distance(&self, vector: &[u8]) -> f64 {
        self.dist.distance(self.query.as_ref(), vector)
    }
}

#[inline(always)]
fn quantize_i8(value: f32, scale: f32) -> i8 {
    (value * scale).round() as i8
}

struct QueryDistance<K: Kernel> {
    _kernel: K,
    query: Vec<i8>,
    scale: f32,
    magnitude: i32,
}

impl<K: Kernel> QueryDistance<K> {
    pub fn new(kernel: K, query: &[f32]) -> Self {
        let max = query
            .iter()
            .copied()
            .map(f32::abs)
            .max_by(f32::total_cmp)
            .unwrap();
        let scale = 127.0 / max;
        let query = query
            .iter()
            .map(|&d| quantize_i8(d, scale))
            .collect::<Vec<_>>();
        let magnitude = query.iter().map(|&d| d as i32 * d as i32).sum::<i32>();
        Self {
            _kernel: kernel,
            query,
            scale,
            magnitude,
        }
    }
}

impl<K: Kernel> QueryVectorDistance for QueryDistance<K> {
    fn distance(&self, vector: &[u8]) -> f64 {
        // Read strong/weak value and encode as i8
        let (header, vector) = Header::split_and_decode(vector);
        let strong = quantize_i8(header.strong, self.scale);
        let weak = quantize_i8(header.weak, self.scale);

        let raw_dist = K::asymmetric_distance(&self.query, vector, weak, strong);
        let doc_magnitude = (strong as i32 * strong as i32 * header.strong_count as i32)
            + (weak as i32 * weak as i32 * (self.query.len() as i32 - header.strong_count as i32));
        let distance_scale = (self.magnitude as f64 * doc_magnitude as f64).sqrt();

        (raw_dist as f64 / distance_scale) * -0.5 + 0.5
    }
}

pub fn new_coder() -> Box<dyn F32VectorCoder> {
    if cfg!(target_arch = "aarch64") {
        Box::new(Coder(aarch64::Neon))
    } else {
        Box::new(Coder(scalar::Scalar))
    }
}

pub fn new_symmetric_distance() -> Box<dyn VectorDistance> {
    if cfg!(target_arch = "aarch64") {
        Box::new(Distance(aarch64::Neon))
    } else {
        Box::new(Distance(scalar::Scalar))
    }
}

pub fn new_symmetric_query_distance<'a>(query: Cow<'a, [u8]>) -> Box<dyn QueryVectorDistance + 'a> {
    if cfg!(target_arch = "aarch64") {
        Box::new(SymmetricalQueryDistance::new(aarch64::Neon, query))
    } else {
        Box::new(SymmetricalQueryDistance::new(scalar::Scalar, query))
    }
}

pub fn new_asymmetric_distance(query: &[f32]) -> Box<dyn QueryVectorDistance> {
    if cfg!(target_arch = "aarch64") {
        Box::new(QueryDistance::new(aarch64::Neon, query))
    } else {
        Box::new(QueryDistance::new(scalar::Scalar, query))
    }
}

// XXX for testing add a mechanism to get scalar implementations
