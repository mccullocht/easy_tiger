use std::{
    borrow::Cow,
    collections::HashMap,
    sync::{LazyLock, RwLock},
};

use crate::{F32VectorCoder, float32, rotate::Rotator};

const ROTATOR_SEED: u64 = 15628395401334080154;
static ROTATORS: LazyLock<RwLock<HashMap<usize, Box<Rotator>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

fn get_rotator(dim: usize) -> &'static Rotator {
    {
        let m = ROTATORS.read().unwrap();
        if let Some(r) = m.get(&dim) {
            // SAFETY: Box provides a consistent address, we never remove anything from the map.
            return unsafe { std::mem::transmute::<&Rotator, &'static Rotator>(r.as_ref()) };
        }
    }
    let mut m = ROTATORS.write().unwrap();
    // SAFETY: Box provides a consistent address, insert only happens if not present.
    let r = m
        .entry(dim)
        .or_insert_with(|| Box::new(Rotator::new(dim, ROTATOR_SEED)));
    unsafe { std::mem::transmute::<&Rotator, &'static Rotator>(r.as_ref()) }
}

#[derive(Debug, Copy, Clone, PartialEq, Default)]
struct Header {
    /// L2 norm of the original centered vector.
    l2_norm: f32,
    /// Inner product of the quantized vector and normalized original vector.
    correction_term: f32,
    /// Sum of all of the component bits.
    component_sum: u32,
}

impl Header {
    const LEN: usize = 12;

    fn split_mut(bytes: &mut [u8]) -> (&mut [u8; Self::LEN], &mut [u8]) {
        let (hbytes, vbytes) = bytes.split_at_mut(Self::LEN);
        (&mut hbytes.as_chunks_mut::<{ Self::LEN }>().0[0], vbytes)
    }

    fn encode(&self, out: &mut [u8; Self::LEN]) {
        let parts = out.as_chunks_mut::<4>().0;
        parts[0] = self.l2_norm.to_le_bytes();
        parts[1] = self.correction_term.to_le_bytes();
        parts[2] = self.component_sum.to_le_bytes();
    }

    fn decode(raw: &[u8]) -> (Header, &[u8]) {
        let (hbytes, vbytes) = raw.split_at(Self::LEN);
        let parts = hbytes.as_chunks::<4>().0;
        (
            Header {
                l2_norm: f32::from_le_bytes(parts[0]),
                correction_term: f32::from_le_bytes(parts[1]),
                component_sum: u32::from_le_bytes(parts[2]),
            },
            vbytes,
        )
    }
}

#[derive(Debug, Clone, Default)]
pub struct Coder {
    center: Option<Vec<f32>>,
}

impl Coder {
    pub fn new(center: Option<Vec<f32>>) -> Self {
        Self { center }
    }
}

impl F32VectorCoder for Coder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let centered_vector: Cow<'_, [f32]> = if let Some(center) = self.center.as_ref() {
            vector
                .iter()
                .zip(center.iter())
                .map(|(v, c)| *v - *c)
                .collect::<Vec<_>>()
                .into()
        } else {
            vector.into()
        };
        let mut header = Header::default();
        header.l2_norm = float32::l2_norm(&centered_vector);
        let unit_vector = float32::l2_normalize(centered_vector);
        let rotator = get_rotator(unit_vector.len());
        let rotated = rotator.forward(vector.as_ref());
        header.correction_term =
            rotated.iter().copied().map(f32::abs).sum::<f32>() / (rotated.len() as f32).sqrt();

        let (hbytes, vbytes) = Header::split_mut(out);
        header.component_sum = rotated
            .chunks(8)
            .zip(vbytes.iter_mut())
            .map(|(i, o)| {
                let b = i
                    .iter()
                    .enumerate()
                    .map(|(i, x)| if x.is_sign_negative() { 1u32 << i } else { 0 })
                    .reduce(|a, b| a | b)
                    .unwrap();
                *o = b as u8;
                b.count_ones()
            })
            .sum::<u32>();

        header.encode(hbytes);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        Header::LEN + dimensions.div_ceil(8)
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        let (_, vector) = Header::decode(encoded);
        let sign_mask = 1u32 << 31;
        let magnitude = 1.0 / (out.len() as f32).sqrt();
        for (&c, o) in vector.iter().zip(out.chunks_mut(8)) {
            let c = c as u32;
            for (i, o) in o.iter_mut().enumerate() {
                *o = f32::from_bits(magnitude.to_bits() ^ ((c << (31 - i)) & sign_mask));
            }
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        (byte_len - Header::LEN) * 8
    }
}
