//! Codebooks for different bit rates based on Gaussian component distributions.
//!
//! This is useful for quantizers that expect preconditioned outputs like those that come out of
//! [`crate::Rotator`].

#![allow(clippy::excessive_precision)]

pub trait Codebook: Send + Sync {
    fn encode(&self, v: f32) -> u8;
    fn decode(&self, c: u8) -> f32;
}

const fn encode_thresholds<const N: usize>(decode: &[f32; N]) -> [f32; N] {
    let mut encode = [f32::INFINITY; N];
    let mut i = 1;
    while i < N {
        encode[i - 1] = (decode[i] + decode[i - 1]) / 2.0;
        i += 1;
    }
    encode
}

#[inline(always)]
fn select_code<const N: usize>(encode: &[f32; N], v: f32) -> u8 {
    encode.iter().map(|&x| u8::from(x < v)).sum()
}

macro_rules! declare_codebook {
    ($name:ident, $len:literal, $codebook:expr) => {
        pub struct $name {
            encode: [f32; $len * 2],
            decode: [f32; $len * 2],
        }

        impl $name {
            const fn new() -> Self {
                let mut decode = [0.0f32; $len * 2];
                let mut i = 0;
                while i < $len {
                    decode[$len + i] = $codebook[i];
                    decode[$len - i - 1] = -$codebook[i];
                    i += 1;
                }
                Self {
                    encode: encode_thresholds(&decode),
                    decode,
                }
            }
        }

        impl Codebook for $name {
            #[inline(always)]
            fn encode(&self, v: f32) -> u8 {
                select_code(&self.encode, v)
            }

            #[inline(always)]
            fn decode(&self, c: u8) -> f32 {
                self.decode[c as usize]
            }
        }
    };
}

declare_codebook!(Codebook1, 1, [0.79788456]);
declare_codebook!(Codebook2, 2, [0.452780, 1.510418]);
declare_codebook!(Codebook3, 3, [0.245094, 0.756005, 1.343909, 2.151946]);
declare_codebook!(
    Codebook4,
    8,
    [
        0.128395, 0.388048, 0.656759, 0.942340, 1.256231, 1.618046, 2.069017, 2.732590,
    ]
);
declare_codebook!(
    Codebook8OneLevel,
    128,
    [
        0.008446, 0.025339, 0.042235, 0.059135, 0.076040, 0.092952, 0.109874, 0.126806, 0.143750,
        0.160707, 0.177680, 0.194671, 0.211680, 0.228709, 0.245761, 0.262836, 0.279937, 0.297066,
        0.314223, 0.331411, 0.348632, 0.365888, 0.383180, 0.400510, 0.417881, 0.435293, 0.452750,
        0.470253, 0.487804, 0.505405, 0.523058, 0.540766, 0.558531, 0.576355, 0.594240, 0.612188,
        0.630203, 0.648286, 0.666439, 0.684667, 0.702970, 0.721353, 0.739817, 0.758365, 0.777001,
        0.795727, 0.814547, 0.833463, 0.852479, 0.871599, 0.890825, 0.910162, 0.929613, 0.949182,
        0.968873, 0.988689, 1.008636, 1.028718, 1.048939, 1.069304, 1.089818, 1.110486, 1.131313,
        1.152305, 1.173468, 1.194808, 1.216330, 1.238042, 1.259950, 1.282061, 1.304384, 1.326925,
        1.349694, 1.372698, 1.395946, 1.419449, 1.443217, 1.467259, 1.491587, 1.516213, 1.541150,
        1.566410, 1.592008, 1.617958, 1.644276, 1.670980, 1.698087, 1.725617, 1.753589, 1.782027,
        1.810953, 1.840393, 1.870375, 1.900928, 1.932084, 1.963878, 1.996348, 2.029536, 2.063485,
        2.098247, 2.133874, 2.170428, 2.207975, 2.246590, 2.286354, 2.327362, 2.369717, 2.413539,
        2.458962, 2.506143, 2.555259, 2.606521, 2.660174, 2.716508, 2.775871, 2.838686, 2.905473,
        2.976882, 3.053742, 3.137133, 3.228500, 3.329848, 3.444072, 3.575588, 3.731666, 3.925638,
        4.186595, 4.603536
    ]
);

pub struct Codebook8 {
    base: Codebook8OneLevel,
    encode_block: [f32; 16],
}

impl Codebook8 {
    const fn new() -> Self {
        let base = Codebook8OneLevel::new();
        let mut encode_block = [0f32; 16];
        let mut i = 0;
        while i < 16 {
            encode_block[i] = base.encode[i * 16 + 15];
            i += 1;
        }
        Self { base, encode_block }
    }
}

impl Codebook for Codebook8 {
    #[inline(always)]
    fn encode(&self, v: f32) -> u8 {
        let block = select_code(&self.encode_block, v);
        block * 16 + select_code(&self.base.encode.as_chunks::<16>().0[block as usize], v)
    }

    #[inline(always)]
    fn decode(&self, c: u8) -> f32 {
        self.base.decode(c)
    }
}

pub const CODEBOOK1: Codebook1 = Codebook1::new();
pub const CODEBOOK2: Codebook2 = Codebook2::new();
#[allow(dead_code)]
pub const CODEBOOK3: Codebook3 = Codebook3::new();
pub const CODEBOOK4: Codebook4 = Codebook4::new();
pub const CODEBOOK8: Codebook8 = Codebook8::new();

#[cfg(test)]
mod test {
    use crate::codebooks::CODEBOOK1;

    #[test]
    fn codebook1() {
        assert_eq!(CODEBOOK1.decode, [-0.7978846, 0.7978846]);
        assert_eq!(CODEBOOK1.encode, [0.0, f32::INFINITY]);
    }
}
