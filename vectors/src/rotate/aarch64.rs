use std::arch::aarch64::{
    float32x4_t, vaddq_f32, veorq_u32, vld1q_f32, vld1q_u32, vreinterpretq_f32_f64,
    vreinterpretq_f32_u32, vreinterpretq_f64_f32, vreinterpretq_u32_f32, vst1q_f32, vsubq_f32,
    vuzp1q_f32, vuzp1q_f64, vuzp2q_f32, vuzp2q_f64, vzip1q_f32, vzip1q_f64, vzip2q_f32, vzip2q_f64,
};

#[inline]
pub fn neon_walsh_hadamard_transform<const F: bool>(v: &mut [f32], signs: &[u32]) {
    assert!(
        v.len().is_power_of_two(),
        "Hadamard transform requires power of 2 length"
    );
    assert_eq!(v.len(), signs.len());
    if v.len() < 64 {
        return super::scalar::walsh_hadamard_transform::<F>(v, signs);
    } else {
        // Perform the early strides of the block transformation together in 64 dimension chunks
        // in an effort to improve locality. v.len() is a power of 2 and there are at least 64
        // entries, so there will be no tail entries.
        let blocks = v.as_chunks_mut::<64>().0;
        let sblocks = signs.as_chunks::<64>().0;
        if F {
            for (b, s) in blocks.iter_mut().zip(sblocks.iter()) {
                wht_fixed_block64::<true>(b, s);
            }
        } else {
            for (b, s) in blocks.iter_mut().zip(sblocks.iter()) {
                wht_fixed_block64::<false>(b, s);
            }
        }
        // Continue butterfly transformation at block size and beyond.
        wht_block::<64>(v);
    }

    // Normalize by 1/sqrt(n) to preserve distances and inner products.
    // For backwards transformation invert the sign flip here too.
    let scale = 1.0 / (v.len() as f32).sqrt();
    if F {
        for x in v.iter_mut() {
            *x *= scale;
        }
    } else {
        for (&s, x) in signs.iter().zip(v.iter_mut()) {
            *x *= f32::from_bits(scale.to_bits() ^ s);
        }
    }
}

#[inline]
fn wht_block<const S: usize>(block: &mut [f32]) {
    let n = block.len();
    assert!(
        n.is_power_of_two(),
        "Hadamard transform requires power of 2 length"
    );
    let mut h = S;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in 0..h {
                let x = block[i + j];
                let y = block[i + j + h];
                block[i + j] = x + y;
                block[i + j + h] = x - y;
            }
        }
        h *= 2;
    }
}

/// Initial base Walsh-Hadamard Transform over a fixed size block.
///
/// This includes the sign flips that are needed before the operation begins if `F` is true.
#[inline]
fn wht_fixed_block64<const F: bool>(block: &mut [f32; 64], signs: &[u32; 64]) {
    // Load 4 entries at a time, optionally flipping signs if F is true.
    fn load4<const F: bool>(b: *const f32, s: *const u32, off: usize) -> float32x4_t {
        unsafe {
            let r = vld1q_f32(b.add(off));
            if F {
                vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(r), vld1q_u32(s.add(off))))
            } else {
                r
            }
        }
    }
    let mut r = [
        load4::<F>(block.as_ptr(), signs.as_ptr(), 0),
        load4::<F>(block.as_ptr(), signs.as_ptr(), 4),
        load4::<F>(block.as_ptr(), signs.as_ptr(), 8),
        load4::<F>(block.as_ptr(), signs.as_ptr(), 12),
        load4::<F>(block.as_ptr(), signs.as_ptr(), 16),
        load4::<F>(block.as_ptr(), signs.as_ptr(), 20),
        load4::<F>(block.as_ptr(), signs.as_ptr(), 24),
        load4::<F>(block.as_ptr(), signs.as_ptr(), 28),
        load4::<F>(block.as_ptr(), signs.as_ptr(), 32),
        load4::<F>(block.as_ptr(), signs.as_ptr(), 36),
        load4::<F>(block.as_ptr(), signs.as_ptr(), 40),
        load4::<F>(block.as_ptr(), signs.as_ptr(), 44),
        load4::<F>(block.as_ptr(), signs.as_ptr(), 48),
        load4::<F>(block.as_ptr(), signs.as_ptr(), 52),
        load4::<F>(block.as_ptr(), signs.as_ptr(), 56),
        load4::<F>(block.as_ptr(), signs.as_ptr(), 60),
    ];

    // Perform butterfly rotation steps across 2 registers (8 values) for strides 1 and 2.
    fn butterfly2(a: float32x4_t, b: float32x4_t) -> (float32x4_t, float32x4_t) {
        unsafe {
            // 32 bit unzip to arranage as [0246] [1357], perform butterfly step, then rezip.
            let x1 = vuzp1q_f32(a, b);
            let y1 = vuzp2q_f32(a, b);
            let (x, y) = (vaddq_f32(x1, y1), vsubq_f32(x1, y1));
            let a1 = vzip1q_f32(x, y);
            let b1 = vzip2q_f32(x, y);

            // 64 bit unzip to arrange [0145] [2367], perform butterfly step, then rezip.
            let x2 = vreinterpretq_f32_f64(vuzp1q_f64(
                vreinterpretq_f64_f32(a1),
                vreinterpretq_f64_f32(b1),
            ));
            let y2 = vreinterpretq_f32_f64(vuzp2q_f64(
                vreinterpretq_f64_f32(a1),
                vreinterpretq_f64_f32(b1),
            ));
            let (x, y) = (vaddq_f32(x2, y2), vsubq_f32(x2, y2));
            (
                vreinterpretq_f32_f64(vzip1q_f64(
                    vreinterpretq_f64_f32(x),
                    vreinterpretq_f64_f32(y),
                )),
                vreinterpretq_f32_f64(vzip2q_f64(
                    vreinterpretq_f64_f32(x),
                    vreinterpretq_f64_f32(y),
                )),
            )
        }
    }

    (r[0], r[1]) = butterfly2(r[0], r[1]);
    (r[2], r[3]) = butterfly2(r[2], r[3]);
    (r[4], r[5]) = butterfly2(r[4], r[5]);
    (r[6], r[7]) = butterfly2(r[6], r[7]);
    (r[8], r[9]) = butterfly2(r[8], r[9]);
    (r[10], r[11]) = butterfly2(r[10], r[11]);
    (r[12], r[13]) = butterfly2(r[12], r[13]);
    (r[14], r[15]) = butterfly2(r[14], r[15]);

    let mut h = 1;
    while h < 16 {
        for i in (0..16).step_by(h * 2) {
            for j in 0..h {
                let x = r[i + j];
                let y = r[i + j + h];
                r[i + j] = unsafe { vaddq_f32(x, y) };
                r[i + j + h] = unsafe { vsubq_f32(x, y) };
            }
        }
        h *= 2;
    }

    for (i, r) in r.into_iter().enumerate() {
        unsafe { vst1q_f32(block.as_mut_ptr().add(i * 4), r) };
    }
}
