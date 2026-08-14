#[inline]
pub fn walsh_hadamard_transform<const F: bool>(v: &mut [f32], signs: &[u32]) {
    assert!(
        v.len().is_power_of_two(),
        "Hadamard transform requires power of 2 length"
    );
    assert_eq!(v.len(), signs.len());
    if v.len() < 64 {
        if F {
            // Forward must sign flip first.
            for (&s, v) in signs.iter().zip(v.iter_mut()) {
                *v = f32::from_bits(v.to_bits() ^ s);
            }
        }
        wht_block::<1>(v)
    } else {
        // Perform the early strides of the block transformation together in 64 dimension chunks
        // in an effort to improve locality. v.len() is a power of 2 and there are at least 64
        // entries, so there will be no tail entries.
        let blocks = v.as_chunks_mut::<64>().0;
        let sblocks = signs.as_chunks::<64>().0;
        if F {
            for (b, s) in blocks.iter_mut().zip(sblocks.iter()) {
                wht_fixed_block::<true, 64>(b, s);
            }
        } else {
            for (b, s) in blocks.iter_mut().zip(sblocks.iter()) {
                wht_fixed_block::<false, 64>(b, s);
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
fn wht_fixed_block<const F: bool, const N: usize>(block: &mut [f32; N], signs: &[u32; N]) {
    if F {
        for (&s, v) in signs.iter().zip(block.iter_mut()) {
            *v = f32::from_bits(v.to_bits() ^ s);
        }
    }

    let mut h = 1;
    while h < N {
        for i in (0..N).step_by(h * 2) {
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
