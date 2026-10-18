#[inline]
pub fn quantize_and_pack(v: &[f32], out: &mut [u8]) -> u32 {
    let mut packer = crate::packing::TurboPacker::<1>::new(out);
    v.iter()
        .map(|x| {
            let s = x.to_bits() >> 31;
            packer.push(s as u8);
            s
        })
        .sum::<u32>()
}

#[inline]
pub fn l1_norm_scaled(v: &[f32], scale: f32) -> f32 {
    v.iter().map(|&x| x.abs() * scale).sum::<f32>() / (v.len() as f32).sqrt()
}

#[inline]
pub fn decode(v: &[u8], magnitude: f32, out: &mut [f32]) {
    let it = crate::packing::TurboUnpacker::<1>::new(v);
    let decode = |x: u8| -> f32 { f32::from_bits(magnitude.to_bits() | (u32::from(x) << 31)) };
    for (i, o) in it.zip(out.iter_mut()) {
        *o = decode(i);
    }
}
