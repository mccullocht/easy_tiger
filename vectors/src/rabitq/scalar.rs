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

pub fn decode(v: &[u8], magnitude: f32, center: Option<&[f32]>, out: &mut [f32]) {
    let it = crate::packing::TurboUnpacker::<1>::new(v);
    let decode = |x: u8| -> f32 { f32::from_bits(magnitude.to_bits() | (u32::from(x) << 31)) };
    if let Some(center) = center {
        for ((i, &c), o) in it.zip(center.iter()).zip(out.iter_mut()) {
            *o = decode(i) + c;
        }
    } else {
        for (i, o) in it.zip(out.iter_mut()) {
            *o = decode(i);
        }
    }
}
