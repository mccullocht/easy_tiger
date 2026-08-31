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
