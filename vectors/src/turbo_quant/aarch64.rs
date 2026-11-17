use std::arch::aarch64::{
    vaddq_s32, vaddvq_s32, vandq_u8, vdotq_s32, vdupq_n_s32, vdupq_n_u8, vld1q_s8, vld1q_u8,
    vqtbl1q_s8, vshrq_n_u8,
};

use super::scalar;

/// Inner product of an i8 query and a 2-bit turbo packed doc decoded through `table`.
///
/// # Safety
/// Requires `dotprod`. `query` must contain at least 64 entries for every complete 16 byte block
/// in `doc`.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn lut_dot2(query: &[i8], table: &[i8; 16], doc: &[u8]) -> i32 {
    let (blocks, tail) = doc.as_chunks::<16>();
    let head = blocks.len() * 64;
    debug_assert!(query.len() >= head);
    let ip = unsafe {
        let lut = vld1q_s8(table.as_ptr());
        let mask = vdupq_n_u8(0x3);
        let mut ip0 = vdupq_n_s32(0);
        let mut ip1 = vdupq_n_s32(0);
        let mut ip2 = vdupq_n_s32(0);
        let mut ip3 = vdupq_n_s32(0);
        for (i, b) in blocks.iter().enumerate() {
            let q = query.as_ptr().add(i * 64);
            let d = vld1q_u8(b.as_ptr());
            ip0 = vdotq_s32(ip0, vld1q_s8(q), vqtbl1q_s8(lut, vandq_u8(d, mask)));
            ip1 = vdotq_s32(
                ip1,
                vld1q_s8(q.add(16)),
                vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8::<2>(d), mask)),
            );
            ip2 = vdotq_s32(
                ip2,
                vld1q_s8(q.add(32)),
                vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8::<4>(d), mask)),
            );
            ip3 = vdotq_s32(
                ip3,
                vld1q_s8(q.add(48)),
                vqtbl1q_s8(lut, vshrq_n_u8::<6>(d)),
            );
        }
        vaddvq_s32(vaddq_s32(vaddq_s32(ip0, ip1), vaddq_s32(ip2, ip3)))
    };
    ip.wrapping_add(scalar::lut_dot::<2>(&query[head..], table, 0, tail))
}

/// Inner product of an i8 query and a 4-bit turbo packed doc decoded through `table`.
///
/// # Safety
/// Requires `dotprod`. `query` must contain at least 32 entries for every complete 16 byte block
/// in `doc`.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn lut_dot4(query: &[i8], table: &[i8; 16], doc: &[u8]) -> i32 {
    let (blocks, tail) = doc.as_chunks::<16>();
    let head = blocks.len() * 32;
    debug_assert!(query.len() >= head);
    let ip = unsafe {
        let lut = vld1q_s8(table.as_ptr());
        let mask = vdupq_n_u8(0xf);
        let mut ip0 = vdupq_n_s32(0);
        let mut ip1 = vdupq_n_s32(0);
        let mut ip2 = vdupq_n_s32(0);
        let mut ip3 = vdupq_n_s32(0);
        // Process blocks in pairs to keep 4 independent accumulators in flight.
        let (pairs, rem) = blocks.as_chunks::<2>();
        for (i, [b0, b1]) in pairs.iter().enumerate() {
            let q = query.as_ptr().add(i * 64);
            let d0 = vld1q_u8(b0.as_ptr());
            let d1 = vld1q_u8(b1.as_ptr());
            ip0 = vdotq_s32(ip0, vld1q_s8(q), vqtbl1q_s8(lut, vandq_u8(d0, mask)));
            ip1 = vdotq_s32(
                ip1,
                vld1q_s8(q.add(16)),
                vqtbl1q_s8(lut, vshrq_n_u8::<4>(d0)),
            );
            ip2 = vdotq_s32(
                ip2,
                vld1q_s8(q.add(32)),
                vqtbl1q_s8(lut, vandq_u8(d1, mask)),
            );
            ip3 = vdotq_s32(
                ip3,
                vld1q_s8(q.add(48)),
                vqtbl1q_s8(lut, vshrq_n_u8::<4>(d1)),
            );
        }
        if let Some(b) = rem.first() {
            let q = query.as_ptr().add(pairs.len() * 64);
            let d = vld1q_u8(b.as_ptr());
            ip0 = vdotq_s32(ip0, vld1q_s8(q), vqtbl1q_s8(lut, vandq_u8(d, mask)));
            ip1 = vdotq_s32(
                ip1,
                vld1q_s8(q.add(16)),
                vqtbl1q_s8(lut, vshrq_n_u8::<4>(d)),
            );
        }
        vaddvq_s32(vaddq_s32(vaddq_s32(ip0, ip1), vaddq_s32(ip2, ip3)))
    };
    ip.wrapping_add(scalar::lut_dot::<4>(&query[head..], table, 0, tail))
}
