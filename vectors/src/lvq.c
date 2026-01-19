#define EXPORT __attribute__((visibility("default")))
#define HIDDEN __attribute__((visibility("hidden")))

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __aarch64__

#include <arm_neon.h>

__attribute__((target("+dotprod"))) EXPORT uint32_t
et_lvq_dot_u2(const uint8_t *a, const uint8_t *b, size_t len) {
  uint32x4_t dot0 = vdupq_n_u32(0);
  uint32x4_t dot1 = vdupq_n_u32(0);
  uint32x4_t dot2 = vdupq_n_u32(0);
  uint32x4_t dot3 = vdupq_n_u32(0);
  uint8x16_t dibit_mask = vdupq_n_u8(0x3);
  size_t len16 = len & ~15;
  for (size_t i = 0; i < len16; i += 16) {
    uint8x16_t av = vld1q_u8(a + i);
    uint8x16_t bv = vld1q_u8(b + i);
    dot0 =
        vdotq_u32(dot0, vandq_u8(av, dibit_mask), vandq_u8(bv, dibit_mask));
    dot1 = vdotq_u32(dot1, vandq_u8(vshrq_n_u8(av, 2), dibit_mask),
                     vandq_u8(vshrq_n_u8(bv, 2), dibit_mask));
    dot2 = vdotq_u32(dot2, vandq_u8(vshrq_n_u8(av, 4), dibit_mask),
                     vandq_u8(vshrq_n_u8(bv, 4), dibit_mask));
    dot3 = vdotq_u32(dot3, vshrq_n_u8(av, 6), vshrq_n_u8(bv, 6));
  }

  uint32_t dot =
      vaddvq_u32(vaddq_u32(vaddq_u32(dot0, dot1), vaddq_u32(dot2, dot3)));
  for (size_t i = len16; i < len; i++) {
    uint32_t av = a[i];
    uint32_t bv = b[i];
    dot += (av & 0x3) * (bv & 0x3) + ((av >> 2) & 0x3) * ((bv >> 2) & 0x3) +
           ((av >> 4) & 0x3) * ((bv >> 4) & 0x3) + ((av >> 6) & 0x3) *
           ((bv >> 6) & 0x3);
  }

  return dot;
}

__attribute__((target("+dotprod"))) EXPORT uint32_t
et_lvq_dot_u4(const uint8_t *a, const uint8_t *b, size_t len) {
  uint32x4_t dot0 = vdupq_n_u32(0);
  uint32x4_t dot1 = vdupq_n_u32(0);
  uint32x4_t dot2 = vdupq_n_u32(0);
  uint32x4_t dot3 = vdupq_n_u32(0);
  uint8x16_t nibble_mask = vdupq_n_u8(0xf);
  size_t len32 = len & ~31;
  for (size_t i = 0; i < len32; i += 32) {
    uint8x16_t av = vld1q_u8(a + i);
    uint8x16_t bv = vld1q_u8(b + i);
    dot0 =
        vdotq_u32(dot0, vandq_u8(av, nibble_mask), vandq_u8(bv, nibble_mask));
    dot1 = vdotq_u32(dot1, vshrq_n_u8(av, 4), vshrq_n_u8(bv, 4));

    av = vld1q_u8(a + i + 16);
    bv = vld1q_u8(b + i + 16);
    dot2 =
        vdotq_u32(dot2, vandq_u8(av, nibble_mask), vandq_u8(bv, nibble_mask));
    dot3 = vdotq_u32(dot3, vshrq_n_u8(av, 4), vshrq_n_u8(bv, 4));
  }

  dot0 = vaddq_u32(dot0, dot2);
  dot1 = vaddq_u32(dot1, dot3);
  size_t len16 = len & ~15;
  if (len32 < len16) {
    uint8x16_t av = vld1q_u8(a + len32);
    uint8x16_t bv = vld1q_u8(b + len32);
    dot0 =
        vdotq_u32(dot0, vandq_u8(av, nibble_mask), vandq_u8(bv, nibble_mask));
    dot1 = vdotq_u32(dot1, vshrq_n_u8(av, 4), vshrq_n_u8(bv, 4));
  }

  uint32_t dot = vaddvq_u32(vaddq_u32(dot0, dot1));
  for (size_t i = len16; i < len; i++) {
    uint32_t av = a[i];
    uint32_t bv = b[i];
    dot += (av & 0xf) * (bv & 0xf) + (av >> 4) * (bv >> 4);
  }

  return dot;
}

__attribute__((target("+dotprod"))) EXPORT uint32_t
et_lvq_dot_u8(const uint8_t *a, const uint8_t *b, size_t len) {
  uint32x4_t dot0 = vdupq_n_u32(0);
  uint32x4_t dot1 = vdupq_n_u32(0);
  uint32x4_t dot2 = vdupq_n_u32(0);
  uint32x4_t dot3 = vdupq_n_u32(0);
  size_t len64 = len & ~63;
  for (size_t i = 0; i < len64; i += 64) {
    uint8x16_t av = vld1q_u8(a + i);
    uint8x16_t bv = vld1q_u8(b + i);
    dot0 = vdotq_u32(dot0, av, bv);

    av = vld1q_u8(a + i + 16);
    bv = vld1q_u8(b + i + 16);
    dot1 = vdotq_u32(dot1, av, bv);

    av = vld1q_u8(a + i + 32);
    bv = vld1q_u8(b + i + 32);
    dot2 = vdotq_u32(dot2, av, bv);

    av = vld1q_u8(a + i + 48);
    bv = vld1q_u8(b + i + 48);
    dot3 = vdotq_u32(dot3, av, bv);
  }

  dot0 = vaddq_u32(vaddq_u32(dot0, dot1), vaddq_u32(dot2, dot3));
  size_t len16 = len & ~15;
  for (size_t i = len64; i < len16; i += 16) {
    uint8x16_t av = vld1q_u8(a + i);
    uint8x16_t bv = vld1q_u8(b + i);
    dot0 = vdotq_u32(dot0, av, bv);
  }

  uint32_t dot = vaddvq_u32(dot0);
  for (size_t i = len16; i < len; i++) {
    uint32_t av = a[i];
    uint32_t bv = b[i];
    dot += av * bv;
  }

  return dot;
}

__attribute__((target("+dotprod"))) EXPORT uint32_t
et_lvq_dot_u8_u1(const uint8_t *q, const uint8_t *d, size_t len) {
  uint32x4_t dot0 = vdupq_n_u32(0);
  uint32x4_t dot1 = vdupq_n_u32(0);
  uint32x4_t dot2 = vdupq_n_u32(0);
  uint32x4_t dot3 = vdupq_n_u32(0);
  uint8x16_t mask = vdupq_n_u8(0x1);
  size_t len128 = len & ~127;
  for (size_t i = 0; i < len128; i += 128) {
    uint8x16_t dv = vld1q_u8(d + i / 128 * 16);

    dot0 = vdotq_u32(dot0, vld1q_u8(q + i), vandq_u8(dv, mask));
    dot1 = vdotq_u32(dot1, vld1q_u8(q + i + 16), vandq_u8(vshrq_n_u8(dv, 1), mask));
    dot2 = vdotq_u32(dot2, vld1q_u8(q + i + 32), vandq_u8(vshrq_n_u8(dv, 2), mask));
    dot3 = vdotq_u32(dot3, vld1q_u8(q + i + 48), vandq_u8(vshrq_n_u8(dv, 3), mask));
    dot0 = vdotq_u32(dot0, vld1q_u8(q + i + 64), vandq_u8(vshrq_n_u8(dv, 4), mask));
    dot1 = vdotq_u32(dot1, vld1q_u8(q + i + 80), vandq_u8(vshrq_n_u8(dv, 5), mask));
    dot2 = vdotq_u32(dot2, vld1q_u8(q + i + 96), vandq_u8(vshrq_n_u8(dv, 6), mask));
    dot3 = vdotq_u32(dot3, vld1q_u8(q + i + 112), vandq_u8(vshrq_n_u8(dv, 7), mask));
  }

  return vaddvq_u32(vaddq_u32(vaddq_u32(dot0, dot1), vaddq_u32(dot2, dot3)));
}

__attribute__((target("+dotprod"))) EXPORT uint32_t
et_lvq_dot_u8_u2(const uint8_t *q, const uint8_t *d, size_t len) {
  uint32x4_t dot0 = vdupq_n_u32(0);
  uint32x4_t dot1 = vdupq_n_u32(0);
  uint32x4_t dot2 = vdupq_n_u32(0);
  uint32x4_t dot3 = vdupq_n_u32(0);
  uint8x16_t mask = vdupq_n_u8(0x3);
  size_t len64 = len & ~63;
  for (size_t i = 0; i < len64; i += 64) {
    uint8x16_t dv = vld1q_u8(d + i / 64 * 16);

    dot0 = vdotq_u32(dot0, vld1q_u8(q + i), vandq_u8(dv, mask));
    dot1 = vdotq_u32(dot1, vld1q_u8(q + i + 16), vandq_u8(vshrq_n_u8(dv, 2), mask));
    dot2 = vdotq_u32(dot2, vld1q_u8(q + i + 32), vandq_u8(vshrq_n_u8(dv, 4), mask));
    dot3 = vdotq_u32(dot3, vld1q_u8(q + i + 48), vandq_u8(vshrq_n_u8(dv, 6), mask));
  }

  return vaddvq_u32(vaddq_u32(vaddq_u32(dot0, dot1), vaddq_u32(dot2, dot3)));
}

__attribute__((target("+dotprod"))) EXPORT uint32_t
et_lvq_dot_u8_u4(const uint8_t *q, const uint8_t *d, size_t len) {
  uint32x4_t dot0 = vdupq_n_u32(0);
  uint32x4_t dot1 = vdupq_n_u32(0);
  uint32x4_t dot2 = vdupq_n_u32(0);
  uint32x4_t dot3 = vdupq_n_u32(0);
  uint8x16_t mask = vdupq_n_u8(0xf);
  size_t len64 = len & ~63;
  for (size_t i = 0; i < len64; i += 64) {
    uint8x16_t dv0 = vld1q_u8(d + i / 64 * 32);
    uint8x16_t dv1 = vld1q_u8(d + i / 64 * 32 + 16);

    dot0 = vdotq_u32(dot0, vld1q_u8(q + i), vandq_u8(dv0, mask));
    dot1 = vdotq_u32(dot1, vld1q_u8(q + i + 16), vshrq_n_u8(dv0, 4));
    dot2 = vdotq_u32(dot2, vld1q_u8(q + i + 32), vandq_u8(dv1, mask));
    dot3 = vdotq_u32(dot3, vld1q_u8(q + i + 48), vshrq_n_u8(dv1, 4));
  }

  dot0 = vaddq_u32(dot0, dot1);
  dot2 = vaddq_u32(dot2, dot3);
  if (len64 < len) {
    uint8x16_t dv = vld1q_u8(d + len64 / 64 * 32);
    dot0 = vdotq_u32(dot0, vld1q_u8(q + len64), vandq_u8(dv, mask));
    dot2 = vdotq_u32(dot2, vld1q_u8(q + len64 + 16), vshrq_n_u8(dv, 4));
  }

  return vaddvq_u32(vaddq_u32(dot0, dot2));
}

struct LVQ2Dot {
  uint32_t ap_dot_bp;
  uint32_t ap_dot_br;
  uint32_t ar_dot_bp;
  uint32_t ar_dot_br;
};

inline HIDDEN uint16_t load_u16_le(const uint8_t *ptr) {
  return (uint16_t)ptr[0] | (uint16_t)ptr[1] << 8;
}

inline HIDDEN uint8x16_t unpack1(uint16_t v) {
  uint8x16_t broadcast = vreinterpretq_u8_u16(vdupq_n_u16(v));
  uint8x16_t shuffle_mask =
      vld1q_u8(((uint8_t[]){0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1}));
  uint8x16_t shuffled = vqtbl1q_u8(broadcast, shuffle_mask);
  int8x16_t shift_mask = vld1q_s8(((int8_t[]){0, -1, -2, -3, -4, -5, -6, -7, 0,
                                              -1, -2, -3, -4, -5, -6, -7}));
  uint8x16_t shifted = vshlq_u8(shuffled, shift_mask);
  uint8x16_t mask = vdupq_n_u8(1);
  return vandq_u8(shifted, mask);
}

__attribute__((target("+dotprod"))) EXPORT struct LVQ2Dot
et_lvq2_dot_u1_u8(const uint8_t *ap, const uint8_t *ar, const uint8_t *bp,
                  const uint8_t *br, size_t len) {
  uint32x4_t ap_dot_bp = vdupq_n_u32(0);
  uint32x4_t ap_dot_br = vdupq_n_u32(0);
  uint32x4_t ar_dot_bp = vdupq_n_u32(0);
  uint32x4_t ar_dot_br = vdupq_n_u32(0);
  size_t len16 = len & ~15;
  for (size_t i = 0; i < len16; i += 16) {
    uint8x16_t apv = unpack1(load_u16_le(ap + i / 8));
    uint8x16_t arv = vld1q_u8(ar + i);
    uint8x16_t bpv = unpack1(load_u16_le(bp + i / 8));
    uint8x16_t brv = vld1q_u8(br + i);
    ap_dot_bp = vdotq_u32(ap_dot_bp, apv, bpv);
    ap_dot_br = vdotq_u32(ap_dot_br, apv, brv);
    ar_dot_bp = vdotq_u32(ar_dot_bp, arv, bpv);
    ar_dot_br = vdotq_u32(ar_dot_br, arv, brv);
  }

  struct LVQ2Dot result = {
      .ap_dot_bp = vaddvq_u32(ap_dot_bp),
      .ap_dot_br = vaddvq_u32(ap_dot_br),
      .ar_dot_bp = vaddvq_u32(ar_dot_bp),
      .ar_dot_br = vaddvq_u32(ar_dot_br),
  };
  for (size_t i = len16; i < len; i++) {
    uint32_t apv = (ap[i / 8] >> (i % 8)) & 1;
    uint32_t arv = ar[i];
    uint32_t bpv = (bp[i / 8] >> (i % 8)) & 1;
    uint32_t brv = br[i];
    result.ap_dot_bp += apv * bpv;
    result.ap_dot_br += apv * brv;
    result.ar_dot_bp += arv * bpv;
    result.ar_dot_br += arv * brv;
  }

  return result;
}

__attribute__((target("+dotprod"))) EXPORT struct LVQ2Dot
et_lvq2_dot_u4_u4(const uint8_t *ap, const uint8_t *ar, const uint8_t *bp,
                  const uint8_t *br, size_t len) {
  uint32x4_t ap_dot_bp = vdupq_n_u32(0);
  uint32x4_t ap_dot_br = vdupq_n_u32(0);
  uint32x4_t ar_dot_bp = vdupq_n_u32(0);
  uint32x4_t ar_dot_br = vdupq_n_u32(0);
  uint8x16_t nibble_mask = vdupq_n_u8(0xf);
  size_t len16 = len & ~15;
  for (size_t i = 0; i < len16; i += 16) {
    uint8x16_t apv = vld1q_u8(ap + i);
    uint8x16_t arv = vld1q_u8(ar + i);
    uint8x16_t bpv = vld1q_u8(bp + i);
    uint8x16_t brv = vld1q_u8(br + i);

    ap_dot_bp = vdotq_u32(ap_dot_bp, vandq_u8(apv, nibble_mask),
                          vandq_u8(bpv, nibble_mask));
    ap_dot_br = vdotq_u32(ap_dot_br, vandq_u8(apv, nibble_mask),
                          vandq_u8(brv, nibble_mask));
    ar_dot_bp = vdotq_u32(ar_dot_bp, vandq_u8(arv, nibble_mask),
                          vandq_u8(bpv, nibble_mask));
    ar_dot_br = vdotq_u32(ar_dot_br, vandq_u8(arv, nibble_mask),
                          vandq_u8(brv, nibble_mask));

    apv = vshrq_n_u8(apv, 4);
    arv = vshrq_n_u8(arv, 4);
    bpv = vshrq_n_u8(bpv, 4);
    brv = vshrq_n_u8(brv, 4);

    ap_dot_bp = vdotq_u32(ap_dot_bp, vandq_u8(apv, nibble_mask),
                          vandq_u8(bpv, nibble_mask));
    ap_dot_br = vdotq_u32(ap_dot_br, vandq_u8(apv, nibble_mask),
                          vandq_u8(brv, nibble_mask));
    ar_dot_bp = vdotq_u32(ar_dot_bp, vandq_u8(arv, nibble_mask),
                          vandq_u8(bpv, nibble_mask));
    ar_dot_br = vdotq_u32(ar_dot_br, vandq_u8(arv, nibble_mask),
                          vandq_u8(brv, nibble_mask));
  }

  struct LVQ2Dot result = {
      .ap_dot_bp = vaddvq_u32(ap_dot_bp),
      .ap_dot_br = vaddvq_u32(ap_dot_br),
      .ar_dot_bp = vaddvq_u32(ar_dot_bp),
      .ar_dot_br = vaddvq_u32(ar_dot_br),
  };
  for (size_t i = len16; i < len; i++) {
    uint32_t apd = ap[i];
    uint32_t ard = ar[i];
    uint32_t bpd = bp[i];
    uint32_t brd = br[i];
    result.ap_dot_bp +=
        (apd & 0xf) * (bpd & 0xf) + ((apd >> 4) & 0xf) * ((bpd >> 4) & 0xf);
    result.ap_dot_br +=
        (apd & 0xf) * (brd & 0xf) + ((apd >> 4) & 0xf) * ((brd >> 4) & 0xf);
    result.ar_dot_bp +=
        (ard & 0xf) * (bpd & 0xf) + ((ard >> 4) & 0xf) * ((bpd >> 4) & 0xf);
    result.ar_dot_br +=
        (ard & 0xf) * (brd & 0xf) + ((ard >> 4) & 0xf) * ((brd >> 4) & 0xf);
  }

  return result;
}

inline HIDDEN uint8x16x2_t unpack_u4_u8(const uint8_t *ptr) {
  uint8x16_t v = vld1q_u8(ptr);
  uint8x16_t mask = vdupq_n_u8(0xf);
  uint8x16_t evens = vandq_u8(v, mask);
  uint8x16_t odds = vandq_u8(vshrq_n_u8(v, 4), mask);
  return (uint8x16x2_t){{vzip1q_u8(evens, odds), vzip2q_u8(evens, odds)}};
}

inline HIDDEN uint8x16x2_t load_u8x2(const uint8_t *ptr) {
  return (uint8x16x2_t){{vld1q_u8(ptr), vld1q_u8(ptr + 16)}};
}

__attribute__((target("+dotprod"))) EXPORT struct LVQ2Dot
et_lvq2_dot_u4_u8(const uint8_t *ap, const uint8_t *ar, const uint8_t *bp,
                  const uint8_t *br, size_t len) {
  uint32x4_t ap_dot_bp = vdupq_n_u32(0);
  uint32x4_t ap_dot_br = vdupq_n_u32(0);
  uint32x4_t ar_dot_bp = vdupq_n_u32(0);
  uint32x4_t ar_dot_br = vdupq_n_u32(0);
  size_t len32 = len & ~31;
  for (size_t i = 0; i < len32; i += 32) {
    uint8x16x2_t apv = unpack_u4_u8(ap + i / 2);
    uint8x16x2_t arv = load_u8x2(ar + i);
    uint8x16x2_t bpv = unpack_u4_u8(bp + i / 2);
    uint8x16x2_t brv = load_u8x2(br + i);

    ap_dot_bp = vdotq_u32(ap_dot_bp, apv.val[0], bpv.val[0]);
    ap_dot_br = vdotq_u32(ap_dot_br, apv.val[0], brv.val[0]);
    ar_dot_bp = vdotq_u32(ar_dot_bp, arv.val[0], bpv.val[0]);
    ar_dot_br = vdotq_u32(ar_dot_br, arv.val[0], brv.val[0]);

    ap_dot_bp = vdotq_u32(ap_dot_bp, apv.val[1], bpv.val[1]);
    ap_dot_br = vdotq_u32(ap_dot_br, apv.val[1], brv.val[1]);
    ar_dot_bp = vdotq_u32(ar_dot_bp, arv.val[1], bpv.val[1]);
    ar_dot_br = vdotq_u32(ar_dot_br, arv.val[1], brv.val[1]);
  }

  struct LVQ2Dot result = {
      .ap_dot_bp = vaddvq_u32(ap_dot_bp),
      .ap_dot_br = vaddvq_u32(ap_dot_br),
      .ar_dot_bp = vaddvq_u32(ar_dot_bp),
      .ar_dot_br = vaddvq_u32(ar_dot_br),
  };
  for (size_t i = len32; i < len; i++) {
    uint32_t apd = (ap[i / 2] >> (i % 2 * 4)) & 0xf;
    uint32_t ard = ar[i];
    uint32_t bpd = (bp[i / 2] >> (i % 2 * 4)) & 0xf;
    uint32_t brd = br[i];

    result.ap_dot_bp += apd * bpd;
    result.ap_dot_br += apd * brd;
    result.ar_dot_bp += ard * bpd;
    result.ar_dot_br += ard * brd;
  }

  return result;
}

__attribute__((target("+dotprod"))) EXPORT struct LVQ2Dot
et_lvq2_dot_u8_u8(const uint8_t *ap, const uint8_t *ar, const uint8_t *bp,
                  const uint8_t *br, size_t len) {
  uint32x4_t ap_dot_bp = vdupq_n_u32(0);
  uint32x4_t ap_dot_br = vdupq_n_u32(0);
  uint32x4_t ar_dot_bp = vdupq_n_u32(0);
  uint32x4_t ar_dot_br = vdupq_n_u32(0);
  size_t len16 = len & ~15;
  for (size_t i = 0; i < len16; i += 16) {
    uint8x16_t apv = vld1q_u8(ap + i);
    uint8x16_t arv = vld1q_u8(ar + i);
    uint8x16_t bpv = vld1q_u8(bp + i);
    uint8x16_t brv = vld1q_u8(br + i);
    ap_dot_bp = vdotq_u32(ap_dot_bp, apv, bpv);
    ap_dot_br = vdotq_u32(ap_dot_br, apv, brv);
    ar_dot_bp = vdotq_u32(ar_dot_bp, arv, bpv);
    ar_dot_br = vdotq_u32(ar_dot_br, arv, brv);
  }

  struct LVQ2Dot result = {
      .ap_dot_bp = vaddvq_u32(ap_dot_bp),
      .ap_dot_br = vaddvq_u32(ap_dot_br),
      .ar_dot_bp = vaddvq_u32(ar_dot_bp),
      .ar_dot_br = vaddvq_u32(ar_dot_br),
  };
  for (size_t i = len16; i < len; i++) {
    uint32_t apd = ap[i];
    uint32_t ard = ar[i];
    uint32_t bpd = bp[i];
    uint32_t brd = br[i];
    result.ap_dot_bp += apd * bpd;
    result.ap_dot_br += apd * brd;
    result.ar_dot_bp += ard * bpd;
    result.ar_dot_br += ard * brd;
  }

  return result;
}

#endif /* __aarch64__ */