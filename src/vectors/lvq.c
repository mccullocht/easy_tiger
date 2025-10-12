#define EXPORT __attribute__((visibility("default")))
#define HIDDEN __attribute__((visibility("hidden")))

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __aarch64__

#include <arm_neon.h>

__attribute__((target("+dotprod"))) EXPORT uint32_t
et_lvq_dot_u4(const uint8_t* a, const uint8_t* b, size_t len) {
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
et_lvq_dot_u8(const uint8_t* a, const uint8_t* b, size_t len) {
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

  dot0 = vaddq_u32(vaddq_u32(dot0, dot1), vaddq_f32(dot2, dot3));
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

#endif /* __aarch64__ */