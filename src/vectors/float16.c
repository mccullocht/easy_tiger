#define EXPORT __attribute__((visibility("default")))
#define HIDDEN __attribute__((visibility("hidden")))

#ifdef __aarch64__

#include <arm_neon.h>
#include <stddef.h>
#include <string.h>

// It's faster to fill out a full 4 value tail entry than it is
// to convert and compute one element at a time.
HIDDEN float16x4_t load_tail_f16x4(const __fp16* v, size_t len) {
  __fp16 tail[4];
  memset(&tail, 0, sizeof(tail));
  for (size_t i = 0; i < len; i++) {
    tail[i] = v[i];
  }
  return vld1_f16(&tail[0]);
}

EXPORT float et_dot_f16_f16(const __fp16* a, const __fp16* b, size_t len) {
  size_t tail_split = len & ~3;
  float32x4_t dotv = vdupq_n_f32(0.0);
  for (size_t i = 0; i < tail_split; i += 4) {
    float32x4_t av = vcvt_f32_f16(vld1_f16(a + i));
    float32x4_t bv = vcvt_f32_f16(vld1_f16(b + i));
    dotv = vfmaq_f32(dotv, av, bv);
  }

  if (tail_split < len) {
    float32x4_t av =
        vcvt_f32_f16(load_tail_f16x4(a + tail_split, len - tail_split));
    float32x4_t bv =
        vcvt_f32_f16(load_tail_f16x4(b + tail_split, len - tail_split));
    dotv = vfmaq_f32(dotv, av, bv);
  }

  return vaddvq_f32(dotv);
}

EXPORT float et_l2_f16_f16(const __fp16* a, const __fp16* b, size_t len) {
  size_t tail_split = len & ~3;
  float32x4_t l2v = vdupq_n_f32(0.0);
  for (size_t i = 0; i < tail_split; i += 4) {
    float32x4_t av = vcvt_f32_f16(vld1_f16(a + i));
    float32x4_t bv = vcvt_f32_f16(vld1_f16(b + i));
    float32x4_t dv = vsubq_f32(av, bv);
    l2v = vfmaq_f32(l2v, dv, dv);
  }

  if (tail_split < len) {
    float32x4_t av =
        vcvt_f32_f16(load_tail_f16x4(a + tail_split, len - tail_split));
    float32x4_t bv =
        vcvt_f32_f16(load_tail_f16x4(b + tail_split, len - tail_split));
    float32x4_t dv = vsubq_f32(av, bv);
    l2v = vfmaq_f32(l2v, dv, dv);
  }

  return vaddvq_f32(l2v);
}

#endif /* __aarch64__ */