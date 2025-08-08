#define EXPORT __attribute__((visibility("default")))
#define HIDDEN __attribute__((visibility("hidden")))

#ifdef __aarch64__

#include <arm_neon.h>
#include <stddef.h>
#include <string.h>

EXPORT void et_serialize_f16(const float* v, size_t len, const float* scale,
                             uint8_t* out) {
  size_t tail_split = len & ~3;
  for (size_t i = 0; i < tail_split; i += 4) {
    float32x4_t in = vld1q_f32(v + i);
    if (scale != NULL) {
      in = vmulq_n_f32(in, *scale);
    }
    vst1_u8(out + i * 2, vreinterpret_u8_f16(vcvt_f16_f32(in)));
  }

  if (tail_split < len) {
    float tail_in[4] = {0, 0, 0, 0};
    for (size_t i = tail_split; i < len; i++) {
      tail_in[i - tail_split] = v[i];
    }
    float32x4_t in = vld1q_f32(&tail_in[0]);
    if (scale != NULL) {
      in = vmulq_n_f32(in, *scale);
    }
    uint8_t tail_out[8];
    vst1_u8(&tail_out[0], vreinterpret_u8_f16(vcvt_f16_f32(in)));
    memcpy(out + tail_split * 2, &tail_out[0],
           (len - tail_split) * sizeof(__fp16));
  }
}

// It's faster to fill out a full 4 value tail entry than it is
// to convert and compute one element at a time.
HIDDEN float16x4_t load_tail_f16x4(const __fp16* v, size_t len) {
  __fp16 tail[4] = {0, 0, 0, 0};
  for (size_t i = 0; i < len; i++) {
    tail[i] = v[i];
  }
  return vld1_f16(&tail[0]);
}

// Load a partial value into a vector register to cover cases comparing to f16
// where we will also have to a partial value into a vector register.
HIDDEN float32x4_t load_tail_f32x4(const float* v, size_t len) {
  float tail[4] = {0, 0, 0, 0};
  for (size_t i = 0; i < len; i++) {
    tail[i] = v[i];
  }
  return vld1q_f32(&tail[0]);
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

EXPORT float et_dot_f32_f16(const float* a, const __fp16* b, size_t len) {
  size_t tail_split = len & ~3;
  float32x4_t dotv = vdupq_n_f32(0.0);
  for (size_t i = 0; i < tail_split; i += 4) {
    float32x4_t av = vld1q_f32(a + i);
    float32x4_t bv = vcvt_f32_f16(vld1_f16(b + i));
    dotv = vfmaq_f32(dotv, av, bv);
  }

  if (tail_split < len) {
    float32x4_t av = load_tail_f32x4(a + tail_split, len - tail_split);
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

EXPORT float et_l2_f32_f16(const float* a, const __fp16* b, size_t len) {
  size_t tail_split = len & ~3;
  float32x4_t l2v = vdupq_n_f32(0.0);
  for (size_t i = 0; i < tail_split; i += 4) {
    float32x4_t av = vld1q_f32(a + i);
    float32x4_t bv = vcvt_f32_f16(vld1_f16(b + i));
    float32x4_t dv = vsubq_f32(av, bv);
    l2v = vfmaq_f32(l2v, dv, dv);
  }

  if (tail_split < len) {
    float32x4_t av = load_tail_f32x4(a + tail_split, len - tail_split);
    float32x4_t bv =
        vcvt_f32_f16(load_tail_f16x4(b + tail_split, len - tail_split));
    float32x4_t dv = vsubq_f32(av, bv);
    l2v = vfmaq_f32(l2v, dv, dv);
  }

  return vaddvq_f32(l2v);
}

#endif /* __aarch64__ */