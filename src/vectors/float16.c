#ifdef __aarch64__

#include <arm_neon.h>
#include <stddef.h>
#include <string.h>

float et_dot_f16(const __fp16* a, const __fp16* b, size_t len) {
  size_t tail_split = len & ~7;
  float32x4_t dotv = vdupq_n_f32(0.0);
  for (int i = 0; i < tail_split; i += 8) {
    float16x8_t av = vld1q_f16(a + i);
    float16x8_t bv = vld1q_f16(b + i);
    dotv = vfmaq_f32(dotv, vcvt_f32_f16(vget_low_f16(av)),
                     vcvt_f32_f16(vget_low_f16(bv)));
    dotv = vfmaq_f32(dotv, vcvt_f32_f16(vget_high_f16(av)),
                     vcvt_f32_f16(vget_high_f16(bv)));
  }

  if (tail_split < len) {
    __fp16 tail_a[8];
    memset(&tail_a, 0, sizeof(tail_a));
    __fp16 tail_b[8];
    memset(&tail_b, 0, sizeof(tail_b));
    for (int i = tail_split; i < len; i++) {
      int dest = i - tail_split;
      tail_a[dest] = a[i];
      tail_b[dest] = b[i];
    }

    float16x8_t av = vld1q_f16(&tail_a[0]);
    float16x8_t bv = vld1q_f16(&tail_b[0]);
    dotv = vfmaq_f32(dotv, vcvt_f32_f16(vget_low_f16(av)),
                     vcvt_f32_f16(vget_low_f16(bv)));
    dotv = vfmaq_f32(dotv, vcvt_f32_f16(vget_high_f16(av)),
                     vcvt_f32_f16(vget_high_f16(bv)));
  }

  return vaddvq_f32(dotv);
}

#endif /* __aarch64__ */