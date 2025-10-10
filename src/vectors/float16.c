#define EXPORT __attribute__((visibility("default")))
#define HIDDEN __attribute__((visibility("hidden")))

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __aarch64__

#include <arm_neon.h>

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
  size_t tail_split = len & ~15;
  float32x4_t dot0 = vdupq_n_f32(0.0);
  float32x4_t dot1 = vdupq_n_f32(0.0);
  float32x4_t dot2 = vdupq_n_f32(0.0);
  float32x4_t dot3 = vdupq_n_f32(0.0);
  for (size_t i = 0; i < tail_split; i += 16) {
    float16x8_t av16 = vld1q_f16(a + i);
    float16x8_t bv16 = vld1q_f16(b + i);
    dot0 = vfmaq_f32(dot0, vcvt_f32_f16(vget_low_f16(av16)), vcvt_f32_f16(vget_low_f16(bv16)));
    dot1 = vfmaq_f32(dot1, vcvt_high_f32_f16(av16), vcvt_high_f32_f16(bv16));

    av16 = vld1q_f16(a + i + 8);
    bv16 = vld1q_f16(b + i + 8);
    dot2 = vfmaq_f32(dot2, vcvt_f32_f16(vget_low_f16(av16)), vcvt_f32_f16(vget_low_f16(bv16)));
    dot3 = vfmaq_f32(dot3, vcvt_high_f32_f16(av16), vcvt_high_f32_f16(bv16));
  }

  dot0 = vaddq_f32(vaddq_f32(dot0, dot1), vaddq_f32(dot2, dot3));
  for (; tail_split + 4 <= len; tail_split += 4) {
    dot0 = vfmaq_f32(dot0, vcvt_f32_f16(vld1_f16(a + tail_split)), vcvt_f32_f16(vld1_f16(b + tail_split)));
  }

  if (tail_split < len) {
    float32x4_t av =
        vcvt_f32_f16(load_tail_f16x4(a + tail_split, len - tail_split));
    float32x4_t bv =
        vcvt_f32_f16(load_tail_f16x4(b + tail_split, len - tail_split));
    dot0 = vfmaq_f32(dot0, av, bv);
  }

  return vaddvq_f32(dot0);
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
  size_t tail_split = len & ~15;
  float32x4_t sum0 = vdupq_n_f32(0.0);
  float32x4_t sum1 = vdupq_n_f32(0.0);
  float32x4_t sum2 = vdupq_n_f32(0.0);
  float32x4_t sum3 = vdupq_n_f32(0.0);
  for (size_t i = 0; i < tail_split; i += 16) {
    float16x8_t av16 = vld1q_f16(a + i);
    float16x8_t bv16 = vld1q_f16(b + i);

    float32x4_t dv = vsubq_f32(vcvt_f32_f16(vget_low_f16(av16)),vcvt_f32_f16(vget_low_f16(bv16)));
    sum0 = vfmaq_f32(sum0, dv, dv);
    dv = vsubq_f32(vcvt_high_f32_f16(av16),vcvt_high_f32_f16(bv16));
    sum1 = vfmaq_f32(sum1, dv, dv);

    av16 = vld1q_f16(a + i + 8);
    bv16 = vld1q_f16(b + i + 8);

    dv = vsubq_f32(vcvt_f32_f16(vget_low_f16(av16)),vcvt_f32_f16(vget_low_f16(bv16)));
    sum2 = vfmaq_f32(sum2, dv, dv);
    dv = vsubq_f32(vcvt_high_f32_f16(av16),vcvt_high_f32_f16(bv16));
    sum3 = vfmaq_f32(sum3, dv, dv);
  }

  sum0 = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
  for (; tail_split + 4 <= len; tail_split += 4) {
    float32x4_t diff = vsubq_f32(vcvt_f32_f16(vld1_f16(a + tail_split)), vcvt_f32_f16(vld1_f16(b + tail_split)));
    sum0 = vfmaq_f32(sum0, diff, diff);
  }

  if (tail_split < len) {
    float32x4_t av =
        vcvt_f32_f16(load_tail_f16x4(a + tail_split, len - tail_split));
    float32x4_t bv =
        vcvt_f32_f16(load_tail_f16x4(b + tail_split, len - tail_split));
    float32x4_t dv = vsubq_f32(av, bv);
    sum0 = vfmaq_f32(sum0, dv, dv);
  }

  return vaddvq_f32(sum0);
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

#if defined(__x86_64__)

#include <immintrin.h>

__attribute__((target("avx,f16c")))
EXPORT void et_serialize_f16_avx512(
    const float* v,
    size_t len,
    const float* scale,
    uint8_t* out) {
  size_t tail_split = len & ~7;
  for (size_t i = 0; i < tail_split; i += 8) {
    __m256 vs = _mm256_loadu_ps(v + i);
    if (scale != NULL) {
      vs = _mm256_mul_ps(vs, _mm256_set1_ps(*scale));
    }
    __m128i vh = _mm256_cvtps_ph(vs, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128((__m128i*)(out + i * 2), vh);
  }

  if (tail_split != len) {
    float vt[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    memcpy(vt, v + tail_split, (len - tail_split) * 4);
    __m256 vs = _mm256_loadu_ps(vt);
    if (scale != NULL) {
      vs = _mm256_mul_ps(vs, _mm256_set1_ps(*scale));
    }
    __m128i vh = _mm256_cvtps_ph(vs, _MM_FROUND_TO_NEAREST_INT);
    uint8_t vo[16];
    _mm_storeu_si128((__m128i*)vo, vh);
    memcpy(out + tail_split * 2, vo, (len - tail_split) * 2);
  }
}

__attribute__((target("avx,f16c")))
HIDDEN __m256 load_f16x8_tail(const uint16_t* v, size_t len) {
  uint16_t r[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  memcpy(r, v, len * 2);
  return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) r));
}

__attribute__((target("avx")))
HIDDEN __m256 load_f32x8_tail(const float* v, size_t len) {
  float r[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  memcpy(r, v, len * 4);
  return _mm256_loadu_ps(r);
}

__attribute__((target("avx")))
HIDDEN float reduce_f32x8(__m256 v) {
  __m128 x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  __m128 y = _mm_shuffle_ps(x, x, _MM_SHUFFLE(0, 0, 3, 2));
  __m128 z = _mm_add_ps(x, y);
  return _mm_cvtss_f32(_mm_hadd_ps(z, z));
}

__attribute__((target("avx,f16c,fma")))
EXPORT float et_dot_f16_f16_avx512(const uint16_t* a, const uint16_t* b, size_t len) {
  size_t tail_split = len & ~7;
  __m256 dotv = _mm256_set1_ps(0.0);
  for (size_t i = 0; i < tail_split; i += 8) {
    __m256 av = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a + i)));
    __m256 bv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + i)));
    dotv = _mm256_fmadd_ps(av, bv, dotv);
  }

  if (tail_split < len) {
    __m256 av = load_f16x8_tail(a + tail_split, len - tail_split);
    __m256 bv = load_f16x8_tail(b + tail_split, len - tail_split);
    dotv = _mm256_fmadd_ps(av, bv, dotv);
  }

  return reduce_f32x8(dotv);
}

__attribute__((target("avx,f16c,fma")))
EXPORT float et_dot_f32_f16_avx512(const float* a, const uint16_t* b, size_t len) {
  size_t tail_split = len & ~7;
  __m256 dotv = _mm256_set1_ps(0.0);
  for (size_t i = 0; i < tail_split; i += 8) {
    __m256 av = _mm256_loadu_ps(a + i);
    __m256 bv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + i)));
    dotv = _mm256_fmadd_ps(av, bv, dotv);
  }

  if (tail_split < len) {
    __m256 av = load_f32x8_tail(a + tail_split, len - tail_split);
    __m256 bv = load_f16x8_tail(b + tail_split, len - tail_split);
    dotv = _mm256_fmadd_ps(av, bv, dotv);
  }

  return reduce_f32x8(dotv);
}

__attribute__((target("avx,f16c,fma")))
EXPORT float et_l2_f16_f16_avx512(const uint16_t* a, const uint16_t* b, size_t len) {
  size_t tail_split = len & ~7;
  __m256 sumv = _mm256_set1_ps(0.0);
  for (size_t i = 0; i < tail_split; i += 8) {
    __m256 av = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a + i)));
    __m256 bv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + i)));
    __m256 diff = _mm256_sub_ps(av, bv);
    sumv = _mm256_fmadd_ps(diff, diff, sumv);
  }

  if (tail_split < len) {
    __m256 av = load_f16x8_tail(a + tail_split, len - tail_split);
    __m256 bv = load_f16x8_tail(b + tail_split, len - tail_split);
    __m256 diff = _mm256_sub_ps(av, bv);
    sumv = _mm256_fmadd_ps(diff, diff, sumv);
  }

  return reduce_f32x8(sumv);
}

__attribute__((target("avx,f16c,fma")))
EXPORT float et_l2_f32_f16_avx512(const float* a, const uint16_t* b, size_t len) {
  size_t tail_split = len & ~7;
  __m256 sumv = _mm256_set1_ps(0.0);
  for (size_t i = 0; i < tail_split; i += 8) {
    __m256 av = _mm256_loadu_ps(a + i);
    __m256 bv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + i)));
    __m256 diff = _mm256_sub_ps(av, bv);
    sumv = _mm256_fmadd_ps(diff, diff, sumv);
  }

  if (tail_split < len) {
    __m256 av = load_f32x8_tail(a + tail_split, len - tail_split);
    __m256 bv = load_f16x8_tail(b + tail_split, len - tail_split);
    __m256 diff = _mm256_sub_ps(av, bv);
    sumv = _mm256_fmadd_ps(diff, diff, sumv);
  }

  return reduce_f32x8(sumv);
}

#endif /* __x86_64__ */