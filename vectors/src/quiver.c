#define EXPORT __attribute__((visibility("default")))
#define HIDDEN __attribute__((visibility("hidden")))

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __aarch64__

#include <arm_neon.h>

__attribute__((target("+dotprod"))) EXPORT int32_t et_quiver_asymmetric_ip(
    const int8_t *query, size_t len, const uint8_t *doc, const int8_t *table) {
  // Load the table we're going to decode from.
  int8_t tmp_table[16];
  memset(tmp_table, 0, 16);
  memcpy(tmp_table, table, 4);
  int8x16_t decode_table = vld1q_s8((const int8_t *)&tmp_table);

  int32x4_t ip0 = vdupq_n_s32(0);
  int32x4_t ip1 = vdupq_n_s32(0);
  int32x4_t ip2 = vdupq_n_s32(0);
  int32x4_t ip3 = vdupq_n_s32(0);
  uint8x16_t mask = vdupq_n_u8(3);

  for (size_t i = 0; i < len; i += 64) {
    int8x16_t q0 = vld1q_u8((const uint8_t *)query + i);
    int8x16_t q1 = vld1q_u8((const uint8_t *)query + i + 16);
    int8x16_t q2 = vld1q_u8((const uint8_t *)query + i + 32);
    int8x16_t q3 = vld1q_u8((const uint8_t *)query + i + 48);
    uint8x16_t d = vld1q_u8(doc + (i / 4));

    int8x16_t d0 = vqtbl1q_s8(decode_table, vandq_u8(d, mask));
    ip0 = vdotq_s32(ip0, q0, d0);
    int8x16_t d1 = vqtbl1q_s8(decode_table, vandq_u8(vshrq_n_u8(d, 2), mask));
    ip1 = vdotq_s32(ip1, q1, d1);
    int8x16_t d2 = vqtbl1q_s8(decode_table, vandq_u8(vshrq_n_u8(d, 4), mask));
    ip2 = vdotq_s32(ip2, q2, d2);
    int8x16_t d3 = vqtbl1q_s8(decode_table, vandq_u8(vshrq_n_u8(d, 6), mask));
    ip3 = vdotq_s32(ip3, q3, d3);
  }

  // XXX support tails.
  return vaddvq_s32(vaddq_s32(vaddq_s32(ip0, ip1), vaddq_s32(ip2, ip3)));
}

#endif /* __aarch64__ */
