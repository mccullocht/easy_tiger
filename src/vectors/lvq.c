#define EXPORT __attribute__((visibility("default")))
#define HIDDEN __attribute__((visibility("hidden")))

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __aarch64__

#include <arm_neon.h>

EXPORT uint32_t et_lvq_dot_u1(const uint8_t* a, const uint8_t* b, size_t len) {
    size_t tail = len & ~31;
    uint16x8_t dotv = vdupq_n_u16(0);
    for (size_t i = 0; i < tail; i += 32) {
        uint8x16_t av = vld1q_u8(a + i);
        uint8x16_t bv = vld1q_u8(b + i);
        uint8x16_t abvcnt = vcntq_u8(vandq_u8(av, bv));

        av = vld1q_u8(a + i + 16);
        bv = vld1q_u8(b + i + 16);
        abvcnt = vaddq_u8(abvcnt, vcntq_u8(vandq_u8(av, bv)));
        dotv = vaddq_u16(dotv, vpaddlq_u8(abvcnt));
    }

    uint32_t dot = vaddlvq_u16(dotv);
    for (size_t i = tail; i < len; i++) {
        dot += __builtin_popcount(a[i] & b[i]);
    }

    return dot;
}

__attribute__((target("+dotprod")))
EXPORT uint32_t et_lvq_dot_u4(const uint8_t* a, const uint8_t* b, size_t len) {
    size_t tail = len & ~31;
    uint32x4_t dot0 = vdupq_n_u32(0);
    uint32x4_t dot1 = vdupq_n_u32(0);
    uint32x4_t dot2 = vdupq_n_u32(0);
    uint32x4_t dot3 = vdupq_n_u32(0);
    uint8x16_t nibble_mask = vdupq_n_u8(0xf);
    for (size_t i = 0; i < tail; i += 32) {
        uint8x16_t av = vld1q_u8(a + i);
        uint8x16_t bv = vld1q_u8(b + i);
        dot0 = vdotq_u32(dot0, vandq_u8(av, nibble_mask), vandq_u8(bv, nibble_mask));
        dot1 = vdotq_u32(dot1, vshrq_n_u8(av, 4), vshrq_n_u8(bv, 4));

        av = vld1q_u8(a + i + 16);
        bv = vld1q_u8(b + i + 16);
        dot2 = vdotq_u32(dot2, vandq_u8(av, nibble_mask), vandq_u8(bv, nibble_mask));
        dot3 = vdotq_u32(dot3, vshrq_n_u8(av, 4), vshrq_n_u8(bv, 4));
    }

    uint32_t dot = vaddvq_u32(vaddq_u32(vaddq_u32(dot0, dot1), vaddq_u32(dot2, dot3)));
    for (size_t i = tail; i < len; i++) {
        uint32_t av = a[i];
        uint32_t bv = b[i];
        dot += (av & 0xf) * (bv & 0xf) + (av >> 4) * (bv >> 4);
    }

    return dot;
}

__attribute__((target("+dotprod")))
EXPORT uint32_t et_lvq_dot_u8(const uint8_t* a, const uint8_t* b, size_t len) {
    // XXX consider unrolling more. this is so stupid!
    size_t tail = len & ~31;
    uint32x4_t dot0 = vdupq_n_u32(0);
    uint32x4_t dot1 = vdupq_n_u32(0);
    for (size_t i = 0; i < tail; i += 32) {
        uint8x16_t av = vld1q_u8(a + i);
        uint8x16_t bv = vld1q_u8(b + i);
        dot0 = vdotq_u32(dot0, av, bv);
        av = vld1q_u8(a + i + 16);
        bv = vld1q_u8(b + i + 16);
        dot1 = vdotq_u32(dot1, av, bv);
    }

    uint32_t dot = vaddvq_u32(vaddq_u32(dot0, dot1));
    for (size_t i = tail; i < len; i++) {
        uint32_t av = a[i];
        uint32_t bv = b[i];
        dot += av * bv;
    }

    return dot;
}

#endif /* __aarch64__ */