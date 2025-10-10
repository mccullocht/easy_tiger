#define EXPORT __attribute__((visibility("default")))
#define HIDDEN __attribute__((visibility("hidden")))

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __aarch64__

#include <arm_neon.h>

__attribute__((target("neon")))
EXPORT uint32_t et_lvq_dot_u1(const uint8_t* a, const uint8_t* b, size_t len) {
    size_t tail = len & ~15;
    uint16x8_t dotv = vdupq_n_u16(0);
    for (size_t i = 0; i < tail; i += 16) {
        uint8x16_t av = vld1q_u8(a + i);
        uint8x16_t bv = vld1q_u8(b + i);
        uint8x16_t abvcnt = vcntq_u8(vandq_u8(av, bv));
        dotv = vaddq_u16(dotv, vpaddlq_u8(abvcnt));
    }

    uint32_t dot = vaddvq_u32(dotv);
    // XXX FIXME scalar tail
    //for (size_t i = tail; i < len; i++) {
    //    dot += ((uint32_t)a[i]) * ((uint32_t)b[i]);
    //}

    return dot;
}

__attribute__((target("neon")))
EXPORT uint32_t et_lvq_dot_u4(const uint8_t* a, const uint8_t* b, size_t len) {
    size_t tail = len & ~15;
    uint32x4_t dotv = vdupq_n_u32(0);
    uint8x16_t nibble_mask = vdupq_n_u8(0xf);
    for (size_t i = 0; i < tail; i += 16) {
        uint8x16_t av = vld1q_u8(a + i);
        uint8x16_t bv = vld1q_u8(b + i);

        uint8x16_t ave = vandq_u8(av, nibble_mask);
        uint8x16_t bve = vandq_u8(bv, nibble_mask);
        dotv = vdotq_u32(dotv, ave, bve);

        uint8x16_t avo = vandq_u8(vshrq_n_u8(av, 4), nibble_mask);
        uint8x16_t bvo = vandq_u8(vshrq_n_u8(bv, 4), nibble_mask);
        dotv = vdotq_u32(dotv, avo, bvo);
    }

    uint16_t dot = vaddvq_u16(dotv);
    // XXX FIXME scalar tail
    //for (size_t i = tail; i < len; i++) {
    //    dot += ((uint32_t)a[i]) * ((uint32_t)b[i]);
    //}

    return dot;
}

EXPORT uint32_t et_lvq_dot_u8(const uint8_t* a, const uint8_t* b, size_t len) {
    size_t tail = len & ~15;
    uint32x4_t dotv = vdupq_n_u32(0);
    for (size_t i = 0; i < tail; i += 16) {
        uint8x16_t av = vld1q_u8(a + i);
        uint8x16_t bv = vld1q_u8(b + i);
        dotv = vdotq_u32(dotv, av, bv);
    }

    uint32_t dot = vaddvq_u32(dotv);
    for (size_t i = tail; i < len; i++) {
        dot += ((uint32_t)a[i]) * ((uint32_t)b[i]);
    }

    return dot;
}

#endif /* __aarch64__ */