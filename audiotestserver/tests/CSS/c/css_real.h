/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
/* css_real.h — working precision of the streaming PLL decoder (libcsspll).
 *
 * The Python reference (css_decoder.decode) does its array DSP in SIG, which
 * is float32 by default and float64 under CSS_SIG_DTYPE=float64. Audio that
 * the reference stores in explicit float32 arrays (the MF ring, AGC output,
 * the raw audio history, the preamble-detector history) stays float32 in
 * BOTH modes; that is mirrored here with plain `float` storage.
 *
 *   default          css_real = float   -> libcsspll    (the shipping build)
 *   -DCSS_PLL_F64    css_real = double  -> libcsspll64  (grading build: must
 *                    match Python run with CSS_SIG_DTYPE=float64; a mismatch
 *                    there is a port bug, a float32-only mismatch is precision)
 *
 * Scalar state the reference keeps in Python floats (PLL integrators, AGC
 * gain, SNR, confidences) is double in both builds, as in Python.
 */
#ifndef CSS_REAL_H
#define CSS_REAL_H
#include <stddef.h>

#ifdef CSS_PLL_F64
typedef double css_real;
#define CSS_REAL_EPS 2.220446049250313e-16
#else
typedef float css_real;
#define CSS_REAL_EPS 1.1920928955078125e-07
#endif

typedef struct { css_real re, im; } css_rcpx;
typedef struct { double re, im; } css_dcpx;

static inline size_t css_al8(size_t v) { return (v + 7u) & ~(size_t)7u; }

/* Python's `1 << x.bit_length()` for x > 0: the power of two strictly above
 * the highest set bit (NOT the smallest pow2 >= x; for x a power of two it is
 * 2x). The reference sizes its MF/retro FFTs this way. */
static inline int css_pow2_above(long x)
{
    int n = 1;
    while (x > 0) { n <<= 1; x >>= 1; }
    return n;
}

/* Python's `1 << (x - 1).bit_length()`: the smallest power of two >= x. */
static inline int css_pow2_atleast(long x)
{
    return x <= 1 ? 1 : css_pow2_above(x - 1);
}

#endif
