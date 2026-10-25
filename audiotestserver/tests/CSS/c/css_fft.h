/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
/* css_fft.h — minimal real FFT, swappable for CMSIS-DSP arm_rfft_fast_f32.
 *
 * Freestanding: no malloc, no stdio. All state lives in caller memory. */
#ifndef CSS_FFT_H
#define CSS_FFT_H
#include <stddef.h>

#include <math.h>

/* C89 math: fabsf()/log10f(), and every other f-suffixed libm entry point,
 * are C99. A strict C89 library (OpenWatcom targeting DOS/Win16/Win32, and
 * some embedded toolchains) does not declare them, and an implicit
 * declaration returning int silently corrupts every magnitude in the
 * matched filter.
 *
 * Define CSS_C89_MATH on such a target to substitute the double-precision
 * entry points, which are C89. Opt-in, not autodetected, so quietly changing
 * which libm routine runs would not blur a precision regression into a
 * build-config difference (see ofdm_fft.h's ofdm_hypot for the same split).
 *
 * css_fabsf is exact either way: fabs only clears a sign bit, and float ->
 * double -> float round-trips a float exactly, so the fallback costs only a
 * widening. css_log10f can differ in the last ulp; it feeds snr_db, a
 * diagnostic and coarse gate, never a bit decision.
 */
#ifdef CSS_C89_MATH
#define css_fabsf(x)  ((float)fabs((double)(x)))
#define css_log10f(x) ((float)log10((double)(x)))
#else
#define css_fabsf(x)  fabsf(x)
#define css_log10f(x) log10f(x)
#endif

typedef struct { float re, im; } css_cpx;

typedef struct {
    int      n;        /* real transform length (power of two)      */
    int      h;        /* n/2 — length of the packed complex FFT    */
    css_cpx *tw_h;     /* h/2 twiddles for the complex FFT          */
    css_cpx *tw_n;     /* h   twiddles for the real pack/unpack     */
    css_cpx *z;        /* h   scratch                               */
} css_fft;

/* Bytes of caller memory css_fft_init needs for this n. */
size_t css_fft_workspace(int n);

/* mem must be >= css_fft_workspace(n) and 8-byte aligned. */
int  css_fft_init(css_fft *f, int n, void *mem, size_t memlen);

/* x[n] real -> X[n/2+1] half spectrum. */
void css_rfft(css_fft *f, const float *x, css_cpx *X);

/* X[n/2+1] half spectrum -> x[n] real, scaled like numpy.fft.irfft. */
void css_irfft(css_fft *f, const css_cpx *X, float *x);

#endif
