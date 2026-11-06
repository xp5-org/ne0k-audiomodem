/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026 XP5.ORG
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/* ofdm_fft.h — minimal real FFT, precision-parameterised, freestanding C99.
 *
 * Same radix-2 algorithm as tests/CSS/c/css_fft.c, generalised so the scalar
 * type is a compile-time choice:
 *
 *     default            ofdm_real = double   (matches the numpy reference,
 *                                              which computes rfft in float64)
 *     -DOFDM_ACQ_F32     ofdm_real = float    (the shipping target: 32-bit
 *                                              MCU / CMSIS-DSP)
 *
 * The port is validated in two stages against the same numpy oracle -- double
 * first, so a mismatch is unambiguously a PORT bug, then float32, where any
 * new difference is unambiguously a PRECISION effect.  One source builds both,
 * so the two stages cannot drift apart.
 *
 * css_fft.c is the float-only ancestor of this file and is still what the CSS
 * matched filter links against.  The two should be unified onto this header
 * once CSS is re-validated; until then css_fft.c is left untouched on purpose
 * -- it is a shipping, regression-covered artifact and this port is not the
 * change that should destabilise it.
 *
 * Swapping in CMSIS-DSP arm_rfft_fast_f32 means providing these four symbols
 * over arm_rfft_fast_instance_f32 and nothing else changes: the real input is
 * a plain ofdm_real[n], the spectrum is ofdm_cpx[n/2+1] interleaved re/im, and
 * ofdm_irfft is scaled to match numpy.fft.irfft so no rescale hides in the
 * caller.  Power-of-two only, which is also CMSIS's constraint.
 *
 * Freestanding: no malloc, no stdio, no globals. All state lives in caller
 * memory.
 */
#ifndef OFDM_FFT_H
#define OFDM_FFT_H
#include <stddef.h>
#include <math.h>

/* hypot()/hypotf() are C99 + POSIX and are NOT in C89.  A strict C89 library
 * (OpenWatcom targeting DOS/Win16/Win32, and some embedded toolchains) does not
 * declare them, and an implicit declaration returning int silently corrupts
 * every magnitude in the acquisition metric.
 *
 * Define OFDM_C89_MATH on such a target to substitute sqrt-based versions.
 * This is deliberately OPT-IN, not autodetected: the substitution is NOT
 * bit-identical -- real hypot is written to avoid intermediate overflow and
 * rounds differently -- and the host equivalence suite grades the C against
 * numpy, whose complex abs IS hypot.  Leaving the default alone keeps that
 * comparison meaningful; a DOS build does not need it and can take the sqrt
 * form, whose overflow range is irrelevant for values bounded by 1. */
#ifdef OFDM_C89_MATH
#define ofdm_hypot(a, b) \
    sqrt((double)(a) * (double)(a) + (double)(b) * (double)(b))
#define ofdm_hypotf(a, b) \
    ((float)sqrt((double)(a) * (double)(a) + (double)(b) * (double)(b)))
#else
#define ofdm_hypot(a, b)  hypot((double)(a), (double)(b))
#define ofdm_hypotf(a, b) hypotf((float)(a), (float)(b))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef OFDM_ACQ_F32
typedef float  ofdm_real;
#else
typedef double ofdm_real;
#endif

typedef struct { ofdm_real re, im; } ofdm_cpx;

typedef struct {
    int       n;       /* real transform length (power of two)      */
    int       h;       /* n/2 — length of the packed complex FFT    */
    ofdm_cpx *tw_h;    /* h/2 twiddles for the complex FFT          */
    ofdm_cpx *tw_n;    /* h   twiddles for the real pack/unpack     */
    ofdm_cpx *z;       /* h   scratch                               */
} ofdm_fft;

/* Bytes of caller memory ofdm_fft_init needs for this n. 0 if n is invalid. */
size_t ofdm_fft_workspace(int n);

/* mem must be >= ofdm_fft_workspace(n) and 8-byte aligned.
 * Returns 0 ok, -1 n not a power of two or < 4, -2 memlen too small. */
int  ofdm_fft_init(ofdm_fft *f, int n, void *mem, size_t memlen);

/* x[n] real -> X[n/2+1] half spectrum. */
void ofdm_rfft(ofdm_fft *f, const ofdm_real *x, ofdm_cpx *X);

/* X[n/2+1] half spectrum -> x[n] real, scaled like numpy.fft.irfft. */
void ofdm_irfft(ofdm_fft *f, const ofdm_cpx *X, ofdm_real *x);

#ifdef __cplusplus
}
#endif
#endif
