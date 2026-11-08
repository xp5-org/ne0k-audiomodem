/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
/* css_pfft.h — FFTs for the streaming PLL decoder, freestanding C99.
 *
 * Two instantiations of one radix-2 implementation:
 *   cssr_*  in css_real  (the reference's SIG-typed array FFTs)
 *   cssd_*  in double    (the reference's explicit float64 paths: the
 *                         whitening filter design and the IF-slope analytic
 *                         signal)
 * plus a Bluestein complex DFT of ANY length in double, for the IF-slope
 * estimator whose segment length is the chirp length (not a power of two).
 *
 * Twiddles are computed in double and rounded once to the working type.
 * Caller memory only; a plan is (struct + workspace) for one length.
 */
#ifndef CSS_PFFT_H
#define CSS_PFFT_H
#include "css_real.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- real FFT, power-of-two n, working precision css_real -------------- */
typedef struct {
    int n, h;
    css_rcpx *tw_h;   /* h-1 per-stage twiddles */
    css_rcpx *tw_n;   /* h   */
    css_rcpx *z;      /* h   */
} cssr_fft;
size_t cssr_fft_workspace(int n);
int    cssr_fft_init(cssr_fft *f, int n, void *mem, size_t memlen);
/* x[n] -> X[n/2+1]                                                         */
void   cssr_rfft(cssr_fft *f, const css_real *x, css_rcpx *X);
/* X[n/2+1] -> x[n], numpy irfft scaling (1/n). X is not modified.          */
void   cssr_irfft(cssr_fft *f, const css_rcpx *X, css_real *x);

/* ---- the same in double -------------------------------------------------- */
typedef struct {
    int n, h;
    css_dcpx *tw_h;
    css_dcpx *tw_n;
    css_dcpx *z;
} cssd_fft;
size_t cssd_fft_workspace(int n);
int    cssd_fft_init(cssd_fft *f, int n, void *mem, size_t memlen);
void   cssd_rfft(cssd_fft *f, const double *x, css_dcpx *X);
void   cssd_irfft(cssd_fft *f, const css_dcpx *X, double *x);

/* In-place complex FFT of power-of-two length n in double.
 * sign -1 forward, +1 inverse (unscaled). tw: n-1 per-stage twiddles from
 * cssd_twiddles(tw, n). */
void   cssd_cfft(css_dcpx *a, int n, const css_dcpx *tw, int sign);
void   cssd_twiddles(css_dcpx *tw, int n);

/* ---- Bluestein DFT of any length N <= nmax, double ----------------------- */
typedef struct {
    int nmax, m;          /* m = pow2 >= 2*nmax - 1                          */
    int n_cached;         /* N the chirp/kernel tables currently hold, or 0  */
    css_dcpx *tw;         /* m-1 per-stage twiddles */
    css_dcpx *w;          /* nmax: e^{-i pi k^2 / N}                          */
    css_dcpx *kern;       /* m : FFT of the conj chirp kernel                 */
    css_dcpx *buf;        /* m */
} cssd_blue;
size_t cssd_blue_workspace(int nmax);
int    cssd_blue_init(cssd_blue *b, int nmax, void *mem, size_t memlen);
/* out[k] = sum_j in[j] e^{sign * 2 pi i jk/N}, unscaled. in/out may alias. */
int    cssd_blue_dft(cssd_blue *b, const css_dcpx *in, css_dcpx *out, int N,
                     int sign);

#ifdef __cplusplus
}
#endif
#endif
