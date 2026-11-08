/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
/* css_pfft.c — radix-2 DIT with per-stage contiguous twiddle tables.
 *
 * Twiddles for stage `len` live at tw[len/2 - 1 .. len - 2] (e^{-2 pi i k /
 * len}, k < len/2), n-1 entries in all, so the inner loop reads them
 * sequentially instead of with a stride of n/len; forward and inverse are
 * separate loops so the conjugate costs nothing.  Twiddles are computed in
 * double and rounded once to the working type. */
#include "css_pfft.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CSS_PFFT_IMPL(PFX, T, C)                                               \
static void PFX##_stage_tw(C *tw, int n)                                       \
{                                                                              \
    int len, k;                                                                \
    for (len = 2; len <= n; len <<= 1) {                                       \
        C *t = tw + (len / 2 - 1);                                             \
        for (k = 0; k < len / 2; k++) {                                        \
            double a = -2.0 * M_PI * (double)k / (double)len;                  \
            t[k].re = (T)cos(a); t[k].im = (T)sin(a);                          \
        }                                                                      \
    }                                                                          \
}                                                                              \
static void PFX##_bitrev(C *a, int n)                                          \
{                                                                              \
    int i, j;                                                                  \
    for (i = 1, j = 0; i < n; i++) {                                           \
        int bit = n >> 1;                                                      \
        for (; j & bit; bit >>= 1) j ^= bit;                                   \
        j ^= bit;                                                              \
        if (i < j) { C t = a[i]; a[i] = a[j]; a[j] = t; }                      \
    }                                                                          \
}                                                                              \
static void PFX##_cfft_s(C *a, int n, const C *tw, int sign)                   \
{                                                                              \
    int i, len, k;                                                             \
    if (n < 2) return;                                                         \
    PFX##_bitrev(a, n);                                                        \
    for (i = 0; i < n; i += 2) {                                               \
        C u = a[i], v = a[i + 1];                                              \
        a[i].re = u.re + v.re; a[i].im = u.im + v.im;                          \
        a[i + 1].re = u.re - v.re; a[i + 1].im = u.im - v.im;                  \
    }                                                                          \
    for (len = 4; len <= n; len <<= 1) {                                       \
        int half = len >> 1;                                                   \
        const C *t = tw + (half - 1);                                          \
        for (i = 0; i < n; i += len) {                                         \
            C *x = a + i, *y = a + i + half;                                   \
            if (sign < 0) {                                                    \
                for (k = 0; k < half; k++) {                                   \
                    T wr = t[k].re, wi = t[k].im;                              \
                    T vr = y[k].re * wr - y[k].im * wi;                        \
                    T vi = y[k].re * wi + y[k].im * wr;                        \
                    T ur = x[k].re, ui = x[k].im;                              \
                    x[k].re = ur + vr; x[k].im = ui + vi;                      \
                    y[k].re = ur - vr; y[k].im = ui - vi;                      \
                }                                                              \
            } else {                                                           \
                for (k = 0; k < half; k++) {                                   \
                    T wr = t[k].re, wi = -t[k].im;                             \
                    T vr = y[k].re * wr - y[k].im * wi;                        \
                    T vi = y[k].re * wi + y[k].im * wr;                        \
                    T ur = x[k].re, ui = x[k].im;                              \
                    x[k].re = ur + vr; x[k].im = ui + vi;                      \
                    y[k].re = ur - vr; y[k].im = ui - vi;                      \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}                                                                              \
size_t PFX##_fft_workspace(int n)                                              \
{                                                                              \
    int h = n / 2;                                                             \
    return css_al8((size_t)(h > 1 ? h : 1) * sizeof(C))                        \
         + css_al8((size_t)h * sizeof(C)) + css_al8((size_t)h * sizeof(C));    \
}                                                                              \
int PFX##_fft_init(PFX##_fft *f, int n, void *mem, size_t memlen)              \
{                                                                              \
    int h = n / 2, i;                                                          \
    unsigned char *p = (unsigned char *)mem;                                   \
    if (n < 4 || (n & (n - 1)) != 0) return -1;                                \
    if (memlen < PFX##_fft_workspace(n)) return -2;                            \
    f->n = n; f->h = h;                                                        \
    f->tw_h = (C *)p; p += css_al8((size_t)(h > 1 ? h : 1) * sizeof(C));       \
    f->tw_n = (C *)p; p += css_al8((size_t)h * sizeof(C));                     \
    f->z    = (C *)p;                                                          \
    PFX##_stage_tw(f->tw_h, h);                                                \
    for (i = 0; i < h; i++) {                                                  \
        double a = -2.0 * M_PI * (double)i / (double)n;                        \
        f->tw_n[i].re = (T)cos(a); f->tw_n[i].im = (T)sin(a);                  \
    }                                                                          \
    return 0;                                                                  \
}                                                                              \
void PFX##_rfft(PFX##_fft *f, const T *x, C *X)                                \
{                                                                              \
    int h = f->h, k;                                                           \
    C *z = f->z;                                                               \
    for (k = 0; k < h; k++) { z[k].re = x[2 * k]; z[k].im = x[2 * k + 1]; }    \
    PFX##_cfft_s(z, h, f->tw_h, -1);                                           \
    X[0].re = z[0].re + z[0].im; X[0].im = 0;                                  \
    X[h].re = z[0].re - z[0].im; X[h].im = 0;                                  \
    for (k = 1; k < h; k++) {                                                  \
        C zk = z[k], zm = z[h - k], w = f->tw_n[k];                            \
        T er = (T)0.5 * (zk.re + zm.re), ei = (T)0.5 * (zk.im - zm.im);        \
        T orr = (T)0.5 * (zk.im + zm.im), oi = (T)-0.5 * (zk.re - zm.re);      \
        X[k].re = er + (orr * w.re - oi * w.im);                               \
        X[k].im = ei + (orr * w.im + oi * w.re);                               \
    }                                                                          \
}                                                                              \
void PFX##_irfft(PFX##_fft *f, const C *X, T *x)                               \
{                                                                              \
    int h = f->h, k;                                                           \
    C *z = f->z;                                                               \
    T sc = (T)2 / (T)f->n;                                                     \
    for (k = 0; k < h; k++) {                                                  \
        C xk = X[k], xm = X[h - k], w = f->tw_n[k];                            \
        T er = (T)0.5 * (xk.re + xm.re), ei = (T)0.5 * (xk.im - xm.im);        \
        T dr = (T)0.5 * (xk.re - xm.re), di = (T)0.5 * (xk.im + xm.im);        \
        T orr = dr * w.re + di * w.im, oi = di * w.re - dr * w.im;             \
        z[k].re = er - oi; z[k].im = ei + orr;                                 \
    }                                                                          \
    PFX##_cfft_s(z, h, f->tw_h, +1);                                           \
    for (k = 0; k < h; k++) { x[2 * k] = z[k].re * sc; x[2 * k + 1] = z[k].im * sc; } \
}

CSS_PFFT_IMPL(cssr, css_real, css_rcpx)
CSS_PFFT_IMPL(cssd, double, css_dcpx)

/* n-1 stage twiddles for a length-n complex transform */
void cssd_twiddles(css_dcpx *tw, int n) { cssd_stage_tw(tw, n); }

void cssd_cfft(css_dcpx *a, int n, const css_dcpx *tw, int sign)
{
    cssd_cfft_s(a, n, tw, sign);
}

/* ---- Bluestein ---------------------------------------------------------- */
static int blue_m(int nmax)
{
    int m = css_pow2_atleast(2L * nmax - 1);
    return m < 2 ? 2 : m;
}

size_t cssd_blue_workspace(int nmax)
{
    int m;
    if (nmax < 1) return 0;
    m = blue_m(nmax);
    return css_al8((size_t)m * sizeof(css_dcpx))          /* stage twiddles */
         + css_al8((size_t)nmax * sizeof(css_dcpx))
         + 2 * css_al8((size_t)m * sizeof(css_dcpx));
}

int cssd_blue_init(cssd_blue *b, int nmax, void *mem, size_t memlen)
{
    unsigned char *p = (unsigned char *)mem;
    int m;
    if (nmax < 1) return -1;
    if (memlen < cssd_blue_workspace(nmax)) return -2;
    m = blue_m(nmax);
    b->nmax = nmax; b->m = m; b->n_cached = 0;
    b->tw   = (css_dcpx *)p; p += css_al8((size_t)m * sizeof(css_dcpx));
    b->w    = (css_dcpx *)p; p += css_al8((size_t)nmax * sizeof(css_dcpx));
    b->kern = (css_dcpx *)p; p += css_al8((size_t)m * sizeof(css_dcpx));
    b->buf  = (css_dcpx *)p;
    cssd_stage_tw(b->tw, m);
    return 0;
}

static void blue_prepare(cssd_blue *b, int N)
{
    int k, m = b->m;
    for (k = 0; k < N; k++) {
        long long kk = ((long long)k * k) % (2LL * N);
        double a = -M_PI * (double)kk / (double)N;
        b->w[k].re = cos(a); b->w[k].im = sin(a);
    }
    for (k = 0; k < m; k++) { b->kern[k].re = 0; b->kern[k].im = 0; }
    for (k = 0; k < N; k++) {
        b->kern[k].re = b->w[k].re; b->kern[k].im = -b->w[k].im;
        if (k) b->kern[m - k] = b->kern[k];
    }
    cssd_cfft_s(b->kern, m, b->tw, -1);
    b->n_cached = N;
}

int cssd_blue_dft(cssd_blue *b, const css_dcpx *in, css_dcpx *out, int N,
                  int sign)
{
    int k, m = b->m;
    double sc = 1.0 / (double)m;
    if (N < 1 || N > b->nmax) return -1;
    if (b->n_cached != N) blue_prepare(b, N);
    for (k = 0; k < N; k++) {
        double xr = in[k].re, xi = sign > 0 ? -in[k].im : in[k].im;
        b->buf[k].re = xr * b->w[k].re - xi * b->w[k].im;
        b->buf[k].im = xr * b->w[k].im + xi * b->w[k].re;
    }
    for (k = N; k < m; k++) { b->buf[k].re = 0; b->buf[k].im = 0; }
    cssd_cfft_s(b->buf, m, b->tw, -1);
    for (k = 0; k < m; k++) {
        double r = b->buf[k].re * b->kern[k].re - b->buf[k].im * b->kern[k].im;
        double i = b->buf[k].re * b->kern[k].im + b->buf[k].im * b->kern[k].re;
        b->buf[k].re = r; b->buf[k].im = i;
    }
    cssd_cfft_s(b->buf, m, b->tw, +1);
    for (k = 0; k < N; k++) {
        double yr = b->buf[k].re * sc, yi = b->buf[k].im * sc;
        double zr = yr * b->w[k].re - yi * b->w[k].im;
        double zi = yr * b->w[k].im + yi * b->w[k].re;
        out[k].re = zr; out[k].im = sign > 0 ? -zi : zi;
    }
    return 0;
}
