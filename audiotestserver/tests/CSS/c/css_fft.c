/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
#include "css_fft.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static size_t align8(size_t v) { return (v + 7u) & ~(size_t)7u; }

size_t css_fft_workspace(int n)
{
    int h = n / 2;
    return align8((size_t)(h / 2) * sizeof(css_cpx))    /* tw_h */
         + align8((size_t)h       * sizeof(css_cpx))    /* tw_n */
         + align8((size_t)h       * sizeof(css_cpx));   /* z    */
}

int css_fft_init(css_fft *f, int n, void *mem, size_t memlen)
{
    int h = n / 2, i;
    unsigned char *p = (unsigned char *)mem;
    if (n < 4 || (n & (n - 1)) != 0) return -1;      /* power of two only */
    if (memlen < css_fft_workspace(n))  return -2;

    f->n = n; f->h = h;
    f->tw_h = (css_cpx *)p; p += align8((size_t)(h / 2) * sizeof(css_cpx));
    f->tw_n = (css_cpx *)p; p += align8((size_t)h       * sizeof(css_cpx));
    f->z    = (css_cpx *)p;

    /* e^{-2*pi*i*j/h}, j < h/2 — butterflies of the length-h complex FFT */
    for (i = 0; i < h / 2; i++) {
        double a = -2.0 * M_PI * (double)i / (double)h;
        f->tw_h[i].re = (float)cos(a);
        f->tw_h[i].im = (float)sin(a);
    }
    /* e^{-2*pi*i*k/n}, k < h — real pack/unpack rotation */
    for (i = 0; i < h; i++) {
        double a = -2.0 * M_PI * (double)i / (double)n;
        f->tw_n[i].re = (float)cos(a);
        f->tw_n[i].im = (float)sin(a);
    }
    return 0;
}

/* In-place iterative radix-2 DIT. sign<0 forward, sign>0 inverse (unscaled). */
static void cfft(css_cpx *a, int n, const css_cpx *tw, int sign)
{
    int i, j, len, k;
    for (i = 1, j = 0; i < n; i++) {                 /* bit reversal */
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { css_cpx t = a[i]; a[i] = a[j]; a[j] = t; }
    }
    for (len = 2; len <= n; len <<= 1) {
        int half = len >> 1, step = n / len;
        for (i = 0; i < n; i += len) {
            for (k = 0; k < half; k++) {
                css_cpx w = tw[(size_t)k * step];
                css_cpx b = a[i + k + half], u = a[i + k], v;
                if (sign > 0) w.im = -w.im;
                v.re = b.re * w.re - b.im * w.im;
                v.im = b.re * w.im + b.im * w.re;
                a[i + k].re        = u.re + v.re;
                a[i + k].im        = u.im + v.im;
                a[i + k + half].re = u.re - v.re;
                a[i + k + half].im = u.im - v.im;
            }
        }
    }
}

void css_rfft(css_fft *f, const float *x, css_cpx *X)
{
    int h = f->h, k;
    css_cpx *z = f->z;
    for (k = 0; k < h; k++) { z[k].re = x[2 * k]; z[k].im = x[2 * k + 1]; }
    cfft(z, h, f->tw_h, -1);

    X[0].re = z[0].re + z[0].im;  X[0].im = 0.0f;
    X[h].re = z[0].re - z[0].im;  X[h].im = 0.0f;
    for (k = 1; k < h; k++) {
        css_cpx zk = z[k], zm = z[h - k];
        /* Ze = (zk + conj(zm))/2 ; Zo = (zk - conj(zm))/(2i) */
        float er =  0.5f * (zk.re + zm.re), ei =  0.5f * (zk.im - zm.im);
        float orr = 0.5f * (zk.im + zm.im), oi = -0.5f * (zk.re - zm.re);
        css_cpx w = f->tw_n[k];
        X[k].re = er + (orr * w.re - oi * w.im);
        X[k].im = ei + (orr * w.im + oi * w.re);
    }
}

void css_irfft(css_fft *f, const css_cpx *X, float *x)
{
    int h = f->h, k;
    css_cpx *z = f->z;
    float s = 1.0f / (float)h;
    for (k = 0; k < h; k++) {
        css_cpx xk = X[k], xm = X[h - k];
        float er =  0.5f * (xk.re + xm.re), ei =  0.5f * (xk.im - xm.im);
        float dr =  0.5f * (xk.re - xm.re), di =  0.5f * (xk.im + xm.im);
        /* Zo = (Xk - conj(Xm))/2 * e^{+2*pi*i*k/n} ; conj twiddle for inverse */
        css_cpx w = f->tw_n[k];
        float orr = dr * w.re + di * w.im;
        float oi  = di * w.re - dr * w.im;
        z[k].re = er - oi;      /* Ze + i*Zo */
        z[k].im = ei + orr;
    }
    cfft(z, h, f->tw_h, +1);
    for (k = 0; k < h; k++) {
        x[2 * k]     = z[k].re * s;
        x[2 * k + 1] = z[k].im * s;
    }
}
