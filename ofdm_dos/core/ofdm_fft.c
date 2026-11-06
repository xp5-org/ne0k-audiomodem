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

#include "ofdm_fft.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static size_t align8(size_t v) { return (v + 7u) & ~(size_t)7u; }

/* Must stay the term-for-term sum of the bumps ofdm_fft_init performs, in the
 * same order.  Edit the two together. */
size_t ofdm_fft_workspace(int n)
{
    int h;
    if (n < 4 || (n & (n - 1)) != 0) return 0;
    h = n / 2;
    return align8((size_t)(h / 2) * sizeof(ofdm_cpx))    /* tw_h */
         + align8((size_t)h       * sizeof(ofdm_cpx))    /* tw_n */
         + align8((size_t)h       * sizeof(ofdm_cpx));   /* z    */
}

int ofdm_fft_init(ofdm_fft *f, int n, void *mem, size_t memlen)
{
    int h, i;
    unsigned char *p = (unsigned char *)mem;
    if (!f || !mem) return -1;
    if (n < 4 || (n & (n - 1)) != 0) return -1;      /* power of two only */
    if (memlen < ofdm_fft_workspace(n)) return -2;
    h = n / 2;

    f->n = n; f->h = h;
    f->tw_h = (ofdm_cpx *)p; p += align8((size_t)(h / 2) * sizeof(ofdm_cpx));
    f->tw_n = (ofdm_cpx *)p; p += align8((size_t)h       * sizeof(ofdm_cpx));
    f->z    = (ofdm_cpx *)p;

    /* Twiddles are generated in double regardless of ofdm_real: this is a
     * one-time init cost, and a float32 build wants accurate twiddles rounded
     * once, not twiddles computed in float32. */
    for (i = 0; i < h / 2; i++) {                    /* e^{-2*pi*i*j/h} */
        double a = -2.0 * M_PI * (double)i / (double)h;
        f->tw_h[i].re = (ofdm_real)cos(a);
        f->tw_h[i].im = (ofdm_real)sin(a);
    }
    for (i = 0; i < h; i++) {                        /* e^{-2*pi*i*k/n} */
        double a = -2.0 * M_PI * (double)i / (double)n;
        f->tw_n[i].re = (ofdm_real)cos(a);
        f->tw_n[i].im = (ofdm_real)sin(a);
    }
    return 0;
}

/* In-place iterative radix-2 DIT. sign<0 forward, sign>0 inverse (unscaled). */
static void cfft(ofdm_cpx *a, int n, const ofdm_cpx *tw, int sign)
{
    int i, j, len, k;
    for (i = 1, j = 0; i < n; i++) {                 /* bit reversal */
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { ofdm_cpx t = a[i]; a[i] = a[j]; a[j] = t; }
    }
    for (len = 2; len <= n; len <<= 1) {
        int half = len >> 1, step = n / len;
        for (i = 0; i < n; i += len) {
            for (k = 0; k < half; k++) {
                ofdm_cpx w = tw[(size_t)k * step];
                ofdm_cpx b = a[i + k + half], u = a[i + k], v;
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

void ofdm_rfft(ofdm_fft *f, const ofdm_real *x, ofdm_cpx *X)
{
    int h = f->h, k;
    ofdm_cpx *z = f->z;
    for (k = 0; k < h; k++) { z[k].re = x[2 * k]; z[k].im = x[2 * k + 1]; }
    cfft(z, h, f->tw_h, -1);

    X[0].re = z[0].re + z[0].im;  X[0].im = (ofdm_real)0;
    X[h].re = z[0].re - z[0].im;  X[h].im = (ofdm_real)0;
    for (k = 1; k < h; k++) {
        ofdm_cpx zk = z[k], zm = z[h - k];
        /* Ze = (zk + conj(zm))/2 ; Zo = (zk - conj(zm))/(2i) */
        ofdm_real er  = (ofdm_real)0.5 * (zk.re + zm.re);
        ofdm_real ei  = (ofdm_real)0.5 * (zk.im - zm.im);
        ofdm_real orr = (ofdm_real)0.5 * (zk.im + zm.im);
        ofdm_real oi  = (ofdm_real)-0.5 * (zk.re - zm.re);
        ofdm_cpx  w   = f->tw_n[k];
        X[k].re = er + (orr * w.re - oi * w.im);
        X[k].im = ei + (orr * w.im + oi * w.re);
    }
}

void ofdm_irfft(ofdm_fft *f, const ofdm_cpx *X, ofdm_real *x)
{
    int h = f->h, k;
    ofdm_cpx *z = f->z;
    ofdm_real s = (ofdm_real)1 / (ofdm_real)h;
    for (k = 0; k < h; k++) {
        ofdm_cpx xk = X[k], xm = X[h - k];
        ofdm_real er = (ofdm_real)0.5 * (xk.re + xm.re);
        ofdm_real ei = (ofdm_real)0.5 * (xk.im - xm.im);
        ofdm_real dr = (ofdm_real)0.5 * (xk.re - xm.re);
        ofdm_real di = (ofdm_real)0.5 * (xk.im + xm.im);
        /* Zo = (Xk - conj(Xm))/2 * e^{+2*pi*i*k/n} ; conj twiddle for inverse */
        ofdm_cpx  w  = f->tw_n[k];
        ofdm_real orr = dr * w.re + di * w.im;
        ofdm_real oi  = di * w.re - dr * w.im;
        z[k].re = er - oi;      /* Ze + i*Zo */
        z[k].im = ei + orr;
    }
    cfft(z, h, f->tw_h, +1);
    for (k = 0; k < h; k++) {
        x[2 * k]     = z[k].re * s;
        x[2 * k + 1] = z[k].im * s;
    }
}
