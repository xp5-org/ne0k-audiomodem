/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_rxf.c -- see ofdm_rxf.h. */
#include <math.h>
#include "ofdm_rxf.h"

#define PI_ 3.14159265358979323846

static size_t al8(size_t v) { return (v + 7u) & ~(size_t)7u; }

/* Modified Bessel I0 by its power series; converges fast for the beta values
 * used here (< ~7). */
static double bessel_i0(double x)
{
    double sum = 1.0, term = 1.0, y = x * x * 0.25;
    int k;
    for (k = 1; k < 60; k++) {
        term *= y / ((double)k * (double)k);
        sum += term;
        if (term < 1e-17 * sum) break;
    }
    return sum;
}

int ofdm_rxf_design(double *taps, int cap, int fs,
                    double far_lo, double far_hi, double my_lo, double my_hi,
                    double stop_db, double *got_stop_db)
{
    double ny, wp, ws, fc, delta, beta, a, w, x, h, sum, i0b;
    int highpass, n, mid, i;

    if (!taps || cap < 3 || fs <= 0) return -1;
    if (stop_db < 20.0) stop_db = 20.0;
    ny = fs / 2.0;

    if (far_lo > my_hi)       { wp = far_lo; ws = my_hi; highpass = 1; }
    else if (far_hi < my_lo)  { wp = far_hi; ws = my_lo; highpass = 0; }
    else return 0;                              /* overlapping bands */
    if (wp < 20.0) wp = 20.0;
    if (ws < 20.0) ws = 20.0;
    if (wp > ny * 0.99) wp = ny * 0.99;
    if (ws > ny * 0.99) ws = ny * 0.99;

    delta = fabs(wp - ws);
    if (delta < 1.0) delta = 1.0;
    fc = 0.5 * (wp + ws);

    /* Kaiser design: order from the transition width, beta from attenuation. */
    a = stop_db;
    n = (int)ceil((a - 7.95) / (2.285 * 2.0 * PI_ * delta / fs)) + 1;
    if (n < 5) n = 5;
    if (n % 2 == 0) n++;
    if (n > cap) {
        n = cap;
        if (n % 2 == 0) n--;
        /* what the shortened filter can still do, from the same formula */
        a = 2.285 * (n - 1) * 2.0 * PI_ * delta / fs + 7.95;
    }
    if (got_stop_db) *got_stop_db = a;
    beta = a > 50.0 ? 0.1102 * (a - 8.7)
         : a >= 21.0 ? 0.5842 * pow(a - 21.0, 0.4) + 0.07886 * (a - 21.0)
         : 0.0;
    i0b = bessel_i0(beta);

    mid = (n - 1) / 2;
    sum = 0.0;
    for (i = 0; i < n; i++) {
        x = (double)(i - mid);
        h = (i == mid) ? 2.0 * fc / fs
                       : sin(2.0 * PI_ * fc * x / fs) / (PI_ * x);
        w = (double)(i - mid) / (double)(mid ? mid : 1);
        w = bessel_i0(beta * sqrt(1.0 - w * w)) / i0b;
        taps[i] = h * w;
        sum += taps[i];
    }
    /* unity gain at DC for the lowpass; the highpass is its spectral inverse */
    for (i = 0; i < n; i++) taps[i] /= sum;
    if (highpass) {
        for (i = 0; i < n; i++) taps[i] = -taps[i];
        taps[mid] += 1.0;
    }
    return n;
}

struct ofdm_rxf {
    int     n, mid, pos;
    double *h;          /* mid+1 distinct taps (symmetric) */
    double *z;          /* delay line, 2*n so a window is always contiguous */
};

size_t ofdm_rxf_workspace(int ntaps)
{
    if (ntaps < 3 || ntaps > OFDM_RXF_MAX_TAPS || ntaps % 2 == 0) return 0;
    return al8(sizeof(struct ofdm_rxf))
         + al8((size_t)((ntaps - 1) / 2 + 1) * sizeof(double))
         + al8((size_t)(2 * ntaps) * sizeof(double));
}

int ofdm_rxf_init(ofdm_rxf **fp, void *mem, size_t memlen, const double *taps, int ntaps)
{
    struct ofdm_rxf *f;
    unsigned char *p = (unsigned char *)mem;
    size_t need = ofdm_rxf_workspace(ntaps);
    int i;

    if (!fp || !mem || !taps || need == 0) return -1;
    if (memlen < need) return -2;
    f = (struct ofdm_rxf *)(void *)p;
    p += al8(sizeof(struct ofdm_rxf));
    f->n = ntaps;
    f->mid = (ntaps - 1) / 2;
    f->h = (double *)(void *)p;
    p += al8((size_t)(f->mid + 1) * sizeof(double));
    f->z = (double *)(void *)p;
    for (i = 0; i <= f->mid; i++) f->h[i] = taps[i];     /* taps are symmetric */
    ofdm_rxf_reset(f);
    *fp = f;
    return 0;
}

void ofdm_rxf_reset(ofdm_rxf *f)
{
    int i;
    if (!f) return;
    for (i = 0; i < 2 * f->n; i++) f->z[i] = 0.0;
    f->pos = 0;
}

int ofdm_rxf_delay(const ofdm_rxf *f) { return f ? f->mid : 0; }

void ofdm_rxf_process(ofdm_rxf *f, const double *in, double *out, long n)
{
    long s;
    int k, N = f->n, mid = f->mid;
    for (s = 0; s < n; s++) {
        const double *w;
        double acc;
        double x = in[s];
        /* newest sample is written at pos and pos+N, so the N most recent
         * samples are always the contiguous run z[pos+1 .. pos+N] */
        f->z[f->pos] = x;
        f->z[f->pos + N] = x;
        w = f->z + f->pos + 1;                  /* oldest .. newest */
        acc = f->h[mid] * w[mid];
        for (k = 0; k < mid; k++)
            acc += f->h[k] * (w[k] + w[N - 1 - k]);
        out[s] = acc;
        if (++f->pos == N) f->pos = 0;
    }
}
