/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

#include "ofdm_cxacq.h"
#include "ofdm_fft.h"
#include <math.h>

/* float32 grid (the memory), double sums (see ofdm_cxacq.h). */
typedef struct { float re, im; } cxacq_cpx;

/* den is a sum of products of squared magnitudes; the guard only matters for
 * a dead carrier, where it turns 0/0 into 0. */
#define CXACQ_DEN_EPS 1e-300

struct ofdm_cxacq {
    int N, CP, SS, O, nsyms, step, top_k;
    int stride;      /* SS / step                 */
    int span;        /* stride * (nsyms - 1)      */
    int ring_rows;   /* span + 1                  */
    ofdm_cfft  fft;
    int       *fidx;     /* O: natural-order FFT index of occupied i */
    float     *q;        /* nsyms                                    */
    ofdm_cpx  *X;        /* N scratch                                */
    cxacq_cpx *ring;     /* ring_rows * O                            */
    double    *accr, *acci, *den, *mag, *ck, *sel;   /* O each       */
};

static size_t al8(size_t v) { return (v + 7u) & ~(size_t)7u; }

static int plan_ok(const ofdm_cxcfg *x, int nsyms, int step, int top_k)
{
    const ofdm_cfg *c;
    if (!x) return 0;
    c = &x->c;
    if (c->n_fft < 4 || (c->n_fft & (c->n_fft - 1)) != 0) return 0;
    if (c->cp_len < 0 || c->n_occupied < 1 || c->n_occupied > c->n_fft) return 0;
    if (c->symbol_samples != c->n_fft + c->cp_len) return 0;
    if (nsyms < 2 || step < 1 || top_k < 1 || top_k > c->n_occupied) return 0;
    if (c->symbol_samples % step != 0) return 0;
    return 1;
}

/* Term-for-term the carving in ofdm_cxacq_init.  Edit the two together. */
size_t ofdm_cxacq_workspace(const ofdm_cxcfg *x, int nsyms, int step, int top_k)
{
    size_t fftw;
    int    O, rows;
    if (!plan_ok(x, nsyms, step, top_k)) return 0;
    fftw = ofdm_cfft_workspace(x->c.n_fft);
    if (fftw == 0) return 0;
    O    = x->c.n_occupied;
    rows = (x->c.symbol_samples / step) * (nsyms - 1) + 1;
    return al8(sizeof(struct ofdm_cxacq))
         + al8(fftw)
         + al8((size_t)O * sizeof(int))                    /* fidx */
         + al8((size_t)nsyms * sizeof(float))              /* q    */
         + al8((size_t)x->c.n_fft * sizeof(ofdm_cpx))      /* X    */
         + al8((size_t)rows * O * sizeof(cxacq_cpx))       /* ring */
         + 6 * al8((size_t)O * sizeof(double));            /* accr acci den mag ck sel */
}

int ofdm_cxacq_init(ofdm_cxacq **ap, void *mem, size_t memlen,
                    const ofdm_cxcfg *x, const double *q,
                    int nsyms, int step, int top_k)
{
    struct ofdm_cxacq *a;
    unsigned char *p = (unsigned char *)mem;
    size_t need, fftw, frow;
    int    i, O, N;

    if (!ap || !mem || !x) return -1;
    if (!plan_ok(x, nsyms, step, top_k)) {
        if (step >= 1 && x->c.symbol_samples > 0 && x->c.symbol_samples % step != 0)
            return -4;
        return -1;
    }
    need = ofdm_cxacq_workspace(x, nsyms, step, top_k);
    if (need == 0) return -1;
    if (memlen < need) return -2;
    fftw = ofdm_cfft_workspace(x->c.n_fft);
    O = x->c.n_occupied;
    N = x->c.n_fft;
    frow = al8((size_t)O * sizeof(double));

    a = (struct ofdm_cxacq *)p; p += al8(sizeof(struct ofdm_cxacq));
    if (ofdm_cfft_init(&a->fft, N, p, fftw) != 0) return -3;
    p += al8(fftw);
    a->N = N; a->CP = x->c.cp_len; a->SS = x->c.symbol_samples; a->O = O;
    a->nsyms = nsyms; a->step = step; a->top_k = top_k;
    a->stride    = a->SS / step;
    a->span      = a->stride * (nsyms - 1);
    a->ring_rows = a->span + 1;

    a->fidx = (int *)p;       p += al8((size_t)O * sizeof(int));
    a->q    = (float *)p;     p += al8((size_t)nsyms * sizeof(float));
    a->X    = (ofdm_cpx *)p;  p += al8((size_t)N * sizeof(ofdm_cpx));
    a->ring = (cxacq_cpx *)p; p += al8((size_t)a->ring_rows * O * sizeof(cxacq_cpx));
    a->accr = (double *)p;    p += frow;
    a->acci = (double *)p;    p += frow;
    a->den  = (double *)p;    p += frow;
    a->mag  = (double *)p;    p += frow;
    a->ck   = (double *)p;    p += frow;
    a->sel  = (double *)p;

    for (i = 0; i < O; i++) {
        int b = ofdm_cx_bin(x, i);
        a->fidx[i] = b < 0 ? b + N : b;
    }
    for (i = 0; i < nsyms; i++) a->q[i] = (float)(q ? q[i] : 1.0);
    *ap = a;
    return 0;
}

int ofdm_cxacq_ring_rows(const ofdm_cxacq *a) { return a ? a->ring_rows : 0; }

int ofdm_cxacq_n_cand(const ofdm_cxacq *a, int n)
{
    int n_pos;
    if (!a || n < a->CP + a->N) return 0;
    n_pos = (n - a->CP - a->N) / a->step + 1;
    return n_pos > a->span ? n_pos - a->span : 0;
}

/* One grid row: FFT the window at pos*step + CP, scale, store float32. */
static void row_at(struct ofdm_cxacq *a, const double *rx, int pos, cxacq_cpx *out)
{
    int i;
    ofdm_real scale = (ofdm_real)(1.0 / sqrt((double)a->N));
    const double *src = rx + 2 * ((long)pos * a->step + a->CP);
    for (i = 0; i < a->N; i++) {
        a->X[i].re = (ofdm_real)src[2 * i];
        a->X[i].im = (ofdm_real)src[2 * i + 1];
    }
    ofdm_cfft_fwd(&a->fft, a->X);
    for (i = 0; i < a->O; i++) {
        ofdm_cpx v = a->X[a->fidx[i]];
        out[i].re = (float)(v.re * scale);
        out[i].im = (float)(v.im * scale);
    }
}

/* k-th largest of v[0..n), destroying v: quickselect, as ofdm_acq. */
#define CXACQ_SWAP(x, y) do { double t_ = (x); (x) = (y); (y) = t_; } while (0)

static double kth_largest(double *v, int n, int k)
{
    int lo = 0, hi = n - 1, target = k - 1;
    while (lo < hi) {
        int   mid = lo + (hi - lo) / 2, i, store;
        double pv;
        if (v[mid] > v[lo]) CXACQ_SWAP(v[mid], v[lo]);
        if (v[hi]  > v[lo]) CXACQ_SWAP(v[hi],  v[lo]);
        if (v[mid] > v[hi]) CXACQ_SWAP(v[mid], v[hi]);
        pv = v[hi];
        store = lo;
        for (i = lo; i < hi; i++)
            if (v[i] > pv) { CXACQ_SWAP(v[i], v[store]); store++; }
        CXACQ_SWAP(v[store], v[hi]);
        if (target == store) return v[store];
        if (target <  store) hi = store - 1;
        else                 lo = store + 1;
    }
    return v[lo];
}

/* Mean of ck over the top_k carriers by mag; ties lowest index first, as
 * ofdm_acq's top_k_mean. */
static double top_k_mean(const double *mag, const double *ck, double *scratch,
                         int n, int k)
{
    int    i, ties;
    double thr, s = 0.0;
    for (i = 0; i < n; i++) scratch[i] = mag[i];
    thr = kth_largest(scratch, n, k);
    ties = k;
    for (i = 0; i < n; i++)
        if (mag[i] > thr) { s += ck[i]; ties--; }
    for (i = 0; i < n && ties > 0; i++)
        if (mag[i] == thr) { s += ck[i]; ties--; }
    return s / (double)k;
}

int ofdm_cxacq_run(ofdm_cxacq *a, const double *rx_iq, int n,
                   double *metric_out, int metric_cap, ofdm_acq_info *info)
{
    int n_pos, n_cand, p, c, i, j;

    if (!a || !rx_iq || !metric_out || !info) return -1;
    info->start_step = a->step;
    info->n_cand = 0;
    info->n_pos = n < a->CP + a->N ? 0 : (n - a->CP - a->N) / a->step + 1;
    n_cand = ofdm_cxacq_n_cand(a, n);
    if (n_cand < 1) return 0;
    if (metric_cap < n_cand) return -2;
    n_pos = info->n_pos;
    info->n_cand = n_cand;

    for (p = 0; p < n_pos; p++) {
        row_at(a, rx_iq, p, a->ring + (size_t)(p % a->ring_rows) * a->O);
        c = p - a->span;
        if (c < 0) continue;

        for (i = 0; i < a->O; i++)
            a->accr[i] = a->acci[i] = a->den[i] = a->mag[i] = 0.0;
        /* Sequential in j, as the numpy reference; products of float32 grid
         * values are exact in double. */
        for (j = 0; j < a->nsyms; j++) {
            const cxacq_cpx *r1 = a->ring
                + (size_t)((c + j * a->stride) % a->ring_rows) * a->O;
            const cxacq_cpx *r0 = j > 0 ? a->ring
                + (size_t)((c + (j - 1) * a->stride) % a->ring_rows) * a->O : 0;
            double qq = j > 0 ? (double)a->q[j - 1] * (double)a->q[j] : 0.0;
            for (i = 0; i < a->O; i++) {
                double ar = r1[i].re, ai = r1[i].im;
                double p1 = ar * ar + ai * ai;
                a->mag[i] += sqrt(p1);
                if (r0) {
                    double br = r0[i].re, bi = r0[i].im;
                    double p0 = br * br + bi * bi;
                    a->accr[i] += qq * (br * ar + bi * ai);   /* conj(r0) r1 */
                    a->acci[i] += qq * (br * ai - bi * ar);
                    a->den[i]  += p0 * p1;
                }
            }
        }
        for (i = 0; i < a->O; i++) {
            double e = a->accr[i] * a->accr[i] + a->acci[i] * a->acci[i];
            a->ck[i] = e / ((double)(a->nsyms - 1) * a->den[i] + CXACQ_DEN_EPS);
        }
        metric_out[c] = top_k_mean(a->mag, a->ck, a->sel, a->O, a->top_k);
    }
    return 0;
}

int ofdm_cxacq_row(ofdm_cxacq *a, const double *rx_iq, int n, int pos, float *out)
{
    int i, n_pos;
    cxacq_cpx *tmp;
    if (!a || !rx_iq || !out || pos < 0) return -1;
    if (n < a->CP + a->N) return -2;
    n_pos = (n - a->CP - a->N) / a->step + 1;
    if (pos >= n_pos) return -2;
    tmp = a->ring;                        /* row 0 is free scratch here */
    row_at(a, rx_iq, pos, tmp);
    for (i = 0; i < a->O; i++) {
        out[2 * i]     = tmp[i].re;
        out[2 * i + 1] = tmp[i].im;
    }
    return 0;
}
