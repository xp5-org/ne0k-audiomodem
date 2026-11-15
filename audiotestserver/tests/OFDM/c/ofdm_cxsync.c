/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

#include "ofdm_cxsync.h"
#include "ofdm_fft.h"
#include <math.h>

#define CX_PI 3.14159265358979323846

struct ofdm_cxsync {
    ofdm_cxcfg x;
    int        L;        /* n_fft/2                                         */
    int        max_int;  /* integer search range, subcarriers               */
    ofdm_cfft  fft;
    double    *ref;      /* 2*SYM: SC reference, mean-removed, unit norm    */
    double    *refc;     /* 2*SYM: ref rotated by the candidate's CFO       */
    double    *ring;     /* SYM: metric history                             */
    double    *ce;       /* n_occupied: known CE values (real BPSK)         */
    int       *bins;     /* n_occupied: natural-order FFT index of each bin */
    ofdm_cpx  *X;        /* n_fft: CE window spectrum                       */
};

static size_t al8(size_t v) { return (v + 7u) & ~(size_t)7u; }

/* Must stay the term-for-term sum of the bumps ofdm_cxsync_init performs. */
size_t ofdm_cxsync_workspace(const ofdm_cxcfg *x)
{
    size_t fftw, SYM, N, O;
    if (!x || x->c.symbol_samples < 1 || x->c.n_occupied < 2) return 0;
    fftw = ofdm_cfft_workspace(x->c.n_fft);
    if (fftw == 0) return 0;
    SYM = (size_t)x->c.symbol_samples;
    N   = (size_t)x->c.n_fft;
    O   = (size_t)x->c.n_occupied;
    return al8(sizeof(struct ofdm_cxsync))
         + fftw
         + al8(2 * SYM * sizeof(double))      /* ref  */
         + al8(2 * SYM * sizeof(double))      /* refc */
         + al8(SYM * sizeof(double))          /* ring */
         + al8(O * sizeof(double))            /* ce   */
         + al8(O * sizeof(int))               /* bins */
         + al8(N * sizeof(ofdm_cpx));         /* X    */
}

int ofdm_cxsync_init(ofdm_cxsync **sp, void *mem, size_t memlen,
                     const ofdm_cxcfg *x, const double *sc_ref_iq,
                     int max_cfo_bins)
{
    struct ofdm_cxsync *s;
    unsigned char *p = (unsigned char *)mem;
    size_t need, fftw;
    int    i, SYM, N, O, b;
    double mr = 0.0, mi = 0.0, nrm = 0.0;

    if (!sp || !mem || !x || !sc_ref_iq || max_cfo_bins < 0) return -1;
    need = ofdm_cxsync_workspace(x);
    if (need == 0) return -1;
    if (memlen < need) return -2;
    SYM  = x->c.symbol_samples;
    N    = x->c.n_fft;
    O    = x->c.n_occupied;
    fftw = ofdm_cfft_workspace(N);

    s = (struct ofdm_cxsync *)p; p += al8(sizeof(struct ofdm_cxsync));
    s->x       = *x;
    s->L       = N / 2;
    s->max_int = max_cfo_bins;
    if (ofdm_cfft_init(&s->fft, N, p, fftw) != 0) return -3;
    p += fftw;
    s->ref  = (double *)p;   p += al8((size_t)2 * SYM * sizeof(double));
    s->refc = (double *)p;   p += al8((size_t)2 * SYM * sizeof(double));
    s->ring = (double *)p;   p += al8((size_t)SYM * sizeof(double));
    s->ce   = (double *)p;   p += al8((size_t)O * sizeof(double));
    s->bins = (int *)p;      p += al8((size_t)O * sizeof(int));
    s->X    = (ofdm_cpx *)p;

    /* Mean-removed (an LO-leakage DC offset then correlates to zero) and unit
     * norm, so the refine score is an NCC in [0,1] by Cauchy-Schwarz. */
    for (i = 0; i < SYM; i++) { mr += sc_ref_iq[2 * i]; mi += sc_ref_iq[2 * i + 1]; }
    mr /= (double)SYM;
    mi /= (double)SYM;
    for (i = 0; i < SYM; i++) {
        s->ref[2 * i]     = sc_ref_iq[2 * i] - mr;
        s->ref[2 * i + 1] = sc_ref_iq[2 * i + 1] - mi;
        nrm += s->ref[2 * i] * s->ref[2 * i] + s->ref[2 * i + 1] * s->ref[2 * i + 1];
    }
    nrm = sqrt(nrm);
    if (nrm > 0.0)
        for (i = 0; i < 2 * SYM; i++) s->ref[i] /= nrm;
    for (i = 0; i < SYM; i++) s->ring[i] = 0.0;

    /* The CE is ofdm_cxmod's: OFDM_SEED_CE BPSK on every occupied bin. */
    ofdm_known_bpsk(s->ce, O, OFDM_SEED_CE, OFDM_REF_SYM_BOOST / sqrt(2.0));
    for (i = 0; i < O; i++) {
        b = ofdm_cx_bin(&s->x, i);
        s->bins[i] = b < 0 ? b + N : b;
    }

    *sp = s;
    return 0;
}

void ofdm_cx_derotate(double *iq, long n, double cfo, int n_fft, long t0)
{
    long   t;
    double ph, c, sn, re, im, w;

    if (!iq || n_fft < 1) return;
    w = -2.0 * CX_PI / (double)n_fft;
    for (t = 0; t < n; t++) {
        /* fmod keeps the argument small, so the phase stays exact over a long
         * stream instead of accumulating a recurrence's rounding. */
        ph = w * fmod(cfo * (double)(t + t0), (double)n_fft);
        c  = cos(ph);
        sn = sin(ph);
        re = iq[2 * t];
        im = iq[2 * t + 1];
        iq[2 * t]     = re * c - im * sn;
        iq[2 * t + 1] = re * sn + im * c;
    }
}

/* Half-symbol delay correlation P at position d (L terms). */
static void delay_corr(const double *r, int d, int L, double *pr, double *pi)
{
    int    k;
    double a = 0.0, b = 0.0;
    const double *u = r + 2L * d, *v = r + 2L * (d + L);
    for (k = 0; k < L; k++) {
        a += u[2 * k] * v[2 * k]     + u[2 * k + 1] * v[2 * k + 1];
        b += u[2 * k] * v[2 * k + 1] - u[2 * k + 1] * v[2 * k];
    }
    *pr = a;
    *pi = b;
}

/* D(q) for the CE spectrum in s->X: the mean of the UNIT phasors
 * conj(a_i) b_i / |a_i b_i| over neighbouring carrier pairs.  Phase-only, so
 * one strong bin -- LO leakage at DC, a CW carrier anywhere in the band, which
 * the SC metric also scores as 1 -- moves at most two of the n-1 terms
 * instead of dominating an energy-weighted sum. */
static double int_score(const struct ofdm_cxsync *s, int q)
{
    int    i, N = s->x.c.n_fft, O = s->x.c.n_occupied, ka, kb;
    double nr = 0.0, ni = 0.0, zr, zi, mag;
    double ar, ai, br, bi, ca, cb;
    for (i = 0; i + 1 < O; i++) {
        ka = (s->bins[i] + q) % N;     if (ka < 0) ka += N;
        kb = (s->bins[i + 1] + q) % N; if (kb < 0) kb += N;
        ca = s->ce[i];
        cb = s->ce[i + 1];
        ar = (double)s->X[ka].re * ca;  ai = (double)s->X[ka].im * ca;
        br = (double)s->X[kb].re * cb;  bi = (double)s->X[kb].im * cb;
        zr = ar * br + ai * bi;                    /* conj(a) * b */
        zi = ar * bi - ai * br;
        mag = sqrt(zr * zr + zi * zi);
        if (mag > 1e-300) { nr += zr / mag; ni += zi / mag; }
    }
    return sqrt(nr * nr + ni * ni) / (double)(O - 1);
}

int ofdm_cxsync_eval(ofdm_cxsync *s, const double *rx, int n, int d,
                     ofdm_cxsync_hit *h)
{
    const ofdm_cfg *c;
    int    SYM, N, CP, L, pf, ce, k, q, lo, hi, i, j, bi;
    double pr, pi, frac, cfo, w, ph, cs, sn, bs, ds, Fr, Fi;

    if (!s || !rx || !h || d < 0 || d >= n) return -1;
    c   = &s->x.c;
    SYM = c->symbol_samples;
    N   = c->n_fft;
    CP  = c->cp_len;
    L   = s->L;
    if (n < 2 * L) return -1;

    /* ---- 2. fractional CFO, mid-plateau ---- */
    pf = d + CP / 2;
    if (pf + 2 * L > n) pf = n - 2 * L;
    delay_corr(rx, pf, L, &pr, &pi);
    frac = (pr == 0.0 && pi == 0.0) ? 0.0 : atan2(pi, pr) / CX_PI;
    if (frac <= -1.0) frac = 1.0;                  /* (-1, 1] */

    /* ---- 3. even integer CFO from the CE, frac removed ---- */
    q  = 0;
    ds = -1.0;
    ce = d + SYM + CP / 2;
    if (ce + N <= n) {
        w = -2.0 * CX_PI / (double)N;
        for (k = 0; k < N; k++) {
            double re = rx[2 * (ce + k)], im = rx[2 * (ce + k) + 1];
            ph = w * fmod(frac * (double)(ce + k), (double)N);
            cs = cos(ph);
            sn = sin(ph);
            s->X[k].re = (ofdm_real)(re * cs - im * sn);
            s->X[k].im = (ofdm_real)(re * sn + im * cs);
        }
        ofdm_cfft_fwd(&s->fft, s->X);
        /* Outward scan 0, +2, -2, +4, -4, ... with a strict >, so a tie
         * keeps the smaller |q|. */
        for (k = 0; k <= s->max_int / 2; k++) {
            for (j = 0; j < (k == 0 ? 1 : 2); j++) {
                int    qq = (j == 0) ? 2 * k : -2 * k;
                double v  = int_score(s, qq);
                if (v > ds) { ds = v; q = qq; }
            }
        }
    }
    cfo = frac + (double)q;

    /* ---- 4. matched filter against the reference rotated by cfo ---- */
    w = 2.0 * CX_PI / (double)N;
    for (j = 0; j < SYM; j++) {
        ph = w * fmod(cfo * (double)j, (double)N);
        cs = cos(ph);
        sn = sin(ph);
        s->refc[2 * j]     = s->ref[2 * j] * cs - s->ref[2 * j + 1] * sn;
        s->refc[2 * j + 1] = s->ref[2 * j] * sn + s->ref[2 * j + 1] * cs;
    }
    /* Rotating the reference by an integer CFO undoes its zero mean, so the
     * window is mean-removed as well: a DC offset (LO leakage) then scores
     * zero whatever the estimate.  With m = mean(r), F = sum conj(refc):
     *     sum conj(refc)(r - m) = sum conj(refc) r - m F
     *     sum |r - m|^2         = sum |r|^2 - SYM |m|^2                     */
    Fr = Fi = 0.0;
    for (j = 0; j < SYM; j++) { Fr += s->refc[2 * j]; Fi -= s->refc[2 * j + 1]; }
    bi = d;
    bs = -1.0;
    if (n >= SYM) {
        lo = d - CP;
        if (lo < 0) lo = 0;
        hi = d + CP;
        if (hi > n - SYM) hi = n - SYM;
        for (i = lo; i <= hi; i++) {
            double nr = 0.0, ni = 0.0, eng = 0.0, sr = 0.0, si = 0.0, v;
            const double *r = rx + 2L * i;
            for (j = 0; j < SYM; j++) {
                double rr = r[2 * j], ri = r[2 * j + 1];
                double fr = s->refc[2 * j], fi = s->refc[2 * j + 1];
                nr  += fr * rr + fi * ri;          /* conj(ref) * r */
                ni  += fr * ri - fi * rr;
                eng += rr * rr + ri * ri;
                sr  += rr;
                si  += ri;
            }
            sr /= (double)SYM;
            si /= (double)SYM;
            nr -= sr * Fr - si * Fi;
            ni -= sr * Fi + si * Fr;
            eng -= (double)SYM * (sr * sr + si * si);
            if (eng < 0.0) eng = 0.0;
            v = sqrt(nr * nr + ni * ni) / (sqrt(eng) + 1e-12);
            if (v > bs) { bs = v; bi = i; }
        }
    }
    if (bs < 0.0) bs = 1.0;                        /* too short to refine */

    h->start     = bi;
    h->score     = bs;
    h->metric    = 0.0;
    h->cfo       = cfo;
    h->cfo_frac  = frac;
    h->cfo_int   = q;
    h->int_score = ds;
    return 0;
}

static double sc_metric_at(double pr, double pi, double R1, double R2)
{
    double den = R1 * R2;
    if (den <= 1e-18) return 0.0;
    return (pr * pr + pi * pi) / (den + 1e-24);
}

int ofdm_cxsync_find(ofdm_cxsync *s, const double *rx, int n, int n_body_syms,
                     double threshold, double ncc_min,
                     ofdm_cxsync_hit *hits, int max_hits)
{
    int    SYM, L, md, d, cnt = 0, min_gap, i, k;
    int    last, in_region, region_d0, region_end, ref_pos, inside;
    double Pr = 0.0, Pi = 0.0, R1 = 0.0, R2 = 0.0, region_max = 0.0;
    double last_sc = 0.0, last_rmax = 0.0, gate;
    const double *a, *b, *e;
    ofdm_cxsync_hit h;

    if (!s || !rx || !hits || max_hits < 1 || n_body_syms < 1) return -1;
    SYM = s->x.c.symbol_samples;
    L   = s->L;
    if (n < 2 * L + 1) return 0;
    md = n - 2 * L;

    min_gap   = ofdm_frame_min_gap(&s->x.c, n_body_syms);
    last      = -(min_gap * 2);
    in_region = 0;
    region_d0 = 0;
    region_end = 0;

    for (i = 0; i < L; i++) {
        a = rx + 2L * i;
        b = rx + 2L * (i + L);
        Pr += a[0] * b[0] + a[1] * b[1];
        Pi += a[0] * b[1] - a[1] * b[0];
        R1 += a[0] * a[0] + a[1] * a[1];
        R2 += b[0] * b[0] + b[1] * b[1];
    }

    for (d = 0; d < md; d++) {
        double M = sc_metric_at(Pr, Pi, R1, R2);
        s->ring[d % SYM] = M;

        if (!in_region) {
            if (M > threshold) {
                in_region  = 1;
                region_d0  = d;
                region_end = d + SYM;
                if (region_end > md) region_end = md;
                region_max = M;
            }
        } else if (M > region_max) {
            region_max = M;
        }

        if (in_region && d + 1 >= region_end) {
            /* Leading edge of the plateau: first sample >= 95% of the max. */
            ref_pos = region_d0;
            gate = 0.95 * region_max;
            for (k = region_d0; k < region_end; k++)
                if (s->ring[k % SYM] >= gate) { ref_pos = k; break; }
            in_region = 0;

            /* Leading edge against leading edge.  The real path tests the
             * threshold edge (region_d0) against the previous ACCEPTED edge,
             * which sits earlier by the metric's ramp from threshold to 95%;
             * min_gap's cp_len allowance absorbs that only when frames are
             * separated by frame_gap_samples.  Back to back (encode gap=0)
             * the stride is exactly min_gap + cp_len, the ramp exceeds cp_len
             * and the next frame was taken for a second region of the last
             * one (replacing it: 22 of 30 hits).  Two 95% edges differ by the
             * stride +- noise. */
            inside = ref_pos - last < min_gap;
            if ((!inside || (cnt > 0 && region_max > last_rmax))
                && ofdm_cxsync_eval(s, rx, n, ref_pos, &h) == 0
                && h.score >= ncc_min
                && h.int_score >= OFDM_CXSYNC_INT_MIN) {
                h.metric = region_max;
                if (!inside) {
                    if (cnt < max_hits) hits[cnt] = h;
                    cnt++;
                    last = ref_pos; last_sc = h.score; last_rmax = region_max;
                } else if (h.score >= OFDM_SC_REPLACE_RATIO * last_sc) {
                    if (cnt <= max_hits) hits[cnt - 1] = h;
                    last = ref_pos; last_sc = h.score; last_rmax = region_max;
                }
            }
        }

        if (d + 1 < md) {
            a = rx + 2L * d;              /* leaves the first half  */
            b = rx + 2L * (d + L);        /* moves first -> second  */
            e = rx + 2L * (d + 2 * L);    /* enters the second half */
            Pr -= a[0] * b[0] + a[1] * b[1];
            Pi -= a[0] * b[1] - a[1] * b[0];
            Pr += b[0] * e[0] + b[1] * e[1];
            Pi += b[0] * e[1] - b[1] * e[0];
            R1 += b[0] * b[0] + b[1] * b[1] - a[0] * a[0] - a[1] * a[1];
            R2 += e[0] * e[0] + e[1] * e[1] - b[0] * b[0] - b[1] * b[1];
        }
    }
    return cnt > max_hits ? max_hits : cnt;
}
