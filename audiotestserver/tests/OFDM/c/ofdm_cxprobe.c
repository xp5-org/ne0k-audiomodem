/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

#include "ofdm_cxprobe.h"
#include "ofdm_cxacq.h"
#include "ofdm_cxsync.h"
#include "ofdm_fft.h"
#include <math.h>

#define CX_PI 3.14159265358979323846
/* As ofdm_cxmod.c: the data-frame SC gain. */
#define CXP_SC_TX_GAIN 2.0
#define CXP_MAX_HITS   4

struct ofdm_cxprobe {
    ofdm_cxcfg   x;
    int          N, CP, SS, T, O, w0, np, nm, step, metric_cap, max_cap;
    ofdm_cxsym  *sym;
    ofdm_cfft    fft;
    ofdm_cxacq  *acq;
    ofdm_cxsync *sync;
    ofdm_cpx    *X;                 /* N                                   */
    int         *fidx, *sbin, *mir; /* O: FFT index, signed bin, mirror i  */
    int         *strong;            /* OFDM_COH_CARRIERS                   */
    double      *pk, *ce, *mk;      /* O: probe, CE, measurement patterns  */
    double      *occ_re, *occ_im;   /* O                                   */
    double      *sc;                /* 2 * SS, unscaled SC preamble        */
    double      *symbuf;            /* 2 * (SS + 2T)                       */
    double      *yr, *yi;           /* O: one window's bins / sqrt(N)      */
    double      *rr, *ri;           /* O: reference (walk; then CE's H)    */
    double      *mre, *mim, *m2re, *m2im;          /* O: Welford           */
    double      *s11, *s12, *s22, *b1r, *b1i, *b2r, *b2i, *yy;  /* O: LS   */
    double      *pdb, *srt;         /* O                                   */
    double      *metric;            /* metric_cap                          */
};

static size_t al8(size_t v) { return (v + 7u) & ~(size_t)7u; }

static int step_for(const ofdm_cfg *c)
{
    int a = c->symbol_samples, b = c->cp_len / 2, g;
    if (b < 8) b = 8;
    while (b) { g = a % b; a = b; b = g; }
    return a < 1 ? 1 : a;
}

static int n_cand_for(const ofdm_cfg *c, int step, int max_cap)
{
    int n_pos = (max_cap - c->cp_len - c->n_fft) / step + 1;
    int span  = (c->symbol_samples / step) * (OFDM_COH_SCAN_SYMS - 1);
    return n_pos - span < 1 ? 1 : n_pos - span;
}

#define CXP_ROWS 23   /* O-long double rows carved below */

/* Term-for-term the carving in ofdm_cxprobe_init.  Edit the two together. */
size_t ofdm_cxprobe_workspace(const ofdm_cxcfg *x, int n_probe_syms,
                              int n_meas_syms, int max_capture_samples,
                              int max_cfo_bins)
{
    const ofdm_cfg *c;
    size_t symw, fftw, acqw, syncw;
    int    step, O;

    if (!x || n_probe_syms < OFDM_COH_SCAN_SYMS || n_meas_syms < 0
        || max_cfo_bins < 0) return 0;
    c = &x->c;
    if (c->n_occupied < 2 || max_capture_samples < c->symbol_samples) return 0;
    symw  = ofdm_cxsym_workspace(x);
    fftw  = ofdm_cfft_workspace(c->n_fft);
    step  = step_for(c);
    acqw  = ofdm_cxacq_workspace(x, OFDM_COH_SCAN_SYMS, step,
                                 OFDM_COH_CARRIERS < c->n_occupied
                                 ? OFDM_COH_CARRIERS : c->n_occupied);
    syncw = ofdm_cxsync_workspace(x);
    if (!symw || !fftw || !acqw || !syncw) return 0;
    O = c->n_occupied;
    return al8(sizeof(struct ofdm_cxprobe))
         + al8(symw) + al8(fftw) + al8(acqw) + al8(syncw)
         + al8((size_t)c->n_fft * sizeof(ofdm_cpx))                     /* X      */
         + 3 * al8((size_t)O * sizeof(int))                             /* fidx sbin mir */
         + al8((size_t)OFDM_COH_CARRIERS * sizeof(int))                 /* strong */
         + al8((size_t)2 * c->symbol_samples * sizeof(double))          /* sc     */
         + al8((size_t)2 * (c->symbol_samples + 2 * c->taper_samples)
               * sizeof(double))                                        /* symbuf */
         + CXP_ROWS * al8((size_t)O * sizeof(double))
         + al8((size_t)n_cand_for(c, step, max_capture_samples)
               * sizeof(double));                                       /* metric */
}

int ofdm_cxprobe_init(ofdm_cxprobe **pp, void *mem, size_t memlen,
                      const ofdm_cxcfg *x, int n_probe_syms, int n_meas_syms,
                      int max_capture_samples, int max_cfo_bins)
{
    struct ofdm_cxprobe *p;
    const ofdm_cfg *c;
    unsigned char *q = (unsigned char *)mem;
    size_t need, symw, fftw, acqw, syncw, row;
    int    i, j, O, N, kk, n_sc, len;
    double **rows[CXP_ROWS];

    if (!pp || !mem || !x) return -1;
    need = ofdm_cxprobe_workspace(x, n_probe_syms, n_meas_syms,
                                  max_capture_samples, max_cfo_bins);
    if (need == 0) return -1;
    if (memlen < need) return -2;
    c = &x->c;
    O = c->n_occupied;
    N = c->n_fft;
    kk = OFDM_COH_CARRIERS < O ? OFDM_COH_CARRIERS : O;
    symw  = ofdm_cxsym_workspace(x);
    fftw  = ofdm_cfft_workspace(N);
    acqw  = ofdm_cxacq_workspace(x, OFDM_COH_SCAN_SYMS, step_for(c), kk);
    syncw = ofdm_cxsync_workspace(x);
    row   = al8((size_t)O * sizeof(double));

    p = (struct ofdm_cxprobe *)q; q += al8(sizeof(struct ofdm_cxprobe));
    p->x  = *x;
    p->N  = N; p->CP = c->cp_len; p->SS = c->symbol_samples;
    p->T  = c->taper_samples; p->O = O;
    p->w0 = (2 * p->T <= p->CP) ? p->T + p->CP / 2 : p->CP;   /* as ofdm_cxdem */
    p->np = n_probe_syms; p->nm = n_meas_syms;
    p->step = step_for(c);
    p->max_cap = max_capture_samples;
    p->metric_cap = n_cand_for(c, p->step, max_capture_samples);

    if (ofdm_cxsym_init(&p->sym, q, symw, x) != 0) return -3;
    q += al8(symw);
    if (ofdm_cfft_init(&p->fft, N, q, fftw) != 0) return -3;
    q += al8(fftw);
    if (ofdm_cxacq_init(&p->acq, q, acqw, x, 0, OFDM_COH_SCAN_SYMS, p->step, kk) != 0)
        return -3;
    q += al8(acqw);
    {
        unsigned char *sync_mem = q;
        q += al8(syncw);
        p->X      = (ofdm_cpx *)q; q += al8((size_t)N * sizeof(ofdm_cpx));
        p->fidx   = (int *)q;      q += al8((size_t)O * sizeof(int));
        p->sbin   = (int *)q;      q += al8((size_t)O * sizeof(int));
        p->mir    = (int *)q;      q += al8((size_t)O * sizeof(int));
        p->strong = (int *)q;      q += al8((size_t)OFDM_COH_CARRIERS * sizeof(int));
        p->sc     = (double *)q;   q += al8((size_t)2 * p->SS * sizeof(double));
        p->symbuf = (double *)q;   q += al8((size_t)2 * (p->SS + 2 * p->T)
                                            * sizeof(double));
        rows[0]  = &p->pk;   rows[1]  = &p->ce;   rows[2]  = &p->mk;
        rows[3]  = &p->occ_re; rows[4] = &p->occ_im;
        rows[5]  = &p->yr;   rows[6]  = &p->yi;   rows[7]  = &p->rr;
        rows[8]  = &p->ri;   rows[9]  = &p->mre;  rows[10] = &p->mim;
        rows[11] = &p->m2re; rows[12] = &p->m2im; rows[13] = &p->s11;
        rows[14] = &p->s12;  rows[15] = &p->s22;  rows[16] = &p->b1r;
        rows[17] = &p->b1i;  rows[18] = &p->b2r;  rows[19] = &p->b2i;
        rows[20] = &p->yy;   rows[21] = &p->pdb;  rows[22] = &p->srt;
        for (i = 0; i < CXP_ROWS; i++) { *rows[i] = (double *)q; q += row; }
        p->metric = (double *)q;

        for (i = 0; i < O; i++) {
            int b = ofdm_cx_bin(x, i);
            p->sbin[i] = b;
            p->fidx[i] = b < 0 ? b + N : b;
        }
        for (i = 0; i < O; i++) {         /* mirror: the occupied index of -bin */
            p->mir[i] = -1;
            for (j = 0; j < O; j++)
                if (p->sbin[j] == -p->sbin[i]) { p->mir[i] = j; break; }
        }
        ofdm_known_bpsk(p->pk, O, OFDM_SEED_PROBE, 1.0 / sqrt(2.0));
        /* the data frame's CE level, so SC + CE are ofdm_cxmod's */
        ofdm_known_bpsk(p->ce, O, OFDM_SEED_CE, OFDM_REF_SYM_BOOST / sqrt(2.0));

        /* SC preamble exactly as ofdm_cxmod_init: PN on the even signed bins. */
        n_sc = 0;
        for (i = 0; i < O; i++) if (p->sbin[i] % 2 == 0) n_sc++;
        if (n_sc < 1) return -1;
        ofdm_known_bpsk(p->occ_im, n_sc, OFDM_SEED_SC, 1.0 / sqrt(2.0));
        for (i = 0, j = 0; i < O; i++)
            p->occ_re[i] = (p->sbin[i] % 2 == 0) ? p->occ_im[j++] : 0.0;
        for (i = 0; i < O; i++) p->occ_im[i] = 0.0;
        len = ofdm_cxsym_make(p->sym, p->occ_re, p->occ_im, 0, p->symbuf);
        if (len != p->SS) return -1;
        for (i = 0; i < 2 * p->SS; i++) p->sc[i] = p->symbuf[i];

        if (ofdm_cxsync_init(&p->sync, sync_mem, syncw, x, p->sc, max_cfo_bins) != 0)
            return -3;
    }
    *pp = p;
    return 0;
}

long ofdm_cxprobe_build_samples(const ofdm_cxprobe *p)
{
    if (!p) return -1;
    return (long)p->SS * (2 + p->np + p->nm) + 2L * p->T;
}

static void ola(struct ofdm_cxprobe *p, double *body, int si)
{
    int  i, len = ofdm_cxsym_make(p->sym, p->occ_re, p->occ_im, 1, p->symbuf);
    long off = 2L * si * p->SS;
    for (i = 0; i < 2 * len; i++) body[off + i] += p->symbuf[i];
}

long ofdm_cxprobe_build(ofdm_cxprobe *p, double *out, long cap)
{
    long need, i;
    int  j, k, si = 0;
    double *body;
    unsigned ms = OFDM_SEED_MEAS;

    if (!p || !out) return -1;
    need = ofdm_cxprobe_build_samples(p);
    if (cap < need) return -2;
    for (i = 0; i < 2 * p->SS; i++) out[i] = p->sc[i] * CXP_SC_TX_GAIN;
    body = out + 2L * p->SS;
    for (i = 0; i < 2 * (need - p->SS); i++) body[i] = 0.0;
    for (k = 0; k < p->O; k++) { p->occ_re[k] = p->ce[k]; p->occ_im[k] = 0.0; }
    ola(p, body, si++);
    for (k = 0; k < p->O; k++) p->occ_re[k] = p->pk[k];
    for (j = 0; j < p->np; j++) ola(p, body, si++);
    for (j = 0; j < p->nm; j++) {
        ms = ofdm_known_bpsk_next(p->occ_re, p->O, ms, 1.0 / sqrt(2.0));
        ola(p, body, si++);
    }
    return need;
}

/* ------------------------------------------------------------------ helpers */

/* Window [a, a + N) -> yr/yi = occupied bins / sqrt(N).  With derot, the CFO
 * is removed with its phase referenced to s0, as ofdm_cxdem's win_fft. */
static void win(struct ofdm_cxprobe *p, const double *rx, long a, int derot,
                long s0, double cfo)
{
    int    t, i;
    double w = -2.0 * CX_PI / (double)p->N, sc = 1.0 / sqrt((double)p->N);
    for (t = 0; t < p->N; t++) {
        double re = rx[2 * (a + t)], im = rx[2 * (a + t) + 1];
        if (derot) {
            double ph = w * fmod(cfo * (double)(a + t - s0), (double)p->N);
            double cs = cos(ph), sn = sin(ph);
            p->X[t].re = (ofdm_real)(re * cs - im * sn);
            p->X[t].im = (ofdm_real)(re * sn + im * cs);
        } else {
            p->X[t].re = (ofdm_real)re;
            p->X[t].im = (ofdm_real)im;
        }
    }
    ofdm_cfft_fwd(&p->fft, p->X);
    for (i = 0; i < p->O; i++) {
        p->yr[i] = (double)p->X[p->fidx[i]].re * sc;
        p->yi[i] = (double)p->X[p->fidx[i]].im * sc;
    }
}

/* k-th smallest (0-based) of v[0..n), reordering v. */
static double kth_smallest(double *v, int n, int k)
{
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int    mid = lo + (hi - lo) / 2, i, st;
        double pv, t;
        if (v[mid] < v[lo]) { t = v[mid]; v[mid] = v[lo]; v[lo] = t; }
        if (v[hi]  < v[lo]) { t = v[hi];  v[hi]  = v[lo]; v[lo] = t; }
        if (v[mid] < v[hi]) { t = v[mid]; v[mid] = v[hi]; v[hi] = t; }
        pv = v[hi];
        st = lo;
        for (i = lo; i < hi; i++)
            if (v[i] < pv) { t = v[i]; v[i] = v[st]; v[st] = t; st++; }
        t = v[st]; v[st] = v[hi]; v[hi] = t;
        if (k == st) return v[st];
        if (k < st) hi = st - 1; else lo = st + 1;
    }
    return v[lo];
}

/* numpy's default (linear) percentile of v[0..n), reordering v. */
static double pctile(double *v, int n, double q)
{
    double pos, lo_v, hi_v;
    int    lo;
    if (n < 1) return 0.0;
    pos = q / 100.0 * (double)(n - 1);
    lo = (int)floor(pos);
    if (lo >= n - 1) return kth_smallest(v, n, n - 1);
    lo_v = kth_smallest(v, n, lo);
    hi_v = kth_smallest(v, n, lo + 1);
    return lo_v + (hi_v - lo_v) * (pos - (double)lo);
}

static void top_k_idx(const double *mag, int n, int k, int *out)
{
    int i, j, cnt = 0;
    for (i = 0; i < n && cnt < k; i++) out[cnt++] = i;
    for (; i < n; i++) {
        int worst = 0;
        for (j = 1; j < cnt; j++)
            if (mag[out[j]] < mag[out[worst]]) worst = j;
        if (mag[i] > mag[out[worst]]) out[worst] = i;
    }
}

/* Phase-only match of the row in yr/yi against the unit reference rr/ri on
 * the strong carriers; the common rotation (the CFO's, per symbol) drops out
 * in the modulus. */
static double match(struct ofdm_cxprobe *p, int kk)
{
    int    j;
    double mr = 0.0, mi = 0.0;
    for (j = 0; j < kk; j++) {
        int    s = p->strong[j];
        double a = sqrt(p->yr[s] * p->yr[s] + p->yi[s] * p->yi[s]) + 1e-30;
        double ur = p->yr[s] / a, ui = p->yi[s] / a;
        mr += ur * p->rr[s] + ui * p->ri[s];
        mi += ui * p->rr[s] - ur * p->ri[s];
    }
    return sqrt(mr * mr + mi * mi) / (double)kk;
}

/* Fit theta = a + b * bin to z = H conj(Href) over all carriers (H already
 * divided by the known pattern), in three stages, each unwrapping the next:
 *   1. slope from neighbours at bin gap 1: unambiguous (|b| < pi) but noisy --
 *      each difference carries two carriers' phase noise and the band edge is
 *      tens of bins away, so at 10 dB it alone leaves ~3 rad at the edges;
 *   2. slope from carriers CXP_LAG bins apart, unwrapped against stage 1:
 *      CXP_LAG times less noise, so the residual phase is well inside +-pi;
 *   3. CXP_REFINE weighted least-squares passes on that residual phase: the
 *      full lever arm, std(b) ~ phase noise / sqrt(sum bin^2).
 * Without stage 2, stage 3 was fed wrapped residuals on a few symbols and the
 * 10 dB measurement swung by 2 dB from run to run.  Pairs are chosen by BIN
 * gap, so the ones straddling the DC hole (one bin wider) drop out.  The
 * intercept is the slope-derotated sum.  Weights |z| ~ |H|^2. */
#define CXP_LAG    8
#define CXP_REFINE 2

static double wrap_pi(double v)
{
    while (v >  CX_PI) v -= 2.0 * CX_PI;
    while (v < -CX_PI) v += 2.0 * CX_PI;
    return v;
}

static void track(struct ofdm_cxprobe *p, const double *hr, const double *hi,
                  double *th_out)
{
    int    i, it;
    double accr = 0.0, acci = 0.0, a = 0.0, b = 0.0, sr = 0.0, si = 0.0;
    double pzr = 0.0, pzi = 0.0;
    for (i = 0; i < p->O; i++) {
        double zr = hr[i] * p->rr[i] + hi[i] * p->ri[i];
        double zi = hi[i] * p->rr[i] - hr[i] * p->ri[i];
        if (i > 0 && p->sbin[i] - p->sbin[i - 1] == 1) {
            accr += zr * pzr + zi * pzi;          /* z_i conj(z_{i-1}) */
            acci += zi * pzr - zr * pzi;
        }
        pzr = zr; pzi = zi;
    }
    if (accr != 0.0 || acci != 0.0) b = atan2(acci, accr);
    accr = acci = 0.0;
    for (i = CXP_LAG; i < p->O; i++) {
        int    k = i - CXP_LAG;
        double zr, zi, qr, qi;
        if (p->sbin[i] - p->sbin[k] != CXP_LAG) continue;
        zr = hr[i] * p->rr[i] + hi[i] * p->ri[i];
        zi = hi[i] * p->rr[i] - hr[i] * p->ri[i];
        qr = hr[k] * p->rr[k] + hi[k] * p->ri[k];
        qi = hi[k] * p->rr[k] - hr[k] * p->ri[k];
        accr += zr * qr + zi * qi;                /* z_i conj(z_k) */
        acci += zi * qr - zr * qi;
    }
    if (accr != 0.0 || acci != 0.0)
        b += wrap_pi(atan2(acci, accr) - CXP_LAG * b) / CXP_LAG;
    for (i = 0; i < p->O; i++) {
        double zr = hr[i] * p->rr[i] + hi[i] * p->ri[i];
        double zi = hi[i] * p->rr[i] - hr[i] * p->ri[i];
        double c = cos(b * p->sbin[i]), s = -sin(b * p->sbin[i]);
        sr += zr * c - zi * s;
        si += zr * s + zi * c;
    }
    if (sr != 0.0 || si != 0.0) a = atan2(si, sr);
    for (it = 0; it < CXP_REFINE; it++) {
        double sw = 0.0, sx = 0.0, sxx = 0.0, se = 0.0, sxe = 0.0, den;
        for (i = 0; i < p->O; i++) {
            double zr = hr[i] * p->rr[i] + hi[i] * p->ri[i];
            double zi = hi[i] * p->rr[i] - hr[i] * p->ri[i];
            double t = a + b * p->sbin[i], c = cos(t), s = -sin(t);
            double w = sqrt(zr * zr + zi * zi), xb = (double)p->sbin[i];
            double e = atan2(zr * s + zi * c, zr * c - zi * s);
            sw += w; sx += w * xb; sxx += w * xb * xb;
            se += w * e; sxe += w * xb * e;
        }
        den = sw * sxx - sx * sx;
        if (sw <= 0.0 || den <= 0.0) break;
        {
            double db = (sw * sxe - sx * se) / den;
            a += (se - db * sx) / sw;
            b += db;
        }
    }
    th_out[0] = a;
    th_out[1] = b;
}

static void rot(double *re, double *im, double th)
{
    double c = cos(th), s = sin(th), r = *re;
    *re = r * c - *im * s;
    *im = r * s + *im * c;
}

/* ---------------------------------------------------------------- estimate */

int ofdm_cxprobe_estimate(ofdm_cxprobe *p, const double *rx, int n,
                          double threshold,
                          double *H_re, double *H_im,
                          double *snr_lin, double *snr_db,
                          double *snr_db_periodic, double *image_db,
                          ofdm_cxprobe_info *info)
{
    ofdm_acq_info   ai;
    ofdm_cxsync_hit hits[CXP_MAX_HITS];
    int    i, j, k, O, SS, kk, bc, n_cand, best, lo_g, first, nh;
    int    cnt, mcnt, n_img;
    long   s0, b0, b1, pos;
    double gate, peak, cfo, th[2];
    double *pdb;
    unsigned ms = OFDM_SEED_MEAS;          /* measurement block: one PRBS stream */

    if (!p || !rx || !info || !H_re || !H_im || !snr_lin || !snr_db) return -1;
    O = p->O; SS = p->SS;
    kk = OFDM_COH_CARRIERS < O ? OFDM_COH_CARRIERS : O;
    pdb = snr_db_periodic ? snr_db_periodic : p->pdb;

    info->valid = 0; info->error = OFDM_PROBE_OK;
    info->coherence = 0.0; info->noise_floor = 0.0;
    info->acq_start = 0; info->frame_start = 0;
    info->cfo = 0.0; info->sc_score = 0.0; info->int_score = 0.0;
    info->n_syms = 0; info->n_meas_syms = 0;
    info->has_isi_penalty = 0; info->isi_penalty_db = 0.0;
    info->has_image = 0; info->image_db = OFDM_CXPROBE_NO_IMAGE_DB;

    gate = threshold > OFDM_COH_GATE ? threshold : OFDM_COH_GATE;
    if (n > p->max_cap
        || n < (OFDM_COH_SCAN_SYMS - 1) * SS + p->CP + p->N + SS) {
        info->error = OFDM_PROBE_TOO_SHORT;
        return 1;
    }

    /* -- 1. coherence scan */
    if (ofdm_cxacq_run(p->acq, rx, n, p->metric, p->metric_cap, &ai) != 0
        || ai.n_cand < 1) {
        info->error = OFDM_PROBE_TOO_SHORT;
        return 1;
    }
    n_cand = ai.n_cand;
    bc = 0;
    peak = p->metric[0];
    for (i = 1; i < n_cand; i++)
        if (p->metric[i] > peak) { peak = p->metric[i]; bc = i; }
    best = bc * p->step;
    info->acq_start = best;
    info->coherence = peak;
    info->noise_floor = pctile(p->metric, n_cand, 25.0);   /* metric is done */
    if (peak < gate) { info->error = OFDM_PROBE_SYNC_FAIL; return 1; }

    /* -- 2. periodic reference from the winner's symbols, each turned onto the
     * first before summing (they rotate by the CFO), then walk down the grid
     * while symbols match it. */
    win(p, rx, (long)best + p->CP, 0, 0, 0.0);
    for (i = 0; i < O; i++) { p->rr[i] = p->yr[i]; p->ri[i] = p->yi[i]; }
    for (j = 1; j < OFDM_COH_SCAN_SYMS; j++) {
        double cr = 0.0, ci = 0.0, ph;
        win(p, rx, (long)best + (long)j * SS + p->CP, 0, 0, 0.0);
        for (i = 0; i < O; i++) {           /* sum conj(ref) y */
            cr += p->rr[i] * p->yr[i] + p->ri[i] * p->yi[i];
            ci += p->rr[i] * p->yi[i] - p->ri[i] * p->yr[i];
        }
        ph = atan2(ci, cr);
        for (i = 0; i < O; i++) {
            double r = p->yr[i], m = p->yi[i];
            rot(&r, &m, -ph);
            p->rr[i] += r;
            p->ri[i] += m;
        }
    }
    for (i = 0; i < O; i++) p->srt[i] = sqrt(p->rr[i] * p->rr[i] + p->ri[i] * p->ri[i]);
    top_k_idx(p->srt, O, kk, p->strong);
    for (j = 0; j < kk; j++) {
        int s = p->strong[j];
        double m = p->srt[s] + 1e-30;
        p->rr[s] /= m;
        p->ri[s] /= m;
    }
    first = best % SS;
    lo_g = best;
    for (;;) {
        int g = lo_g - SS;
        if (g < first) break;
        win(p, rx, (long)g + p->CP, 0, 0, 0.0);
        if (match(p, kk) <= OFDM_COH_MATCH) break;
        lo_g = g;
    }

    /* -- 3. the SC preamble two symbols ahead of the periodic block: exact
     * start, CFO, and the CE-sequence check a tone cannot pass. */
    b0 = (long)lo_g - 4L * SS;
    if (b0 < 0) b0 = 0;
    b1 = (long)lo_g + SS;
    if (b1 > n) b1 = n;
    nh = ofdm_cxsync_find(p->sync, rx + 2 * b0, (int)(b1 - b0),
                          2 + p->np + p->nm, OFDM_SC_THRESHOLD, OFDM_SC_NCC_MIN,
                          hits, CXP_MAX_HITS);
    k = -1;
    for (i = 0; i < nh; i++) {
        long d = hits[i].start + b0 - ((long)lo_g - 2L * SS);
        if (d < -SS || d > SS) continue;
        if (k < 0 || hits[i].score > hits[k].score) k = i;
    }
    if (k < 0) { info->error = OFDM_CXPROBE_NO_SC; return 1; }
    s0  = hits[k].start + b0;
    cfo = hits[k].cfo;
    info->frame_start = (int)s0;
    info->cfo = cfo;
    info->sc_score = hits[k].score;
    info->int_score = hits[k].int_score;

    /* -- 4. phase reference: the CE's H */
    pos = s0 + SS;
    win(p, rx, pos + p->w0, 1, s0, cfo);
    for (i = 0; i < O; i++) {
        p->rr[i] = p->yr[i] / p->ce[i];
        p->ri[i] = p->yi[i] / p->ce[i];
    }

    /* -- 5. periodic block, Welford over tracked H */
    for (i = 0; i < O; i++) p->mre[i] = p->mim[i] = p->m2re[i] = p->m2im[i] = 0.0;
    cnt = 0;
    for (j = 0; j < p->np; j++) {
        pos = s0 + (long)(2 + j) * SS;
        if (pos + p->w0 + p->N > n) break;
        win(p, rx, pos + p->w0, 1, s0, cfo);
        for (i = 0; i < O; i++) { p->yr[i] /= p->pk[i]; p->yi[i] /= p->pk[i]; }
        track(p, p->yr, p->yi, th);
        cnt++;
        for (i = 0; i < O; i++) {
            double hr = p->yr[i], hi = p->yi[i], d;
            rot(&hr, &hi, -(th[0] + th[1] * p->sbin[i]));
            d = hr - p->mre[i]; p->mre[i] += d / cnt; p->m2re[i] += d * (hr - p->mre[i]);
            d = hi - p->mim[i]; p->mim[i] += d / cnt; p->m2im[i] += d * (hi - p->mim[i]);
        }
    }
    if (cnt < 2) { info->error = OFDM_PROBE_TOO_FEW_SYMS; return 1; }
    info->n_syms = cnt;
    for (i = 0; i < O; i++) {
        double nv = (p->m2re[i] + p->m2im[i]) / (double)(cnt - 1), s;
        if (nv < 1e-12) nv = 1e-12;
        s = 2.0 * (p->mre[i] * p->mre[i] + p->mim[i] * p->mim[i]) / nv;
        if (s < 1e-10) s = 1e-10;
        snr_lin[i] = s;
        pdb[i] = 10.0 * log10(s + 1e-10);
        snr_db[i] = pdb[i];
        H_re[i] = p->mre[i];
        H_im[i] = p->mim[i];
        if (image_db) image_db[i] = OFDM_CXPROBE_NO_IMAGE_DB;
    }

    /* -- 6. measurement block: G and I by least squares per carrier */
    for (i = 0; i < O; i++)
        p->s11[i] = p->s12[i] = p->s22[i] = p->b1r[i] = p->b1i[i]
                  = p->b2r[i] = p->b2i[i] = p->yy[i] = 0.0;
    mcnt = 0;
    for (j = 0; j < p->nm; j++) {
        pos = s0 + (long)(2 + p->np + j) * SS;
        if (pos + p->w0 + p->N > n) break;
        win(p, rx, pos + p->w0, 1, s0, cfo);
        ms = ofdm_known_bpsk_next(p->mk, O, ms, 1.0 / sqrt(2.0));
        for (i = 0; i < O; i++) {           /* H = y / x1 for the phase fit */
            p->occ_re[i] = p->yr[i] / p->mk[i];
            p->occ_im[i] = p->yi[i] / p->mk[i];
        }
        track(p, p->occ_re, p->occ_im, th);
        mcnt++;
        for (i = 0; i < O; i++) {
            double yr = p->yr[i], yi = p->yi[i];
            double x1 = p->mk[i], x2 = p->mir[i] >= 0 ? p->mk[p->mir[i]] : 0.0;
            rot(&yr, &yi, -(th[0] + th[1] * p->sbin[i]));
            p->s11[i] += x1 * x1; p->s12[i] += x1 * x2; p->s22[i] += x2 * x2;
            p->b1r[i] += x1 * yr; p->b1i[i] += x1 * yi;
            p->b2r[i] += x2 * yr; p->b2i[i] += x2 * yi;
            p->yy[i]  += yr * yr + yi * yi;
        }
    }
    if (mcnt >= 4) {
        double med_p, med_m;
        n_img = 0;
        for (i = 0; i < O; i++) {
            double det = p->s11[i] * p->s22[i] - p->s12[i] * p->s12[i];
            double gr, gi, ir = 0.0, ii = 0.0, rss, nv, g2, i2 = 0.0, s, g2u, i2u, den_s;
            int    dof;
            if (p->mir[i] >= 0 && mcnt >= 3 && det > 1e-3 * p->s11[i] * p->s22[i]) {
                gr = (p->s22[i] * p->b1r[i] - p->s12[i] * p->b2r[i]) / det;
                gi = (p->s22[i] * p->b1i[i] - p->s12[i] * p->b2i[i]) / det;
                ir = (p->s11[i] * p->b2r[i] - p->s12[i] * p->b1r[i]) / det;
                ii = (p->s11[i] * p->b2i[i] - p->s12[i] * p->b1i[i]) / det;
                rss = p->yy[i] - (gr * p->b1r[i] + gi * p->b1i[i]
                                + ir * p->b2r[i] + ii * p->b2i[i]);
                dof = mcnt - 2;
                i2 = ir * ir + ii * ii;
            } else {
                gr = p->b1r[i] / p->s11[i];
                gi = p->b1i[i] / p->s11[i];
                rss = p->yy[i] - (gr * p->b1r[i] + gi * p->b1i[i]);
                dof = mcnt - 1;
            }
            g2 = gr * gr + gi * gi;
            nv = rss / (double)dof;
            if (nv < 1e-12) nv = 1e-12;
            /* |G^|^2 and |I^|^2 each carry the fit's noise, nv * (A^-1)_kk on
             * average; take it out so the SINR is unbiased at short blocks
             * (it read ~0.4 dB low at 16 symbols).  image_db stays the raw
             * ratio: its floor is that noise, ~ -10 log10(SNR * M / 2). */
            if (i2 > 0.0) {
                g2u = g2 - nv * p->s22[i] / det;
                i2u = i2 - nv * p->s11[i] / det;
            } else {
                g2u = g2 - nv / p->s11[i];
                i2u = 0.0;
            }
            /* i2u is NOT clamped at 0: it is zero-mean when there is no
             * image, and clamping made the denominator biased high by
             * ~0.37 nv s11/det (0.4 dB at 8 measurement symbols).  The
             * denominator stays >= nv (1 - s11/det) > 0. */
            if (g2u < 0.0) g2u = 0.0;
            den_s = i2u + nv;
            if (den_s < 1e-12) den_s = 1e-12;
            s = g2u / den_s;
            if (s < 1e-10) s = 1e-10;
            snr_lin[i] = s;
            snr_db[i] = 10.0 * log10(s + 1e-10);
            if (p->mir[i] >= 0 && i2 > 0.0) {
                double idb = 10.0 * log10(i2 / (g2 + 1e-300) + 1e-30);
                if (image_db) image_db[i] = idb;
                p->srt[n_img++] = idb;
            }
        }
        if (n_img > 0) {
            info->has_image = 1;
            info->image_db = pctile(p->srt, n_img, 50.0);
        }
        for (i = 0; i < O; i++) p->srt[i] = pdb[i];
        med_p = pctile(p->srt, O, 50.0);
        for (i = 0; i < O; i++) p->srt[i] = snr_db[i];
        med_m = pctile(p->srt, O, 50.0);
        info->n_meas_syms = mcnt;
        info->has_isi_penalty = 1;
        info->isi_penalty_db = med_p - med_m;
    }
    info->valid = 1;
    return 0;
}
