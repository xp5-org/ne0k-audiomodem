/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
#include "css_retro.h"
#include "css_pfft.h"
#include "css_util.h"
#include <math.h>

#define NRS 8
static const double RS_FRAC[NRS][2] = {
    {0.33, 0.00}, {0.33, 0.33}, {0.33, 0.67},
    {0.50, 0.00}, {0.50, 0.25}, {0.50, 0.50},
    {0.67, 0.00}, {0.67, 0.33}
};

struct css_retro {
    int sps, fsmax, nfft_max, lnfft_max;
    long nmax;
    const float *up_ref, *dn_ref;     /* caller-owned, length >= sps          */
    cssr_fft big; int big_n; void *big_mem; size_t big_len;
    cssr_fft loc; int loc_n; void *loc_mem; size_t loc_len;
    css_real *x;                      /* nfft_max                             */
    css_rcpx *F, *T, *P;              /* nfft_max/2+1                         */
    css_real *up, *dn, *comb;         /* nmax                                 */
    css_real *lx, *lu, *ld;           /* lnfft_max                            */
    css_rcpx *LF, *LT, *LP;           /* lnfft_max/2+1                        */
    double *gp, *dk, *dp, *res;       /* grid peaks (nmax/sps+2), frame_symbols */
    unsigned char *mask, *nmask;
    double *pu, *pd, *pt, *tmp;       /* per-symbol pass-5 data               */
    long N; int sps_eff_d, M, fsym;   /* anchor state                         */
};

static long grid_cap(long nmax, int sps) { return nmax / (sps > 1 ? sps / 2 : 1) + 4; }

size_t css_retro_workspace(int sps, long n_max, int fsm)
{
    int nf, ln;
    long gc;
    size_t w;
    if (sps < 8 || n_max < sps || fsm < 1) return 0;
    nf = css_pow2_above(n_max + sps - 1);
    ln = css_pow2_above(2L * sps + 1);
    gc = grid_cap(n_max, sps);
    if (gc < fsm) gc = fsm;
    w = css_al8(sizeof(struct css_retro));
    w += cssr_fft_workspace(nf) + cssr_fft_workspace(ln);
    w += css_al8((size_t)nf * sizeof(css_real));
    w += 3 * css_al8((size_t)(nf / 2 + 1) * sizeof(css_rcpx));
    w += 3 * css_al8((size_t)n_max * sizeof(css_real));
    w += 3 * css_al8((size_t)ln * sizeof(css_real));
    w += 3 * css_al8((size_t)(ln / 2 + 1) * sizeof(css_rcpx));
    w += 4 * css_al8((size_t)gc * sizeof(double));
    w += 2 * css_al8((size_t)gc);
    w += 4 * css_al8((size_t)gc * sizeof(double));
    return w;
}

int css_retro_init(css_retro **rp, void *mem, size_t memlen,
                   const float *up_ref, const float *dn_ref, int sps,
                   long n_max, int fsm)
{
    unsigned char *p = (unsigned char *)mem;
    struct css_retro *r;
    int nf, ln;
    long gc;
    if (!rp || !mem || !up_ref || !dn_ref) return -1;
    if (css_retro_workspace(sps, n_max, fsm) == 0) return -1;
    if (memlen < css_retro_workspace(sps, n_max, fsm)) return -2;
    nf = css_pow2_above(n_max + sps - 1);
    ln = css_pow2_above(2L * sps + 1);
    gc = grid_cap(n_max, sps);
    if (gc < fsm) gc = fsm;
    r = (struct css_retro *)p; p += css_al8(sizeof(struct css_retro));
    r->sps = sps; r->nmax = n_max; r->fsmax = fsm;
    r->nfft_max = nf; r->lnfft_max = ln;
    r->up_ref = up_ref; r->dn_ref = dn_ref;
    r->big_mem = p; r->big_len = cssr_fft_workspace(nf); p += r->big_len; r->big_n = 0;
    r->loc_mem = p; r->loc_len = cssr_fft_workspace(ln); p += r->loc_len; r->loc_n = 0;
    r->x = (css_real *)p; p += css_al8((size_t)nf * sizeof(css_real));
    r->F = (css_rcpx *)p; p += css_al8((size_t)(nf / 2 + 1) * sizeof(css_rcpx));
    r->T = (css_rcpx *)p; p += css_al8((size_t)(nf / 2 + 1) * sizeof(css_rcpx));
    r->P = (css_rcpx *)p; p += css_al8((size_t)(nf / 2 + 1) * sizeof(css_rcpx));
    r->up = (css_real *)p; p += css_al8((size_t)n_max * sizeof(css_real));
    r->dn = (css_real *)p; p += css_al8((size_t)n_max * sizeof(css_real));
    r->comb = (css_real *)p; p += css_al8((size_t)n_max * sizeof(css_real));
    r->lx = (css_real *)p; p += css_al8((size_t)ln * sizeof(css_real));
    r->lu = (css_real *)p; p += css_al8((size_t)ln * sizeof(css_real));
    r->ld = (css_real *)p; p += css_al8((size_t)ln * sizeof(css_real));
    r->LF = (css_rcpx *)p; p += css_al8((size_t)(ln / 2 + 1) * sizeof(css_rcpx));
    r->LT = (css_rcpx *)p; p += css_al8((size_t)(ln / 2 + 1) * sizeof(css_rcpx));
    r->LP = (css_rcpx *)p; p += css_al8((size_t)(ln / 2 + 1) * sizeof(css_rcpx));
    r->gp = (double *)p; p += css_al8((size_t)gc * sizeof(double));
    r->dk = (double *)p; p += css_al8((size_t)gc * sizeof(double));
    r->dp = (double *)p; p += css_al8((size_t)gc * sizeof(double));
    r->res = (double *)p; p += css_al8((size_t)gc * sizeof(double));
    r->mask = p; p += css_al8((size_t)gc);
    r->nmask = p; p += css_al8((size_t)gc);
    r->pu = (double *)p; p += css_al8((size_t)gc * sizeof(double));
    r->pd = (double *)p; p += css_al8((size_t)gc * sizeof(double));
    r->pt = (double *)p; p += css_al8((size_t)gc * sizeof(double));
    r->tmp = (double *)p;
    r->M = 0; r->N = 0; r->fsym = 0; r->sps_eff_d = 0;
    *rp = r;
    return 0;
}

static css_real rabs(css_real v) { return v < 0 ? -v : v; }

/* up/dn/comb envelopes of audio[N] (one rfft, two irfft at nfft) */
static int envelopes(css_retro *r, const float *audio, long N)
{
    int nf = css_pow2_above(N + r->sps - 1), k, nh = nf / 2 + 1, pass;
    long i;
    if (nf > r->nfft_max) return -1;
    if (r->big_n != nf) {
        if (cssr_fft_init(&r->big, nf, r->big_mem, r->big_len)) return -1;
        r->big_n = nf;
    }
    for (i = 0; i < nf; i++) r->x[i] = 0;
    for (i = 0; i < N; i++) r->x[i] = (css_real)audio[i];
    cssr_rfft(&r->big, r->x, r->F);
    for (pass = 0; pass < 2; pass++) {
        const float *ref = pass ? r->dn_ref : r->up_ref;
        css_real *env = pass ? r->dn : r->up;
        for (i = 0; i < nf; i++) r->x[i] = 0;
        for (k = 0; k < r->sps; k++) r->x[k] = (css_real)ref[r->sps - 1 - k];
        cssr_rfft(&r->big, r->x, r->T);
        for (k = 0; k < nh; k++) {
            css_rcpx a = r->F[k], b = r->T[k];
            r->P[k].re = a.re * b.re - a.im * b.im;
            r->P[k].im = a.re * b.im + a.im * b.re;
        }
        cssr_irfft(&r->big, r->P, r->x);
        for (i = 0; i < N; i++) env[i] = rabs(r->x[i]);
    }
    for (i = 0; i < N; i++) r->comb[i] = r->up[i] >= r->dn[i] ? r->up[i] : r->dn[i];
    return 0;
}

/* fold comb[:usable] at period P, first argmax; -1 if usable < 2P */
static long fold_t0(css_retro *r, long N, int P)
{
    long usable = (N / P) * P, i;
    int j, best = 0;
    css_real *acc = r->x;          /* scratch, P <= nfft */
    if (usable < 2L * P) return -1;
    for (j = 0; j < P; j++) acc[j] = 0;
    for (i = 0; i < usable; i += P)
        for (j = 0; j < P; j++) acc[j] = acc[j] + r->comb[i + j];
    for (j = 1; j < P; j++) if (acc[j] > acc[best]) best = j;
    return best;
}

static long argmax_r(const css_real *v, long lo, long hi)
{
    long i, b = lo;
    for (i = lo + 1; i < hi; i++) if (v[i] > v[b]) b = i;
    return b;
}

static double maxd_r(const css_real *v, long lo, long hi)
{
    return (double)v[argmax_r(v, lo, hi)];
}

/* 6-iteration IRLS line fit over p[0..n); mask in/out. */
static void irls(css_retro *r, const double *p, int n, int search_r,
                 double *sps_fit, double *t0_fit, int *n_out)
{
    int it, k;
    for (k = 0; k < n; k++) r->mask[k] = 1;
    for (it = 0; it < 6; it++) {
        int m = 0, cnt_old = 0, cnt_new = 0;
        double km = 0, pm = 0, sxy = 0, sxx = 0;
        for (k = 0; k < n; k++) if (r->mask[k]) { r->dk[m] = k; r->dp[m] = p[k]; m++; }
        if (m < 4) break;
        km = css_pmean(r->dk, (size_t)m);
        pm = css_pmean(r->dp, (size_t)m);
        for (k = 0; k < m; k++) {
            sxy += (r->dk[k] - km) * (r->dp[k] - pm);
            sxx += (r->dk[k] - km) * (r->dk[k] - km);
        }
        *sps_fit = sxy / sxx;
        *t0_fit = pm - *sps_fit * km;
        for (k = 0; k < n; k++) {
            double res = fabs(p[k] - (*t0_fit + (double)k * *sps_fit));
            r->nmask[k] = res < (double)search_r;
            cnt_new += r->nmask[k];
            cnt_old += r->mask[k];
        }
        if (cnt_new == cnt_old) break;
        for (k = 0; k < n; k++) r->mask[k] = r->nmask[k];
    }
    *n_out = 0;
    for (k = 0; k < n; k++) if (!r->mask[k]) (*n_out)++;
}

int css_retro_decode(css_retro *r, const float *audio, long N,
                     int fsym, int sps_eff, double rate_est,
                     unsigned char *bits, double *confs, double *t0,
                     double *sps_fit_o, int *n_out)
{
    int sps = r->sps, k, P, search_r = sps / 6, half_w, nnz = 0;
    long coarse;
    double sps_fit, t0_fit, peak_ref;
    if (N < (long)sps * (fsym / 2)) return -1;
    if (N > r->nmax || fsym > r->fsmax) return -2;
    if (envelopes(r, audio, N)) return -2;
    P = sps_eff + (int)css_iround(rate_est);
    if (P < 1) P = 1;
    coarse = fold_t0(r, N, P);
    if (coarse < 0) return -1;

    for (k = 0; k < fsym; k++) {
        long ex = coarse + (long)k * P, lo = ex - search_r, hi = ex + search_r;
        long ul, dl;
        css_real uv, dv, tot, cf;
        if (lo < 0) lo = 0;
        if (hi > N) hi = N;
        if (hi <= lo) { r->gp[k] = (double)ex; bits[k] = 0; confs[k] = 0.0; continue; }
        ul = argmax_r(r->up, lo, hi); dl = argmax_r(r->dn, lo, hi);
        uv = r->up[ul]; dv = r->dn[dl];
        r->gp[k] = (double)(uv >= dv ? ul : dl);
        /* numpy scalars in SIG: (u - d) / (total + 1e-12) */
        tot = uv + dv;
        cf = (uv - dv) / (tot + (css_real)1e-12);
        bits[k] = cf > 0 ? 1 : 0;
        confs[k] = (double)cf;
    }
    sps_fit = (double)P; t0_fit = (double)coarse;
    irls(r, r->gp, fsym, search_r, &sps_fit, &t0_fit, n_out);

    half_w = search_r / 4; if (half_w < 4) half_w = 4;
    for (k = 0; k < fsym; k++) {
        long pk = (long)css_iround(t0_fit + (double)k * sps_fit);
        long lo = pk - half_w, hi = pk + half_w;
        if (lo < 0) lo = 0;
        if (hi > N) hi = N;
        r->res[k] = (double)pk;
        if (hi <= lo) { r->pu[k] = r->pd[k] = r->pt[k] = 0.0; continue; }
        r->pu[k] = maxd_r(r->up, lo, hi);
        r->pd[k] = maxd_r(r->dn, lo, hi);
        r->pt[k] = r->pu[k] + r->pd[k];
    }
    for (k = 0; k < fsym; k++) if (r->pt[k] > 1e-12) r->tmp[nnz++] = r->pt[k];
    peak_ref = nnz ? css_median_d(r->tmp, (size_t)nnz) : 1.0;

    for (k = 0; k < fsym; k++) {
        long pk = (long)r->res[k];
        double u = r->pu[k], d = r->pd[k], total = r->pt[k], conf, snr_w;
        if (total < 1e-12) continue;
        conf = (u - d) / (total + 1e-12);
        if (fabs(conf) < 0.35) {
            long loc_lo = pk - sps, loc_hi = pk + 1;
            int loc_n;
            if (loc_lo < 0) loc_lo = 0;
            if (loc_hi > N) loc_hi = N;
            loc_n = (int)(loc_hi - loc_lo);
            if (loc_n >= 8) {
                int lnf = css_pow2_above((long)loc_n + sps - 1), lh = lnf / 2 + 1, s, i;
                double best = conf;
                if (lnf > r->lnfft_max) goto skip_local;
                if (r->loc_n != lnf) {
                    if (cssr_fft_init(&r->loc, lnf, r->loc_mem, r->loc_len)) goto skip_local;
                    r->loc_n = lnf;
                }
                for (i = 0; i < lnf; i++) r->lx[i] = 0;
                for (i = 0; i < loc_n; i++) r->lx[i] = (css_real)audio[loc_lo + i];
                cssr_rfft(&r->loc, r->lx, r->LF);
                for (s = 0; s < NRS; s++) {
                    int sl = (int)(sps * RS_FRAC[s][0]), s0, poff, cen, slo, shi, pass;
                    double frac, upk = 0, dpk = 0, tt, sc, wgt;
                    if (sl < 8) sl = 8;
                    s0 = (int)(sps * RS_FRAC[s][1]);
                    if (s0 > sps - sl) s0 = sps - sl;
                    poff = sps - s0 - sl;
                    frac = (double)sl / (double)sps;
                    if (sl > loc_n) continue;
                    cen = (int)(pk - loc_lo) - poff;
                    slo = cen - half_w; if (slo < 0) slo = 0;
                    shi = cen + half_w; if (shi > loc_n) shi = loc_n;
                    if (shi <= slo) continue;
                    for (pass = 0; pass < 2; pass++) {
                        const float *ref = pass ? r->dn_ref : r->up_ref;
                        css_real *env = pass ? r->ld : r->lu;
                        for (i = 0; i < lnf; i++) r->lx[i] = 0;
                        for (i = 0; i < sl; i++) r->lx[i] = (css_real)ref[s0 + sl - 1 - i];
                        cssr_rfft(&r->loc, r->lx, r->LT);
                        for (i = 0; i < lh; i++) {
                            css_rcpx a = r->LF[i], b = r->LT[i];
                            r->LP[i].re = a.re * b.re - a.im * b.im;
                            r->LP[i].im = a.re * b.im + a.im * b.re;
                        }
                        cssr_irfft(&r->loc, r->LP, env);
                        for (i = 0; i < loc_n; i++) env[i] = rabs(env[i]);
                    }
                    upk = maxd_r(r->lu, slo, shi);
                    dpk = maxd_r(r->ld, slo, shi);
                    tt = upk + dpk;
                    if (tt < 1e-12) continue;
                    sc = (upk - dpk) / (tt + 1e-12);
                    wgt = fabs(sc) * pow(frac, 0.5) * (tt / (peak_ref + 1e-12));
                    if (wgt > fabs(best) * 1.0) { best = sc; total = tt; }
                }
                if (fabs(best) > fabs(conf)) conf = best;
            }
        }
skip_local:
        snr_w = total / (peak_ref + 1e-12);
        if (snr_w < 0.0) snr_w = 0.0;
        if (snr_w > 1.0) snr_w = 1.0;
        bits[k] = conf > 0 ? 1 : 0;
        confs[k] = conf * snr_w;
    }
    *t0 = t0_fit; *sps_fit_o = sps_fit;
    return 0;
}

int css_retro_anchor_prepare(css_retro *r, const float *audio, long N,
                             int sps_eff, double rate_est, int fsym)
{
    int sps = r->sps, P, search_r = sps / 6;
    long coarse, M, m;
    r->M = 0;
    if (N > r->nmax || fsym > r->fsmax) return -1;
    if (envelopes(r, audio, N)) return -1;
    P = sps_eff + (int)css_iround(rate_est);
    if (P < 1) P = 1;
    coarse = fold_t0(r, N, P);
    if (coarse < 0) return -1;
    /* slots whose MF peak lies in the buffer (coarse + m*P <= N-1), not (N - coarse) / P: the anchor history
     * ends ring_lookahead past the frame's last peak, so that dropped the frame's last symbol and shift 0 never
     * held the aligned frame (css_retro.py retro_offset_candidates, 2026-11-08) */
    M = N > coarse ? (N - 1 - coarse) / P + 1 : 0;
    if (M < fsym) return -1;
    if (M > grid_cap(r->nmax, sps)) return -1;
    for (m = 0; m < M; m++) {
        long ex = coarse + m * P, lo = ex - search_r, hi = ex + search_r, ul, dl;
        if (lo < 0) lo = 0;
        if (hi > N) hi = N;
        if (hi <= lo) { r->gp[m] = (double)ex; continue; }
        ul = argmax_r(r->up, lo, hi); dl = argmax_r(r->dn, lo, hi);
        r->gp[m] = (double)(r->up[ul] >= r->dn[dl] ? ul : dl);
    }
    r->N = N; r->sps_eff_d = P; r->M = (int)M; r->fsym = fsym;
    return (int)M;
}

int css_retro_anchor_candidate(css_retro *r, int shift, unsigned char *bits,
                               double *confs, double *t0, double *sps_fit_o,
                               int *n_out)
{
    int sps = r->sps, search_r = sps / 6, half_w, k, fsym = r->fsym, nnz = 0;
    long m0 = (long)(r->M - fsym) + shift, N = r->N;
    double sps_fit, t0_fit, peak_ref;
    const double *raw;
    if (r->M <= 0 || m0 < 0 || m0 + fsym > r->M) return -1;
    raw = r->gp + m0;
    sps_fit = (double)r->sps_eff_d; t0_fit = raw[0];
    irls(r, raw, fsym, search_r, &sps_fit, &t0_fit, n_out);
    half_w = search_r / 4; if (half_w < 4) half_w = 4;
    for (k = 0; k < fsym; k++) {
        long pk = (long)css_iround(t0_fit + (double)k * sps_fit);
        long lo = pk - half_w, hi = pk + half_w;
        if (lo < 0) lo = 0;
        if (hi > N) hi = N;
        if (hi <= lo) { r->pu[k] = r->pd[k] = r->pt[k] = 0.0; continue; }
        r->pu[k] = maxd_r(r->up, lo, hi);
        r->pd[k] = maxd_r(r->dn, lo, hi);
        r->pt[k] = r->pu[k] + r->pd[k];
    }
    for (k = 0; k < fsym; k++) if (r->pt[k] > 1e-12) r->tmp[nnz++] = r->pt[k];
    peak_ref = nnz ? css_median_d(r->tmp, (size_t)nnz) : 1.0;
    for (k = 0; k < fsym; k++) {
        double u = r->pu[k], d = r->pd[k], tot = r->pt[k], c, w;
        if (tot < 1e-12) { bits[k] = 0; confs[k] = 0.0; continue; }
        c = (u - d) / (tot + 1e-12);
        w = tot / (peak_ref + 1e-12);
        if (w < 0) w = 0;
        if (w > 1) w = 1;
        bits[k] = c > 0 ? 1 : 0;
        confs[k] = c * w;
    }
    *t0 = t0_fit; *sps_fit_o = sps_fit;
    return 0;
}
