/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
#include "css_corr.h"
#include "css_pfft.h"
#include "css_win.h"
#include "css_util.h"
#include <math.h>

#define NSEG_MAX   6
#define NSLICE     8
#define MIN_SPS_FOR_SEGMENTS 96

/* _SLICE_SPEC_12 */
static const int SPEC12[NSLICE][2] = {
    {0, 4}, {4, 8}, {8, 12}, {0, 6}, {3, 9}, {6, 12}, {0, 8}, {4, 12}
};
/* sorted({0,12} | starts | ends) */
static const int EDGES12[] = {0, 3, 4, 6, 8, 9, 12};
#define NEDGES 7

typedef struct {
    double frac;
    int s0, sl, peak_offset;
    int a, b;               /* segment edge indices (segment path)            */
    css_rcpx *Fu, *Fd;      /* slice template spectra (legacy path)           */
    int valid;
} slice_t;

struct css_corr {
    int sps, buf, nfft, nh, sr, use_seg, nslice;
    cssr_fft fft;
    css_rcpx *F_up, *F_dn;              /* full templates (legacy path)        */
    css_rcpx *seg_Fu[NSEG_MAX], *seg_Fd[NSEG_MAX];
    slice_t   sl[NSLICE];
    css_rcpx *B, *P;                    /* buffer spectrum, product scratch    */
    css_real *tmp;                      /* nfft                                */
    css_real *Pu, *Pd;                  /* (S+1) x buf prefix sums              */
    css_real *up_env, *dn_env;          /* buf                                  */
    css_real *xin;                      /* nfft: zero-padded input              */
    double   *dsc;                      /* buf : reduction scratch              */
};

static int use_segments(int sps, int hann) { return !hann && sps >= MIN_SPS_FOR_SEGMENTS; }

static int count_legacy_slices(int sps)
{
    int k, n = 0;
    for (k = 0; k < NSLICE; k++) {
        int s0 = (SPEC12[k][0] * sps) / 12, s1 = (SPEC12[k][1] * sps) / 12;
        if (s1 - s0 >= 8) n++;
    }
    return n;
}

size_t css_corr_workspace(int sps, int buf_size, int hann)
{
    int nfft, nh;
    size_t cs, rs, w;
    if (sps < 1 || buf_size < sps) return 0;
    nfft = css_pow2_above((long)buf_size + sps - 1);
    nh = nfft / 2 + 1;
    cs = css_al8((size_t)nh * sizeof(css_rcpx));
    rs = css_al8((size_t)nfft * sizeof(css_real));
    w  = css_al8(sizeof(struct css_corr)) + cssr_fft_workspace(nfft);
    w += 2 * cs;                         /* B, P  */
    w += 2 * rs;                         /* tmp, xin */
    w += 2 * css_al8((size_t)buf_size * sizeof(css_real));   /* envs */
    w += css_al8((size_t)buf_size * sizeof(double));         /* dsc  */
    if (use_segments(sps, hann)) {
        w += 2 * NSEG_MAX * cs;
        w += 2 * css_al8((size_t)(NSEG_MAX + 1) * buf_size * sizeof(css_real));
    } else {
        w += 2 * cs;                     /* F_up, F_dn */
        w += 2 * (size_t)count_legacy_slices(sps) * cs;
    }
    return w;
}

/* spectrum of a length-sps template, reversed, at offset 0 of an nfft frame,
 * optionally masked to [lo, hi) in the reversed coordinates. */
static void tpl_spectrum(css_corr *c, const css_real *rev, int len, int lo,
                         int hi, css_rcpx *out)
{
    int i;
    for (i = 0; i < c->nfft; i++) c->xin[i] = 0;
    for (i = lo; i < hi && i < len; i++) c->xin[i] = rev[i];
    cssr_rfft(&c->fft, c->xin, out);
}

int css_corr_init(css_corr **cp, void *mem, size_t memlen,
                  const float *up_ref, const float *dn_ref, int sps,
                  double search_frac, int buf_size, int hann)
{
    unsigned char *p = (unsigned char *)mem;
    struct css_corr *c;
    size_t need = css_corr_workspace(sps, buf_size, hann), cs, rs;
    int nfft, nh, i, k;
    if (!cp || !mem || !up_ref || !dn_ref || need == 0) return -1;
    if (memlen < need) return -2;
    nfft = css_pow2_above((long)buf_size + sps - 1);
    nh = nfft / 2 + 1;
    cs = css_al8((size_t)nh * sizeof(css_rcpx));
    rs = css_al8((size_t)nfft * sizeof(css_real));

    c = (struct css_corr *)p; p += css_al8(sizeof(struct css_corr));
    c->sps = sps; c->buf = buf_size; c->nfft = nfft; c->nh = nh;
    c->sr = (int)(sps * search_frac);
    if (c->sr < 16) c->sr = 16;
    c->use_seg = use_segments(sps, hann);
    if (cssr_fft_init(&c->fft, nfft, p, cssr_fft_workspace(nfft)) != 0) return -1;
    p += cssr_fft_workspace(nfft);
    c->B = (css_rcpx *)p; p += cs;
    c->P = (css_rcpx *)p; p += cs;
    c->tmp = (css_real *)p; p += rs;
    c->xin = (css_real *)p; p += rs;
    c->up_env = (css_real *)p; p += css_al8((size_t)buf_size * sizeof(css_real));
    c->dn_env = (css_real *)p; p += css_al8((size_t)buf_size * sizeof(css_real));
    c->dsc = (double *)p; p += css_al8((size_t)buf_size * sizeof(double));

    /* reversed templates in working precision; tmp/P double as scratch */
    {
        css_real *urev = c->up_env, *drev = c->dn_env;   /* buf >= sps */
        if (hann) {
            for (i = 0; i < sps; i++) {
                double w = css_hanning(i, sps);
                urev[sps - 1 - i] = (css_real)((css_real)up_ref[i] * (css_real)w);
                drev[sps - 1 - i] = (css_real)((css_real)dn_ref[i] * (css_real)w);
            }
        } else {
            for (i = 0; i < sps; i++) {
                urev[sps - 1 - i] = (css_real)up_ref[i];
                drev[sps - 1 - i] = (css_real)dn_ref[i];
            }
        }
        if (c->use_seg) {
            int edge[NEDGES];
            for (k = 0; k < NEDGES; k++) edge[k] = (EDGES12[k] * sps) / 12;
            for (k = 0; k < NEDGES - 1; k++) {
                int a = edge[k], b = edge[k + 1];
                c->seg_Fu[k] = (css_rcpx *)p; p += cs;
                c->seg_Fd[k] = (css_rcpx *)p; p += cs;
                tpl_spectrum(c, urev, sps, sps - b, sps - a, c->seg_Fu[k]);
                tpl_spectrum(c, drev, sps, sps - b, sps - a, c->seg_Fd[k]);
            }
            c->Pu = (css_real *)p; p += css_al8((size_t)(NSEG_MAX + 1) * buf_size * sizeof(css_real));
            c->Pd = (css_real *)p; p += css_al8((size_t)(NSEG_MAX + 1) * buf_size * sizeof(css_real));
            c->nslice = NSLICE;
            for (k = 0; k < NSLICE; k++) {
                int ai = 0, bi = 0, s0, s1;
                for (i = 0; i < NEDGES; i++) {
                    if (EDGES12[i] == SPEC12[k][0]) ai = i;
                    if (EDGES12[i] == SPEC12[k][1]) bi = i;
                }
                s0 = edge[ai]; s1 = edge[bi];
                c->sl[k].frac = (double)(SPEC12[k][1] - SPEC12[k][0]) / 12.0;
                c->sl[k].s0 = s0; c->sl[k].sl = s1 - s0;
                c->sl[k].peak_offset = sps - s1;
                c->sl[k].a = ai; c->sl[k].b = bi;
                c->sl[k].Fu = c->sl[k].Fd = 0; c->sl[k].valid = 1;
            }
            c->F_up = c->F_dn = 0;
        } else {
            c->F_up = (css_rcpx *)p; p += cs;
            c->F_dn = (css_rcpx *)p; p += cs;
            tpl_spectrum(c, urev, sps, 0, sps, c->F_up);
            tpl_spectrum(c, drev, sps, 0, sps, c->F_dn);
            c->Pu = c->Pd = 0;
            c->nslice = 0;
            for (k = 0; k < NSLICE; k++) {
                int s0 = (SPEC12[k][0] * sps) / 12, s1 = (SPEC12[k][1] * sps) / 12;
                int sl = s1 - s0, j;
                slice_t *S;
                if (sl < 8) continue;
                S = &c->sl[c->nslice++];
                S->frac = (double)(SPEC12[k][1] - SPEC12[k][0]) / 12.0;
                S->s0 = s0; S->sl = sl; S->peak_offset = sps - s1;
                S->a = S->b = 0; S->valid = 1;
                S->Fu = (css_rcpx *)p; p += cs;
                S->Fd = (css_rcpx *)p; p += cs;
                /* (ref[s0:s1] * hanning(sl))[::-1] or ref[s0:s1][::-1] */
                for (i = 0; i < c->nfft; i++) c->xin[i] = 0;
                for (j = 0; j < sl; j++) {
                    css_real v = (css_real)up_ref[s0 + j];
                    if (hann) v = (css_real)(v * (css_real)css_hanning(j, sl));
                    c->xin[sl - 1 - j] = v;
                }
                cssr_rfft(&c->fft, c->xin, S->Fu);
                for (i = 0; i < c->nfft; i++) c->xin[i] = 0;
                for (j = 0; j < sl; j++) {
                    css_real v = (css_real)dn_ref[s0 + j];
                    if (hann) v = (css_real)(v * (css_real)css_hanning(j, sl));
                    c->xin[sl - 1 - j] = v;
                }
                cssr_rfft(&c->fft, c->xin, S->Fd);
            }
        }
    }
    *cp = c;
    return 0;
}

int css_corr_search_range(const css_corr *c) { return c->sr; }
int css_corr_nfft(const css_corr *c) { return c->nfft; }

/* irfft(B * F)[:n] into out */
static void mf_into(css_corr *c, const css_rcpx *F, css_real *out, int n)
{
    int k;
    for (k = 0; k < c->nh; k++) {
        css_rcpx a = c->B[k], b = F[k];
        c->P[k].re = a.re * b.re - a.im * b.im;
        c->P[k].im = a.re * b.im + a.im * b.re;
    }
    cssr_irfft(&c->fft, c->P, c->tmp);
    for (k = 0; k < n; k++) out[k] = c->tmp[k];
}

static css_real rabs(css_real v) { return v < 0 ? -v : v; }

/* max and first argmax of |v| over [lo, hi) */
static css_real absmax(const css_real *v, int lo, int hi, int *arg)
{
    int i, ai = lo;
    css_real m = rabs(v[lo]);
    for (i = lo + 1; i < hi; i++) {
        css_real a = rabs(v[i]);
        if (a > m) { m = a; ai = i; }
    }
    if (arg) *arg = ai;
    return m;
}

void css_corr_decide(css_corr *c, const float *buf, int n, int centre,
                     int sr_override, css_corr_out *o)
{
    int i, j, lo, hi, sr, S = NSEG_MAX;
    double floor_, up_pk, dn_pk, total, full_conf, full_wpk;
    double best_conf, best_wpk, used_frac;
    int full_peak, best_peak;
    css_real scale = 0;

    for (i = 0; i < c->nfft; i++) c->xin[i] = 0;
    for (i = 0; i < n; i++) {
        css_real v = (css_real)buf[i];
        c->xin[i] = v;
        if (rabs(v) > scale) scale = rabs(v);
    }
    cssr_rfft(&c->fft, c->xin, c->B);

    if (c->use_seg) {
        css_real *Pu = c->Pu, *Pd = c->Pd;
        for (i = 0; i < n; i++) { Pu[i] = 0; Pd[i] = 0; }
        for (j = 0; j < S; j++) {
            css_real *pu0 = Pu + (size_t)j * n, *pu1 = Pu + (size_t)(j + 1) * n;
            css_real *pd0 = Pd + (size_t)j * n, *pd1 = Pd + (size_t)(j + 1) * n;
            mf_into(c, c->seg_Fu[j], c->up_env, n);
            for (i = 0; i < n; i++) pu1[i] = pu0[i] + c->up_env[i];
            mf_into(c, c->seg_Fd[j], c->dn_env, n);
            for (i = 0; i < n; i++) pd1[i] = pd0[i] + c->dn_env[i];
        }
        for (i = 0; i < n; i++) {
            c->up_env[i] = rabs(Pu[(size_t)S * n + i]);
            c->dn_env[i] = rabs(Pd[(size_t)S * n + i]);
        }
    } else {
        mf_into(c, c->F_up, c->up_env, n);
        mf_into(c, c->F_dn, c->dn_env, n);
        for (i = 0; i < n; i++) {
            c->up_env[i] = rabs(c->up_env[i]);
            c->dn_env[i] = rabs(c->dn_env[i]);
        }
    }

    sr = sr_override >= 0 ? sr_override : c->sr;
    lo = centre - sr; if (lo < 0) lo = 0;
    hi = centre + sr; if (hi > n) hi = n;
    if (hi <= lo) {
        o->bit = 0; o->conf = 0.0; o->found_peak = centre;
        o->snr_db = -60.0; o->winner_pk = 0.0;
        return;
    }
    /* corr_noise_floor: eps(SIG) * max|win| * sps */
    floor_ = scale > 0 ? CSS_REAL_EPS * (double)scale * (double)c->sps : 0.0;
    {
        int au, ad;
        up_pk = (double)absmax(c->up_env, lo, hi, &au);
        dn_pk = (double)absmax(c->dn_env, lo, hi, &ad);
        total = up_pk + dn_pk;
        if (total < 1e-12 || (up_pk > dn_pk ? up_pk : dn_pk) <= floor_) {
            o->bit = 0; o->conf = 0.0; o->found_peak = centre;
            o->snr_db = -60.0; o->winner_pk = 0.0;
            return;
        }
        full_conf = (up_pk - dn_pk) / total;
        full_peak = up_pk >= dn_pk ? au : ad;
        full_wpk  = up_pk > dn_pk ? up_pk : dn_pk;
    }
    best_conf = full_conf; best_peak = full_peak; best_wpk = full_wpk;
    used_frac = 1.0;

    for (j = 0; j < (c->use_seg ? NSLICE : c->nslice); j++) {
        slice_t *s = &c->sl[j];
        int sl_lo = s->peak_offset > centre - sr ? s->peak_offset : centre - sr;
        int sl_hi = n < centre + sr ? n : centre + sr;
        double u_pk, d_pk, tot, sl_conf, s0_frac, bias, weighted;
        int au = 0, ad = 0, m = 0;
        css_real *uw = c->tmp, *dw;        /* reuse tmp as two windows */
        if (sl_hi <= sl_lo) continue;
        m = sl_hi - sl_lo;
        dw = c->tmp + m;
        if (c->use_seg) {
            const css_real *pa = c->Pu + (size_t)s->a * n, *pb = c->Pu + (size_t)s->b * n;
            const css_real *qa = c->Pd + (size_t)s->a * n, *qb = c->Pd + (size_t)s->b * n;
            for (i = 0; i < m; i++) {
                uw[i] = rabs(pb[sl_lo + i] - pa[sl_lo + i]);
                dw[i] = rabs(qb[sl_lo + i] - qa[sl_lo + i]);
            }
        } else {
            /* irfft(B*a)[:n][l0:l1], l = sl_* - peak_offset; P/xin scratch */
            int l0 = sl_lo - s->peak_offset, k;
            css_real *full = c->xin;     /* nfft scratch (input already FFT'd) */
            for (k = 0; k < c->nh; k++) {
                css_rcpx a = c->B[k], b = s->Fu[k];
                c->P[k].re = a.re * b.re - a.im * b.im;
                c->P[k].im = a.re * b.im + a.im * b.re;
            }
            cssr_irfft(&c->fft, c->P, full);
            for (i = 0; i < m; i++) uw[i] = rabs(full[l0 + i]);
            for (k = 0; k < c->nh; k++) {
                css_rcpx a = c->B[k], b = s->Fd[k];
                c->P[k].re = a.re * b.re - a.im * b.im;
                c->P[k].im = a.re * b.im + a.im * b.re;
            }
            cssr_irfft(&c->fft, c->P, full);
            for (i = 0; i < m; i++) dw[i] = rabs(full[l0 + i]);
        }
        u_pk = (double)absmax(uw, 0, m, &au);
        d_pk = (double)absmax(dw, 0, m, &ad);
        tot = u_pk + d_pk;
        if (tot < 1e-12 || (u_pk > d_pk ? u_pk : d_pk) <= floor_) continue;
        sl_conf = (u_pk - d_pk) / tot;
        s0_frac = (s->sl + s->s0) > 0 ? (double)s->s0 / (double)(s->sl + s->s0) : 0.0;
        if      (s0_frac >= 0.60) bias = 1.35;
        else if (s0_frac >= 0.40) bias = 1.15;
        else if (s0_frac >= 0.25) bias = 1.00;
        else                      bias = 0.75;
        weighted = fabs(sl_conf) * pow(s->frac, 0.5) * bias;
        if (weighted > fabs(best_conf) * pow(used_frac, 0.5)
                && (u_pk > d_pk ? u_pk : d_pk) > full_wpk) {
            int conv = (u_pk >= d_pk ? au : ad) + sl_lo;
            if (!(conv >= 0 && conv < n)) continue;
            best_conf = sl_conf;
            best_peak = conv;
            best_wpk  = u_pk > d_pk ? u_pk : d_pk;
            used_frac = s->frac;
        }
    }

    o->conf = best_conf;
    o->bit = best_conf > 0 ? 1 : 0;
    o->found_peak = best_peak;
    o->winner_pk = best_wpk;
    {
        const css_real *env = best_conf >= 0 ? c->up_env : c->dn_env;
        int m0 = best_peak - c->sps / 4, m1 = best_peak + c->sps / 4, cnt = 0;
        double acc = 0.0, noise;
        if (m0 < 0) m0 = 0;
        if (m1 > n) m1 = n;
        /* np.sqrt(np.mean(winner_env[mask] ** 2)): square in SIG, then a
         * pairwise mean over the compacted masked copy */
        for (i = 0; i < n; i++) {
            css_real e;
            if (i >= m0 && i < m1) continue;
            e = env[i] * env[i];
            c->dsc[cnt++] = (double)e;
        }
        (void)acc;
        noise = cnt ? sqrt(css_pmean(c->dsc, (size_t)cnt)) : 1e-9;
        o->snr_db = 20.0 * log10(best_wpk / (noise + 1e-12) + 1e-12);
    }
}
