/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
#include "css_mf.h"
#include "css_fft.h"
#include <math.h>
#include <float.h>

#define CSS_EPS_F32 1.1920929e-7f     /* FLT_EPSILON */

struct css_mf {
    int sps, track, decide, seg_M, seg_trim, keep, ntop, max_win, nfft, nbin;
    css_fft   fft;
    css_cpx  *Fseg;      /* 2 * seg_M * nbin — up segments then down segments */
    css_cpx  *Xin;       /* nbin  input spectrum                              */
    css_cpx  *Xmul;      /* nbin  product scratch                             */
    float    *tmp;       /* nfft  irfft output / init scratch                 */
    float    *tot_u, *tot_d;   /* max_win each — running per-lag totals       */
    float    *top_u, *top_d;   /* ntop * max_win — the ntop largest per lag   */
    float    *env;             /* max_win                                     */
    float    *msc;             /* max_win — median scratch                    */
    int      *seg_off;         /* seg_M                                       */
};

static size_t al8(size_t v) { return (v + 7u) & ~(size_t)7u; }
static int next_pow2(int v) { int p = 1; while (p < v) p <<= 1; return p; }

static void plan_sizes(int sps, int track, int seg_M, int max_win,
                       int *nfft, int *nbin, int *ntop, int *keep, int seg_trim)
{
    int seg_L = sps / seg_M;
    int max_L = sps - (seg_M - 1) * seg_L;      /* last segment is the longest */
    if (seg_L > max_L) max_L = seg_L;
    (void)track;
    *nfft = next_pow2(max_win + max_L - 1);
    *nbin = *nfft / 2 + 1;
    *keep = seg_M - (seg_trim > 0 ? seg_trim : 0);
    if (*keep < 1) *keep = 1;
    *ntop = seg_M - *keep;
}

size_t css_mf_workspace(int sps, int track_search, int seg_M, int seg_trim,
                        int max_win)
{
    int nfft, nbin, ntop, keep;
    if (seg_M < 2 || sps < 2 || max_win < 2) return 0;
    plan_sizes(sps, track_search, seg_M, max_win, &nfft, &nbin, &ntop, &keep,
               seg_trim);
    return al8(sizeof(struct css_mf))
         + al8(css_fft_workspace(nfft))
         + al8((size_t)2 * seg_M * nbin * sizeof(css_cpx))
         + al8((size_t)nbin * sizeof(css_cpx)) * 2
         + al8((size_t)nfft * sizeof(float))
         + al8((size_t)max_win * sizeof(float)) * 4          /* tot_u/d, env, msc */
         + al8((size_t)ntop * max_win * sizeof(float)) * 2   /* top_u, top_d      */
         + al8((size_t)seg_M * sizeof(int));
}

int css_mf_init(css_mf **out, void *mem, size_t memlen,
                const float *up_ref, const float *dn_ref,
                int sps, int track_search, int decide_search,
                int seg_M, int seg_trim, int max_win)
{
    unsigned char *p = (unsigned char *)mem;
    struct css_mf *m;
    int nfft, nbin, ntop, keep, seg_L, i, mi, ntop_max;

    if (!out || !mem || seg_M < 2 || sps < 2 || max_win < 2) return -1;
    if (memlen < css_mf_workspace(sps, track_search, seg_M, seg_trim, max_win))
        return -2;
    plan_sizes(sps, track_search, seg_M, max_win, &nfft, &nbin, &ntop, &keep, seg_trim);
    ntop_max = ntop;

    m = (struct css_mf *)p;                 p += al8(sizeof(struct css_mf));
    if (css_fft_init(&m->fft, nfft, p, memlen)) return -3;
    p += al8(css_fft_workspace(nfft));
    m->Fseg  = (css_cpx *)p; p += al8((size_t)2 * seg_M * nbin * sizeof(css_cpx));
    m->Xin   = (css_cpx *)p; p += al8((size_t)nbin * sizeof(css_cpx));
    m->Xmul  = (css_cpx *)p; p += al8((size_t)nbin * sizeof(css_cpx));
    m->tmp   = (float *)p;   p += al8((size_t)nfft * sizeof(float));
    m->tot_u = (float *)p;   p += al8((size_t)max_win * sizeof(float));
    m->tot_d = (float *)p;   p += al8((size_t)max_win * sizeof(float));
    m->env   = (float *)p;   p += al8((size_t)max_win * sizeof(float));
    m->msc   = (float *)p;   p += al8((size_t)max_win * sizeof(float));
    m->top_u = (float *)p;   p += al8((size_t)ntop_max * max_win * sizeof(float));
    m->top_d = (float *)p;   p += al8((size_t)ntop_max * max_win * sizeof(float));
    m->seg_off = (int *)p;

    m->sps = sps; m->track = track_search; m->decide = decide_search;
    m->seg_M = seg_M; m->seg_trim = seg_trim; m->keep = keep; m->ntop = ntop;
    m->max_win = max_win; m->nfft = nfft; m->nbin = nbin;

    /* Segment templates, compacted to offset 0: segment m of the reversed
     * template is rev[a:b], and rev[j] = fwd[sps-1-j], so
     * rev[a+t] = fwd[sps-1-a-t]. The lag shift `a` is restored in run(). */
    seg_L = sps / seg_M;
    for (mi = 0; mi < seg_M; mi++) {
        int a = mi * seg_L;
        int b = (mi == seg_M - 1) ? sps : (mi + 1) * seg_L;
        int L = b - a, t;
        m->seg_off[mi] = a;
        for (i = 0; i < nfft; i++) m->tmp[i] = 0.0f;
        for (t = 0; t < L; t++) m->tmp[t] = up_ref[sps - 1 - a - t];
        css_rfft(&m->fft, m->tmp, m->Fseg + (size_t)mi * nbin);
        for (i = 0; i < nfft; i++) m->tmp[i] = 0.0f;
        for (t = 0; t < L; t++) m->tmp[t] = dn_ref[sps - 1 - a - t];
        css_rfft(&m->fft, m->tmp, m->Fseg + (size_t)(seg_M + mi) * nbin);
    }
    *out = m;
    return 0;
}

int css_mf_nfft(const css_mf *m) { return m ? m->nfft : 0; }

/* Accumulate one segment envelope into (total, top-ntop) state for one
 * direction: a running total plus the ntop largest seen so far per lag, so
 * the trimmed sum is total - sum(top) at the end. The (M, n) stack is never
 * materialised; peak memory drops from M*n to (1+ntop)*n per direction. */
static void accum(float *tot, float *top, int ntop, const float *seg,
                  int a, int n, int max_win)
{
    int k;
    for (k = a; k < n; k++) {
        float v = seg[k - a];
        int j;
        tot[k] += v;
        if (ntop <= 0) continue;
        if (v <= top[k]) continue;                   /* top[0] = smallest kept */
        /* insert v, keeping top[0..ntop-1] ascending (ntop is 1..M-1, tiny) */
        for (j = 0; j < ntop - 1 && top[(size_t)(j + 1) * max_win + k] < v; j++)
            top[(size_t)j * max_win + k] = top[(size_t)(j + 1) * max_win + k];
        top[(size_t)j * max_win + k] = v;
    }
}

static float quickselect(float *a, int n, int k)
{
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        float pv = a[(lo + hi) >> 1];
        int i = lo, j = hi;
        while (i <= j) {
            while (a[i] < pv) i++;
            while (a[j] > pv) j--;
            if (i <= j) { float t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
        }
        if (k <= j) hi = j; else if (k >= i) lo = i; else break;
    }
    return a[k];
}

int css_mf_run(css_mf *m, const float *win, int n, int centre, css_mf_out *out)
{
    int nbin = m->nbin, nfft = m->nfft, ntop = m->ntop, i, mi, k;
    int lo, hi, dlo, dhi, pk, dp;
    float u, d, du, dd, scale, floor_, med, mx;

    if (!m || !win || !out || n < 2 || n > m->max_win) return -1;

    for (i = 0; i < nfft; i++) m->tmp[i] = (i < n) ? win[i] : 0.0f;
    css_rfft(&m->fft, m->tmp, m->Xin);

    for (k = 0; k < n; k++) { m->tot_u[k] = 0.0f; m->tot_d[k] = 0.0f; }
    for (i = 0; i < ntop; i++)
        for (k = 0; k < n; k++) {
            m->top_u[(size_t)i * m->max_win + k] = -1.0f;   /* envelopes are >= 0 */
            m->top_d[(size_t)i * m->max_win + k] = -1.0f;
        }

    for (mi = 0; mi < 2 * m->seg_M; mi++) {
        const css_cpx *F = m->Fseg + (size_t)mi * nbin;
        int a = m->seg_off[mi % m->seg_M];
        if (a >= n) continue;
        for (i = 0; i < nbin; i++) {
            float ar = m->Xin[i].re, ai = m->Xin[i].im;
            m->Xmul[i].re = ar * F[i].re - ai * F[i].im;
            m->Xmul[i].im = ar * F[i].im + ai * F[i].re;
        }
        css_irfft(&m->fft, m->Xmul, m->tmp);
        for (i = 0; i < n - a; i++) m->tmp[i] = css_fabsf(m->tmp[i]);
        if (mi < m->seg_M) accum(m->tot_u, m->top_u, ntop, m->tmp, a, n, m->max_win);
        else               accum(m->tot_d, m->top_d, ntop, m->tmp, a, n, m->max_win);
    }

    for (i = 0; i < ntop; i++)
        for (k = 0; k < n; k++) {
            float tu = m->top_u[(size_t)i * m->max_win + k];
            float td = m->top_d[(size_t)i * m->max_win + k];
            if (tu > 0.0f) m->tot_u[k] -= tu;
            if (td > 0.0f) m->tot_d[k] -= td;
        }
    for (k = 0; k < n; k++)
        m->env[k] = (m->tot_u[k] > m->tot_d[k]) ? m->tot_u[k] : m->tot_d[k];

    lo = centre - m->track; if (lo < 0) lo = 0;
    hi = centre + m->track; if (hi > n) hi = n;
    if (hi <= lo) { out->conf = 0.0f; out->peak = centre;
                    out->snr_db = -60.0f; out->winner_pk = 0.0f; return 0; }

    u = m->tot_u[lo]; d = m->tot_d[lo]; pk = lo;
    for (k = lo; k < hi; k++) {
        if (m->tot_u[k] > u) u = m->tot_u[k];
        if (m->tot_d[k] > d) d = m->tot_d[k];
        if (m->env[k] > m->env[pk]) pk = k;
    }
    dlo = centre - m->decide; if (dlo < 0) dlo = 0;
    dhi = centre + m->decide; if (dhi > n) dhi = n;
    if (dhi <= dlo) dp = pk;
    else { dp = dlo; for (k = dlo; k < dhi; k++) if (m->env[k] > m->env[dp]) dp = k; }
    du = m->tot_u[dp]; dd = m->tot_d[dp];

    scale = 0.0f;
    for (k = 0; k < n; k++) { float v = css_fabsf(win[k]); if (v > scale) scale = v; }
    floor_ = (scale > 0.0f) ? CSS_EPS_F32 * scale * (float)m->sps : 0.0f;

    for (k = 0; k < n; k++) m->msc[k] = m->env[k];
    if (n & 1) med = quickselect(m->msc, n, n / 2);
    else {
        float a1 = quickselect(m->msc, n, n / 2 - 1);
        float a2 = quickselect(m->msc, n, n / 2);
        med = 0.5f * (a1 + a2);
    }
    if (med < floor_) med = floor_;
    med += 1e-9f;

    mx = (u > d) ? u : d;
    if (mx < floor_) mx = floor_;
    if (mx < 1e-9f)  mx = 1e-9f;
    out->snr_db    = 20.0f * css_log10f(mx / med);
    out->winner_pk = (u > d) ? u : d;
    out->peak      = pk;
    out->conf      = (((du > dd) ? du : dd) <= floor_)
                   ? 0.0f : (du - dd) / (du + dd + 1e-12f);
    return 0;
}
