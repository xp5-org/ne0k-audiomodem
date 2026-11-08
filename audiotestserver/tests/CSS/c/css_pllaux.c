/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
#include "css_pllaux.h"
#include "css_util.h"
#include "css_win.h"
#include <math.h>

/* =========================================================================
 * SoftPreambleDetector
 * ====================================================================== */
struct css_syncdet {
    int n, lookback, pos, filled, best_lag, has_win;
    double threshold;
    double exp_norm;
    double conf_floor, cov_knee;
    double raw_score; int raw_lag, raw_reason;
    float *expected;        /* n        */
    float *hist;            /* lookback */
    float *flat;            /* lookback */
    float *best_win;        /* n        */
};

size_t css_syncdet_workspace(int n, int lookback)
{
    if (n < 1 || lookback < n) return 0;
    return css_al8(sizeof(struct css_syncdet))
         + 2 * css_al8((size_t)n * sizeof(float))
         + 2 * css_al8((size_t)lookback * sizeof(float));
}

int css_syncdet_init(css_syncdet **dp, void *mem, size_t memlen,
                     const unsigned char *pre, int n, int lookback,
                     double threshold, double conf_floor, double cov_knee)
{
    unsigned char *p = (unsigned char *)mem;
    struct css_syncdet *d;
    int i;
    float ss;
    if (!dp || !mem || !pre || css_syncdet_workspace(n, lookback) == 0) return -1;
    if (memlen < css_syncdet_workspace(n, lookback)) return -2;
    d = (struct css_syncdet *)p; p += css_al8(sizeof(struct css_syncdet));
    d->expected = (float *)p; p += css_al8((size_t)n * sizeof(float));
    d->best_win = (float *)p; p += css_al8((size_t)n * sizeof(float));
    d->hist     = (float *)p; p += css_al8((size_t)lookback * sizeof(float));
    d->flat     = (float *)p;
    d->n = n; d->lookback = lookback; d->threshold = threshold;
    d->conf_floor = conf_floor; d->cov_knee = cov_knee;
    for (i = 0; i < n; i++) d->expected[i] = pre[i] ? 1.0f : -1.0f;
    /* float(np.sqrt(np.dot(expected, expected))) + 1e-9: float32 dot/sqrt,
     * then a Python double */
    ss = 0.0f;
    for (i = 0; i < n; i++) ss += d->expected[i] * d->expected[i];
    d->exp_norm = (double)sqrtf(ss) + 1e-9;
    css_syncdet_clear(d);
    *dp = d;
    return 0;
}

void css_syncdet_clear(css_syncdet *d)
{
    int i;
    for (i = 0; i < d->lookback; i++) d->hist[i] = 0.0f;
    d->pos = 0; d->filled = 0; d->best_lag = 0; d->has_win = 0;
    d->raw_score = 0.0; d->raw_lag = 0; d->raw_reason = 0;
}

int css_syncdet_best_lag(const css_syncdet *d) { return d->best_lag; }
int css_syncdet_lookback(const css_syncdet *d) { return d->lookback; }
double css_syncdet_raw_score(const css_syncdet *d) { return d->raw_score; }
int css_syncdet_raw_lag(const css_syncdet *d) { return d->raw_lag; }
int css_syncdet_raw_reason(const css_syncdet *d) { return d->raw_reason; }

/* Score of window i (see SoftPreambleDetector.__init__ for the energy and
 * coverage terms).  Without either term it is the float32 cosine. */
static double sd_score(const struct css_syncdet *d, int i, float enorm)
{
    double dot = 0.0, sq = 0.0, asum = 0.0, g = 1.0;
    float fd, wn, sc;
    int j;
    for (j = 0; j < d->n; j++) {
        double v = (double)d->flat[i + j];
        dot += v * (double)d->expected[j];
        sq  += (double)(d->flat[i + j] * d->flat[i + j]);
        asum += v < 0 ? -v : v;
    }
    fd = (float)dot;
    wn = sqrtf((float)sq) + 1e-9f;
    sc = fd / (wn * enorm);
    if (d->conf_floor <= 0.0 && d->cov_knee <= 0.0) return (double)sc;
    if (d->conf_floor > 0.0) {
        double e = sqrt(sq / d->n) / d->conf_floor;
        g *= e < 1.0 ? e : 1.0;
    }
    if (d->cov_knee > 0.0) {
        double pr = asum * asum / (d->n * sq + 1e-30) / d->cov_knee;
        g *= pr < 1.0 ? pr : 1.0;
    }
    return (double)sc * g;
}

static double sd_alt_at(const struct css_syncdet *d, int i)
{
    int j, ch = 0;
    for (j = 1; j < d->n; j++)
        if ((signbit(d->flat[i + j]) != 0) != (signbit(d->flat[i + j - 1]) != 0)) ch++;
    return (double)ch / (double)(d->n - 1);
}

static int sd_fill(struct css_syncdet *d, double conf)
{
    int i, nf;
    d->hist[d->pos] = (float)conf;
    d->pos = (d->pos + 1) % d->lookback;
    d->filled = d->filled + 1 < d->lookback ? d->filled + 1 : d->lookback;
    if (d->filled < d->n) return 0;
    nf = d->filled;
    if (d->filled < d->lookback) {
        for (i = 0; i < nf; i++) d->flat[i] = d->hist[i];
    } else {
        for (i = 0; i < nf; i++) d->flat[i] = d->hist[(d->pos + i) % d->lookback];
    }
    return nf - d->n + 1;
}

double css_syncdet_push_masked(css_syncdet *d, double conf, int *acquired,
                               int lag_max, int use_early, long early_lag,
                               int soft, double trust_thr, double alt_thr)
{
    int i, j, W, raw_idx = 0, best_idx = -1;
    double raw = 0.0, best = 0.0;
    float enorm = (float)d->exp_norm;
    if (acquired) *acquired = 0;
    W = sd_fill(d, conf);
    if (W <= 0) return 0.0;
    for (i = 0; i < W; i++) {
        int lag = W - 1 - i, ok = 1;
        double sc = sd_score(d, i, enorm);
        /* np.argmax: first maximum wins */
        if (i == 0 || sc > raw) { raw = sc; raw_idx = i; }
        if (lag_max >= 0 && lag > lag_max) ok = 0;
        if (ok && use_early && (long)lag > early_lag) {
            ok = soft && (sc >= trust_thr || !(sd_alt_at(d, i) >= alt_thr));
        }
        if (ok && (best_idx < 0 || sc > best)) { best = sc; best_idx = i; }
    }
    d->raw_score = raw;
    d->raw_lag = W - 1 - raw_idx;
    if (best_idx == raw_idx) d->raw_reason = 0;
    else if (lag_max >= 0 && d->raw_lag > lag_max) d->raw_reason = 1;
    else d->raw_reason = 2;
    d->has_win = 1;
    if (best_idx < 0) {
        d->best_lag = d->raw_lag;
        for (j = 0; j < d->n; j++) d->best_win[j] = d->flat[raw_idx + j];
        return 0.0;
    }
    d->best_lag = W - 1 - best_idx;
    for (j = 0; j < d->n; j++) d->best_win[j] = d->flat[best_idx + j];
    if (acquired) *acquired = best >= d->threshold;
    return best;
}

double css_syncdet_push(css_syncdet *d, double conf, int *acquired)
{
    return css_syncdet_push_masked(d, conf, acquired, -1, 0, 0, 0, 0.0, 0.0);
}

double css_syncdet_alternation(const css_syncdet *d)
{
    int j, ch = 0;
    if (!d->has_win || d->n < 2) return -1.0;
    for (j = 1; j < d->n; j++)
        if ((signbit(d->best_win[j]) != 0) != (signbit(d->best_win[j - 1]) != 0)) ch++;
    return (double)ch / (double)(d->n - 1);
}

/* =========================================================================
 * if_slope_confidence
 * ====================================================================== */
struct css_ifslope {
    int nmax;
    cssd_blue blue;
    css_dcpx *a, *b;        /* nmax */
    double *p, *f;          /* nmax */
};

size_t css_ifslope_workspace(int nmax)
{
    if (nmax < 1) return 0;
    return css_al8(sizeof(struct css_ifslope)) + cssd_blue_workspace(nmax)
         + 2 * css_al8((size_t)nmax * sizeof(css_dcpx))
         + 2 * css_al8((size_t)nmax * sizeof(double));
}

int css_ifslope_init(css_ifslope **sp, void *mem, size_t memlen, int nmax)
{
    unsigned char *p = (unsigned char *)mem;
    struct css_ifslope *s;
    size_t bw = cssd_blue_workspace(nmax);
    if (!sp || !mem || css_ifslope_workspace(nmax) == 0) return -1;
    if (memlen < css_ifslope_workspace(nmax)) return -2;
    s = (struct css_ifslope *)p; p += css_al8(sizeof(struct css_ifslope));
    if (cssd_blue_init(&s->blue, nmax, p, bw) != 0) return -1;
    p += bw;
    s->a = (css_dcpx *)p; p += css_al8((size_t)nmax * sizeof(css_dcpx));
    s->b = (css_dcpx *)p; p += css_al8((size_t)nmax * sizeof(css_dcpx));
    s->p = (double *)p;   p += css_al8((size_t)nmax * sizeof(double));
    s->f = (double *)p;
    s->nmax = nmax;
    *sp = s;
    return 0;
}

void css_ifslope_run(css_ifslope *s, const float *seg, int N, double f_min,
                     double f_max, int sps, int fs, double *conf, double *r2)
{
    int k, half, L, trim;
    double val, lo_f = f_min * 0.9, hi_f = f_max * 1.1;
    double t_m, f_m, t_var, slope, expect, ss_res, ss_tot, rr, c;
    *conf = 0.0; *r2 = 0.0;
    if (N < 16 || N > s->nmax) return;
    half = N / 2;
    /* S = rfft(seg) */
    for (k = 0; k < N; k++) { s->a[k].re = (double)(css_real)seg[k]; s->a[k].im = 0.0; }
    cssd_blue_dft(&s->blue, s->a, s->a, N, -1);
    /* freqs = rfftfreq(N, 1/fs) = k * (1/(N*(1/fs))); mask out of band */
    val = 1.0 / ((double)N * (1.0 / (double)fs));
    for (k = 0; k <= half; k++) {
        double fr = (double)k * val;
        if (fr < lo_f || fr > hi_f) { s->a[k].re = 0.0; s->a[k].im = 0.0; }
    }
    /* bp = irfft(S, N): Hermitian extension, DC (and even-N Nyquist) real */
    s->a[0].im = 0.0;
    if (N % 2 == 0) s->a[half].im = 0.0;
    for (k = half + 1; k < N; k++) {
        s->a[k].re = s->a[N - k].re; s->a[k].im = -s->a[N - k].im;
    }
    /* fft(irfft(S, N)) is S's Hermitian extension (with DC and even-N
     * Nyquist made real) up to round-off, so the bp round trip is folded
     * away: two Bluestein transforms per symbol instead of four. */
    for (k = 0; k < N; k++) {
        double h;
        if (N % 2 == 0) h = (k == 0 || k == half) ? 1.0 : (k < half ? 2.0 : 0.0);
        else            h = (k == 0) ? 1.0 : (k < (N + 1) / 2 ? 2.0 : 0.0);
        s->a[k].re *= h; s->a[k].im *= h;
    }
    cssd_blue_dft(&s->blue, s->a, s->b, N, +1);
    for (k = 0; k < N; k++) {
        double re = s->b[k].re / (double)N, im = s->b[k].im / (double)N;
        s->p[k] = atan2(im, re);
    }
    /* np.unwrap (period 2pi, discont pi), then diff * fs / (2pi) */
    {
        double cum = 0.0, prev_up = s->p[0];
        for (k = 1; k < N; k++) {
            double dd = s->p[k] - s->p[k - 1];
            double ddmod = css_pymod(dd + M_PI, 2.0 * M_PI) - M_PI;
            double corr, up;
            if (ddmod == -M_PI && dd > 0) ddmod = M_PI;
            corr = ddmod - dd;
            if (fabs(dd) < M_PI) corr = 0.0;
            cum += corr;
            up = s->p[k] + cum;
            s->f[k - 1] = (up - prev_up) * (double)fs / (2.0 * M_PI);
            prev_up = up;
        }
    }
    trim = N / 20; if (trim < 1) trim = 1;
    L = (N - 1) - 2 * trim;
    if (L < 4) return;
    {
        double *f = s->f + trim;
        double *t = s->p;               /* reuse: t, then deviations */
        for (k = 0; k < L; k++) t[k] = (double)k;
        t_m = css_pmean(t, (size_t)L);
        f_m = css_pmean(f, (size_t)L);
        for (k = 0; k < L; k++) t[k] = ((double)k - t_m) * ((double)k - t_m);
        t_var = css_psum(t, (size_t)L);
        if (t_var < 1e-12) return;
        for (k = 0; k < L; k++) t[k] = ((double)k - t_m) * (f[k] - f_m);
        slope = css_psum(t, (size_t)L) / t_var * (double)fs;
        expect = (f_max - f_min) / ((double)sps / (double)fs);
        for (k = 0; k < L; k++) {
            double pred = f_m + (slope / (double)fs) * ((double)k - t_m);
            t[k] = (f[k] - pred) * (f[k] - pred);
        }
        ss_res = css_psum(t, (size_t)L);
        for (k = 0; k < L; k++) t[k] = (f[k] - f_m) * (f[k] - f_m);
        ss_tot = css_psum(t, (size_t)L);
    }
    rr = 1.0 - ss_res / (ss_tot + 1e-12);
    *r2 = rr > 0.0 ? rr : 0.0;
    c = slope / (expect + 1e-12);
    *conf = c < -1.0 ? -1.0 : (c > 1.0 ? 1.0 : c);
}

/* =========================================================================
 * whitening
 * ====================================================================== */
#define WH_PLANS 2
struct css_whiten {
    int fs, win, hop, nfs, nhs, taps, flen, redesign_every, count, nr_max;
    int have_psd, have_fir, fmax;
    double bg_alpha, max_gain_db;
    cssr_fft stft;                    /* nfs                                   */
    cssd_fft design;                  /* nfs, double                           */
    cssr_fft conv[WH_PLANS];          /* cached convolution plans              */
    int conv_n[WH_PLANS], conv_next;
    void *conv_mem[WH_PLANS];
    size_t conv_memlen;
    css_real *window;                 /* win                                   */
    css_real *psd;                    /* nhs                                   */
    css_real *mags;                   /* fmax * nhs                            */
    css_real *fir;                    /* taps+1                                */
    css_real *x, *y;                  /* nconv_max                             */
    css_rcpx *X, *H;                  /* nconv_max/2+1                         */
    css_real *col;                    /* fmax  median scratch                  */
    double *d1, *d2;                  /* nfs   design scratch                  */
    css_dcpx *dX;                     /* nhs                                   */
};

static int wh_win(int fs, double s, int win_max)
{
    int w = (int)css_iround(s * (double)fs);
    if (w < 64) w = 64;
    if (win_max > 0 && w > (win_max < 64 ? 64 : win_max)) w = win_max < 64 ? 64 : win_max;
    return w;
}

static int wh_nconv_max(int nr_max, int taps) { return css_pow2_atleast((long)nr_max + taps); }

size_t css_whiten_workspace(int fs, double stft_win_s, int win_max, int taps, int nr_max)
{
    int win = wh_win(fs, stft_win_s, win_max), hop = win / 2 > 0 ? win / 2 : 1;
    int nfs = css_pow2_atleast(win > taps ? win : taps), nhs = nfs / 2 + 1;
    int fmax = nr_max >= win ? (nr_max - win) / hop + 1 : 1;
    int nc = wh_nconv_max(nr_max, taps + 1);
    size_t w = css_al8(sizeof(struct css_whiten));
    if (fmax < 1) fmax = 1;
    w += cssr_fft_workspace(nfs) + cssd_fft_workspace(nfs);
    w += WH_PLANS * cssr_fft_workspace(nc);
    w += css_al8((size_t)win * sizeof(css_real));
    w += css_al8((size_t)nhs * sizeof(css_real));
    w += css_al8((size_t)fmax * nhs * sizeof(css_real));
    w += css_al8((size_t)(taps + 2) * sizeof(css_real));
    w += 2 * css_al8((size_t)nc * sizeof(css_real));
    w += 2 * css_al8((size_t)(nc / 2 + 1) * sizeof(css_rcpx));
    w += css_al8((size_t)fmax * sizeof(css_real));
    w += 2 * css_al8((size_t)nfs * sizeof(double));
    w += css_al8((size_t)nhs * sizeof(css_dcpx));
    return w;
}

int css_whiten_init(css_whiten **wp, void *mem, size_t memlen, int fs,
                    double stft_win_s, int win_max, double bg_alpha, int redesign_every,
                    int taps, double max_gain_db, int nr_max)
{
    unsigned char *p = (unsigned char *)mem;
    struct css_whiten *w;
    int win = wh_win(fs, stft_win_s, win_max), hop = win / 2 > 0 ? win / 2 : 1, i;
    int nfs = css_pow2_atleast(win > taps ? win : taps), nhs = nfs / 2 + 1;
    int fmax = nr_max >= win ? (nr_max - win) / hop + 1 : 1;
    int nc = wh_nconv_max(nr_max, taps + 1);
    if (!wp || !mem || taps < 1 || nr_max < 1) return -1;
    if (memlen < css_whiten_workspace(fs, stft_win_s, win_max, taps, nr_max)) return -2;
    if (fmax < 1) fmax = 1;
    w = (struct css_whiten *)p; p += css_al8(sizeof(struct css_whiten));
    w->fs = fs; w->win = win; w->hop = hop; w->nfs = nfs; w->nhs = nhs;
    w->taps = taps; w->redesign_every = redesign_every; w->count = 0;
    w->nr_max = nr_max; w->fmax = fmax; w->bg_alpha = bg_alpha;
    w->max_gain_db = max_gain_db; w->have_psd = 0; w->have_fir = 0;
    w->flen = 0;
    if (cssr_fft_init(&w->stft, nfs, p, cssr_fft_workspace(nfs))) return -1;
    p += cssr_fft_workspace(nfs);
    if (cssd_fft_init(&w->design, nfs, p, cssd_fft_workspace(nfs))) return -1;
    p += cssd_fft_workspace(nfs);
    w->conv_memlen = cssr_fft_workspace(nc);
    for (i = 0; i < WH_PLANS; i++) {
        w->conv_mem[i] = p; p += w->conv_memlen; w->conv_n[i] = 0;
    }
    w->conv_next = 0;
    w->window = (css_real *)p; p += css_al8((size_t)win * sizeof(css_real));
    w->psd    = (css_real *)p; p += css_al8((size_t)nhs * sizeof(css_real));
    w->mags   = (css_real *)p; p += css_al8((size_t)fmax * nhs * sizeof(css_real));
    w->fir    = (css_real *)p; p += css_al8((size_t)(taps + 2) * sizeof(css_real));
    w->x      = (css_real *)p; p += css_al8((size_t)nc * sizeof(css_real));
    w->y      = (css_real *)p; p += css_al8((size_t)nc * sizeof(css_real));
    w->X      = (css_rcpx *)p; p += css_al8((size_t)(nc / 2 + 1) * sizeof(css_rcpx));
    w->H      = (css_rcpx *)p; p += css_al8((size_t)(nc / 2 + 1) * sizeof(css_rcpx));
    w->col    = (css_real *)p; p += css_al8((size_t)fmax * sizeof(css_real));
    w->d1     = (double *)p;   p += css_al8((size_t)nfs * sizeof(double));
    w->d2     = (double *)p;   p += css_al8((size_t)nfs * sizeof(double));
    w->dX     = (css_dcpx *)p;
    for (i = 0; i < win; i++) w->window[i] = (css_real)css_hanning(i, win);
    *wp = w;
    return 0;
}

static cssr_fft *wh_plan(css_whiten *w, int n)
{
    int i;
    for (i = 0; i < WH_PLANS; i++)
        if (w->conv_n[i] == n) return &w->conv[i];
    i = w->conv_next; w->conv_next = (w->conv_next + 1) % WH_PLANS;
    if (cssr_fft_init(&w->conv[i], n, w->conv_mem[i], w->conv_memlen)) return 0;
    w->conv_n[i] = n;
    return &w->conv[i];
}

static css_real rmed(css_real *a, size_t n)
{
#ifdef CSS_PLL_F64
    return css_median_d(a, n);
#else
    return css_median_f(a, n);
#endif
}

static void wh_redesign(css_whiten *w)
{
    int k, nhs = w->nhs, nfs = w->nfs, nv = 0, half = w->taps / 2, c, flen;
    double med, flo, ref, maxlin, dc;
    for (k = 0; k < nhs; k++) if ((double)w->psd[k] > 0) w->d1[nv++] = (double)w->psd[k];
    med = nv ? css_median_d(w->d1, (size_t)nv) : 1.0;
    flo = med * 1e-3 + 1e-20;
    ref = 1.0 / sqrt(med + flo);
    maxlin = pow(10.0, w->max_gain_db / 20.0);
    for (k = 0; k < nhs; k++) {
        double g = 1.0 / sqrt((double)w->psd[k] + flo);
        g = g / (ref + 1e-20);
        if (g < 1.0 / maxlin) g = 1.0 / maxlin;
        if (g > maxlin) g = maxlin;
        w->dX[k].re = g; w->dX[k].im = 0.0;
    }
    cssd_irfft(&w->design, w->dX, w->d1);          /* full_ir */
    for (k = 0; k < nfs; k++) w->d2[k] = w->d1[(k + nfs / 2) % nfs];  /* fftshift */
    c = nfs / 2;
    if (nfs >= w->taps) {
        flen = 2 * half + 1;
        for (k = 0; k < flen; k++) w->d1[k] = w->d2[c - half + k];
    } else {
        flen = w->taps;
        for (k = 0; k < flen; k++) w->d1[k] = k < nfs ? w->d2[k] : 0.0;
    }
    for (k = 0; k < flen; k++) w->d1[k] = w->d1[k] * css_hamming(k, flen);
    dc = css_psum(w->d1, (size_t)flen);
    for (k = 0; k < flen; k++) {
        double v = fabs(dc) > 1e-9 ? w->d1[k] / dc : w->d1[k];
        w->fir[k] = (css_real)v;
    }
    w->flen = flen;
    w->have_fir = 1;
}

int css_whiten_run(css_whiten *w, const float *raw, int nr, css_real *out)
{
    int i, k, nf;
    if (nr > w->nr_max) return -1;
    /* _n_frames = max(1, (len - win) // hop + 1), Python floor division */
    {
        long q = (long)nr - w->win;
        long fl = q >= 0 ? q / w->hop : -((-q + w->hop - 1) / w->hop);
        nf = (int)(fl + 1);
        if (nf < 1) nf = 1;
    }
    if (nf >= 3) {
        for (i = 0; i < nf; i++) {
            int s0 = i * w->hop;
            css_real *row = w->mags + (size_t)i * w->nhs;
            for (k = 0; k < w->nfs; k++) w->x[k] = 0;
            for (k = 0; k < w->win; k++)
                w->x[k] = (css_real)((css_real)raw[s0 + k] * w->window[k]);
            cssr_rfft(&w->stft, w->x, w->X);
            for (k = 0; k < w->nhs; k++) {
#ifdef CSS_PLL_F64
                row[k] = hypot(w->X[k].re, w->X[k].im);
#else
                row[k] = hypotf(w->X[k].re, w->X[k].im);
#endif
            }
        }
        for (k = 0; k < w->nhs; k++) {
            css_real m, pk;
            for (i = 0; i < nf; i++) w->col[i] = w->mags[(size_t)i * w->nhs + k];
            m = rmed(w->col, (size_t)nf);
            pk = m * m;
            if (!w->have_psd) w->psd[k] = pk;
            else w->psd[k] = w->psd[k] + (css_real)w->bg_alpha * (pk - w->psd[k]);
        }
        w->have_psd = 1;
    }
    w->count++;
    if (w->have_psd && (!w->have_fir || w->count >= w->redesign_every)) {
        w->count = 0;
        wh_redesign(w);
    }
    if (w->have_fir) {
        int L = w->flen, ncv = nr + L - 1, nfc = css_pow2_atleast(ncv);
        int start = (L - 1) / 2;
        cssr_fft *pl = wh_plan(w, nfc);
        if (!pl) return -1;
        for (k = 0; k < nfc; k++) w->x[k] = 0;
        for (k = 0; k < nr; k++) w->x[k] = (css_real)raw[k];
        cssr_rfft(pl, w->x, w->X);
        for (k = 0; k < nfc; k++) w->y[k] = 0;
        for (k = 0; k < L; k++) w->y[k] = w->fir[k];
        cssr_rfft(pl, w->y, w->H);
        for (k = 0; k < nfc / 2 + 1; k++) {
            css_rcpx a = w->X[k], b = w->H[k];
            w->X[k].re = a.re * b.re - a.im * b.im;
            w->X[k].im = a.re * b.im + a.im * b.re;
        }
        cssr_irfft(pl, w->X, w->y);
        for (k = 0; k < nr; k++) out[k] = w->y[start + k];
        return 1;
    }
    for (k = 0; k < nr; k++) out[k] = (css_real)raw[k];
    return 0;
}
