/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
/* css_trk.c — see css_trk.h. */
#include "css_trk.h"
#include "css_real.h"
#include "css_pfft.h"
#include "css_util.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define REC_MAX   96          /* recent detected chirps kept for echo / dedupe */
#define OWN_MAX   8192        /* own chirps remembered                          */
#define LAGBIN_MAX 4096
#define RES_HIST  3

typedef struct {
    int    active, id, label, peer, echo_of, settled, hits, misses, off_run;
    long   anchor_k; double anchor_x, period;
    long   last_k, last_abs;
    double res[RES_HIST]; int nres;
    double off_res[8];
    double power_db, delay_s, gain_db, last_residual;
    int    echo_cons; double echo_dsum, echo_dmean;
    double acq_k[32], acq_x[32]; int nacq;
    long   collide_mark;
} grid_t;

typedef struct { long abs; int dir; double mag; int grid; int label; } rec_t;

struct css_trk {
    css_trk_cfg c;
    css_trk_cb cb;
    int sps, sps_eff, W, H, nfft, mainlobe, settle_w, acq_w;
    cssr_fft fft;
    css_rcpx *Fu, *Fd, *B, *P;
    css_real *x, *cu, *cd;
    float *win; int win_n, pend;
    double *e2;                 /* prefix energy, W+1                         */
    long abs_end;               /* abs index one past the last sample pushed  */
    double eref;
    grid_t g[CSS_TRK_MAX_GRIDS]; int next_id;
    rec_t rec[REC_MAX]; int nrec, rec_head;
    long own_end[OWN_MAX]; unsigned char own_dir[OWN_MAX]; int own_n, own_head;
    int *lagbin; int nlagbin; int own_lag_hits; double own_lag; int own_lag_known;
    /* carrier sense */
    double cal_sum_n; double *cal; int ncal, cal_cap, calibrated;
    double on_thr, off_thr, score_s; double sh[16]; int nsh, shh;
    int busy_ref; long below_since; long last_peak_abs;
    int linked; long linked_abs;
};

void css_trk_defaults(css_trk_cfg *c, int fs, double dur, double f_min,
                      double f_max, double gap_frac)
{
    c->fs = fs; c->dur = dur; c->f_min = f_min; c->f_max = f_max; c->gap_frac = gap_frac;
    c->peak_thr = 0.30; c->rel_floor = 0.06;
    c->acq_alpha = 0.25; c->alpha = 0.10; c->rate_alpha = 0.02; c->acq_hits = 20;
    c->settle_frac = 0.04; c->acq_frac = 0.25; c->lost_syms = 12;
    c->echo_min_s = 0.002; c->echo_max_s = 0.6; c->echo_tol_frac = 0.02;
    c->echo_min_db = 3.0; c->echo_hits = 4; c->max_ppm = 800.0; c->phase_m = 3;
    c->collide_db = 6.0; c->own_max_lag_s = 1.0; c->cal_s = 3.0;
    c->carrier_k = 2.2; c->carrier_lo = 0.05; c->carrier_hi = 0.12;
    c->busy_hold_s = 2.0; c->smooth_n = 4;
}

static int geo_sps(const css_trk_cfg *c) { return (int)(c->fs * c->dur); }
static int geo_eff(const css_trk_cfg *c) { int s = geo_sps(c); return s + (int)(s * c->gap_frac); }

size_t css_trk_workspace(const css_trk_cfg *c)
{
    int sps = geo_sps(c), W = 2 * sps, nfft, nh, ncal, nlag;
    size_t w;
    if (sps < 32) return 0;
    nfft = css_pow2_atleast((long)W + sps - 1);
    nh = nfft / 2 + 1;
    ncal = (int)(c->cal_s * c->fs / (geo_eff(c) / 2)) + 4;
    nlag = (int)(c->own_max_lag_s * c->fs) / 8 + 2;
    if (nlag > LAGBIN_MAX) nlag = LAGBIN_MAX;
    w = css_al8(sizeof(struct css_trk)) + cssr_fft_workspace(nfft);
    w += 4 * css_al8((size_t)nh * sizeof(css_rcpx));
    w += 3 * css_al8((size_t)nfft * sizeof(css_real));
    w += css_al8((size_t)W * sizeof(float));
    w += css_al8((size_t)(W + 1) * sizeof(double));
    w += css_al8((size_t)ncal * sizeof(double));
    w += css_al8((size_t)nlag * sizeof(int));
    return w;
}

static void ev(css_trk *t, int kind, long abs, int a, int b, double x, double y)
{
    if (t->cb.on_event) t->cb.on_event(t->cb.user, kind, abs, a, b, x, y);
}

int css_trk_init(css_trk **tp, void *mem, size_t memlen, const css_trk_cfg *c,
                 const css_trk_cb *cb)
{
    unsigned char *p = (unsigned char *)mem;
    struct css_trk *t;
    int sps = geo_sps(c), W = 2 * sps, nfft, nh, i, k, ncal, nlag;
    double bw;
    if (!tp || !mem || css_trk_workspace(c) == 0) return -1;
    if (memlen < css_trk_workspace(c)) return -2;
    nfft = css_pow2_atleast((long)W + sps - 1);
    nh = nfft / 2 + 1;
    ncal = (int)(c->cal_s * c->fs / (geo_eff(c) / 2)) + 4;
    nlag = (int)(c->own_max_lag_s * c->fs) / 8 + 2;
    if (nlag > LAGBIN_MAX) nlag = LAGBIN_MAX;
    t = (struct css_trk *)p; p += css_al8(sizeof(struct css_trk));
    t->c = *c;
    if (cb) t->cb = *cb; else { t->cb.user = 0; t->cb.on_chirp = 0; t->cb.on_event = 0; }
    t->sps = sps; t->sps_eff = geo_eff(c); t->W = W; t->H = t->sps_eff / 2;
    if (t->H < 1) t->H = 1;
    t->nfft = nfft;
    bw = fabs(c->f_max - c->f_min);
    t->mainlobe = (int)(4.0 * c->fs / (bw > 1 ? bw : 1));
    if (t->mainlobe < 8) t->mainlobe = 8;
    t->settle_w = (int)(sps * c->settle_frac); if (t->settle_w < 16) t->settle_w = 16;
    t->acq_w = (int)(sps * c->acq_frac);
    if (cssr_fft_init(&t->fft, nfft, p, cssr_fft_workspace(nfft))) return -1;
    p += cssr_fft_workspace(nfft);
    t->Fu = (css_rcpx *)p; p += css_al8((size_t)nh * sizeof(css_rcpx));
    t->Fd = (css_rcpx *)p; p += css_al8((size_t)nh * sizeof(css_rcpx));
    t->B  = (css_rcpx *)p; p += css_al8((size_t)nh * sizeof(css_rcpx));
    t->P  = (css_rcpx *)p; p += css_al8((size_t)nh * sizeof(css_rcpx));
    t->x  = (css_real *)p; p += css_al8((size_t)nfft * sizeof(css_real));
    t->cu = (css_real *)p; p += css_al8((size_t)nfft * sizeof(css_real));
    t->cd = (css_real *)p; p += css_al8((size_t)nfft * sizeof(css_real));
    t->win = (float *)p; p += css_al8((size_t)W * sizeof(float));
    t->e2 = (double *)p; p += css_al8((size_t)(W + 1) * sizeof(double));
    t->cal = (double *)p; p += css_al8((size_t)ncal * sizeof(double));
    t->lagbin = (int *)p;
    t->cal_cap = ncal; t->ncal = 0; t->calibrated = 0;
    t->nlagbin = nlag;
    for (i = 0; i < nlag; i++) t->lagbin[i] = 0;
    /* reversed chirp templates (same waveform as the encoder / decoder) */
    {
        double step = c->dur / (double)sps, kk, e = 0;
        int pass;
        for (pass = 0; pass < 2; pass++) {
            double f0 = pass ? c->f_max : c->f_min, f1 = pass ? c->f_min : c->f_max;
            kk = (f1 - f0) / c->dur;
            for (i = 0; i < nfft; i++) t->x[i] = 0;
            for (k = 0; k < sps; k++) {
                double tt = (double)k * step;
                float v = (float)sin((2.0 * M_PI) * (f0 * tt + (0.5 * kk) * (tt * tt)));
                t->x[sps - 1 - k] = (css_real)v;
                if (!pass) e += (double)v * (double)v;
            }
            cssr_rfft(&t->fft, t->x, pass ? t->Fd : t->Fu);
        }
        t->eref = sqrt(e);
    }
    for (i = 0; i < W; i++) t->win[i] = 0.0f;
    t->win_n = 0; t->pend = 0; t->abs_end = 0;
    for (i = 0; i < CSS_TRK_MAX_GRIDS; i++) { t->g[i].active = 0; t->g[i].id = -1; }
    t->next_id = 0;
    t->nrec = 0; t->rec_head = 0;
    t->own_n = 0; t->own_head = 0;
    t->own_lag_hits = 0; t->own_lag = -1.0; t->own_lag_known = 0;
    t->on_thr = c->carrier_hi; t->off_thr = 0.7 * c->carrier_hi; t->score_s = 0;
    t->nsh = 0; t->shh = 0; t->busy_ref = 0; t->below_since = -1;
    t->last_peak_abs = -(1L << 40);
    t->linked = -1; t->linked_abs = -1;
    *tp = t;
    return 0;
}

/* ------------------------------------------------------------ helpers */
static double grid_pred(const grid_t *g, long k) { return g->anchor_x + (double)(k - g->anchor_k) * g->period; }

static int slot_of(css_trk *t, int id)
{
    int i;
    for (i = 0; i < CSS_TRK_MAX_GRIDS; i++) if (t->g[i].active && t->g[i].id == id) return i;
    return -1;
}

static void rec_add(css_trk *t, long abs, int dir, double mag, int grid, int label)
{
    int idx;
    if (t->nrec < REC_MAX) { idx = (t->rec_head + t->nrec) % REC_MAX; t->nrec++; }
    else { idx = t->rec_head; t->rec_head = (t->rec_head + 1) % REC_MAX; }
    t->rec[idx].abs = abs; t->rec[idx].dir = dir; t->rec[idx].mag = mag;
    t->rec[idx].grid = grid; t->rec[idx].label = label;
}

static int rec_dup(const css_trk *t, long abs, int dir)
{
    int i;
    for (i = 0; i < t->nrec; i++) {
        const rec_t *r = &t->rec[(t->rec_head + i) % REC_MAX];
        long d = r->abs - abs;
        if (d < 0) d = -d;
        if (d <= t->mainlobe && r->dir == dir) return 1;
    }
    return 0;
}

static double med3(const double *v, int n)
{
    double a[RES_HIST]; int i, j;
    for (i = 0; i < n; i++) a[i] = v[i];
    for (i = 1; i < n; i++) { double x = a[i]; for (j = i - 1; j >= 0 && a[j] > x; j--) a[j + 1] = a[j]; a[j + 1] = x; }
    return a[n / 2];
}

static int spawn(css_trk *t, long abs, double db, int label, int echo_of)
{
    int i, s = -1;
    for (i = 0; i < CSS_TRK_MAX_GRIDS; i++) if (!t->g[i].active) { s = i; break; }
    if (s < 0) {          /* replace the weakest unsettled grid, else the oldest */
        double worst = 1e9; long oldest = -1;
        for (i = 0; i < CSS_TRK_MAX_GRIDS; i++)
            if (!t->g[i].settled && t->g[i].power_db < worst) { worst = t->g[i].power_db; s = i; }
        if (s < 0) for (i = 0; i < CSS_TRK_MAX_GRIDS; i++)
            if (oldest < 0 || t->g[i].last_abs < oldest) { oldest = t->g[i].last_abs; s = i; }
    }
    {
        grid_t *g = &t->g[s];
        g->active = 1; g->id = t->next_id++; g->label = label; g->peer = -1;
        g->echo_of = echo_of; g->settled = 0; g->hits = 1; g->misses = 0; g->off_run = 0;
        g->anchor_k = 0; g->anchor_x = (double)abs; g->period = (double)t->sps_eff;
        g->last_k = 0; g->last_abs = abs; g->nres = 0; g->power_db = db;
        g->delay_s = 0; g->gain_db = 0; g->last_residual = 0;
        g->echo_cons = 0; g->echo_dsum = 0; g->echo_dmean = 0;
        g->acq_k[0] = 0; g->acq_x[0] = (double)abs; g->nacq = 1;
        g->collide_mark = -(1L << 40);
        ev(t, CSS_TRK_EV_NEW, abs, g->id, label, 0, 0);
        return s;
    }
}

/* the two-layer loop: LS line fit over the acquisition hits (the fast
 * layer), then a phase IIR + Type-2 rate integrator on the median-filtered
 * residual (the slow layer) */
static void grid_update(css_trk *t, grid_t *g, long k, long abs, double r)
{
    double filt, dk = (double)(k - g->anchor_k);
    if (g->nres < RES_HIST) g->res[g->nres++] = r;
    else { g->res[0] = g->res[1]; g->res[1] = g->res[2]; g->res[2] = r; }
    filt = med3(g->res, g->nres);
    if (!g->settled && g->nacq < 32) {
        int i, n;
        double km = 0, xm = 0, sxy = 0, sxx = 0;
        g->acq_k[g->nacq] = (double)k; g->acq_x[g->nacq] = (double)abs; g->nacq++;
        n = g->nacq;
        for (i = 0; i < n; i++) { km += g->acq_k[i]; xm += g->acq_x[i]; }
        km /= n; xm /= n;
        for (i = 0; i < n; i++) {
            sxy += (g->acq_k[i] - km) * (g->acq_x[i] - xm);
            sxx += (g->acq_k[i] - km) * (g->acq_k[i] - km);
        }
        if (n >= 3 && sxx > 0) {
            double per = sxy / sxx, lim = t->sps_eff * t->c.max_ppm * 1e-6;
            if (per < t->sps_eff - lim) per = t->sps_eff - lim;
            if (per > t->sps_eff + lim) per = t->sps_eff + lim;
            g->period = per;
            g->anchor_k = k; g->anchor_x = xm + (double)(k - km) * per;
        } else {
            g->anchor_x = grid_pred(g, k) + t->c.acq_alpha * r; g->anchor_k = k;
        }
    } else {
        double a = g->hits < t->c.acq_hits ? t->c.acq_alpha : t->c.alpha;
        double lim = t->sps_eff * t->c.max_ppm * 1e-6;
        g->anchor_x = grid_pred(g, k) + a * filt; g->anchor_k = k;
        if (dk >= 1) g->period += t->c.rate_alpha * a * filt / dk * 4.0;
        if (g->period < t->sps_eff - lim) g->period = t->sps_eff - lim;
        if (g->period > t->sps_eff + lim) g->period = t->sps_eff + lim;
    }
    g->hits++; g->misses = 0; g->last_k = k; g->last_abs = abs; g->last_residual = r;
    if (!g->settled && g->hits >= t->c.acq_hits) {
        g->settled = 1;
        ev(t, CSS_TRK_EV_LOCK, abs, g->id, g->hits, (g->period / t->sps_eff - 1.0) * 1e6, 0);
    }
}

/* own schedule: does abs land on our own transmission + lag? */
static int own_match(css_trk *t, long abs, int dir)
{
    int i;
    long tol = t->mainlobe + t->sps / 50;
    if (!t->own_n) return 0;
    if (t->own_lag_known) {
        long L = (long)(t->own_lag * t->c.fs + 0.5);
        for (i = 0; i < t->own_n; i++) {
            int idx = (t->own_head + i) % OWN_MAX;
            long d = abs - (t->own_end[idx] + L);
            if (d < 0) d = -d;
            if (d <= tol && t->own_dir[idx] == dir) return 1;
        }
        return 0;
    }
    /* learning: histogram lags of direction-matched own chirps */
    for (i = 0; i < t->own_n; i++) {
        int idx = (t->own_head + i) % OWN_MAX;
        long L = abs - t->own_end[idx];
        int b;
        if (t->own_dir[idx] != dir || L < 0 || L > (long)(t->c.own_max_lag_s * t->c.fs)) continue;
        b = (int)(L / 8);
        if (b >= t->nlagbin) continue;
        t->lagbin[b]++;
        if (b > 0) t->lagbin[b - 1]++;
        if (b + 1 < t->nlagbin) t->lagbin[b + 1]++;
        if (t->lagbin[b] >= 4) {
            t->own_lag = (double)(b * 8 + 4) / t->c.fs;
            t->own_lag_known = 1;
            ev(t, CSS_TRK_EV_OWN_LAG, abs, t->lagbin[b], 0, t->own_lag, 0);
            return 1;
        }
    }
    return 0;
}

static void handle_chirp(css_trk *t, long abs, int dir, double mag, double score)
{
    css_trk_chirp ch;
    int i, best = -1, own;
    long bk = 0;
    double br = 0, db = 20.0 * log10(mag + 1e-30);
    ch.abs_peak = abs; ch.dir = dir; ch.mag = mag; ch.score = score;
    ch.grid = -1; ch.label = CSS_TRK_UNKNOWN; ch.peer = -1; ch.echo_of = -1; ch.residual = 0;
    t->last_peak_abs = abs;

    own = own_match(t, abs, dir);
    if (own) {
        ch.label = CSS_TRK_OWN;
        rec_add(t, abs, dir, mag, -1, CSS_TRK_OWN);
        if (t->cb.on_chirp) t->cb.on_chirp(t->cb.user, &ch);
        return;
    }
    /* nearest grid prediction */
    for (i = 0; i < CSS_TRK_MAX_GRIDS; i++) {
        grid_t *g = &t->g[i];
        long k; double pred, r, wmax;
        if (!g->active) continue;
        k = (long)floor(((double)abs - g->anchor_x) / g->period + 0.5) + g->anchor_k;
        if (k <= g->last_k) continue;
        pred = grid_pred(g, k); r = (double)abs - pred;
        wmax = g->settled ? (double)t->acq_w : (double)t->acq_w;
        if (fabs(r) > wmax) continue;
        if (best < 0 || fabs(r) < fabs(br)) { best = i; br = r; bk = k; }
    }
    if (best >= 0) {
        grid_t *g = &t->g[best];
        if (g->settled && fabs(br) > t->settle_w / 2) {
            /* off-grid hit on a settled grid (the decoder's settled outlier gate,
             * post_acq/2): not used to steer; M consistent ones are a phase step */
            if (g->off_run > 0 && fabs(br - g->off_res[g->off_run - 1]) > t->settle_w / 2) g->off_run = 0;
            if (g->off_run < 8) g->off_res[g->off_run++] = br;
            if (g->off_run >= t->c.phase_m) {
                double tmp[8]; int n = g->off_run, j; double d;
                for (j = 0; j < n; j++) tmp[j] = g->off_res[j];
                d = css_median_d(tmp, (size_t)n);
                g->anchor_x = grid_pred(g, bk) + d; g->anchor_k = bk;
                g->last_k = bk; g->last_abs = abs; g->nres = 0; g->off_run = 0;
                ev(t, CSS_TRK_EV_PHASE_STEP, abs, g->id, 0, d, d / t->c.fs);
            }
            /* still this station's chirp: attribute it, do not steer on it */
            ch.grid = g->id; ch.label = g->label; ch.peer = g->peer;
            ch.echo_of = g->echo_of; ch.residual = br;
            rec_add(t, abs, dir, mag, ch.grid, ch.label);
            if (t->cb.on_chirp) t->cb.on_chirp(t->cb.user, &ch);
            return;
        } else {
            g->off_run = 0;
            grid_update(t, g, bk, abs, br);
            g->power_db = g->power_db + 0.2 * (db - g->power_db);
            ch.grid = g->id; ch.label = g->label; ch.peer = g->peer;
            ch.echo_of = g->echo_of; ch.residual = br;
        }
    }
    if (best < 0 || t->g[best].label == CSS_TRK_ECHO) {
        /* echo: an earlier, stronger chirp of the same direction at a delay in range */
        int pr = -1; double pmag = 0; long pd = 0;
        long dmin = (long)(t->c.echo_min_s * t->c.fs), dmax = (long)(t->c.echo_max_s * t->c.fs);
        double need = mag * pow(10.0, t->c.echo_min_db / 20.0);
        for (i = 0; i < t->nrec; i++) {
            const rec_t *r = &t->rec[(t->rec_head + i) % REC_MAX];
            long d = abs - r->abs;
            if (r->dir != dir || d < dmin || d > dmax || r->mag < need) continue;
            if (r->label == CSS_TRK_ECHO) continue;
            if (r->mag > pmag) { pmag = r->mag; pr = i; pd = d; }
        }
        if (pr >= 0) {
            const rec_t *pa = &t->rec[(t->rec_head + pr) % REC_MAX];
            int gs = best >= 0 ? best : -1;
            grid_t *g;
            if (gs < 0) {
                /* an echo grid of this parent at this delay already? */
                for (i = 0; i < CSS_TRK_MAX_GRIDS; i++) {
                    grid_t *h = &t->g[i];
                    if (h->active && h->label == CSS_TRK_ECHO && h->echo_of == pa->grid
                            && fabs(h->echo_dmean - (double)pd) <= t->sps * t->c.echo_tol_frac + t->mainlobe) { gs = i; break; }
                }
                if (gs < 0) gs = spawn(t, abs, db, CSS_TRK_ECHO, pa->grid);
                else {
                    g = &t->g[gs];
                    g->last_abs = abs; g->hits++; g->misses = 0;
                }
            }
            g = &t->g[gs];
            g->label = CSS_TRK_ECHO; g->echo_of = pa->grid;
            if (g->echo_cons > 0 && fabs((double)pd - g->echo_dmean) > t->sps * t->c.echo_tol_frac + t->mainlobe) {
                g->echo_cons = 0; g->echo_dsum = 0;
            }
            g->echo_cons++; g->echo_dsum += (double)pd; g->echo_dmean = g->echo_dsum / g->echo_cons;
            g->delay_s = g->echo_dmean / t->c.fs;
            g->gain_db = db - 20.0 * log10(pmag + 1e-30);
            if (g->echo_cons == t->c.echo_hits)
                ev(t, CSS_TRK_EV_ECHO, abs, g->id, pa->grid, g->delay_s, g->gain_db);
            ch.grid = g->id; ch.label = CSS_TRK_ECHO; ch.echo_of = pa->grid;
            ch.residual = 0;
            rec_add(t, abs, dir, mag, g->id, CSS_TRK_ECHO);
            if (t->cb.on_chirp) t->cb.on_chirp(t->cb.user, &ch);
            return;
        }
        if (best < 0) {
            int s = spawn(t, abs, db, CSS_TRK_UNKNOWN, -1);
            ch.grid = t->g[s].id; ch.label = CSS_TRK_UNKNOWN;
            best = s;
        }
    }
    /* collision: another non-echo grid with a hit within half a symbol, at similar power */
    if (best >= 0) {
        grid_t *g = &t->g[best];
        for (i = 0; i < CSS_TRK_MAX_GRIDS; i++) {
            grid_t *h = &t->g[i];
            long d;
            if (i == best || !h->active || h->label == CSS_TRK_ECHO || g->label == CSS_TRK_ECHO) continue;
            d = h->last_abs - abs; if (d < 0) d = -d;
            if (d < t->sps_eff / 2 && fabs(h->power_db - g->power_db) <= t->c.collide_db
                    && abs - g->collide_mark > 2L * t->c.fs) {
                g->collide_mark = abs; h->collide_mark = abs;
                ev(t, CSS_TRK_EV_COLLISION, abs, g->id, h->id, g->power_db - h->power_db, 0);
            }
        }
    }
    rec_add(t, abs, dir, mag, ch.grid, ch.label);
    if (t->cb.on_chirp) t->cb.on_chirp(t->cb.user, &ch);
}

/* one analysis window over win[0..W) whose last sample is abs_end-1 */
static void analyse(css_trk *t)
{
    int W = t->W, sps = t->sps, nh = t->nfft / 2 + 1, i, k;
    long base = t->abs_end - W;
    double ewin = 0, pk_u = 0, pk_d = 0, wscore;
    for (i = 0; i < t->nfft; i++) t->x[i] = 0;
    t->e2[0] = 0;
    for (i = 0; i < W; i++) {
        double v = t->win[i];
        t->x[i] = (css_real)t->win[i];
        t->e2[i + 1] = t->e2[i] + v * v;
    }
    ewin = t->e2[W];
    cssr_rfft(&t->fft, t->x, t->B);
    for (k = 0; k < nh; k++) {
        css_rcpx a = t->B[k], b = t->Fu[k];
        t->P[k].re = a.re * b.re - a.im * b.im; t->P[k].im = a.re * b.im + a.im * b.re;
    }
    cssr_irfft(&t->fft, t->P, t->cu);
    for (k = 0; k < nh; k++) {
        css_rcpx a = t->B[k], b = t->Fd[k];
        t->P[k].re = a.re * b.re - a.im * b.im; t->P[k].im = a.re * b.im + a.im * b.re;
    }
    cssr_irfft(&t->fft, t->P, t->cd);
    for (i = 0; i < W; i++) {
        double u = fabs((double)t->cu[i]), d = fabs((double)t->cd[i]);
        t->cu[i] = (css_real)u; t->cd[i] = (css_real)d;
        if (u > pk_u) pk_u = u;
        if (d > pk_d) pk_d = d;
    }
    /* window chirp score (reference carrier sense) */
    wscore = (pk_u > pk_d ? pk_u : pk_d) / (sqrt(ewin) * t->eref + 1e-12);
    {
        int n = t->c.smooth_n > 16 ? 16 : (t->c.smooth_n < 1 ? 1 : t->c.smooth_n), j;
        double tmp[16], sm;
        t->sh[(t->shh + t->nsh) % 16] = wscore;
        if (t->nsh < n) t->nsh++; else t->shh = (t->shh + 1) % 16;
        for (j = 0; j < t->nsh; j++) tmp[j] = t->sh[(t->shh + j) % 16];
        sm = css_median_d(tmp, (size_t)t->nsh);
        t->score_s = sm;
        if (!t->calibrated) {
            if (t->ncal < t->cal_cap) t->cal[t->ncal++] = wscore;
            if ((double)t->abs_end >= t->c.cal_s * t->c.fs && t->ncal > 0) {
                double amb = css_median_d(t->cal, (size_t)t->ncal);
                double on = t->c.carrier_k * amb;
                if (on < t->c.carrier_lo) on = t->c.carrier_lo;
                if (on > t->c.carrier_hi) on = t->c.carrier_hi;
                t->on_thr = on;
                t->off_thr = 0.7 * on;
                if (t->off_thr > 1.3 * amb && 1.3 * amb > 0.045) t->off_thr = 1.3 * amb;
                if (t->off_thr < 0.045) t->off_thr = 0.045 < 0.7 * on ? 0.045 : 0.7 * on;
                t->calibrated = 1;
            }
        }
        if (!t->busy_ref) {
            if (sm > t->on_thr) { t->busy_ref = 1; t->below_since = -1; ev(t, CSS_TRK_EV_BUSY, t->abs_end, 1, 0, sm, 0); }
        } else if (sm < t->off_thr) {
            if (t->below_since < 0) t->below_since = t->abs_end;
            else if ((double)(t->abs_end - t->below_since) >= t->c.busy_hold_s * t->c.fs) {
                t->busy_ref = 0; t->below_since = -1;
                ev(t, CSS_TRK_EV_BUSY, t->abs_end, 0, 0, sm, 0);
            }
        } else t->below_since = -1;
    }
    /* peaks: greedy, strongest first, in the valid lag range [sps-1, W) */
    {
        double gmax = pk_u > pk_d ? pk_u : pk_d, floor_ = gmax * t->c.rel_floor;
        long found_abs[8]; int found_dir[8], nf = 0, it;
        double found_mag[8], found_sc[8];
        for (it = 0; it < 8; it++) {
            int bi = -1, bd = 0, j;
            double bv = 0;
            for (i = sps - 1; i < W; i++) {
                double u = (double)t->cu[i], d = (double)t->cd[i];
                if (u > bv) { bv = u; bi = i; bd = 1; }
                if (d > bv) { bv = d; bi = i; bd = 0; }
            }
            if (bi < 0 || bv < floor_ || bv <= 0) break;
            {
                double es = t->e2[bi + 1] - t->e2[bi + 1 - sps];
                double sc = bv / (sqrt(es > 0 ? es : 0) * t->eref + 1e-12);
                int lo = bi - t->mainlobe, hi = bi + t->mainlobe;
                if (lo < 0) lo = 0;
                if (hi > W - 1) hi = W - 1;
                for (j = lo; j <= hi; j++) { t->cu[j] = 0; t->cd[j] = 0; }
                /* weaker candidates may still qualify as echoes */
                if (sc >= t->c.peak_thr || (nf > 0 && sc >= 0.5 * t->c.peak_thr)) {
                    found_abs[nf] = base + bi; found_dir[nf] = bd;
                    found_mag[nf] = bv; found_sc[nf] = sc; nf++;
                }
            }
        }
        /* chronological order */
        for (i = 1; i < nf; i++) {
            long a = found_abs[i]; int d = found_dir[i]; double m = found_mag[i], s = found_sc[i];
            int j = i - 1;
            while (j >= 0 && found_abs[j] > a) {
                found_abs[j + 1] = found_abs[j]; found_dir[j + 1] = found_dir[j];
                found_mag[j + 1] = found_mag[j]; found_sc[j + 1] = found_sc[j]; j--;
            }
            found_abs[j + 1] = a; found_dir[j + 1] = d; found_mag[j + 1] = m; found_sc[j + 1] = s;
        }
        for (i = 0; i < nf; i++) {
            if (rec_dup(t, found_abs[i], found_dir[i])) continue;
            if (found_sc[i] < t->c.peak_thr) {
                /* sub-threshold: only as an echo of a stronger recent chirp */
                int j, ok = 0;
                for (j = 0; j < t->nrec; j++) {
                    const rec_t *r = &t->rec[(t->rec_head + j) % REC_MAX];
                    long dd = found_abs[i] - r->abs;
                    if (r->dir == found_dir[i] && dd >= (long)(t->c.echo_min_s * t->c.fs)
                            && dd <= (long)(t->c.echo_max_s * t->c.fs) && r->mag > found_mag[i]) { ok = 1; break; }
                }
                if (!ok) continue;
            }
            handle_chirp(t, found_abs[i], found_dir[i], found_mag[i], found_sc[i]);
        }
    }
    /* retire silent grids */
    for (i = 0; i < CSS_TRK_MAX_GRIDS; i++) {
        grid_t *g = &t->g[i];
        if (!g->active) continue;
        if ((double)(t->abs_end - g->last_abs) > t->c.lost_syms * g->period) {
            ev(t, CSS_TRK_EV_LOST, t->abs_end, g->id, g->hits, 0, 0);
            g->active = 0;
            if (t->linked == g->id) t->linked = -1;
        }
    }
}

int css_trk_push(css_trk *t, const float *x, int n)
{
    int off = 0, ran = 0;
    while (off < n) {
        int room = t->W - t->win_n, take = n - off;
        if (t->win_n < t->W) {
            if (take > room) take = room;
            for (int i = 0; i < take; i++) t->win[t->win_n + i] = x[off + i];
            t->win_n += take; off += take; t->abs_end += take;
            if (t->win_n == t->W) { analyse(t); ran++; t->pend = 0; }
            continue;
        }
        /* window full: slide by H once H new samples are in */
        take = t->H - t->pend;
        if (take > n - off) take = n - off;
        {
            int i, keep = t->W - take;
            for (i = 0; i < keep; i++) t->win[i] = t->win[i + take];
            for (i = 0; i < take; i++) t->win[keep + i] = x[off + i];
        }
        t->pend += take; off += take; t->abs_end += take;
        if (t->pend >= t->H) { analyse(t); ran++; t->pend = 0; }
    }
    return ran;
}

int css_trk_tx(css_trk *t, long abs_first, int nsym, const unsigned char *dirs)
{
    int k;
    for (k = 0; k < nsym; k++) {
        int idx;
        if (t->own_n < OWN_MAX) { idx = (t->own_head + t->own_n) % OWN_MAX; t->own_n++; }
        else { idx = t->own_head; t->own_head = (t->own_head + 1) % OWN_MAX; }
        t->own_end[idx] = abs_first + (long)k * t->sps_eff + t->sps - 1;
        t->own_dir[idx] = dirs ? (unsigned char)(dirs[k] ? 1 : 0) : 0;
    }
    return nsym;
}

void css_trk_decoder_peak(css_trk *t, long abs_peak)
{
    int i, best = -1;
    long bd = 0;
    for (i = 0; i < CSS_TRK_MAX_GRIDS; i++) {
        grid_t *g = &t->g[i];
        long d;
        if (!g->active || g->label == CSS_TRK_ECHO) continue;
        d = g->last_abs - abs_peak; if (d < 0) d = -d;
        if (d <= t->acq_w && (best < 0 || d < bd)) { best = i; bd = d; }
    }
    if (best >= 0) { t->linked = t->g[best].id; t->linked_abs = abs_peak; }
}

void css_trk_decoder_frame(css_trk *t, int src)
{
    int s = t->linked >= 0 ? slot_of(t, t->linked) : -1;
    if (s < 0) return;
    if (t->g[s].label != CSS_TRK_PEER || t->g[s].peer != src) {
        t->g[s].label = CSS_TRK_PEER; t->g[s].peer = src;
        ev(t, CSS_TRK_EV_PEER, t->linked_abs, t->g[s].id, src, (t->g[s].period / t->sps_eff - 1.0) * 1e6, 0);
    }
}

int css_trk_busy(const css_trk *t)
{
    return t->busy_ref || (t->abs_end - t->last_peak_abs) <= (long)(t->sps_eff + t->H);
}

double css_trk_score(const css_trk *t) { return t->score_s; }
long   css_trk_abs(const css_trk *t) { return t->abs_end; }
double css_trk_own_lag(const css_trk *t) { return t->own_lag_known ? t->own_lag : -1.0; }

int css_trk_get_grid(const css_trk *t, int i, css_trk_grid *o)
{
    const grid_t *g;
    if (i < 0 || i >= CSS_TRK_MAX_GRIDS) return -1;
    g = &t->g[i];
    o->active = g->active; o->id = g->id; o->label = g->label; o->peer = g->peer;
    o->echo_of = g->echo_of; o->settled = g->settled; o->hits = g->hits; o->misses = g->misses;
    o->t0 = g->anchor_x - (double)g->anchor_k * g->period; o->period = g->period;
    o->ppm = (g->period / t->sps_eff - 1.0) * 1e6; o->power_db = g->power_db;
    o->delay_s = g->delay_s; o->gain_db = g->gain_db; o->last_residual = g->last_residual;
    o->last_abs = g->last_abs;
    return 0;
}
