/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
/* css_pll.c — see css_pll.h.  Section letters (§K .. §AB) refer to the
 * reference css_decoder.decode() layout; every branch below mirrors one in
 * that function, in the same order, so the two can be read side by side. */
#include "css_pll.h"
#include "css_real.h"
#include "css_corr.h"
#include "css_pllaux.h"
#include "css_retro.h"
#include "css_util.h"
#include "css_dec.h"
#include "../../c/rs255.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TERR_MAX 64
static int iabs(int v) { return v < 0 ? -v : v; }

struct css_pll {
    css_pll_cfg c;
    css_pll_cb cb;
    float *up_ref, *dn_ref;
    css_corr *corr; css_syncdet *sd; css_ifslope *ifs; css_whiten *wh;
    css_retro *rt; rs255 *rs;
    css_enc_cfg g; void *decw; size_t decw_len;
    int nr_max;
    float *ring, *raw, *agc;
    css_real *wraw;
    float *fifo; int fifo_n, fifo_cap; long abs_pos;
    /* _audio_hist: deque(maxlen=audio_hist_len) of raw chunks */
    float *hist; long hist_cap; long hist_head, hist_n;
    int *hq; int hq_cap, hq_head, hq_n;
    float *lin; long lin_cap;
    /* _frame_audio */
    float *fa; long fa_cap, fa_n; int *fal; int fal_cap, fal_n;
    long retro_nmax;
    /* bit_stream, pre_sync_buf */
    unsigned char *bs_b; double *bs_c; int bs_n, bs_cap;
    unsigned char *ps_b; double *ps_c; int ps_n, ps_head, ps_cap;
    int terr[TERR_MAX]; int terr_n, terr_head;
    int acqdev[5]; int acqdev_n, acqdev_head;
    double *snrw; int snrw_cap, snrw_n, snrw_head;
    /* frame scratch */
    unsigned char *fb, *cb_bits, *rawb, *rsout; double *fc, *cc;
    float *fc32, *bconf; int *era;
    /* state */
    int expected_peak, peak_est, timing_settled, n_read, acq_left, timing_err;
    double timing_frac, rate_est, wpk_ema; int have_wpk;
    int sync_found, frames_decoded, frames_errored, consec_errors;
    long sym_count, last_resync_sym; int last_resync_soft;
    int quiet_run;
    double onset_floor; int have_onset_floor;   /* onset guard's room floor (rms) */
    int quiet_rel_run;                          /* symbols under that floor + onset_rel_db */
    /* phase-step recovery */
    double step_ref_snr; int step_ref_ok;
    int step_cand, step_cand_d, step_cand_idx; double step_cand_conf;
    int step_fix_pending, step_fix_idx; double step_fix_conf;
    int have_snap; double snap_tf; int snap_nr, snap_pe;
    int one_shot;
    int have_bg; double bg_amp_ema; double syndet_bg_ema;
    int audio_hist_len, anchor_enabled, anchor_lookahead, max_shift;
    int anchor_retry; double anchor_retry_snr;
    unsigned char decoded_seqs[256];
    double agc_gain;
    int retro_max_syms;
    css_pll_diag d;
};

/* ---------------------------------------------------------------- sizing */
static int cfg_nr_max(const css_pll_cfg *c)
{
    return c->nr_max > 0 ? c->nr_max : c->sps_eff + c->sps;
}

static int hist_len_of(const css_pll_cfg *c)
{
    return c->frame_symbols + c->training_symbols + c->n_preamble
         + c->anchor_lookahead + 8;
}

static int retro_syms_of(const css_pll_cfg *c)
{
    long m = c->max_retro_smp / c->sps_eff;
    return (int)(m > 4 ? m : 4);
}

static long retro_nmax_of(const css_pll_cfg *c, int nr_max)
{
    long per = (long)c->sps_eff + c->sps / 2;          /* n_read with headroom */
    long a, f;
    int hl = hist_len_of(c), rs = retro_syms_of(c), fs = c->frame_symbols;
    (void)nr_max;
    a = ((long)hl * c->sps_eff <= c->max_retro_smp) ? (long)hl * per : 0;
    f = (long)(fs < rs ? fs : rs) * per;
    return (a > f ? a : f) + c->sps;
}

typedef struct { size_t off[24]; size_t total; } layout_t;

static size_t blk(size_t *acc, size_t n) { size_t o = *acc; *acc += css_al8(n); return o; }

static int geometry(const css_pll_cfg *c, css_enc_cfg *g)
{
    if (css_enc_geometry_crc(g, c->chunk_size, c->rs_t, c->conv_k7 ? 1 : 0,
                             c->diff_enc ? 1 : 0, c->interleave_rows,
                             c->crc_bytes ? c->crc_bytes : 4) != 0) return -1;
    if (g->base_bits != c->base_bits) return -1;
    return 0;
}

size_t css_pll_workspace(const css_pll_cfg *c)
{
    css_enc_cfg g;
    size_t a = 0;
    int nr = cfg_nr_max(c), hl;
    long rn;
    if (!c || c->sps < 8 || c->buf_size < c->sps || c->frame_symbols < 1) return 0;
    if (c->n_preamble < 1 || c->n_preamble > CSS_PLL_MAX_PREAMBLE) return 0;
    if (geometry(c, &g)) return 0;
    hl = hist_len_of(c);
    rn = retro_nmax_of(c, nr);
    blk(&a, sizeof(struct css_pll));
    blk(&a, 2 * (size_t)c->sps * sizeof(float));
    blk(&a, css_corr_workspace(c->sps, c->buf_size, c->mf_hann));
    blk(&a, css_syncdet_workspace(c->n_preamble, c->syndet_lookback));
    blk(&a, css_ifslope_workspace(c->sps));
    if (c->whiten_enable)
        blk(&a, css_whiten_workspace(c->fs, c->whiten_stft_win_s, c->whiten_stft_win_max, c->whiten_taps, nr));
    blk(&a, css_retro_workspace(c->sps, rn, c->frame_symbols));
    if (c->rs_t > 0) blk(&a, rs255_workspace(c->rs_t * 2));
    blk(&a, css_dec_workspace(&g));
    blk(&a, (size_t)c->buf_size * sizeof(float));
    blk(&a, 2 * (size_t)nr * sizeof(float));
    blk(&a, (size_t)nr * sizeof(css_real));
    blk(&a, 2 * (size_t)nr * sizeof(float));           /* fifo */
    blk(&a, (size_t)hl * nr * sizeof(float));          /* hist */
    blk(&a, (size_t)hl * sizeof(int));
    blk(&a, (size_t)rn * sizeof(float));               /* lin */
    blk(&a, (size_t)(c->frame_symbols + 2) * nr * sizeof(float));
    blk(&a, (size_t)(c->frame_symbols + 2) * sizeof(int));
    blk(&a, (size_t)(2 * c->frame_symbols + c->syndet_lookback + 2));
    blk(&a, (size_t)(2 * c->frame_symbols + c->syndet_lookback + 2) * sizeof(double));
    blk(&a, (size_t)c->syndet_lookback);
    blk(&a, (size_t)c->syndet_lookback * sizeof(double));
    blk(&a, (size_t)c->frame_symbols * sizeof(double));      /* snr window */
    blk(&a, 2 * (size_t)c->frame_symbols);                   /* fb, cb_bits */
    blk(&a, 2 * (size_t)c->frame_symbols * sizeof(double));  /* fc, cc */
    blk(&a, (size_t)c->frame_symbols * sizeof(float));       /* fc32 */
    blk(&a, 2 * (size_t)(c->base_bits / 8 + 8));             /* rawb, rsout */
    blk(&a, (size_t)(c->base_bits / 8 + 8) * sizeof(float));
    blk(&a, (size_t)(c->base_bits / 8 + 8) * sizeof(int));
    return a;
}

/* generate_chirp: t = linspace(0, dur, n, endpoint=False) = i*(dur/n);
 * phi = 2*pi*(f0*t + 0.5*k*t**2); sin -> float32 */
static void chirp(float *out, int n, double f0, double f1, double dur)
{
    double step = dur / (double)n, k = (f1 - f0) / dur;
    int i;
    for (i = 0; i < n; i++) {
        double t = (double)i * step + 0.0;
        double phi = (2.0 * M_PI) * (f0 * t + (0.5 * k) * (t * t));
        out[i] = (float)sin(phi);
    }
}

static void ev(css_pll *p, int kind, int a, int b, double x, double y)
{
    if (p->cb.on_event) p->cb.on_event(p->cb.user, kind, p->sym_count, a, b, x, y);
}

int css_pll_init(css_pll **pp, void *mem, size_t memlen, const css_pll_cfg *c,
                 const css_pll_cb *cb)
{
    unsigned char *base = (unsigned char *)mem;
    size_t a = 0, need = css_pll_workspace(c), o;
    struct css_pll *p;
    int nr, hl, i;
    long rn;
    if (!pp || !mem || need == 0) return -1;
    if (memlen < need) return -2;
    nr = cfg_nr_max(c); hl = hist_len_of(c); rn = retro_nmax_of(c, nr);
    p = (struct css_pll *)(base + blk(&a, sizeof(struct css_pll)));
    p->c = *c;
    if (cb) p->cb = *cb; else { p->cb.user = 0; p->cb.on_symbol = 0; p->cb.on_frame = 0; p->cb.on_event = 0; }
    p->nr_max = nr;
    geometry(c, &p->g);
    o = blk(&a, 2 * (size_t)c->sps * sizeof(float));
    p->up_ref = (float *)(base + o); p->dn_ref = p->up_ref + c->sps;
    chirp(p->up_ref, c->sps, c->f_min, c->f_max, c->dur);
    chirp(p->dn_ref, c->sps, c->f_max, c->f_min, c->dur);
    o = blk(&a, css_corr_workspace(c->sps, c->buf_size, c->mf_hann));
    if (css_corr_init(&p->corr, base + o, css_corr_workspace(c->sps, c->buf_size, c->mf_hann),
                      p->up_ref, p->dn_ref, c->sps, c->search_frac, c->buf_size, c->mf_hann)) return -3;
    o = blk(&a, css_syncdet_workspace(c->n_preamble, c->syndet_lookback));
    if (css_syncdet_init(&p->sd, base + o, css_syncdet_workspace(c->n_preamble, c->syndet_lookback),
                         c->preamble, c->n_preamble, c->syndet_lookback, c->sync_threshold,
                         c->syndet_conf_floor, c->syndet_cov_knee)) return -4;
    o = blk(&a, css_ifslope_workspace(c->sps));
    if (css_ifslope_init(&p->ifs, base + o, css_ifslope_workspace(c->sps), c->sps)) return -5;
    p->wh = 0;
    if (c->whiten_enable) {
        size_t wl = css_whiten_workspace(c->fs, c->whiten_stft_win_s, c->whiten_stft_win_max, c->whiten_taps, nr);
        o = blk(&a, wl);
        if (css_whiten_init(&p->wh, base + o, wl, c->fs, c->whiten_stft_win_s, c->whiten_stft_win_max, c->whiten_bg_alpha,
                            c->whiten_redesign_every, c->whiten_taps, c->whiten_max_gain_db, nr)) return -6;
    }
    o = blk(&a, css_retro_workspace(c->sps, rn, c->frame_symbols));
    if (css_retro_init(&p->rt, base + o, css_retro_workspace(c->sps, rn, c->frame_symbols),
                       p->up_ref, p->dn_ref, c->sps, rn, c->frame_symbols)) return -7;
    p->retro_nmax = rn;
    p->rs = 0;
    if (c->rs_t > 0) {
        o = blk(&a, rs255_workspace(c->rs_t * 2));
        if (rs255_init(&p->rs, base + o, rs255_workspace(c->rs_t * 2), c->rs_t * 2)) return -8;
    }
    p->decw_len = css_dec_workspace(&p->g);
    p->decw = base + blk(&a, p->decw_len);
    p->ring = (float *)(base + blk(&a, (size_t)c->buf_size * sizeof(float)));
    o = blk(&a, 2 * (size_t)nr * sizeof(float));
    p->raw = (float *)(base + o); p->agc = p->raw + nr;
    p->wraw = (css_real *)(base + blk(&a, (size_t)nr * sizeof(css_real)));
    p->fifo = (float *)(base + blk(&a, 2 * (size_t)nr * sizeof(float)));
    p->fifo_cap = 2 * nr; p->fifo_n = 0; p->abs_pos = 0;
    p->hist = (float *)(base + blk(&a, (size_t)hl * nr * sizeof(float)));
    p->hist_cap = (long)hl * nr; p->hist_head = 0; p->hist_n = 0;
    p->hq = (int *)(base + blk(&a, (size_t)hl * sizeof(int)));
    p->hq_cap = hl; p->hq_head = 0; p->hq_n = 0;
    p->lin = (float *)(base + blk(&a, (size_t)rn * sizeof(float)));
    p->lin_cap = rn;
    p->fa = (float *)(base + blk(&a, (size_t)(c->frame_symbols + 2) * nr * sizeof(float)));
    p->fa_cap = (long)(c->frame_symbols + 2) * nr; p->fa_n = 0;
    p->fal = (int *)(base + blk(&a, (size_t)(c->frame_symbols + 2) * sizeof(int)));
    p->fal_cap = c->frame_symbols + 2; p->fal_n = 0;
    p->bs_cap = 2 * c->frame_symbols + c->syndet_lookback + 2;
    p->bs_b = base + blk(&a, (size_t)p->bs_cap);
    p->bs_c = (double *)(base + blk(&a, (size_t)p->bs_cap * sizeof(double)));
    p->bs_n = 0;
    p->ps_cap = c->syndet_lookback;
    p->ps_b = base + blk(&a, (size_t)p->ps_cap);
    p->ps_c = (double *)(base + blk(&a, (size_t)p->ps_cap * sizeof(double)));
    p->ps_n = 0; p->ps_head = 0;
    p->snrw = (double *)(base + blk(&a, (size_t)c->frame_symbols * sizeof(double)));
    p->snrw_cap = c->frame_symbols; p->snrw_n = 0; p->snrw_head = 0;
    o = blk(&a, 2 * (size_t)c->frame_symbols);
    p->fb = base + o; p->cb_bits = p->fb + c->frame_symbols;
    o = blk(&a, 2 * (size_t)c->frame_symbols * sizeof(double));
    p->fc = (double *)(base + o); p->cc = p->fc + c->frame_symbols;
    p->fc32 = (float *)(base + blk(&a, (size_t)c->frame_symbols * sizeof(float)));
    o = blk(&a, 2 * (size_t)(c->base_bits / 8 + 8));
    p->rawb = base + o; p->rsout = p->rawb + (c->base_bits / 8 + 8);
    p->bconf = (float *)(base + blk(&a, (size_t)(c->base_bits / 8 + 8) * sizeof(float)));
    p->era = (int *)(base + blk(&a, (size_t)(c->base_bits / 8 + 8) * sizeof(int)));

    for (i = 0; i < c->buf_size; i++) p->ring[i] = 0.0f;
    p->expected_peak = c->buf_size - (c->ring_lookahead > c->gap_samps ? c->ring_lookahead : c->gap_samps) - 1;
    p->peak_est = p->expected_peak;
    p->timing_frac = 0.0; p->timing_settled = 0; p->rate_est = 0.0;
    p->have_wpk = 0; p->wpk_ema = 0.0;
    p->sync_found = 0; p->frames_decoded = 0; p->frames_errored = 0; p->consec_errors = 0;
    p->sym_count = 0; p->timing_err = 0; p->n_read = c->sps_eff; p->acq_left = 0;
    p->last_resync_sym = 0; p->last_resync_soft = 0;
    p->quiet_run = 0;
    p->onset_floor = 0.0; p->have_onset_floor = 0; p->quiet_rel_run = 0;
    p->step_ref_ok = 0; p->step_cand = 0; p->step_fix_pending = 0;
    p->terr_n = 0; p->terr_head = 0; p->acqdev_n = 0; p->acqdev_head = 0;
    p->have_snap = 0; p->one_shot = 0;
    p->have_bg = 0; p->bg_amp_ema = 0.0; p->syndet_bg_ema = 0.0;
    p->audio_hist_len = hl;
    p->anchor_lookahead = c->anchor_lookahead;
    p->anchor_enabled = ((long)hl * c->sps_eff <= c->max_retro_smp);
    if (!p->anchor_enabled) p->anchor_lookahead = 0;
    p->max_shift = c->retro_anchor_max_shift >= 0 ? c->retro_anchor_max_shift
                                                  : hl - c->frame_symbols;
    p->anchor_retry = 0; p->anchor_retry_snr = 0.0;
    for (i = 0; i < 256; i++) p->decoded_seqs[i] = 0;
    p->agc_gain = 1.0;
    p->retro_max_syms = retro_syms_of(c);
    {
        css_pll_diag *d = &p->d;
        d->symbols = 0; d->frames_decoded = 0; d->frames_errored = 0;
        d->sync_acquired = 0; d->sync_lag = 0; d->sync_at_symbol = -1; d->sync_score = 0.0;
        d->snr_sum = 0.0; d->snr_n = 0; d->rms_max = 0.0; d->rms_sum = 0.0; d->rms_n = 0;
    }
    *pp = p;
    return 0;
}

/* --------------------------------------------------------------- deques */
static void terr_clear(css_pll *p) { p->terr_n = 0; p->terr_head = 0; }
static void terr_push(css_pll *p, int v)
{
    int m = p->c.timing_median_n;
    if (m > TERR_MAX) m = TERR_MAX;
    if (p->terr_n < m) { p->terr[(p->terr_head + p->terr_n) % m] = v; p->terr_n++; }
    else { p->terr[p->terr_head] = v; p->terr_head = (p->terr_head + 1) % m; }
}
/* sorted(buf)[len // 2] */
static int terr_median(const css_pll *p)
{
    int tmp[TERR_MAX], n = p->terr_n, i, j, m = p->c.timing_median_n;
    if (m > TERR_MAX) m = TERR_MAX;
    for (i = 0; i < n; i++) tmp[i] = p->terr[(p->terr_head + i) % m];
    for (i = 1; i < n; i++) {
        int v = tmp[i];
        for (j = i - 1; j >= 0 && tmp[j] > v; j--) tmp[j + 1] = tmp[j];
        tmp[j + 1] = v;
    }
    return tmp[n / 2];
}
static void acqdev_clear(css_pll *p) { p->acqdev_n = 0; p->acqdev_head = 0; }
static void acqdev_push(css_pll *p, int v)
{
    if (p->acqdev_n < 5) { p->acqdev[(p->acqdev_head + p->acqdev_n) % 5] = v; p->acqdev_n++; }
    else { p->acqdev[p->acqdev_head] = v; p->acqdev_head = (p->acqdev_head + 1) % 5; }
}
static void ps_clear(css_pll *p) { p->ps_n = 0; p->ps_head = 0; }
static void ps_push(css_pll *p, int b, double c)
{
    int idx;
    if (p->ps_n < p->ps_cap) { idx = (p->ps_head + p->ps_n) % p->ps_cap; p->ps_n++; }
    else { idx = p->ps_head; p->ps_head = (p->ps_head + 1) % p->ps_cap; }
    p->ps_b[idx] = (unsigned char)b; p->ps_c[idx] = c;
}
static void fa_clear(css_pll *p) { p->fa_n = 0; p->fal_n = 0; }
static void fa_append(css_pll *p, const float *x, int n)
{
    int i;
    if (p->fal_n >= p->fal_cap || p->fa_n + n > p->fa_cap) {
        /* drop the oldest chunk (beyond what retro can ever read) */
        int l0 = p->fal_n ? p->fal[0] : 0;
        long k;
        for (k = 0; k + l0 < p->fa_n; k++) p->fa[k] = p->fa[k + l0];
        p->fa_n -= l0;
        for (i = 1; i < p->fal_n; i++) p->fal[i - 1] = p->fal[i];
        if (p->fal_n) p->fal_n--;
    }
    for (i = 0; i < n; i++) p->fa[p->fa_n + i] = x[i];
    p->fa_n += n;
    p->fal[p->fal_n++] = n;
}
static void hist_append(css_pll *p, const float *x, int n)
{
    long i;
    if (p->hq_n == p->hq_cap) {            /* deque maxlen: drop oldest chunk */
        int l0 = p->hq[p->hq_head];
        p->hq_head = (p->hq_head + 1) % p->hq_cap; p->hq_n--;
        p->hist_head = (p->hist_head + l0) % p->hist_cap; p->hist_n -= l0;
    }
    while (p->hist_n + n > p->hist_cap && p->hq_n > 0) {
        int l0 = p->hq[p->hq_head];
        p->hq_head = (p->hq_head + 1) % p->hq_cap; p->hq_n--;
        p->hist_head = (p->hist_head + l0) % p->hist_cap; p->hist_n -= l0;
    }
    for (i = 0; i < n; i++) p->hist[(p->hist_head + p->hist_n + i) % p->hist_cap] = x[i];
    p->hist_n += n;
    p->hq[(p->hq_head + p->hq_n) % p->hq_cap] = n; p->hq_n++;
}
/* np.concatenate(list(_audio_hist)) into lin; returns length or -1 */
static long hist_concat(css_pll *p)
{
    long i;
    if (p->hist_n > p->lin_cap) return -1;
    for (i = 0; i < p->hist_n; i++) p->lin[i] = p->hist[(p->hist_head + i) % p->hist_cap];
    return p->hist_n;
}
static void snr_push(css_pll *p, double v)
{
    if (p->snrw_n < p->snrw_cap) { p->snrw[(p->snrw_head + p->snrw_n) % p->snrw_cap] = v; p->snrw_n++; }
    else { p->snrw[p->snrw_head] = v; p->snrw_head = (p->snrw_head + 1) % p->snrw_cap; }
}

/* ------------------------------------------------------------ do_resync */
static void do_resync(css_pll *p, int reason, int reset_timing)
{
    ev(p, CSS_EV_RESYNC, reset_timing, reason, 0, 0);
    p->sync_found = 0;
    p->bs_n = 0;
    p->consec_errors = 0;
    p->timing_settled = !reset_timing;
    p->acq_left = 0;
    p->last_resync_sym = p->sym_count;
    p->last_resync_soft = !reset_timing;
    terr_clear(p);
    fa_clear(p);
    p->have_snap = 0;
    p->one_shot = 0;
    p->step_ref_ok = 0; p->step_cand = 0; p->step_fix_pending = 0;
    if (reset_timing) {
        p->peak_est = p->expected_peak;
        p->timing_frac = 0.0;
        p->n_read = p->c.sps_eff;
        p->rate_est = 0.0;
        acqdev_clear(p);
    }
    css_syncdet_clear(p->sd);
    ps_clear(p);
}

/* -------------------------------------------------------- frame pipeline */
/* _frame_pipeline_c + rs_decode_frame: returns ok; fills seq/data/rs_err;
 * *rs_fail = 1 when RS raised. data points into rsout or rawb. */
static int pipeline(css_pll *p, const unsigned char *bits, const double *conf,
                    int *seq, const unsigned char **data, int *rs_err, int *rs_fail)
{
    const css_pll_cfg *c = &p->c;
    int n = c->frame_symbols, i, nraw, ne = 0, crcb = c->crc_bytes ? c->crc_bytes : 4,
        need = 1 + c->chunk_size + crcb, s = 0, ok;
    const unsigned char *raw;
    *seq = 0; *data = p->rawb; *rs_err = 0; *rs_fail = 0;
    for (i = 0; i < n; i++) p->fc32[i] = (float)conf[i];
    nraw = css_dec_raw(&p->g, bits, p->fc32, n, c->preamble[c->n_preamble - 1],
                       p->decw, p->decw_len, p->rawb, c->base_bits / 8 + 8, p->bconf);
    if (nraw < 0) return 0;
    raw = p->rawb;
    if (c->rs_enabled && p->rs) {
        int errc = 0, got;
        if (c->era_thr > 0)
            for (i = 0; i < nraw; i++) if ((double)p->bconf[i] < c->era_thr) p->era[ne++] = i;
        got = rs255_decode_era(p->rs, p->rawb, nraw, p->era, ne, p->rsout,
                               c->base_bits / 8 + 8, &errc);
        if (got < 0) { *rs_fail = 1; nraw = 0; }
        else { *rs_err = errc; raw = p->rsout; nraw = got; }
    }
    if (nraw < need) { *data = raw; return 0; }
    ok = css_check_crc_n(raw, nraw, c->chunk_size, crcb, &s) == 1;
    *seq = ok ? s : raw[0];
    *data = raw + 1;
    return ok;
}

static void emit_frame(css_pll *p, int seq, const unsigned char *data,
                       double snr, int rs_err, int src)
{
    css_pll_frame f;
    f.sym = p->sym_count; f.seq = seq; f.data = data; f.ndata = p->c.chunk_size;
    f.snr_db = snr; f.rs_errors = rs_err; f.source = src;
    if (p->cb.on_frame) p->cb.on_frame(p->cb.user, &f);
}

/* _anchor_decode over the current _audio_hist; ok -> seq/data */
static int anchor_decode(css_pll *p, int *seq, const unsigned char **data)
{
    const css_pll_cfg *c = &p->c;
    long N;
    int j, M;
    if (!(p->anchor_enabled && p->hq_n >= c->frame_symbols + 2)) return 0;
    N = hist_concat(p);
    if (N < 0) return 0;
    M = css_retro_anchor_prepare(p->rt, p->lin, N, c->sps_eff, p->rate_est, c->frame_symbols);
    if (M < 0) return 0;
    for (j = 0; j <= 2 * p->max_shift; j++) {
        int shift = j == 0 ? 0 : ((j % 2) ? -((j + 1) / 2) : (j / 2));
        double t0, sf; int no, rse, rsf;
        if (css_retro_anchor_candidate(p->rt, shift, p->cb_bits, p->cc, &t0, &sf, &no)) continue;
        if (pipeline(p, p->cb_bits, p->cc, seq, data, &rse, &rsf)) return 1;
    }
    return 0;
}

/* ---------------------------------------------------------------- symbol */
int css_pll_next_nr(const css_pll *p) { return p->n_read + p->one_shot; }

static double psum_sq(const css_real *a, long n)
{
    if (n < 8) {
        long i; double res = 0.0;
        for (i = 0; i < n; i++) { css_real s = a[i] * a[i]; res += (double)s; }
        return res;
    } else if (n <= 128) {
        double r[8], res; long i; int k;
        for (k = 0; k < 8; k++) { css_real s = a[k] * a[k]; r[k] = (double)s; }
        for (i = 8; i < n - (n % 8); i += 8)
            for (k = 0; k < 8; k++) { css_real s = a[i + k] * a[i + k]; r[k] += (double)s; }
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) { css_real s = a[i] * a[i]; res += (double)s; }
        return res;
    } else {
        long n2 = n / 2; n2 -= n2 % 8;
        return psum_sq(a, n2) + psum_sq(a + n2, n - n2);
    }
}

/* Pre-sync full-period search: a peak found more than half a period early is
 * the same chirp phase as one that far late (not yet in the ring); steer the
 * short way round. */
static int presync_wrap(const css_pll *p, int e)
{
    int P = p->c.sps_eff;
    if (p->sync_found || !p->c.presync_full_period) return e;
    if (e < -(P / 2)) e += P;
    else if (e > P / 2) e -= P;
    return e;
}

int css_pll_symbol(css_pll *p, const float *raw_in, int nr)
{
    const css_pll_cfg *c = &p->c;
    css_corr_out mf;
    css_pll_sym S;
    int i, n_blanked = 0, whitened = 0, signal_present, abs_ok, tim_snr_ok;
    int clamped_err = 0, gate_q, outlier = 0, wpk_strong = 1;
    int step_sym = 0;                      /* 1 = step candidate, 2 = confirmed */
    double abs_conf, pll_conf = 0.0, pll_r2 = 0.0, final_conf, rms, msq;
    long abs_start = p->abs_pos;
    float *raw = p->raw;

    if (nr != css_pll_next_nr(p)) return -1;
    if (nr > p->nr_max) { ev(p, CSS_EV_NR_CLAMP, nr, p->nr_max, 0, 0); return -2; }
    p->one_shot = 0;                       /* _one_shot_correction = 0 */
    p->abs_pos += nr;
    for (i = 0; i < nr; i++) raw[i] = raw_in[i];

    /* §K: raw (pre-AGC, pre-blank) into _audio_hist */
    hist_append(p, raw, nr);

    /* §L: deferred anchor re-decode */
    if (p->anchor_retry > 0) {
        p->anchor_retry--;
        if (p->anchor_retry == 0) {
            int sq; const unsigned char *dat;
            if (anchor_decode(p, &sq, &dat) && !p->decoded_seqs[sq & 255]) {
                emit_frame(p, sq, dat, p->anchor_retry_snr, 0, CSS_PLL_SRC_DEFERRED);
                p->frames_decoded++;
                p->frames_errored = p->frames_errored > 0 ? p->frames_errored - 1 : 0;
                p->decoded_seqs[sq & 255] = 1;
            }
        }
    }

    /* §M: impulse blanker (float32 throughout, thr compared as float32) */
    {
        float med;
        for (i = 0; i < nr; i++) p->agc[i] = raw[i] < 0 ? -raw[i] : raw[i];
        int silent;
        med = css_median_f(p->agc, (size_t)nr);
        /* digital silence neither seeds nor moves the baseline */
        silent = (double)med < c->impulse_silence_floor;
        if (!p->have_bg && !silent) { p->bg_amp_ema = (double)med; p->have_bg = 1; }
        if (c->impulse_blank_enable && p->have_bg) {
            double thr = c->impulse_blank_k * p->bg_amp_ema + 1e-9;
            float thr_f;
            double frac;
            if (c->impulse_clip_floor < thr) thr = c->impulse_clip_floor;
            thr_f = (float)thr;
            for (i = 0; i < nr; i++) {
                float a = raw[i] < 0 ? -raw[i] : raw[i];
                if (a > thr_f) n_blanked++;
            }
            frac = (double)n_blanked / (double)nr;
            if (n_blanked > 0 && frac > c->impulse_blank_max_frac) {
                ev(p, CSS_EV_IMPULSE_BLANK, n_blanked, 1, thr, 0);
                n_blanked = 0;
            } else if (n_blanked > 0) {
                for (i = 0; i < nr; i++) {
                    float a = raw[i] < 0 ? -raw[i] : raw[i];
                    if (a > thr_f) raw[i] = 0.0f;
                }
                ev(p, CSS_EV_IMPULSE_BLANK, n_blanked, 0, thr, 0);
            }
        }
        if (p->have_bg && !silent
                && (p->bg_amp_ema == 0 || (double)med < 3.0 * p->bg_amp_ema))
            p->bg_amp_ema += c->impulse_bg_alpha * ((double)med - p->bg_amp_ema);
    }

    /* §N: whitening */
    if (p->wh) {
        whitened = css_whiten_run(p->wh, raw, nr, p->wraw);
        if (whitened < 0) { whitened = 0; for (i = 0; i < nr; i++) p->wraw[i] = (css_real)raw[i]; }
    } else {
        for (i = 0; i < nr; i++) p->wraw[i] = (css_real)raw[i];
    }

    /* §O: RMS, signal_present, ring, AGC apply */
    msq = psum_sq(p->wraw, nr) / (double)nr;
    rms = sqrt(msq);
    p->d.rms_sum += rms; p->d.rms_n++;
    if (rms > p->d.rms_max) p->d.rms_max = rms;
    signal_present = rms > c->silence_thr;
    {
        int keep = c->buf_size - nr;
        if (keep > 0) {
            for (i = 0; i < keep; i++) p->ring[i] = p->ring[i + nr];
        } else keep = 0;
        for (i = 0; i < nr; i++) {
            float v;
            if (whitened) {
                css_real m = p->wraw[i] * (css_real)p->agc_gain;
                v = (float)(m < -1 ? -1 : (m > 1 ? 1 : m));
            } else {
                float m = raw[i] * (float)p->agc_gain;
                v = m < -1.0f ? -1.0f : (m > 1.0f ? 1.0f : m);
            }
            p->agc[i] = v;
        }
        if (keep > 0) for (i = 0; i < nr; i++) p->ring[keep + i] = p->agc[i];
        else for (i = 0; i < c->buf_size; i++) p->ring[i] = p->agc[nr - c->buf_size + i];
    }
    p->sym_count++;
    p->d.symbols = p->sym_count;
    if (p->sync_found) {
        /* one chirp+gap per chunk: take it fa_delay back from the ring end */
        int dly = c->buf_size - 1 - p->expected_peak - c->gap_samps;
        if (dly > 0 && dly + nr <= c->buf_size) fa_append(p, p->ring + c->buf_size - dly - nr, nr);
        else fa_append(p, p->agc, nr);
    }

    /* §P: MF decision + IF slope */
    if (!p->sync_found && c->presync_full_period) {
        /* one whole symbol period ending at the ring end: every onset phase is
         * visible, not just the +-search_range around peak_est */
        int half = c->sps_eff / 2;
        css_corr_decide(p->corr, p->ring, c->buf_size, c->buf_size - half, half, &mf);
    } else {
        css_corr_decide(p->corr, p->ring, c->buf_size, p->peak_est,
                        (p->timing_settled && p->sync_found && p->acq_left == 0)
                            ? c->post_acq_search_smp : -1, &mf);
    }
    /* §P2: phase-step recovery while synced.  Tracking (settled): the narrow
     * window lost the chirp (snr drop) -> search +-search_range.  Acquisition:
     * the +-search_range window already sees it, so a strong peak more than
     * post_acq_search from expected is the candidate. */
    if (c->step_recover && p->sync_found && signal_present) {
        int cand = 0, have_w = 0, tracking = p->timing_settled && p->acq_left == 0;
        css_corr_out w;
        if (p->step_ref_ok) {
            if (tracking && mf.snr_db < p->step_ref_snr - c->step_drop_db) {
                css_corr_decide(p->corr, p->ring, c->buf_size, p->expected_peak, -1, &w);
                have_w = 1;
            } else if (!tracking
                       && iabs(mf.found_peak - p->expected_peak) > c->post_acq_search_smp) {
                w = mf;
                have_w = 1;
            }
        }
        if (have_w) {
            int d = w.found_peak - p->expected_peak;
            if (iabs(d) > c->post_acq_search_smp / 2 && fabs(w.conf) >= c->step_min_conf
                    && w.snr_db >= p->step_ref_snr - c->step_accept_db) {
                if (p->step_cand && iabs(d - p->step_cand_d) <= c->step_tol_smp) {
                    ev(p, CSS_EV_PHASE_STEP, d, 1, w.snr_db, mf.snr_db);
                    mf = w;
                    step_sym = 2;
                    p->one_shot = d;
                    p->peak_est = p->expected_peak;
                    terr_clear(p);
                    acqdev_clear(p);
                    p->step_fix_pending = 1;
                    p->step_fix_idx = p->step_cand_idx;
                    p->step_fix_conf = p->step_cand_conf;
                } else {
                    ev(p, CSS_EV_PHASE_STEP, d, 0, w.snr_db, mf.snr_db);
                    cand = 1;
                    step_sym = 1;
                    p->step_cand_d = d;
                    p->step_cand_conf = w.conf;
                }
            }
        }
        p->step_cand = cand;
        if (!step_sym) {
            if (!p->step_ref_ok) { p->step_ref_snr = mf.snr_db; p->step_ref_ok = 1; }
            else if (mf.snr_db >= p->step_ref_snr - c->step_drop_db)
                p->step_ref_snr += 0.1 * (mf.snr_db - p->step_ref_snr);
        }
    }
    abs_conf = fabs(mf.conf);
    if (c->pll_weight > 0 && signal_present) {
        int s0 = mf.found_peak - c->sps + 1, s1 = mf.found_peak + 1;
        if (s0 < 0) s0 = 0;
        if (s1 > c->buf_size) s1 = c->buf_size;
        if (s1 > s0)
            css_ifslope_run(p->ifs, p->ring + s0, s1 - s0, c->f_min, c->f_max,
                            c->sps, c->fs, &pll_conf, &pll_r2);
    }

    /* §Q: measurement, acquisition peak_est IIR */
    tim_snr_ok = mf.snr_db >= c->timing_min_snr_db;
    abs_ok = abs_conf >= c->min_tim_conf;
    gate_q = signal_present && abs_ok && tim_snr_ok && !step_sym;
    if (gate_q) {
        int raw_err = presync_wrap(p, mf.found_peak - p->expected_peak);
        int eff = p->timing_settled ? c->max_terr_smp : c->presync_max_terr_smp;
        clamped_err = raw_err < -eff ? -eff : (raw_err > eff ? eff : raw_err);
        if (p->sync_found && (!p->timing_settled || p->acq_left > 0))
            p->peak_est = (int)css_iround((1.0 - c->tim_track_alpha) * p->peak_est
                                          + c->tim_track_alpha * (p->expected_peak + clamped_err));
        p->timing_err = clamped_err;
    } else {
        p->timing_err = step_sym ? 0 : presync_wrap(p, mf.found_peak - p->expected_peak);
    }

    snr_push(p, mf.snr_db);
    p->d.snr_sum += mf.snr_db; p->d.snr_n++;
    /* §S: AGC update, acquisition countdown, rate capture */
    if (signal_present && mf.winner_pk >= 1e-9) {
        double desired = c->agc_target / mf.winner_pk;
        double a = desired < p->agc_gain ? c->agc_attack : c->agc_release;
        p->agc_gain += a * (desired - p->agc_gain);
        if (p->agc_gain < 1.0 / c->agc_max) p->agc_gain = 1.0 / c->agc_max;
        if (p->agc_gain > c->agc_max) p->agc_gain = c->agc_max;
    }
    if (p->sync_found && p->acq_left > 0) {
        p->acq_left--;
        if (p->acq_left == 0) {
            p->timing_settled = 1;
            p->peak_est = p->expected_peak;
            if (p->acqdev_n) {
                double tmp[5]; int k;
                for (k = 0; k < p->acqdev_n; k++) tmp[k] = (double)p->acqdev[(p->acqdev_head + k) % 5];
                p->rate_est = css_pmean(tmp, (size_t)p->acqdev_n);
            }
            p->timing_frac = 0.0;
            terr_clear(p);
        }
    }

    /* §T: outlier gate, strength gate */
    if (p->sync_found && p->terr_n > 0) {
        int med = terr_median(p);
        int thr = (p->timing_settled && p->acq_left == 0) ? c->post_acq_search_smp / 2
                                                          : c->post_acq_search_smp;
        if (iabs(p->timing_err - med) > thr) {
            outlier = 1;
            ev(p, CSS_EV_PLL_OUTLIER, p->timing_err, med, thr, 0);
        }
    }
    if (!p->have_wpk) { p->wpk_ema = mf.winner_pk; p->have_wpk = 1; }
    if (c->wpk_track_enable && p->sync_found && p->timing_settled && p->acq_left == 0)
        wpk_strong = mf.winner_pk >= c->wpk_track_k * p->wpk_ema;
    if (!(p->timing_settled && p->acq_left == 0) || !wpk_strong)
        p->wpk_ema += c->wpk_track_alpha * (mf.winner_pk - p->wpk_ema);

    /* §U: timing PLL update, else revert */
    if (gate_q && !outlier && wpk_strong) {
        double cur_alpha, step, max_step, max_frac;
        int filt, int_corr, rate_int;
        if (p->sync_found && p->acq_left > 0) cur_alpha = c->alpha_acq;
        else if (p->sync_found)               cur_alpha = c->alpha;
        else                                  cur_alpha = c->alpha_acq;
        terr_push(p, p->timing_err);
        filt = terr_median(p);
        if (p->timing_settled && p->acq_left == 0)
            p->peak_est = (int)css_iround((1.0 - c->tim_track_alpha) * p->peak_est
                                          + c->tim_track_alpha * (p->expected_peak + clamped_err));
        max_step = c->sps * c->timing_max_step_frac;
        step = cur_alpha * filt;
        if (step < -max_step) step = -max_step;
        if (step > max_step) step = max_step;
        p->timing_frac += step;
        max_frac = (p->timing_settled && p->sync_found && p->acq_left == 0)
                       ? (double)c->post_acq_search_smp : (double)(c->sps / 4);
        if (p->timing_frac < -max_frac) p->timing_frac = -max_frac;
        if (p->timing_frac > max_frac) p->timing_frac = max_frac;
        int_corr = (int)p->timing_frac;            /* int(): toward zero */
        p->timing_frac -= int_corr;
        rate_int = (int)css_iround(p->rate_est);
        p->n_read = c->sps_eff + rate_int + int_corr;
        if (p->n_read < c->sps / 2) p->n_read = c->sps / 2;
        if (p->sync_found && p->timing_settled && p->acq_left == 0 && c->rate_refine_alpha > 0)
            p->rate_est += c->rate_refine_alpha * int_corr;
        if (p->sync_found && p->acq_left > 0)
            acqdev_push(p, p->n_read - c->sps_eff);
        if (p->sync_found && p->acq_left == 0 && p->timing_settled
                && iabs(p->timing_err) < c->post_acq_search_smp / 2
                && abs_conf > 0.55 && mf.snr_db > c->timing_min_snr_db + 3.0) {
            p->have_snap = 1; p->snap_tf = p->timing_frac;
            p->snap_nr = p->n_read; p->snap_pe = p->peak_est;
        }
    } else {
        int target = c->sps_eff + (int)css_iround(p->rate_est);
        p->n_read = (int)css_iround(p->n_read + c->nread_revert * (target - p->n_read));
    }

    /* §W: confidence blend */
    final_conf = mf.conf;
    if (c->pll_weight > 0 && signal_present && pll_r2 > 0.10) {
        double w = c->pll_weight * pll_r2;
        final_conf = (1 - w) * mf.conf + w * pll_conf;
    }
    if (!tim_snr_ok) final_conf = 0.0;

    /* trace record */
    S.sym = p->sym_count; S.abs_start = abs_start; S.nr = nr;
    S.found_peak = mf.found_peak; S.bit = mf.bit; S.conf = mf.conf;
    S.snr_db = mf.snr_db; S.winner_pk = mf.winner_pk; S.pll_conf = pll_conf; S.pll_r2 = pll_r2;
    S.timing_err = p->timing_err; S.peak_est = p->peak_est; S.n_read = p->n_read;
    S.acq_left = p->acq_left; S.settled = p->timing_settled; S.sync_found = p->sync_found;
    S.rate_est = p->rate_est; S.timing_frac = p->timing_frac; S.final_conf = final_conf;
    S.agc_gain = p->agc_gain; S.outlier = outlier; S.wpk_strong = wpk_strong;
    S.signal_present = signal_present; S.n_blanked = n_blanked; S.whitened = whitened;
    if (p->cb.on_symbol) p->cb.on_symbol(p->cb.user, &S);

    /* transmission onset anchors the early-sync guard (decode() onset guard) */
    {
        /* "silent" for the onset guard is relative to the room (2026-11-17: a MacBook mic floor of
         * -55..-44 dBFS sat over silence_thr 0.002, no symbol was ever silent, the guard never anchored
         * and the training run-in locked at ~0.5, 3-4 s early -> FRAME_ERROR) */
        double othr = c->silence_thr;
        int onset_sig;
        if (c->onset_rel_db > 0.0 && p->have_onset_floor) {
            double rel = p->onset_floor * pow(10.0, c->onset_rel_db / 20.0);
            if (rel > othr) othr = rel;
        }
        onset_sig = rms > othr;
        if (!p->sync_found && c->onset_guard_silence_syms > 0) {
            if (signal_present && p->quiet_run >= c->onset_guard_silence_syms) {
                p->last_resync_sym = p->sym_count - 2;     /* silence -> signal: hard anchor */
                p->last_resync_soft = 0;
            } else if (onset_sig && p->quiet_rel_run >= c->onset_guard_silence_syms) {
                /* room-relative onset: soft, so a high-score or non-alternating (real preamble)
                 * window stays eligible -- noise can pass floor+rel mid-burst at low SNR */
                p->last_resync_sym = p->sym_count - 2;
                p->last_resync_soft = 1;
            }
        }
        p->quiet_run = signal_present ? 0 : p->quiet_run + 1;
        p->quiet_rel_run = onset_sig ? 0 : p->quiet_rel_run + 1;
        if (!p->sync_found) {
            if (!p->have_onset_floor || rms < p->onset_floor) { p->onset_floor = rms; p->have_onset_floor = 1; }
            else p->onset_floor *= pow(10.0, c->onset_floor_rise_db / 20.0);
        }
    }

    /* §X: preamble detection */
    if (!p->sync_found) {
        double snr_w, score, eff_thr, alt;
        int acquired, min_pre_end, looks_training, trust_early;
        long window_end_rel;
        if (iabs(p->timing_err) <= c->syndet_flush_terr) p->timing_settled = 1;
        ps_push(p, final_conf > 0 ? 1 : 0, final_conf);
        snr_w = (mf.snr_db - c->syndet_min_snr_db)
                / (c->syndet_snr_ref_db - c->syndet_min_snr_db > 1.0
                       ? c->syndet_snr_ref_db - c->syndet_min_snr_db : 1.0);
        if (snr_w < 0.0) snr_w = 0.0;
        if (snr_w > 1.0) snr_w = 1.0;
        min_pre_end = c->training_symbols + c->n_preamble - 1;
        if (c->syndet_mask_ineligible)
            score = css_syncdet_push_masked(p->sd, final_conf * snr_w, &acquired,
                                            c->syndet_lag_guard ? c->syndet_max_lag : -1,
                                            1, p->sym_count - p->last_resync_sym - min_pre_end,
                                            p->last_resync_soft, c->sync_threshold,
                                            c->syndet_alt_thr);
        else
            score = css_syncdet_push(p->sd, final_conf * snr_w, &acquired);
        eff_thr = p->syndet_bg_ema + c->syndet_margin;
        if (eff_thr < c->syndet_thr_floor) eff_thr = c->syndet_thr_floor;
        if (eff_thr > c->sync_threshold) eff_thr = c->sync_threshold;
        acquired = score >= eff_thr;
        if (c->syndet_mask_ineligible && !acquired && css_syncdet_raw_reason(p->sd)
                && css_syncdet_raw_score(p->sd) >= eff_thr) {
            /* diagnostics only: the overall best window was left out */
            int rl = css_syncdet_raw_lag(p->sd);
            if (css_syncdet_raw_reason(p->sd) == 1)
                ev(p, CSS_EV_SYNC_STALE_REJECT, rl, 1, css_syncdet_raw_score(p->sd), score);
            else
                ev(p, CSS_EV_SYNC_TOO_EARLY, (int)(p->sym_count - rl),
                   (int)(p->sym_count - rl - p->last_resync_sym), css_syncdet_raw_score(p->sd), -2.0);
        }
        if (acquired && c->syndet_lag_guard && !c->syndet_mask_ineligible
                && css_syncdet_best_lag(p->sd) > c->syndet_max_lag) {
            ev(p, CSS_EV_SYNC_STALE_REJECT, css_syncdet_best_lag(p->sd), 0, score, 0);
            acquired = 0;
        }
        window_end_rel = (p->sym_count - css_syncdet_best_lag(p->sd)) - p->last_resync_sym;
        alt = css_syncdet_alternation(p->sd);
        looks_training = alt >= 0.0 && alt >= c->syndet_alt_thr;
        trust_early = p->last_resync_soft && (score >= c->sync_threshold || !looks_training);
        if (acquired && window_end_rel < min_pre_end && !trust_early && !c->syndet_mask_ineligible) {
            ev(p, CSS_EV_SYNC_TOO_EARLY, (int)(p->sym_count - css_syncdet_best_lag(p->sd)),
               (int)window_end_rel, score, alt);
            acquired = 0;
        }
        if (!acquired) {
            p->syndet_bg_ema += c->syndet_bg_alpha * (score - p->syndet_bg_ema);
            if (p->syndet_bg_ema < 0.0) p->syndet_bg_ema = 0.0;
        }
        if (acquired) {
            int lag = css_syncdet_best_lag(p->sd);
            p->sync_found = 1;
            p->step_ref_ok = 0; p->step_cand = 0; p->step_fix_pending = 0;
            p->acq_left = c->timing_acq_symbols;
            p->timing_settled = 0;
            if (iabs(p->timing_err) > c->post_acq_search_smp) {
                int snapped = p->expected_peak + p->timing_err;
                if (snapped < 0) snapped = 0;
                if (snapped > c->buf_size - 1) snapped = c->buf_size - 1;
                ev(p, CSS_EV_SYNC_SNAP, p->peak_est, snapped, 0, 0);
                p->peak_est = snapped;
            }
            terr_clear(p);
            acqdev_clear(p);
            p->bs_n = 0;
            if (lag > 0) {
                int take = lag < p->ps_n ? lag : p->ps_n, k;
                for (k = 0; k < take; k++) {
                    int idx = (p->ps_head + p->ps_n - take + k) % p->ps_cap;
                    p->bs_b[k] = p->ps_b[idx]; p->bs_c[k] = p->ps_c[idx];
                }
                p->bs_n = take;
            }
            p->d.sync_acquired = 1;
            p->d.sync_at_symbol = p->sym_count - lag;
            p->d.sync_score = score;
            p->d.sync_lag = lag;
            ev(p, CSS_EV_SYNC_ACQUIRED, lag, (int)(p->sym_count - lag), score, eff_thr);
        }
        return 0;
    }

    /* §Y: frame assembly */
    if (p->step_fix_pending) {
        /* confirmed step: the candidate symbol is re-decided where its chirp was */
        if (p->step_fix_idx >= 0 && p->step_fix_idx < p->bs_n) {
            p->bs_b[p->step_fix_idx] = p->step_fix_conf > 0 ? 1 : 0;
            p->bs_c[p->step_fix_idx] = p->step_fix_conf;
        }
        p->step_fix_pending = 0;
    }
    p->bs_b[p->bs_n] = final_conf > 0 ? 1 : 0;
    p->bs_c[p->bs_n] = final_conf;
    p->bs_n++;
    if (step_sym == 1) p->step_cand_idx = p->bs_n - 1;
    if (p->bs_n >= c->frame_symbols) {
        p->step_cand = 0;
        int fs = c->frame_symbols, k, used_retro = 0, ok, seq, rse, rsf;
        const unsigned char *dat;
        double frame_snr;
        unsigned char *sb = p->fb;        /* stream bits copy */
        double *scf = p->fc;
        unsigned char *rb = p->cb_bits;   /* final frame bits */
        double *rcf = p->cc;
        for (k = 0; k < fs; k++) { sb[k] = p->bs_b[k]; scf[k] = p->bs_c[k]; }
        for (k = fs; k < p->bs_n; k++) { p->bs_b[k - fs] = p->bs_b[k]; p->bs_c[k - fs] = p->bs_c[k]; }
        p->bs_n -= fs;
        if ((long)p->snrw_n >= fs) {
            /* np.mean(snr_history[-frame_symbols:]), oldest first; reuse the
             * stream-conf slot after the frame's own confs (bs_c has room) */
            double *ord = p->bs_c + (p->bs_cap - fs);
            for (k = 0; k < fs; k++) ord[k] = p->snrw[(p->snrw_head + k) % p->snrw_cap];
            frame_snr = css_pmean(ord, (size_t)fs);
        } else frame_snr = mf.snr_db;
        for (k = 0; k < fs; k++) { rb[k] = sb[k]; rcf[k] = scf[k]; }

        if (p->fal_n > 0) {
            int n_retro = p->fal_n < p->retro_max_syms ? p->fal_n : p->retro_max_syms;
            const float *ra = p->fa;
            long rN = 0;
            int have = 0, rbits_ok = 0;
            int end_off = c->gap_samps;     /* chirp-end distance from retro audio end */
            double t0 = 0, sf = 0; int nout = 0;
            {   /* retro_chunks = _frame_audio[-n_retro:] (contiguous tail) */
                long off = 0;
                for (k = 0; k < p->fal_n - n_retro; k++) off += p->fal[k];
                ra = p->fa + off; rN = p->fa_n - off;
            }
            /* CRC-anchored search first */
            if (p->anchor_enabled && p->hq_n >= fs + 2) {
                long N = hist_concat(p);
                if (N >= 0) {
                    int M = css_retro_anchor_prepare(p->rt, p->lin, N, c->sps_eff, p->rate_est, fs);
                    if (M >= 0) {
                        int j;
                        for (j = 0; j <= 2 * p->max_shift; j++) {
                            int shift = j == 0 ? 0 : ((j % 2) ? -((j + 1) / 2) : (j / 2));
                            int cseq, crse, crsf; const unsigned char *cdat;
                            if (css_retro_anchor_candidate(p->rt, shift, rb, rcf, &t0, &sf, &nout)) continue;
                            if (pipeline(p, rb, rcf, &cseq, &cdat, &crse, &crsf)) {
                                have = 1; ra = p->lin; rN = N; n_retro = fs;
                                end_off = c->buf_size - 1 - p->expected_peak;
                                ev(p, CSS_EV_RETRO_ANCHOR, shift, 0, 0, 0);
                                break;
                            }
                        }
                    }
                }
                if (!have) for (k = 0; k < fs; k++) { rb[k] = sb[k]; rcf[k] = scf[k]; }
            }
            if (!have) {
                int rc = css_retro_decode(p->rt, ra, rN, n_retro, c->sps_eff, p->rate_est,
                                          rb + (fs - n_retro), rcf + (fs - n_retro), &t0, &sf, &nout);
                if (rc == 0) have = 1;
                else for (k = 0; k < fs; k++) { rb[k] = sb[k]; rcf[k] = scf[k]; }
            }
            rbits_ok = have;
            if (rbits_ok) {
                double p_last = t0 + (double)(n_retro - 1) * sf;
                double delta = p_last - (double)(rN - end_off - 1);
                double drift = sf - c->sps_eff;
                int n_in = n_retro - nout;
                double in_frac = (double)n_in / (double)(n_retro > 1 ? n_retro : 1);
                int gate = 0;
                used_retro = 1;
                /* rb[:n_prefix] already holds the stream prefix */
                if (in_frac < 0.5) gate = 1;
                else if (fabs(delta) > (double)(c->post_acq_search_smp * 2)) gate = 2;
                if (gate) {
                    ev(p, CSS_EV_RETRO_GATE, n_in, n_retro, delta, (double)gate);
                    p->one_shot = 0;
                    p->acq_left = c->timing_acq_symbols;
                    p->timing_settled = 0;
                    terr_clear(p);
                    p->rate_est = 0.0;
                    acqdev_clear(p);
                } else {
                    double dc = delta;
                    int re = 0;
                    if (dc < -(c->sps / 2)) dc = -(c->sps / 2);
                    if (dc > c->sps / 2) dc = c->sps / 2;
                    p->one_shot = (int)css_iround(dc);
                    if (iabs(p->one_shot) > c->post_acq_search_smp) {
                        p->acq_left = c->timing_acq_symbols;
                        p->timing_settled = 0;
                        terr_clear(p);
                        re = 1;
                    }
                    if (in_frac >= c->retro_rate_adopt_inlier) p->rate_est = drift;
                    ev(p, CSS_EV_RETRO_SYNC, p->one_shot, re, drift, p->rate_est);
                }
                p->have_snap = 0;
            } else {
                if (p->have_snap) {
                    p->timing_frac = p->snap_tf;
                    p->n_read = p->snap_nr;
                    if (!p->timing_settled) p->peak_est = p->snap_pe;
                    ev(p, CSS_EV_SNAPSHOT_RESTORE, p->n_read, 0, 0, 0);
                    p->have_snap = 0;
                }
            }
        } else {
            if (p->have_snap) {
                p->timing_frac = p->snap_tf;
                p->n_read = p->snap_nr;
                if (!p->timing_settled) p->peak_est = p->snap_pe;
                ev(p, CSS_EV_SNAPSHOT_RESTORE, p->n_read, 1, 0, 0);
                p->have_snap = 0;
            }
        }

        /* §AB: FEC call, outcome, resync */
        fa_clear(p);
        ok = pipeline(p, rb, rcf, &seq, &dat, &rse, &rsf);
        if (!ok && used_retro) {
            int s2, r2, f2; const unsigned char *d2;
            int ok2 = pipeline(p, sb, scf, &s2, &d2, &r2, &f2);
            if (ok2) { ok = ok2; seq = s2; dat = d2; rse = r2; rsf = f2; }
            else {
                /* reference keeps the FIRST (retro) result's rs_fail state */
                int s1, r1, f1; const unsigned char *d1;
                pipeline(p, rb, rcf, &s1, &d1, &r1, &f1);
                seq = s1; dat = d1; rse = r1; rsf = f1;
            }
        }
        if (rsf && !ok) {
            p->frames_errored++;
            p->consec_errors++;
            ev(p, CSS_EV_FRAME_ERROR, seq, 1, frame_snr, 0);
            if (c->preamble_per_frame) do_resync(p, CSS_RS_PPF_RS, 0);
            else if (p->consec_errors >= c->resync_after) do_resync(p, CSS_RS_RS_FAIL, 1);
            return 0;
        }
        if (ok) {
            emit_frame(p, seq, dat, frame_snr, rse, used_retro ? CSS_PLL_SRC_RETRO : CSS_PLL_SRC_STREAM);
            p->frames_decoded++;
            p->consec_errors = 0;
            p->decoded_seqs[seq & 255] = 1;
            if (c->preamble_per_frame) do_resync(p, CSS_RS_PPF_OK, 0);
        } else {
            ev(p, CSS_EV_FRAME_ERROR, seq, 0, frame_snr, 0);
            p->frames_errored++;
            p->consec_errors++;
            if (p->anchor_enabled && p->anchor_lookahead > 0) {
                p->anchor_retry = p->anchor_lookahead;
                p->anchor_retry_snr = frame_snr;
            }
            if (c->preamble_per_frame) do_resync(p, CSS_RS_PPF_CRC, 0);
        }
        if (!c->preamble_per_frame && p->consec_errors >= c->resync_after)
            do_resync(p, CSS_RS_CRC_FAIL, 1);
    }
    return 0;
}

int css_pll_push(css_pll *p, const float *x, int n)
{
    int ran = 0, off = 0;
    while (off < n) {
        int need = css_pll_next_nr(p), take, i;
        if (need > p->nr_max) {
            ev(p, CSS_EV_NR_CLAMP, need, p->nr_max, 0, 0);
            p->one_shot = 0;
            if (p->n_read > p->nr_max) p->n_read = p->nr_max;
            need = css_pll_next_nr(p);
        }
        if (p->fifo_n >= need) {           /* an align shortened the next read */
            int rest = p->fifo_n - need;
            css_pll_symbol(p, p->fifo, need);
            for (i = 0; i < rest; i++) p->fifo[i] = p->fifo[need + i];
            p->fifo_n = rest;
            ran++;
            continue;
        }
        take = need - p->fifo_n;
        if (take > n - off) take = n - off;
        for (i = 0; i < take; i++) p->fifo[p->fifo_n + i] = x[off + i];
        p->fifo_n += take; off += take;
        if (p->fifo_n == need) {
            css_pll_symbol(p, p->fifo, need);
            p->fifo_n = 0;
            ran++;
        }
    }
    return ran;
}

int css_pll_align(css_pll *p, long abs_peak, double period)
{
    long P = (long)(period > 0 ? period + 0.5 : p->c.sps_eff), d;
    long nr, target;
    if (p->sync_found) return 0;
    nr = p->n_read;
    /* after the next symbol, ring index expected_peak holds RX sample target */
    target = p->abs_pos + nr - p->c.buf_size + p->expected_peak;
    d = (abs_peak - target) % P;
    if (d < 0) d += P;
    if (d > P / 2) d -= P;
    if (nr + d < p->c.sps / 2) d += P;
    p->one_shot = (int)d;
    p->peak_est = p->expected_peak;
    p->timing_frac = 0.0;
    terr_clear(p);
    return (int)d;
}

int css_pll_mark_onset(css_pll *p, long abs_onset)
{
    long ahead;
    if (p->sync_found) return 0;
    /* symbol index (sym_count units) whose read contains abs_onset */
    ahead = (abs_onset - p->abs_pos) / p->c.sps_eff;
    /* 2 symbols of slack: the tracker can see a burst's first chirp a symbol
     * late and the mapping floors; training-sidelobe matches (what the guard
     * rejects) sit many symbols earlier than the real preamble end */
    p->last_resync_sym = p->sym_count + (ahead > 0 ? ahead : 0) - 2;
    p->last_resync_soft = 0;          /* a fresh transmission: training first */
    return 1;
}

void css_pll_rearm(css_pll *p)
{
    if (p->sync_found || p->bs_n > 0) do_resync(p, CSS_RS_REARM, 0);
}

void css_pll_skip(css_pll *p, long n)
{
    if (n <= 0) return;
    css_pll_rearm(p);
    p->fifo_n = 0;
    p->abs_pos += n;
}

void css_pll_get_diag(const css_pll *p, css_pll_diag *d)
{
    *d = p->d;
    d->frames_decoded = p->frames_decoded;
    d->frames_errored = p->frames_errored;
    d->sync_found = p->sync_found;
    d->timing_settled = p->timing_settled;
    d->acq_left = p->acq_left;
    d->rate_est = p->rate_est;
    d->agc_gain = p->agc_gain;
    d->peak_est = p->peak_est;
    d->n_read = p->n_read;
}
