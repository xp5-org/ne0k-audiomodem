/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
/* css_pllaux.h — the streaming PLL decoder's helper stages:
 *   css_syncdet   port of css_syncdet.SoftPreambleDetector (float32, as the
 *                 reference's np.float32 history is in both SIG modes)
 *   css_ifslope   port of css_fec.if_slope_confidence (double; arbitrary
 *                 length via Bluestein)
 *   css_whiten    port of decode()'s STFT-median noise whitening
 * Freestanding, caller memory. */
#ifndef CSS_PLLAUX_H
#define CSS_PLLAUX_H
#include "css_real.h"
#include "css_pfft.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- SoftPreambleDetector ------------------------------------------------ */
typedef struct css_syncdet css_syncdet;
size_t css_syncdet_workspace(int n, int lookback);
int    css_syncdet_init(css_syncdet **d, void *mem, size_t memlen,
                        const unsigned char *preamble, int n, int lookback,
                        double threshold, double conf_floor, double cov_knee);
/* returns the best cosine score (0 while filling); *acquired = score>=thr.
 * Plain argmax over every window of the ring. */
double css_syncdet_push(css_syncdet *d, double conf, int *acquired);
/* push(conf, lag_max, early_lag, soft, trust_thr, alt_thr) of the reference:
 * windows older than lag_max (< 0: no limit) are stale; windows older than
 * early_lag (use_early) count only when soft and (score >= trust_thr or
 * alternation < alt_thr).  Returns the best ELIGIBLE score (0 when none). */
double css_syncdet_push_masked(css_syncdet *d, double conf, int *acquired,
                               int lag_max, int use_early, long early_lag,
                               int soft, double trust_thr, double alt_thr);
/* unmasked best of the last push: score, lag, reason 0 ok / 1 stale / 2 early */
double css_syncdet_raw_score(const css_syncdet *d);
int    css_syncdet_raw_lag(const css_syncdet *d);
int    css_syncdet_raw_reason(const css_syncdet *d);
int    css_syncdet_best_lag(const css_syncdet *d);
/* fraction of sign changes in the winning window; < 0 means None */
double css_syncdet_alternation(const css_syncdet *d);
void   css_syncdet_clear(css_syncdet *d);
int    css_syncdet_lookback(const css_syncdet *d);

/* ---- if_slope_confidence --------------------------------------------------- */
typedef struct css_ifslope css_ifslope;
size_t css_ifslope_workspace(int nmax);
int    css_ifslope_init(css_ifslope **s, void *mem, size_t memlen, int nmax);
void   css_ifslope_run(css_ifslope *s, const float *seg, int N, double f_min,
                       double f_max, int sps, int fs, double *conf, double *r2);

/* ---- whitening ------------------------------------------------------------- */
typedef struct css_whiten css_whiten;
/* win_max: cap on the STFT window in samples (0 = none) */
size_t css_whiten_workspace(int fs, double stft_win_s, int win_max, int taps, int nr_max);
int    css_whiten_init(css_whiten **w, void *mem, size_t memlen, int fs,
                       double stft_win_s, int win_max, double bg_alpha, int redesign_every,
                       int taps, double max_gain_db, int nr_max);
/* raw[nr] (float32 audio) -> out[nr]. Returns 1 when the FIR ran (out is SIG
 * data), 0 when it passed raw through (the reference keeps float32 then, which
 * changes the precision of the AGC multiply), -1 if nr > nr_max. */
int    css_whiten_run(css_whiten *w, const float *raw, int nr, css_real *out);

#ifdef __cplusplus
}
#endif
#endif
