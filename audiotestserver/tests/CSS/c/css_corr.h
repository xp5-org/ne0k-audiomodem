/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
/* css_corr.h — port of css_correlator.ChirpCorrelator (symbol_decision), the
 * per-symbol matched filter of the streaming PLL decoder.  NOT the same
 * algorithm as css_mf.c (that is MFSymbolFilter, the long-symbol decoder's).
 *
 * Both reference paths are ported: the 6-segment prefix-sum decomposition
 * (mf_window "none", sps >= 96) and the legacy per-slice transforms
 * (mf_window "hann", or sps < 96).  Freestanding, caller memory. */
#ifndef CSS_CORR_H
#define CSS_CORR_H
#include "css_real.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct css_corr css_corr;

typedef struct {
    int    bit;
    double conf;
    int    found_peak;
    double snr_db;
    double winner_pk;
} css_corr_out;

/* hann: 1 = mf_window "hann", 0 = "none". */
size_t css_corr_workspace(int sps, int buf_size, int hann);
int    css_corr_init(css_corr **c, void *mem, size_t memlen,
                     const float *up_ref, const float *dn_ref, int sps,
                     double search_frac, int buf_size, int hann);
/* buf[n] (n == buf_size), centre = predicted peak, sr_override < 0 = none. */
void   css_corr_decide(css_corr *c, const float *buf, int n, int centre,
                       int sr_override, css_corr_out *o);
int    css_corr_search_range(const css_corr *c);
int    css_corr_nfft(const css_corr *c);

#ifdef __cplusplus
}
#endif
#endif
