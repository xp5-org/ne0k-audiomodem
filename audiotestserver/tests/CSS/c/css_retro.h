/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
/* css_retro.h — port of css_retro.retro_timing_decode and
 * retro_offset_candidates (the streaming decoder's whole-frame timing refit
 * and its CRC-anchored whole-symbol shift search).  Freestanding, caller
 * memory; the full-buffer FFT is sized from n_max at init. */
#ifndef CSS_RETRO_H
#define CSS_RETRO_H
#include "css_real.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct css_retro css_retro;

size_t css_retro_workspace(int sps, long n_max, int frame_symbols_max);
int    css_retro_init(css_retro **r, void *mem, size_t memlen,
                      const float *up_ref, const float *dn_ref, int sps,
                      long n_max, int frame_symbols_max);

/* retro_timing_decode(audio[N], frame_symbols, sps_eff, rate_est).
 * Returns 0 and fills bits/confs/t0/sps_fit/n_out, or -1 for None,
 * -2 if N exceeds n_max. */
int    css_retro_decode(css_retro *r, const float *audio, long N,
                        int frame_symbols, int sps_eff, double rate_est,
                        unsigned char *bits, double *confs, double *t0,
                        double *sps_fit, int *n_out);

/* retro_offset_candidates: prepare once per buffer (returns M >=
 * frame_symbols, or -1 when the generator would yield nothing), then ask
 * for candidates by shift.  candidate() returns 0, or -1 when that shift's
 * window is out of range (the generator skips it). */
int    css_retro_anchor_prepare(css_retro *r, const float *audio, long N,
                                int sps_eff, double rate_est,
                                int frame_symbols);
int    css_retro_anchor_candidate(css_retro *r, int shift,
                                  unsigned char *bits, double *confs,
                                  double *t0, double *sps_fit, int *n_out);

#ifdef __cplusplus
}
#endif
#endif
