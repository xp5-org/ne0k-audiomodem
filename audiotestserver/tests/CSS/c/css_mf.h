/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
/* css_mf.h — segmented chirp matched filter, freestanding C99.
 *
 * The decoder's hot inner loop (~97% of decode CPU), written as the shipping
 * artifact for the embedded target:
 *   - no malloc / free: all state lives in one caller-provided block
 *   - no stdio, no OS calls, no globals
 *   - float32 throughout, matching the validated Python signal path
 * The host build is loaded in-process via ctypes so the regression suite
 * exercises this exact object code.
 */
#ifndef CSS_MF_H
#define CSS_MF_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float conf;       /* (du-dd)/(du+dd) on-grid; 0 when nothing was detected */
    int   peak;       /* argmax lag within the wide track window              */
    float snr_db;
    float winner_pk;
} css_mf_out;

typedef struct css_mf css_mf;

/* Bytes of caller memory needed. Same arguments as css_mf_init. */
size_t css_mf_workspace(int sps, int track_search, int seg_M, int seg_trim,
                        int max_win);

/* Carves `mem` into the plan. Returns 0 on success, negative on bad args.
 * up_ref/dn_ref are the FORWARD chirps, length >= sps (reversal is internal). */
int css_mf_init(css_mf **mf, void *mem, size_t memlen,
                const float *up_ref, const float *dn_ref,
                int sps, int track_search, int decide_search,
                int seg_M, int seg_trim, int max_win);

/* One symbol. `win` is n samples, `centre` the predicted peak index in win. */
int css_mf_run(css_mf *mf, const float *win, int n, int centre,
               css_mf_out *out);

/* Introspection for the host harness / sizing reports. */
int css_mf_nfft(const css_mf *mf);

#ifdef __cplusplus
}
#endif
#endif
