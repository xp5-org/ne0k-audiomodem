/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_acq.h — OFDM sign-matched coherence acquisition, freestanding C99.
 *
 * Measured: 77% of RobustLink.decode, 78% of Modem.analyze_probe, 40% of the
 * handshake sim. Written as the SHIPPING artifact for the 32-bit target, not
 * a host accelerator:
 *   - no malloc / free: all state lives in one caller-provided block
 *   - no stdio, no OS calls, no globals, no statics
 *   - the H grid is a RING, so memory is O(1) in capture length
 * The host build is loaded in-process via ctypes (tests/OFDM/ofdm_acq_c.py) so
 * the regression suite exercises this exact object code.
 *
 * Ring sizing. Candidate c only ever reads grid rows c + j*stride for
 * j < nsyms, so the grid never needs to be wider than stride*(nsyms-1)+1
 * rows: 71 rows / 178 KB at the production geometry (n_fft=1024), against
 * 9.6 MB for a fully materialised grid over a 10 s capture, and 520 KB of
 * SRAM on an RP2350.
 *
 * Precision. The FFT and the per-carrier normalisation run in ofdm_real
 * (double by default, float under -DOFDM_ACQ_F32). The ring and the metric
 * accumulators are ALWAYS float32, matching the numpy reference, which
 * stores the grid as complex64 and accumulates in float32. See ofdm_fft.h
 * for the two-stage validation argument.
 */
#ifndef OFDM_ACQ_H
#define OFDM_ACQ_H
#include <stddef.h>
#include "ofdm_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n_cand;      /* metric values written; 0 when the capture is too short */
    int n_pos;       /* grid positions walked                                  */
    int start_step;  /* candidate i starts at sample i*start_step              */
} ofdm_acq_info;

typedef struct ofdm_acq ofdm_acq;

/* Bytes of caller memory needed.  Same arguments as ofdm_acq_init.
 * Returns 0 for parameters ofdm_acq_init would reject. */
size_t ofdm_acq_workspace(int n_fft, int cp_len, int n_occupied,
                          int nsyms, int step, int top_k);

/* Carves `mem` into the plan.  Returns 0 on success, negative on bad args.
 *   known : n_occupied reference symbols; H = (X*scale)/known, as numpy does.
 *   q     : nsyms sign key, or NULL for an all-ones key (the periodic probe).
 * Both are copied into the workspace, so the caller may free them after.
 * Errors: -1 bad args / null, -2 memlen too small, -3 FFT init failed,
 *         -4 step does not divide (n_fft+cp_len). */
int ofdm_acq_init(ofdm_acq **a, void *mem, size_t memlen,
                  const double *known_re, const double *known_im,
                  const double *q,
                  int n_fft, int cp_len, int bin_lo, int n_occupied,
                  int nsyms, int step, int top_k);

/* Walk a whole capture.  Writes info->n_cand metric values to metric_out.
 * Errors: -1 bad args / null, -2 metric_cap smaller than n_cand. */
int ofdm_acq_run(ofdm_acq *a, const double *rx, int n,
                 double *metric_out, int metric_cap, ofdm_acq_info *info);

/* Introspection for the host harness / sizing reports. 0 on NULL. */
int ofdm_acq_ring_rows(const ofdm_acq *a);
int ofdm_acq_nfft(const ofdm_acq *a);

/* One H row, for stage-wise debugging: computes grid position `pos` and writes
 * n_occupied complex values as interleaved float32 re/im.  Lets the harness
 * diff the FFT + normalisation stage against numpy separately from the metric
 * reduction, which is the difference between "the port is wrong" and "float32
 * rounds differently".  Errors: -1 bad args, -2 pos out of range for n. */
int ofdm_acq_row(ofdm_acq *a, const double *rx, int n, int pos, float *out);

#ifdef __cplusplus
}
#endif
#endif
