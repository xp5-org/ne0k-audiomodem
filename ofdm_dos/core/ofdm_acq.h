/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026 XP5.ORG
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/* ofdm_acq.h — OFDM sign-matched coherence acquisition, freestanding C99.
 *
 * This is the whole cost of OFDM acquisition (measured: 77% of
 * RobustLink.decode, 78% of Modem.analyze_probe, 40% of the handshake sim) and
 * it is written to be the SHIPPING artifact for the 32-bit target, not a host
 * accelerator:
 *   - no malloc / free: all state lives in one caller-provided block
 *   - no stdio, no OS calls, no globals, no statics
 *   - the H grid is a RING, so memory is O(1) in capture length
 * The host build is loaded in-process via ctypes (tests/OFDM/ofdm_acq_c.py) so
 * the regression suite exercises this exact object code.
 *
 * Why the ring matters.  The numpy ancestor materialised H for the whole
 * capture and then gathered (n_cand, nsyms, n_occupied) batches out of it:
 * 9.6 MB of grid plus a 77 MB gather for a 10 s capture at n_fft=1024, against
 * 520 KB of SRAM on an RP2350.  Candidate c only ever reads grid rows
 * c + j*stride for j < nsyms, so the grid never needs to be wider than
 * stride*(nsyms-1)+1 rows: 71 rows / 178 KB at the production geometry, and
 * constant in capture length.
 *
 * Precision.  The FFT and the per-carrier normalisation run in ofdm_real
 * (double by default, float under -DOFDM_ACQ_F32).  The ring and the metric
 * accumulators are ALWAYS float32, because that is what the numpy reference
 * does -- it stores the grid as complex64 and accumulates in float32 -- and
 * matching the reference is the point.  See ofdm_fft.h for the two-stage
 * validation argument.
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
