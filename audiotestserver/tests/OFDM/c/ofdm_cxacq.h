/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_cxacq.h — complex-baseband (I/Q) coherence acquisition, freestanding C89.
 *
 * The counterpart of ofdm_acq for an ofdm_cxcfg: walk a capture on a grid of
 * candidate starts and score how much each run of nsyms symbols spaced one
 * symbol apart looks like the SAME symbol repeated -- the periodic block of a
 * channel probe.  Same ring (O(1) in capture length), same top-k carrier
 * selection, same float32 grid, same gate scale [0, 1].
 *
 * What changes is the metric, because a radio has a carrier frequency offset
 * and an audio channel does not.  ofdm_acq scores, per carrier,
 *
 *     |sum_j H_j|^2 / (nsyms * sum_j |H_j|^2)
 *
 * which needs every H_j in phase.  A CFO of e subcarriers turns symbol j by
 * 2*pi*e*j*SS/N: at e = 0.4 and SS/N = 1.25 that is pi per symbol and the sum
 * cancels.  Here the score is DIFFERENTIAL in time,
 *
 *     a_j  = q_j q_{j+1} conj(Y_j) Y_{j+1}            j = 0 .. nsyms-2
 *     ck   = |sum_j a_j|^2 / ((nsyms-1) * sum_j |a_j|^2)
 *
 * so a constant per-symbol rotation cancels: any fractional CFO, any integer
 * CFO (a shifted spectrum still repeats bin for bin), and the ICI a fractional
 * CFO smears in (it repeats too).  By Cauchy-Schwarz ck is in [0, 1], 1 when
 * every a_j is equal; on noise it is about 1/(nsyms-1).
 *
 * The rows are the raw window spectrum Y / sqrt(n_fft), with no division by a
 * known pattern: for a periodic block the pattern is the same in every symbol
 * and cancels from ck exactly, so the acquisition needs no reference at all.
 *
 * The price of CFO invariance: ANY stationary signal repeats symbol to symbol
 * up to a phase -- a CW tone, a DC offset, the LO leak -- and scores 1 on the
 * bins it occupies, where ofdm_acq rejects an off-grid tone by its rotation.
 * A coherence peak is therefore a CANDIDATE, not a detection; ofdm_cxprobe
 * confirms it with the SC preamble and the CE sequence (ofdm_cxsync's
 * integer-CFO score), which a tone does not carry.
 *
 * q: an optional +-1 sign key (NULL = all ones, the periodic probe), as
 * ofdm_acq; with a key the score is invariant to the same rotation.
 *
 * Grid: candidate c starts at sample c*step, its FFT window at
 * c*step + cp_len (as ofdm_acq); symbol j of it at + j*symbol_samples.
 * Samples interleaved I,Q doubles, n counts complex samples.
 *
 * Precision: the FFT in ofdm_real (float under -DOFDM_ACQ_F32), the ring
 * float32 as ofdm_acq (it is the memory), but the accumulators DOUBLE: the
 * denominator is a fourth power of the sample level and underflows float32
 * for a quiet input (|Y| ~ 1e-5), which would make the score depend on level.
 */
#ifndef OFDM_CXACQ_H
#define OFDM_CXACQ_H
#include <stddef.h>
#include "ofdm_cx.h"
#include "ofdm_acq.h"   /* ofdm_acq_info */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ofdm_cxacq ofdm_cxacq;

/* Bytes of caller memory needed; 0 for parameters ofdm_cxacq_init rejects. */
size_t ofdm_cxacq_workspace(const ofdm_cxcfg *x, int nsyms, int step, int top_k);

/* nsyms >= 2 (one difference at least), 1 <= top_k <= n_occupied,
 * step must divide symbol_samples.  q: nsyms signs or NULL, COPIED.
 * Errors: -1 bad args, -2 memlen too small, -3 FFT init failed,
 *         -4 step does not divide symbol_samples. */
int ofdm_cxacq_init(ofdm_cxacq **a, void *mem, size_t memlen,
                    const ofdm_cxcfg *x, const double *q,
                    int nsyms, int step, int top_k);

/* Walk rx_iq[0..n).  Writes info->n_cand metric values (0 when the capture is
 * shorter than one candidate).  Errors: -1 bad args, -2 metric_cap < n_cand. */
int ofdm_cxacq_run(ofdm_cxacq *a, const double *rx_iq, int n,
                   double *metric_out, int metric_cap, ofdm_acq_info *info);

/* Candidates a capture of n samples yields (what metric_cap must hold). */
int ofdm_cxacq_n_cand(const ofdm_cxacq *a, int n);

int ofdm_cxacq_ring_rows(const ofdm_cxacq *a);

/* One grid row (window at pos*step + cp_len): n_occupied values as interleaved
 * float32 re/im, the stage the harness grades separately from the metric.
 * Errors: -1 bad args, -2 pos out of range for n. */
int ofdm_cxacq_row(ofdm_cxacq *a, const double *rx_iq, int n, int pos, float *out);

#ifdef __cplusplus
}
#endif
#endif
