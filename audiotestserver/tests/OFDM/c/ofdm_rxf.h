/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_rxf.h -- receive filter that rejects this node's own band, C89.
 *
 * Port of ofdm_fdd.rx_filter for a STREAMING receiver.  The reference designs an
 * elliptic filter and runs it forward and backward (sosfiltfilt) over a whole
 * capture; that is zero-phase and therefore not causal, so a core that must
 * consume samples as they arrive cannot use it.  This is a linear-phase FIR
 * (Kaiser window) instead:
 *
 *   - causal, so it streams; the group delay is a constant (ntaps-1)/2 samples
 *     and every downstream stage works on the delayed stream, so nothing needs
 *     to know about it except a caller that wants absolute timestamps;
 *   - designed in C from the same four band edges, so a freestanding node needs
 *     no scipy and no coefficient tables;
 *   - ONE-SIDED like the reference: only the side the other band is on is
 *     rejected.  A two-sided bandpass rang long enough to eat the first frame.
 *
 * Same spec as the reference: passband edge = the far band's near edge, stopband
 * edge = this node's near edge, so the whole guard is spent as transition.
 *
 * Contract: no malloc, no stdio, no globals, one caller-owned workspace.
 */
#ifndef OFDM_RXF_H
#define OFDM_RXF_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OFDM_RXF_MAX_TAPS   1025
#define OFDM_RXF_STOP_DB    60.0

/* Design the taps.  Band edges in Hz.  `taps` takes up to `cap` values; the
 * count is always odd.  If the transition is so narrow that the requested
 * attenuation needs more than `cap` taps, the filter is shortened (less
 * rejection) rather than refused -- *got_stop_db reports what it achieves, and
 * may be NULL.
 *
 * Returns the tap count, 0 if the bands overlap (nothing to reject), -1 bad args. */
int ofdm_rxf_design(double *taps, int cap, int fs,
                    double far_lo, double far_hi, double my_lo, double my_hi,
                    double stop_db, double *got_stop_db);

/* Bytes of caller memory for an ntaps-long filter.  0 if ntaps is unusable. */
size_t ofdm_rxf_workspace(int ntaps);

typedef struct ofdm_rxf ofdm_rxf;

/* `taps` is copied.  Errors: -1 bad args, -2 memlen too small. */
int ofdm_rxf_init(ofdm_rxf **f, void *mem, size_t memlen, const double *taps, int ntaps);

/* Filter n samples; in may equal out.  State carries across calls. */
void ofdm_rxf_process(ofdm_rxf *f, const double *in, double *out, long n);

void ofdm_rxf_reset(ofdm_rxf *f);
int  ofdm_rxf_delay(const ofdm_rxf *f);         /* (ntaps-1)/2 */

#ifdef __cplusplus
}
#endif
#endif
