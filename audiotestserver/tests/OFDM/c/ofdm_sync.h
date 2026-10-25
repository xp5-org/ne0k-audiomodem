/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_sync.h — frame acquisition for the data path, freestanding C89.
 *
 * The last piece between a captured buffer and ofdm_dem_frame(): the
 * Schmidl-Cox delay-correlation metric, peak selection with the false-lock
 * guard, and matched-filter refinement of each accepted start.
 *
 * Mirrors _sc_metric / _find_sc_peaks / _sc_match in ofdm_modem.py.  With this
 * in place a freestanding node can go capture -> payload with no Python at all:
 *   ofdm_sync_find() -> ofdm_dem_frame() -> ofdm_rs_decode_il().
 *
 * STREAMING, NOT MATERIALISED.  The reference builds three cumulative sums the
 * length of the capture (~5.7 MB for 5 s at 48 kHz) and then a full metric
 * array.  This keeps running window sums and a ring of the last
 * `symbol_samples` metric values instead, so working set is O(symbol_samples)
 * and flat in capture length -- the same restructure the acquisition scan
 * needed, and for the same reason.
 *
 * That is also no less accurate.  np.cumsum IS a sequential running sum, and
 * the reference then takes a DIFFERENCE of two large cumulative values, which
 * is worse conditioned than the local add/subtract used here.
 *
 * Same contract: no malloc, no stdio, no globals, one caller-owned workspace.
 */
#ifndef OFDM_SYNC_H
#define OFDM_SYNC_H
#include <stddef.h>
#include "ofdm_cfg.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ofdm_sync ofdm_sync;

/* Minimum normalised correlation with the known preamble for a metric peak to
 * be allowed to SUPPRESS its neighbours.  Deliberately low: this asks "is this
 * a preamble at all?", not "is this a good frame?".  An audio-stream startup
 * click scores ~0.037 and real frames 0.29-0.90, so 0.10 sits well clear of
 * both.  Raising it to reject weak-but-real frames costs frames, because the
 * populations overlap badly at low SNR -- rejecting junk that BLOCKS is the
 * job; rejecting junk that merely fails is what RS is for. */
#define OFDM_SC_NCC_MIN 0.10

/* Default metric threshold, matching the reference's 0.25. */
#define OFDM_SC_THRESHOLD 0.25

size_t ofdm_sync_workspace(const ofdm_cfg *c);

/* `sc_ref` is the SC preamble waveform (symbol_samples samples) -- take it
 * from ofdm_mod_sc_preamble(), so the transmitter and the matched filter can
 * never disagree about what is on the wire.  It is COPIED, mean-removed and
 * normalised at init.
 *
 * Errors: -1 bad args, -2 memlen too small. */
int ofdm_sync_init(ofdm_sync **s, void *mem, size_t memlen,
                   const ofdm_cfg *c, const double *sc_ref);

/* Frame starts in rx[0..n).  `n_body_syms` is ofdm_mod_body_syms() (CE + data
 * + SYNCs) and sets min_gap.  Writes at most `max_starts` refined sample
 * offsets, ascending.  Returns the count, or -1 on bad args.
 *
 * `threshold` selects candidates (OFDM_SC_THRESHOLD is the reference value);
 * `ncc_min` gates which candidates may suppress their neighbours
 * (OFDM_SC_NCC_MIN). */
int ofdm_sync_find(ofdm_sync *s, const double *rx, int n, int n_body_syms,
                   double threshold, double ncc_min,
                   int *starts, int max_starts);

/* Matched-filter refinement of one candidate: the best-correlating start
 * within +-cp_len of `d`, and its normalised score in [0,1].
 *
 * Late is the destructive direction.  Starting EARLY inside the cyclic prefix
 * costs only a phase ramp the equaliser already removes; starting LATE drags
 * the next symbol into the FFT window, which is ICI nothing can undo.  The SC
 * metric plateaus across the prefix and ramps into it under noise, so its
 * crossing lands part-way in and the raw start is systematically late. */
int ofdm_sync_refine(ofdm_sync *s, const double *rx, int n, int d,
                     int *best, double *score);

#ifdef __cplusplus
}
#endif
#endif
