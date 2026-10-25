/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_dem.h — OFDM data-frame demodulator, freestanding C99.
 *
 * The DSP half of AdaptiveDemodulator._decode_frame: per-symbol real FFT,
 * channel estimation from the CE symbol and periodic SYNC refreshes, common
 * phase error from the pilots, one-tap zero-forcing equalisation,
 * hard-decision demapping, EVM, then PRBS descrambling and bit packing.
 *
 * Measured at 39% of Modem.decode_data. Same contract as ofdm_acq: no
 * malloc/stdio/globals, one caller-owned workspace, float32 or float64
 * selected at compile time.
 *
 * Returns descrambled, packed BYTES -- everything up to and including the
 * PRBS/packbits step. Reed-Solomon stays in Python (`reedsolo`): the caller
 * does ofdm_dem_frame() -> RS decode -> sequence number and chunk.
 *
 * Constellations are passed in, not compiled in. The Gray-coded tables live
 * in ofdm_modem.py and are handed over at init, so a table change on the
 * Python side cannot silently diverge from the C.
 *
 * Precision. numpy's rfft returns complex64 for float32 input and complex128
 * for float64 input, so the reference decoder's arithmetic depends on the
 * dtype of the capture it is handed. This core takes double samples and
 * computes in ofdm_real:
 *   - the default (double) build is the exact analogue of numpy fed float64,
 *     used as the equivalence harness's oracle;
 *   - the -DOFDM_ACQ_F32 build is the analogue of the production path,
 *     where captures arrive as float32.
 * Neither is bit-identical to the other, by construction.
 */
#ifndef OFDM_DEM_H
#define OFDM_DEM_H
#include <stddef.h>
#include "ofdm_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double evm;        /* mean per-symbol EVM, linear (not %) */
    int    data_syms;  /* data symbols actually demodulated   */
    int    n_bytes;    /* bytes written to `out`              */
} ofdm_dem_out;

typedef struct ofdm_dem ofdm_dem;

/* Bytes of caller memory needed.  0 for parameters init would reject. */
size_t ofdm_dem_workspace(int n_fft, int cp_len, int n_occupied,
                          int n_pilots, int n_data_carriers,
                          int data_syms, int sync_period,
                          int payload_bytes, int n_pts_total,
                          int n_bits_total);

/* Carves `mem`.  Everything pointed at is COPIED, so the caller may free.
 *
 * Constellation tables are flat and concatenated in ascending `tab_b` order:
 *   tab_b[t]   bits-per-carrier this table serves (1, 2, 4, 6)
 *   tab_n[t]   points in it (2, 4, 16, 64)
 *   pts_re/im  all points, table 0 first
 *   tab_bits   all bit patterns, tab_b[t] bits per point, same order
 *
 * Errors: -1 bad args / null, -2 memlen too small, -3 FFT init failed,
 *         -4 a bits_per_bin value has no table. */
int ofdm_dem_init(ofdm_dem **d, void *mem, size_t memlen,
                  int n_fft, int cp_len, int bin_lo, int n_occupied,
                  const double *ce_known_re, const double *ce_known_im,
                  const double *sync_known_re, const double *sync_known_im,
                  const int *pilot_offsets, int n_pilots,
                  double pilot_re, double pilot_im,
                  const int *bits_per_bin, int n_data_carriers,
                  int data_syms, int sync_period, int payload_bytes,
                  unsigned prbs_seed,
                  const int *tab_b, const int *tab_n, int n_tabs,
                  const double *pts_re, const double *pts_im,
                  const unsigned char *tab_bits);

/* Demodulate ONE frame starting at sample `frame_start`.
 * Errors: -1 bad args, -2 capture too short for the whole frame,
 *         -3 no data symbol survived (caller treats as "no frame"),
 *         -4 out_cap smaller than payload_bytes. */
int ofdm_dem_frame(ofdm_dem *d, const double *rx, int n, int frame_start,
                   unsigned char *out, int out_cap, ofdm_dem_out *info);

/* Introspection for sizing reports. 0 on NULL. */
int ofdm_dem_total_body_syms(const ofdm_dem *d);

#ifdef __cplusplus
}
#endif
#endif
