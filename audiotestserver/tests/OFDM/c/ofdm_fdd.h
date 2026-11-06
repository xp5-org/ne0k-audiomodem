/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_fdd.h -- full-duplex (FDD) planning, freestanding C89.
 *
 * Port of the DSP-free half of ofdm_fdd.py: split one band into a downlink and
 * an uplink around a guard gap, size frames, and re-index a full-band bit
 * table onto a sub-band.  A sub-band's ofdm_cfg is just ofdm_cfg_build() with
 * that band's freq_lo/freq_hi (the Python band_cfg is the same call), so no
 * wrapper is needed for it.
 *
 * NOT here yet: the receive filter (ofdm_fdd.rx_filter).  It is an elliptic
 * design plus zero-phase sosfiltfilt in the reference; a streaming C core
 * must be causal, which changes the acquisition behaviour and has to be
 * graded on its own.
 *
 * Same contract as the other cores: C89, no malloc, no stdio, no globals.
 * Rounding differs from the reference in one place: Python rounds the reported
 * band edges to 0.1 Hz; this returns them unrounded.
 */
#ifndef OFDM_FDD_H
#define OFDM_FDD_H
#include "ofdm_cfg.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OFDM_FDD_SIDE_HIGH 0     /* downlink on the high band (measured-correct) */
#define OFDM_FDD_SIDE_LOW  1

typedef struct {
    double dl_lo, dl_hi;         /* Hz */
    double ul_lo, ul_hi;
    double guard_lo, guard_hi;   /* the gap between the two bands */
    double dl_frac_actual;
    double min_bw_hz;
    double symbol_rate_hz;
    double carrier_spacing_hz;
    int    ratio_clamped;
    int    within_one_octave;    /* freq_hi < 2*freq_lo */
    int    harm2_overlap;        /* 2nd harmonic of the lower band lands in the upper */
    int    harm3_overlap;
} ofdm_fdd_plan;

/* Occupied carriers needed for `min_baud` at 1 bit/carrier, counting pilots. */
int ofdm_fdd_min_occupied(double min_baud, double symbol_rate, int pilot_spacing);

/* Returns 0 ok, -1 bad args, -2 guard consumes the band, -3 band too narrow
 * for min_baud in both directions. */
int ofdm_fdd_plan_bands(ofdm_fdd_plan *p, double freq_lo, double freq_hi,
                        double dl_frac, double guard_hz, int n_fft,
                        double cp_ratio, int sample_rate, int pilot_spacing,
                        double min_baud, int dl_side);

/* Largest chunk size from {256,192,128,96,64,48,32,24,16} whose single frame
 * fits in max_frame_s on a table summing to bits_per_ofdm_sym; 16 if none. */
int ofdm_fdd_chunk_for_frame_len(const ofdm_cfg *c, int bits_per_ofdm_sym,
                                 double max_frame_s);

/* Re-index a full-band table onto a sub-band's data bins.  A bin that was a
 * pilot in the full band was never measured as data and gets 1 (BPSK).
 * `out` has sub->n_data_carriers entries.  Returns 0 ok, -1 bad args. */
int ofdm_fdd_sub_bat(const ofdm_cfg *full, const int *full_bat,
                     const ofdm_cfg *sub, int *out);

#ifdef __cplusplus
}
#endif
#endif
