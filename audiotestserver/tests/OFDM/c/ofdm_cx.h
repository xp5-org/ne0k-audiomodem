/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_cx.h — complex-baseband (I/Q) OFDM: geometry and one symbol, both ways.
 *
 * The audio modem is REAL passband: carriers only at positive bins
 * 1..n_fft/2-1, the negative half implied by Hermitian symmetry, samples are
 * one real number each.  An SDR (GNU Radio, PlutoSDR) carries COMPLEX
 * baseband: the channel is centred on DC, carriers sit at signed bins
 * -n_fft/2+1 .. n_fft/2-1, and every sample is an I,Q pair.  Same sample rate,
 * twice the usable bandwidth, and no mirror image to waste power on.
 *
 * This file is ADDITIVE.  Nothing in the real path changes: ofdm_cfg_build,
 * ofdm_sym, ofdm_mod/dem/acq keep their ABI and their regression oracles.
 * What complex mode reuses unchanged, through the embedded ofdm_cfg:
 *   ofdm_interleave_geometry, ofdm_data_syms_per_pkt, ofdm_n_syncs,
 *   ofdm_frame_min_gap, ofdm_prbs / ofdm_known_bpsk, the constellations.
 * Those only read the byte/carrier COUNTS, never a bin number.
 *
 * Signed bins.  In an ofdm_cxcfg, c.bin_lo / c.bin_hi are SIGNED bin numbers
 * relative to the centre (DC), and c.freq_lo_hz / c.freq_hi_hz are offsets
 * from the RF centre, so a 100 kHz channel is freq_lo = -50e3, freq_hi = +50e3.
 * NEVER hand cx.c to a real-path function that indexes bins (ofdm_sym_*,
 * ofdm_mod_*, ofdm_dem_*): it would index a negative bin.
 *
 * DC null.  A direct-conversion front end (AD936x / PlutoSDR) leaks its LO
 * into bin 0 and its DC-tracking loop notches it, so with dc_null the DC bin
 * carries nothing and is NOT counted in n_occupied: occupied index i maps to
 * bin_lo + i below the hole and bin_lo + i + 1 above it (ofdm_cx_bin).
 * Nyquist (-n_fft/2) is never used, as in the real path.
 *
 * Samples.  Complex samples are INTERLEAVED doubles: out[2t] = I, out[2t+1] = Q,
 * the cf32 wire order.  Lengths returned by these functions count complex
 * samples, so a buffer needs 2 * length doubles.
 *
 * Level.  Same convention as the real path: inverse transform times sqrt(n_fft),
 * so a unit-power constellation on every one of n_occupied carriers gives a
 * mean |x|^2 of n_occupied / n_fft -- at most 1, i.e. RMS within full scale
 * for a full band.  Levelling for the radio stays the caller's job.
 *
 * Same contract as the other cores: C89, no malloc, no stdio, no globals.
 */
#ifndef OFDM_CX_H
#define OFDM_CX_H
#include <stddef.h>
#include "ofdm_cfg.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    ofdm_cfg c;        /* counts as usual; bins and freqs SIGNED, see above   */
    int      dc_null;  /* 1: bin 0 carries nothing                            */
    int      dc_slot;  /* first occupied index above the hole; -1 = no hole   */
} ofdm_cxcfg;

/* Complex-baseband geometry: occupied bins ceil(freq_lo/spacing) ..
 * floor(freq_hi/spacing), clamped to -n_fft/2+1 .. n_fft/2-1, minus DC when
 * dc_null.  freq_lo/freq_hi are offsets from the centre in Hz and may be
 * negative.  Other arguments exactly as ofdm_cfg_build.
 * Returns 0 ok, -1 n_fft not a power of two >= 4, -2 no carriers in the band,
 * -3 other bad argument. */
int ofdm_cxcfg_build(ofdm_cxcfg *x, int n_fft, double cp_ratio,
                     double freq_lo, double freq_hi, int pilot_spacing,
                     int sample_rate, int ecc_bytes, int chunk_bytes,
                     int seq_bytes, double taper_ms, double frame_gap_ms,
                     int sync_period, int dc_null);

/* Occupied index (0 .. n_occupied-1) -> signed FFT bin.  Out of range -> 0
 * (which is never an occupied bin when dc_null, so a caller can test for it). */
int ofdm_cx_bin(const ofdm_cxcfg *x, int i);

/* ---- one OFDM symbol, complex: the counterpart of ofdm_sym. */
typedef struct ofdm_cxsym ofdm_cxsym;

size_t ofdm_cxsym_workspace(const ofdm_cxcfg *x);
/* Returns 0 ok, -1 bad args, -2 memlen too small, -3 FFT init failed. */
int ofdm_cxsym_init(ofdm_cxsym **sp, void *mem, size_t memlen, const ofdm_cxcfg *x);

/* Longest symbol ofdm_cxsym_make writes, in complex samples:
 * symbol_samples + 2 * taper_samples. */
int ofdm_cxsym_max_len(const ofdm_cxsym *s);

/* occ_re/occ_im: n_occupied carrier values in occupied order (occ_im may be
 * NULL = all real).  Writes [CP | body] (use_taper 0) or [CP+T | body | T]
 * with the raised-cosine rise/fall (use_taper 1) -- the same layout and window
 * as ofdm_sym_make, so the frame-level overlap-add carries over unchanged.
 * out_iq: interleaved I,Q, 2 * ofdm_cxsym_max_len doubles.
 * Returns complex samples written, or -1 bad args. */
int ofdm_cxsym_make(ofdm_cxsym *s, const double *occ_re, const double *occ_im,
                    int use_taper, double *out_iq);

/* The receive half: n_fft complex samples of ONE symbol body (cyclic prefix
 * already removed, timing already found) -> n_occupied carrier values,
 * undoing make's sqrt(n_fft) scale, so make -> strip CP -> demod returns the
 * carriers exactly (to rounding) on a clean channel.
 * Returns n_occupied, or -1 bad args. */
int ofdm_cxsym_demod(ofdm_cxsym *s, const double *body_iq,
                     double *occ_re, double *occ_im);

#ifdef __cplusplus
}
#endif
#endif
