/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_cxsync.h — complex-baseband (I/Q) frame acquisition with carrier
 * frequency offset (CFO) estimation, freestanding C89.
 *
 * The counterpart of ofdm_sync for an ofdm_cxcfg, plus the one thing a real
 * audio channel never needed: two radios' local oscillators never agree, so
 * the whole received spectrum is shifted by some fraction-plus-integer number
 * of subcarriers, e.  Left in, it rotates every carrier by 2*pi*e*t/n_fft and
 * (for |e| beyond a few percent) smears each carrier into its neighbours.
 *
 * Per candidate, in this order:
 *
 *  1. Schmidl-Cox on complex samples.  The SC preamble's body is two
 *     identical halves (ofdm_cxmod puts its PN on even signed bins), so
 *         P[d] = sum_k conj(r[d+k]) r[d+k+L],  L = n_fft/2
 *         M[d] = |P|^2 / (R1 R2)
 *     plateaus across the cyclic prefix exactly as the real metric does, and
 *     the same region / leading-edge / min_gap / NCC-gate / replace logic as
 *     ofdm_sync_find picks the frames.
 *
 *  2. FRACTIONAL CFO from the phase of P.  A shift of e subcarriers rotates
 *     the second half by exp(j*pi*e), so e_frac = arg(P)/pi in (-1, 1].
 *     P is taken in the MIDDLE of the CP plateau (leading edge + cp_len/2), so
 *     a timing error of up to cp_len/2 either way still sees two clean halves.
 *     The phase repeats every 2 subcarriers, so what remains is an EVEN
 *     integer: e = e_frac + q, q in {.., -2, 0, 2, ..}.
 *
 *  3. EVEN INTEGER CFO from the CE symbol, after removing e_frac.  The CE
 *     carries a known BPSK sequence c_i on every occupied bin b_i; shifted by q
 *     the received spectrum is Y[b_i + q] ~ H c_i.  With
 *         z_i(q) = conj(Y[b_i+q] c_i) Y[b_{i+1}+q] c_{i+1}
 *     the search scores D(q) = |mean_i z_i / |z_i||  over |q| <= max_cfo_bins,
 *     q even.  DIFFERENTIAL between neighbouring carriers so a timing error (a
 *     linear phase across bins) and the channel phase cancel; PHASE-ONLY so a
 *     single strong bin (DC leakage, a CW carrier) cannot carry it.  D is in
 *     [0,1]: ~1 at the true q on a clean channel, ~1/sqrt(n) on noise.
 *
 *     D is also the second gate.  On complex samples ANY CW tone, and a DC
 *     offset, is periodic in L and scores the SC metric at 1; and at a small
 *     n_fft the matched filter's noise floor (~1/sqrt(symbol_samples), max of
 *     2*cp_len+1 positions) can reach the real path's OFDM_SC_NCC_MIN.  A
 *     candidate must also have D >= OFDM_CXSYNC_INT_MIN, which noise and tones
 *     do not reach and a CE at any decodable SNR clears.
 *
 *  4. Matched-filter refinement as ofdm_sync_refine, against the reference
 *     ROTATED by the full estimate.  Unrotated, an offset of one whole
 *     subcarrier turns the correlation over n_fft samples into exactly zero.
 *
 * The receiver then removes e from the samples (ofdm_cx_derotate) and hands
 * them to the demodulator; the pilots track what is left.  e in Hz is
 * e * sample_rate / n_fft.
 *
 * Positions and samples as the rest of complex mode: interleaved I,Q doubles,
 * every n / start / length counts COMPLEX samples.  A frame start is the first
 * sample of the SC preamble's cyclic prefix, as in ofdm_sync.
 *
 * Same contract: no malloc, no stdio, no globals, one caller-owned workspace;
 * working set O(symbol_samples), flat in capture length.
 */
#ifndef OFDM_CXSYNC_H
#define OFDM_CXSYNC_H
#include <stddef.h>
#include "ofdm_cx.h"
#include "ofdm_sync.h"   /* OFDM_SC_THRESHOLD, OFDM_SC_NCC_MIN, OFDM_SC_REPLACE_RATIO */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ofdm_cxsync ofdm_cxsync;

/* Minimum integer-CFO score D for ofdm_cxsync_find to accept a candidate.
 * Noise: ~0.08, max over the search ~0.2. */
#define OFDM_CXSYNC_INT_MIN 0.35

typedef struct {
    int    start;      /* refined frame start (first sample of the SC CP)      */
    double score;      /* matched-filter NCC after CFO removal, [0,1]          */
    double metric;     /* SC metric region maximum                             */
    double cfo;        /* subcarriers: cfo_frac + cfo_int                      */
    double cfo_frac;   /* (-1, 1]                                              */
    int    cfo_int;    /* even                                                 */
    double int_score;  /* D(cfo_int) in [0,1]; -1 = CE not in the buffer, so
                          cfo_int was not measured and is 0                    */
} ofdm_cxsync_hit;

size_t ofdm_cxsync_workspace(const ofdm_cxcfg *x);

/* sc_ref_iq: the unscaled SC preamble, symbol_samples complex samples, from
 * ofdm_cxmod_sc_preamble(), COPIED, mean-removed and unit-normalised.
 * max_cfo_bins: the integer search range in subcarriers (>= 0; the fractional
 * part always covers +-1 on top).  Pick it from the worst LO error:
 * 2 * ppm * f_rf / spacing.
 * Errors: -1 bad args, -2 memlen too small, -3 FFT init failed. */
int ofdm_cxsync_init(ofdm_cxsync **s, void *mem, size_t memlen,
                     const ofdm_cxcfg *x, const double *sc_ref_iq,
                     int max_cfo_bins);

/* Frames in rx_iq[0..n).  n_body_syms is ofdm_cxmod_body_syms() and sets
 * min_gap, threshold / ncc_min exactly as ofdm_sync_find, plus the
 * OFDM_CXSYNC_INT_MIN gate (see step 3).  A candidate whose CE symbol runs
 * past n cannot be scored and is NOT reported (it could not be demodulated from
 * this buffer either): a streaming caller overlaps successive buffers by at
 * least one frame, as it must for the demodulator anyway.  Writes at most
 * max_hits hits, ascending start.  Returns the count, or -1 bad args. */
int ofdm_cxsync_find(ofdm_cxsync *s, const double *rx_iq, int n, int n_body_syms,
                     double threshold, double ncc_min,
                     ofdm_cxsync_hit *hits, int max_hits);

/* Steps 2-4 for one candidate whose coarse start is d (the SC metric's leading
 * edge, or any position within +-cp_len/2 of the true start).  Fills every
 * field of *h except metric (set to 0).  Returns 0, or -1 bad args / d out of
 * range. */
int ofdm_cxsync_eval(ofdm_cxsync *s, const double *rx_iq, int n, int d,
                     ofdm_cxsync_hit *h);

/* Remove a CFO of cfo subcarriers in place:
 *     iq[t] *= exp(-j 2 pi cfo (t + t0) / n_fft),  t = 0 .. n-1
 * t0 is the absolute sample index of iq[0], so blocks of one stream derotate
 * with a continuous phase. */
void ofdm_cx_derotate(double *iq, long n, double cfo, int n_fft, long t0);

#ifdef __cplusplus
}
#endif
#endif
