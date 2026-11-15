/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_cxdem.h — complex-baseband (I/Q) OFDM frame demodulator, freestanding C89.
 *
 * The counterpart of ofdm_dem for an ofdm_cxcfg, and the receive half of
 * ofdm_cxmod: given a frame start and a CFO estimate from ofdm_cxsync, it
 * returns the frame's sequence number and chunk.  Per frame:
 *
 *  1. Every symbol window [pos + w0, + n_fft) is derotated by the CFO
 *     estimate, phase referenced to frame_start, and FFT'd.  The constant
 *     phase that reference leaves is the same in every symbol, so the channel
 *     estimate absorbs it.  The caller's buffer is not touched.
 *     w0 = taper + cp_len/2, the middle of the ISI-free stretch [2T, CP]
 *     (the previous symbol's taper ends at 2T, the next one starts at the
 *     symbol length).  ofdm_dem uses cp_len, its right edge.  Any w0 in the
 *     stretch only rotates each bin by a constant, which the CE takes out.
 *     The margin either side is CP/2 - T: zero at the cfg's T = CP/2 clamp,
 *     so a multipath channel wants a short taper (step 8's iq profile).
 *
 *  2. Channel estimate from the CE, H = (Y / sqrt(N)) / ce.  The FIRST SYNC
 *     (right after the CE, identical sequence) is averaged in; later SYNCs
 *     replace H, as ofdm_dem.
 *
 *  3. Per data symbol, the pilots give a phase a + b * bin.
 *       a  common phase: what the residual CFO (ofdm_cxsync is good to about
 *          0.003 subcarriers, i.e. ~0.02 rad per symbol) and phase noise leave;
 *       b  slope: a sample-clock offset is a timing DRIFT, a phase linear in
 *          frequency that grows across the frame.
 *     The slope is in SIGNED BIN, not occupied index: across the DC hole
 *     index i and bin differ by one, and a timing phase is linear in bin.  As
 *     in ofdm_dem the slope comes from the differential phase between
 *     adjacent pilots at the smallest bin gap (so the pair that
 *     straddles the hole, one bin further apart, is left out of it), weighted
 *     |H|^2, and a is the slope-derotated weighted circular mean.
 *
 *  4. One-tap equalisation by H * e^{j(a + b*bin)}, nearest-point hard
 *     decision on the bit-loaded constellations (ofdm_mod_table: the same
 *     tables ofdm_cxmod maps with), EVM, OFDM_SEED_DATA descramble, MSB-first
 *     pack, then de-interleave + RS as ofdm_frame_decode_il, with an
 *     erasure retry (GMD) per codeword that fails: ofdm_cxdem_set_erasures.
 *
 * Positions and samples as the rest of complex mode: interleaved I,Q doubles,
 * every n / start counts COMPLEX samples, frame_start is the first sample of
 * the SC preamble's cyclic prefix (ofdm_cxsync_hit.start).
 *
 * Precision as ofdm_dem: double by default, the FFT and the per-carrier
 * equalisation in float under -DOFDM_ACQ_F32.  Same contract: no malloc, no
 * stdio, no globals, one caller-owned workspace (RS carved from it).
 */
#ifndef OFDM_CXDEM_H
#define OFDM_CXDEM_H
#include <stddef.h>
#include "ofdm_cx.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ofdm_cxdem ofdm_cxdem;

typedef struct {
    double   evm;        /* mean per-symbol EVM, linear (not %)                */
    int      data_syms;  /* data symbols demodulated                           */
    int      errata;     /* RS symbols corrected over all codewords            */
    unsigned seq;        /* sequence number (big-endian, seq_bytes)            */
    int      erased;     /* bytes decoded as erasures (0: errors-only sufficed) */
} ofdm_cxdem_out;

/* Bytes of caller memory needed.  0 if the parameters would be rejected. */
size_t ofdm_cxdem_workspace(const ofdm_cxcfg *x, const int *bits_per_bin);

/* bits_per_bin: x->c.n_data_carriers entries from {0,1,2,4,6}, COPIED; must be
 * the transmitter's.  Errors as ofdm_cxmod_init: -1 bad args, -2 memlen too
 * small, -3 FFT init failed, -4 a width has no constellation, -5 empty table,
 * -6 RS init failed. */
int ofdm_cxdem_init(ofdm_cxdem **d, void *mem, size_t memlen,
                    const ofdm_cxcfg *x, const int *bits_per_bin);

int  ofdm_cxdem_data_syms(const ofdm_cxdem *d);
int  ofdm_cxdem_body_syms(const ofdm_cxdem *d);       /* CE + data + SYNCs   */
int  ofdm_cxdem_payload_bytes(const ofdm_cxdem *d);   /* wire bytes, pre-RS  */
/* Samples from frame_start the demodulator reads (through the last window);
 * at most ofdm_cxmod_frame_samples. */
long ofdm_cxdem_frame_samples(const ofdm_cxdem *d);

/* Demodulate the frame at frame_start with a CFO of cfo subcarriers
 * (ofdm_cxsync_hit.cfo; 0 if the caller already derotated).  Writes
 * c.chunk_bytes chunk bytes to chunk_out.
 * Returns 0, or -1 bad args, -2 rx too short for the whole frame
 * (frame_start + ofdm_cxdem_frame_samples > n), -3 no data symbol,
 * -4 out_cap < chunk_bytes, -5 RS uncorrectable (info filled except seq and
 * errata; the hard-decision wire bytes are still in ofdm_cxdem_wire). */
int ofdm_cxdem_frame(ofdm_cxdem *d, const double *rx_iq, int n, int frame_start,
                     double cfo, unsigned char *chunk_out, int out_cap,
                     ofdm_cxdem_out *info);

/* Erasure (GMD) decoding, on by default: a codeword the errors-only RS
 * decode cannot correct is retried with its least reliable bytes erased
 * (per-cell decision distance, per-carrier mean, dead-carrier |H|^2), keeping
 * 4 parity bytes unused as a miscorrection check.  0 = errors-only, the
 * behaviour before 2026-11-14. */
void ofdm_cxdem_set_erasures(ofdm_cxdem *d, int on);

/* With an inner code (ofdm_cxfec.h), as ofdm_cxmod_*_fec: soft (max-log,
 * |He|^2 weighted) LLRs into a Viterbi before the RS. */
size_t ofdm_cxdem_workspace_fec(const ofdm_cxcfg *x, const int *bits_per_bin, int fec);
int ofdm_cxdem_init_fec(ofdm_cxdem **d, void *mem, size_t memlen,
                        const ofdm_cxcfg *x, const int *bits_per_bin, int fec);

/* Tracking between SYNCs, off by default (the selftest oracles grade the plain
 * demodulator).  TRACK_PILOTS: pilots whose phase coherence is under 0.6 are
 * replaced by the previous symbol's phase, then a decision-directed pass
 * refines a + b*bin from the reliable data cells (survives dead or noisy
 * pilots).  TRACK_H: the channel estimate follows the reliable decisions and
 * the pilots (mu 0.15), so a fade between SYNCs is tracked. */
#define OFDM_CXDEM_TRACK_PILOTS 1
#define OFDM_CXDEM_TRACK_H      2
void ofdm_cxdem_set_tracking(ofdm_cxdem *d, int flags);
int  ofdm_cxdem_pilot_fallbacks(const ofdm_cxdem *d);   /* symbols the pilots were overruled */

/* Decoder-aided channel re-estimation (inner code only), off by default.  When
 * the first pass's RS fails, the Viterbi output (with every RS codeword that
 * did decode replaced by its corrected re-encoding) is re-encoded into the
 * transmitted cells; per loaded carrier, H at symbol t is refit by least
 * squares over that carrier's known / re-encoded cells in symbols t-half_win
 * .. t+half_win (CE and SYNCs included, t itself excluded), plus one common
 * phase per symbol; then re-decide, LLRs, Viterbi, RS + GMD.  Up to `passes`
 * such passes (0 = off, max 8); half_win 1..16 (default 2).  Costs a body x
 * n_occupied x 2 complex buffer in the workspace whenever fec != NONE.
 * ofdm_cxdem_iter_passes: passes the last frame used (0 = first pass decoded
 * or off). */
void ofdm_cxdem_set_iterate(ofdm_cxdem *d, int passes, int half_win);
int  ofdm_cxdem_iter_passes(const ofdm_cxdem *d);

/* FFT window offset in the symbol (complex samples after the symbol start):
 * -1 = the default, the middle of the ISI-free stretch (taper + cp/2); up to
 * cp_len.  A later window tolerates a longer echo (delay <= w0 - 2*taper) at
 * the cost of timing margin (late sync <= cp_len - w0).  Call before reading
 * ofdm_cxdem_frame_samples.  Returns the offset in use, -1 bad args. */
int ofdm_cxdem_set_window(ofdm_cxdem *d, int w0);

/* Per data carrier, from the last frame demodulated (RS ok or not): the mean
 * cell score (squared decision distance over (d_min/2)^2; 1 = on the decision
 * boundary) and the mean |H|^2 over the loaded carriers' mean.  n_data_carriers
 * entries each; -1 for a 0-bit carrier.  Returns the loaded count, -1 bad args.
 * What a receiver-side table update (iq_link ARQ) is graded from. */
int ofdm_cxdem_carrier_quality(const ofdm_cxdem *d, double *score, double *hrel);

/* The last frame's descrambled hard-decision bytes before RS,
 * ofdm_cxdem_payload_bytes long.  NULL if d is NULL. */
const unsigned char *ofdm_cxdem_wire(const ofdm_cxdem *d);

#ifdef __cplusplus
}
#endif
#endif
