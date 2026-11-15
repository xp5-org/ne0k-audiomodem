/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_cxmod.h — complex-baseband (I/Q) OFDM frame modulator, freestanding C89.
 *
 * The counterpart of ofdm_mod for an ofdm_cxcfg (see ofdm_cx.h).  Same frame,
 * same bytes on the wire, same bit pipeline -- RS encode + interleave
 * (ofdm_frame_encode_il), MSB-first bits, the OFDM_SEED_DATA scrambler, the
 * bit-loaded constellations, pilots every pilot_spacing-th occupied carrier --
 * only the carriers sit at signed bins and every sample is an I,Q pair:
 *
 *     [ SC preamble ][ CE ][ SYNC ][ data * sync_period ][ SYNC ][ data ... ]
 *
 * with frame_gap_samples of zeros before every frame and after the last, the
 * body overlap-added with the same taper as the real path.  Given the same
 * bits_per_bin, chunk and sequence number, occupied carrier i of every body
 * symbol carries exactly the value occupied carrier i carries in ofdm_mod's
 * frame: the selftest checks that against the real modulator.
 *
 * The one deliberate difference: the Schmidl-Cox preamble.  The real path puts
 * its PN on EVEN occupied INDICES, which are bins of one parity because a real
 * band has no hole.  Across the DC hole index parity flips bin parity, which
 * would break the two-identical-halves repetition the detector locks on.  So
 * here the PN (OFDM_SEED_SC, amplitude 1/sqrt(2), ascending bin order) goes on
 * the occupied bins whose SIGNED BIN NUMBER IS EVEN: the body is then exactly
 * periodic in n_fft/2, and with a carrier frequency offset of e subcarriers the
 * second half is the first times exp(j*pi*e) -- the fractional CFO estimate of
 * step 5.  It is transmitted at the real path's SC gain (2.0); the reference
 * returned by ofdm_cxmod_sc_preamble is unscaled, as ofdm_mod_sc_preamble.
 *
 * Samples are INTERLEAVED doubles (out[2t] = I, out[2t+1] = Q) and every length
 * and capacity here counts COMPLEX samples: a buffer of cap samples is
 * 2 * cap doubles.  Level: raw, as ofdm_mod; levelling is the caller's job.
 *
 * Same contract as the other cores: no malloc, no stdio, no globals, one
 * caller-owned workspace (the RS workspace is carved from it too).
 */
#ifndef OFDM_CXMOD_H
#define OFDM_CXMOD_H
#include <stddef.h>
#include "ofdm_cx.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ofdm_cxmod ofdm_cxmod;

/* Bytes of caller memory needed.  0 if the parameters would be rejected. */
size_t ofdm_cxmod_workspace(const ofdm_cxcfg *x, const int *bits_per_bin);

/* bits_per_bin: x->c.n_data_carriers entries from {0,1,2,4,6}, COPIED.
 * Errors as ofdm_mod_init: -1 bad args, -2 memlen too small, -3 FFT init
 * failed, -4 a width has no constellation, -5 empty table, -6 RS init failed. */
int ofdm_cxmod_init(ofdm_cxmod **m, void *mem, size_t memlen,
                    const ofdm_cxcfg *x, const int *bits_per_bin);

/* The same with an inner code (ofdm_cxfec.h: OFDM_CXFEC_NONE / R12 / R23 / R34)
 * between the scrambled RS bytes and the mapper; the frame gets as many data
 * symbols as the coded bits need.  Both ends must use the same fec. */
size_t ofdm_cxmod_workspace_fec(const ofdm_cxcfg *x, const int *bits_per_bin, int fec);
int ofdm_cxmod_init_fec(ofdm_cxmod **m, void *mem, size_t memlen,
                        const ofdm_cxcfg *x, const int *bits_per_bin, int fec);

int  ofdm_cxmod_bits_per_sym(const ofdm_cxmod *m);
int  ofdm_cxmod_data_syms(const ofdm_cxmod *m);
int  ofdm_cxmod_body_syms(const ofdm_cxmod *m);      /* CE + data + SYNCs */
int  ofdm_cxmod_payload_bytes(const ofdm_cxmod *m);
long ofdm_cxmod_frame_samples(const ofdm_cxmod *m);  /* one frame, no gap */

/* Complex samples ofdm_cxmod_encode_ex writes for n_bytes; gap 0 leaves out
 * every inter-frame gap.  Negative on bad args. */
long ofdm_cxmod_signal_samples_ex(const ofdm_cxmod *m, int n_bytes, int gap);

/* As ofdm_mod_encode_ex: NUL-pads to whole chunks, numbers them seq0, seq0+1,
 * ... mod 2^(8*seq_bytes).  out_iq holds out_cap complex samples.
 * Returns complex samples written, or -1 bad args, -2 out_cap too small,
 * -3 RS encode failed. */
long ofdm_cxmod_encode_ex(ofdm_cxmod *m, const unsigned char *data, int n_bytes,
                          unsigned long seq0, int gap,
                          double *out_iq, long out_cap);

/* One frame (ofdm_cxmod_frame_samples long, no gap) for an explicit sequence
 * number and exactly c.chunk_bytes of chunk.  Same error codes. */
long ofdm_cxmod_frame(ofdm_cxmod *m, unsigned seq, const unsigned char *chunk,
                      double *out_iq, long out_cap);

/* The unscaled SC preamble, symbol_samples complex samples interleaved: the
 * receiver's matched-filter reference.  NULL if m is NULL. */
const double *ofdm_cxmod_sc_preamble(const ofdm_cxmod *m);

#ifdef __cplusplus
}
#endif
#endif
