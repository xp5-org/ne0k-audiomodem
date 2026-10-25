/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_mod.h — OFDM modulator (transmit), freestanding C89.
 *
 * AdaptiveModulator.encode_bytes: RS-encode and interleave each chunk, scramble,
 * map onto the bit-loaded carriers, insert the CE and periodic SYNC reference
 * symbols, inverse-transform with a cyclic prefix and a raised-cosine taper,
 * overlap-add, and prepend the Schmidl-Cox preamble.
 *
 * `tests/regression/ofdm_mod_regress.py` compares the emitted waveform
 * sample-for-sample against ofdm_modem.
 *
 * Frame layout, per RS-coded chunk (unchanged from the reference):
 *     [ SC preamble ][ CE ][ SYNC ][ data * sync_period ][ SYNC ][ data ... ]
 * with `frame_gap_samples` of silence before every frame and after the last.
 *
 * Same contract as the other cores: no malloc, no stdio, no globals, one
 * caller-owned workspace.  The Reed-Solomon workspace is carved from the same
 * block, so a caller needs exactly one allocation for the whole transmitter.
 *
 * LEVEL.  Output is the reference's RAW signal, NOT peak-normalised: levelling
 * is the caller's business and must be RMS-matched to the probe, never peak
 * normalised (peak normalisation ships messages several dB quiet).
 */
#ifndef OFDM_MOD_H
#define OFDM_MOD_H
#include <stddef.h>
#include "ofdm_cfg.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ofdm_mod ofdm_mod;

/* Bytes of caller memory needed.  0 if the parameters would be rejected. */
size_t ofdm_mod_workspace(const ofdm_cfg *c, const int *bits_per_bin);

/* Carves `mem`.  `bits_per_bin` has c->n_data_carriers entries from
 * {0,1,2,4,6}; it is COPIED.  A table summing to zero is rejected (-5) rather
 * than silently producing a frame that carries nothing -- an empty BAT is the
 * caller's decision to make, not this core's to paper over.
 *
 * Errors: -1 bad args/null, -2 memlen too small, -3 FFT init failed,
 *         -4 a bits_per_bin value has no constellation, -5 empty table,
 *         -6 RS init failed. */
int ofdm_mod_init(ofdm_mod **m, void *mem, size_t memlen,
                  const ofdm_cfg *c, const int *bits_per_bin);

/* Frame geometry, for sizing and for the receiver's min_gap. */
int  ofdm_mod_bits_per_sym(const ofdm_mod *m);
int  ofdm_mod_data_syms(const ofdm_mod *m);
int  ofdm_mod_body_syms(const ofdm_mod *m);   /* CE + data + SYNCs */
int  ofdm_mod_payload_bytes(const ofdm_mod *m);
long ofdm_mod_frame_samples(const ofdm_mod *m);  /* one frame, no gap */

/* Samples ofdm_mod_encode will write for an n-byte payload, including the
 * leading and trailing gaps.  Negative on bad args. */
long ofdm_mod_signal_samples(const ofdm_mod *m, int n_bytes);

/* Encode `n_bytes` into `out`.  The payload is NUL-padded up to a whole number
 * of chunks, exactly as the reference does -- so a payload whose own tail is
 * NUL cannot be distinguished from padding on the wire, and callers that care
 * must carry an explicit length.
 *
 * Returns samples written, or -1 bad args, -2 out_cap too small, -3 RS encode
 * failed. */
long ofdm_mod_encode(ofdm_mod *m, const unsigned char *data, int n_bytes,
                     double *out, long out_cap);

/* One frame for an explicit sequence number and one chunk of exactly
 * c->chunk_bytes.  Lets a caller drive framing itself (retransmission, a mesh
 * MAC) instead of handing over a whole message.  Writes
 * ofdm_mod_frame_samples() samples, no gap.  Same error codes. */
long ofdm_mod_frame(ofdm_mod *m, unsigned seq, const unsigned char *chunk,
                    double *out, long out_cap);

/* The Schmidl-Cox preamble waveform (symbol_samples long), which the receiver
 * needs as the matched-filter reference.  Returns NULL if m is NULL. */
const double *ofdm_mod_sc_preamble(const ofdm_mod *m);

#ifdef __cplusplus
}
#endif
#endif
