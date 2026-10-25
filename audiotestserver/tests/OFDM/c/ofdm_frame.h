/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_frame.h — RS + interleave for one frame body, both directions, C89.
 *
 * The transform that pairs rs_encode_interleaved / rs_decode_interleaved in
 * ofdm_modem.py: split the message into `depth` equal RS codewords and read
 * them out column-major, so a burst that destroys consecutive wire bytes is
 * spread one byte per codeword.
 *
 * Its own unit rather than a corner of ofdm_mod.c because the RECEIVE path
 * needs the decode half and has no business linking a modulator to get it.
 *
 * The geometry (depth, k, wire_len) comes from ofdm_interleave_geometry, and
 * both ends derive it from cfg -- the zero padding up to depth*k is never sent
 * as a length, so a disagreement here shifts every byte rather than failing
 * cleanly.
 */
#ifndef OFDM_FRAME_H
#define OFDM_FRAME_H
#include <stddef.h>
#include "rs255.h"

#ifdef __cplusplus
extern "C" {
#endif

/* msg[msg_len] -> wire[depth*(k+ecc)], zero-padded to depth*k and interleaved.
 * `blk` is caller scratch of at least (k + ecc) bytes.
 * Returns bytes written, or -1 bad args, -3 an RS encode failed. */
int ofdm_frame_encode_il(rs255 *rs, const unsigned char *msg, int msg_len,
                         int depth, int k, int ecc,
                         unsigned char *wire, int wire_cap,
                         unsigned char *blk);

/* Inverse.  Writes min(msg_len, depth*k) message bytes to `out`.
 * `*errata` receives the total symbols corrected across all codewords, which is
 * what the callers use as their error count.
 * `blk` is caller scratch of at least (k + ecc) bytes.
 *
 * Returns bytes written, or -1 bad args, -2 out_cap too small,
 * -3 a codeword was uncorrectable.  A partially recovered frame is not useful,
 * so ANY uncorrectable codeword fails the whole frame -- the caller already
 * treats a failed decode as a lost frame. */
int ofdm_frame_decode_il(rs255 *rs, const unsigned char *wire, int wire_len,
                         int depth, int k, int ecc, int msg_len,
                         unsigned char *out, int out_cap,
                         unsigned char *blk, int *errata);

#ifdef __cplusplus
}
#endif
#endif
