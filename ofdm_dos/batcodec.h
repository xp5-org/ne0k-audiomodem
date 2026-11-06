/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026 XP5.ORG
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/* batcodec.h -- the BPSK back-channel that carries a bit-allocation table
 * between two OFDMBAT ends, C89, caller-owned memory, no stdio.
 *
 * WHY THIS EXISTS.  The probe half of this work is one-directional by
 * construction: the RECEIVER measures the channel and computes the BAT for the
 * direction it just heard. For the link to actually be used, the TRANSMITTER
 * needs that table -- so the receiver has to send it back. Without a return
 * path there is no exchange, only two ends each holding half the answer.
 *
 * WHY NOT THE FULL DATA PATH.  ofdm_mod/ofdm_dem/rs255 would do this
 * faithfully (it is what ofdm_modem.py's NEGOTIATE uses) but they are not
 * vendored into the DOS build, and their 486 cost is unmeasured. The payload
 * here is a few hundred BITS. Pulling in a full framed, interleaved,
 * Reed-Solomon-coded modem to move 300 bits over a 39 dB channel is the wrong
 * trade; this is deliberately the smallest thing that is still honest.
 *
 * THE SCHEME, and why each piece is the way it is:
 *
 *   Sync      A preamble of BAT_SYNC_SYMS symbols, every occupied carrier
 *             carrying the same known BPSK reference. This is EXACTLY the
 *             shape ofdm_acq already searches for (a periodic block with an
 *             all-ones sign key), so acquisition is reused rather than
 *             reinvented -- the same code the probe's own sync is built on,
 *             already graded against the numpy reference.
 *
 *   Channel   H is estimated by averaging the preamble symbols against the
 *             known reference. The payload is then equalised with it. The
 *             preamble pays for itself twice: it finds the frame AND measures
 *             the channel that decodes it.
 *
 *   Payload   Carriers are split into BAT_GROUPS contiguous groups, and every
 *             carrier in a group carries the SAME bit as +-1 * known. So one
 *             symbol carries BAT_GROUPS bits, and each bit is spread over
 *             n_occupied/BAT_GROUPS carriers. That frequency diversity is the
 *             entire error-correction strategy: a bit survives unless most of
 *             its carriers are simultaneously bad, which on a channel this
 *             flat does not happen. It is also why no FEC is needed.
 *
 *   Framing   magic + carrier count + CRC16 over the payload. The CRC is what
 *             makes a decode SELF-VERIFYING -- a receiver that cannot tell a
 *             good table from a plausible-looking wrong one would silently
 *             configure the link wrong, which is worse than not decoding.
 *
 *   Repeat    The whole frame is emitted BAT_REPEATS times back to back so the
 *             far end can arm at any moment and still catch a complete copy,
 *             the same reason the probe transmitter loops. The decoder accepts
 *             the first repetition whose CRC passes rather than voting, since
 *             a passing CRC is already stronger evidence than a majority of
 *             unverified copies.
 *
 * Bits are packed 3-per-carrier because ofdm_allocate's output is 0..6.
 */
#ifndef BATCODEC_H
#define BATCODEC_H
#include <stddef.h>
#include "ofdm_cfg.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BAT_SYNC_SYMS   16      /* preamble length, symbols                  */
#define BAT_GROUPS       8      /* bits per symbol; byte-aligned on purpose  */
#define BAT_REPEATS      3      /* whole-frame repetitions                   */
#define BAT_MAGIC     0xBAu     /* frame marker, checked before the CRC      */
#define BAT_SEED_SYNC 0x5A17u   /* distinct from every OFDM_SEED_* so a probe
                                   can never be mistaken for a BAT frame     */
#define BAT_MAX_CARRIERS 256    /* bounds the on-stack packing buffers       */

/* Bytes of scratch bat_encode/bat_decode need. 0 for parameters they would
   reject. */
size_t bat_workspace(const ofdm_cfg *c);

/* Sync diagnostics from the LAST bat_decode call -- the autocorrelation
   plateau's length and end position. Meaningful only after a decode, and
   worth reporting on failure: "no frame" and "a plateau of the wrong length"
   are different faults, and on hardware the difference is the whole
   investigation. */
extern long bat_dbg_runlen;
extern long bat_dbg_runend;

/* Samples bat_encode will write for `n_bits_entries` carriers -- call before
   allocating, the same contract ofdm_probe_build_samples has. <=0 on bad
   args. */
long bat_encode_samples(const ofdm_cfg *c, int n_entries);

/* Modulate a bit-allocation table into `out` (bat_encode_samples doubles).
   `bits` holds n_entries values in 0..6. Returns samples written, or
   -1 bad args, -2 cap too small, -3 workspace too small. */
long bat_encode(const ofdm_cfg *c, void *mem, size_t memlen,
                const int *bits, int n_entries, double *out, long cap);

/* Recover a table from a capture. `bits_out` takes up to `cap` entries.
   Returns the number of entries recovered, or
   -1 bad args, -2 workspace too small, -3 no frame found (sync failed),
   -4 a frame was found but no repetition passed CRC (so the capture is real
      but corrupt -- a different fault from -3 and worth distinguishing),
   -5 the table has more entries than `cap`. */
int bat_decode(const ofdm_cfg *c, void *mem, size_t memlen,
               const double *rx, long n, int *bits_out, int cap);

#ifdef __cplusplus
}
#endif
#endif
