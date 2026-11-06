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

/* rs255.h -- Reed-Solomon(255,*) over GF(2^8), freestanding C89.
 *
 * Wire-compatible with the Python reference's `rs255_c.RSCodec` fallback,
 * the pure-Python `reedsolo` package: prim=0x11d, generator=2, fcr=0,
 * c_exp=8 (all reedsolo defaults -- see reedsolo.RSCodec.__init__). One
 * rs255 instance is configured with a fixed ECC-byte count (`nsym`) and
 * encodes/decodes ONE codeword per call -- exactly what ofdm_frame.c's
 * interleaving needs (it drives its own chunking across `depth` codewords),
 * and it deliberately does NOT reproduce reedsolo's automatic >255-byte
 * message chunking.
 *
 * Pure GF(2^8) integer arithmetic (table lookups + XOR) -- there is no
 * floating point anywhere in this file, unlike the OFDM DSP cores (fft,
 * probe, acq) which need a real or emulated FPU. That means this component
 * does not care whether a target 486 has an FPU, and its cost is expected
 * to be negligible on either the 25MHz or the 50MHz test host -- see the
 * self-test's own tick measurement rather than assuming that, the same way
 * the XCHG slot-timing investigation replaced assumption with a measurement.
 *
 * Same contract as the other cores here: no malloc, no stdio, no globals,
 * one caller-owned workspace via rs255_workspace()/rs255_init().
 *
 * ecc_bytes is capped at RS255_MAX_NSYM: everything past init is fixed-size
 * local scratch (not carved from the workspace), which is a deliberate
 * simplification given every real use in this project asks for well under
 * that (OFDM_modem.py's own default is 16). Raise the cap if a future config
 * genuinely needs more.
 */
#ifndef RS255_H
#define RS255_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RS255_MAX_NSYM 64

typedef struct rs255 rs255;

/* Bytes of caller memory needed. 0 if nsym is out of range (1..RS255_MAX_NSYM). */
size_t rs255_workspace(int nsym);

/* Carves `mem`. Builds the GF(256) log/exp tables and the degree-`nsym`
 * generator polynomial once, so every encode/decode call is table lookups
 * only. Errors: -1 bad args/null, -2 memlen too small, -3 nsym out of range. */
int rs255_init(rs255 **rs, void *mem, size_t memlen, int nsym);

/* Encode `msg_len` data bytes into `out` as one systematic RS codeword:
 * the first msg_len bytes of `out` are the message (byte-identical, even
 * when out==msg, i.e. in-place is supported and is what ofdm_frame.c uses),
 * followed by nsym parity bytes. msg_len + nsym must be <= 255.
 * Returns msg_len+nsym bytes written, or -1 bad args, -2 out_cap too small. */
int rs255_encode(rs255 *rs, const unsigned char *msg, int msg_len,
                 unsigned char *out, int out_cap);

/* Decode one codeword of `in_len` bytes (message + nsym parity), correcting
 * up to floor(nsym/2) symbol errors. Writes the corrected message
 * (in_len - nsym bytes) to `out`. `*errata` (if non-NULL) receives the
 * number of corrected symbols.
 * Returns bytes written (in_len - nsym), or -1 bad args, -2 out_cap too
 * small, -3 uncorrectable (too many errors, or correction did not verify). */
int rs255_decode(rs255 *rs, const unsigned char *in, int in_len,
                 unsigned char *out, int out_cap, int *errata);

#ifdef __cplusplus
}
#endif
#endif
