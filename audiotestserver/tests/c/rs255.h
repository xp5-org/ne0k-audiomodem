/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* rs255.h — Reed-Solomon over GF(256), freestanding C99.
 *
 * Byte-compatible with the `reedsolo` package: RSCodec(nsym) with library
 * defaults, i.e. nsize=255, fcr=0, prim=0x11D, generator=2, c_exp=8. Fixed
 * here rather than configurable.
 *
 * Shared by CSS (css_fec, css_encoder), OFDM (ofdm_modem, ofdm_reader/writer)
 * and QAM. Pure integer GF arithmetic, no floating point, builds standalone
 * into librs255.so.
 *
 * Blocking: reedsolo splits a message into blocks of (255 - nsym) bytes and
 * appends nsym parity to each, so a 240-byte message at nsym=16 encodes to
 * 272 bytes, not 256. Frame geometry is derived from this.
 *
 * Same contract as the other cores: no malloc, no stdio, no globals, one
 * caller-owned workspace block. GF log/antilog tables are built in init
 * rather than compiled in as static data.
 */
#ifndef RS255_H
#define RS255_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rs255 rs255;

/* Bytes of caller memory needed for this nsym. 0 if nsym is out of range
 * (1..254). */
size_t rs255_workspace(int nsym);

/* Carves `mem`.  Returns 0 ok, -1 bad args, -2 memlen too small. */
int rs255_init(rs255 **rs, void *mem, size_t memlen, int nsym);

/* Encoded length of an n-byte message, without encoding it. Lets a caller
 * size a buffer without a dry run. Negative on bad args. */
int rs255_encoded_len(const rs255 *rs, int n);

/* msg[n] -> out, block-chunked.  Returns bytes written, or
 * -1 bad args, -2 out_cap too small. */
int rs255_encode(rs255 *rs, const unsigned char *msg, int n,
                 unsigned char *out, int out_cap);

/* buf[n] (a full encoded stream) -> out, message bytes only.
 * `*errata` receives the total number of symbols corrected across all blocks,
 * which is what the callers use as their error count.
 * Returns message bytes written, or
 * -1 bad args, -2 out_cap too small, -3 a block was uncorrectable
 * (reedsolo raises ReedSolomonError here; the bridge turns -3 back into that). */
int rs255_decode(rs255 *rs, const unsigned char *buf, int n,
                 unsigned char *out, int out_cap, int *errata);

/* Introspection: parity symbols per block, and the message bytes per block. */
int rs255_nsym(const rs255 *rs);
int rs255_block_k(const rs255 *rs);

#ifdef __cplusplus
}
#endif
#endif
