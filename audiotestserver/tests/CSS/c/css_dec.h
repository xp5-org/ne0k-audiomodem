/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
/* css_dec.h — CSS receive frame pipeline, freestanding C89.
 *
 * Everything between "one soft bit per received chirp" and "decoded payload
 * bytes": deinterleave, soft-decision Viterbi, differential decode, bit
 * packing and the CRC check.  Port of css_fec.py and css_decoder._frame_pipeline().
 *
 * Reed-Solomon is not linked here; the chain splits either side of it:
 *
 *     css_dec_raw()    -> deinterleave -> viterbi -> diff -> bytes
 *     [ rs255_decode() ]  optional, caller's call
 *     css_check_crc()  -> seq + payload, or rejected
 *
 * css_dec_raw hands back the per-byte minimum soft confidence so a caller
 * that wants RS erasure marking can build the list itself. The C RS core has
 * no erasure decoder, and only the legacy streaming css_decoder.decode()
 * marks erasures (decode_longsym_mf does not), so the policy stays with the
 * caller.
 *
 * C89: every declaration sits at the top of its block, before any statement
 * in that block; `make c89check` enforces it.
 */
#ifndef CSS_DEC_H
#define CSS_DEC_H
#include <stddef.h>
/* css_enc.h, not a forward declaration: the RX pipeline is driven by the same
 * css_enc_cfg the transmitter derived, and reuses css_crc32. A decoder whose
 * base_bits disagrees with the encoder's does not fail loudly: it CRC-rejects
 * every frame and looks like a bad link. The cost is an RX-only build also
 * linking css_enc.c for one CRC routine and a struct. */
#include "css_enc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Path-metric precision. Metrics accumulate |bm| <= 2 per step, so a
 * 1000-pair frame reaches |pm| ~ 2000, where float32 still resolves ~1e-4.
 * The risk is not magnitude but ties: two survivors whose metrics differ by
 * less than float32 precision can resolve the other way under float64.
 *
 * float is the default; -DCSS_DEC_F64_METRIC selects double so the harness
 * can grade the port against Python's float64 arithmetic separately from the
 * precision choice itself, per regression/css_dec_regress.py.
 */
#ifdef CSS_DEC_F64_METRIC
typedef double css_metric;
#else
typedef float  css_metric;
#endif

#define CSS_CONV_TAIL 6         /* K-1 flush bits the encoder appended */

/* Bytes of caller memory css_viterbi_k7 needs for a frame of `npairs`
 * received symbol pairs. 0 if npairs is out of range.
 *
 * The survivor store is one bit per (step, state), not a (prev_state, bit)
 * pair: given the next state, the input bit is its top bit and the previous
 * state is determined up to its own lowest bit, so a single bit recovers the
 * whole transition. That is 8 bytes a step instead of 64. */
size_t css_viterbi_workspace(int npairs);

/* Soft-decision Viterbi for the rate-1/2 K=7 NASA code (0o171 / 0o133).
 *
 * conf[2*i], conf[2*i+1] are the soft values of received pair i, in [-1, +1]:
 * positive means '1' (up-chirp). npairs pairs decode to npairs-6 bits, the
 * encoder's flush tail being dropped.
 *
 * `bits` receives npairs-6 values (0/1). `energy`, if non-NULL, receives the
 * same count of per-bit soft energies: (|conf0| + |conf1|) / 2 for the pair
 * that produced each bit, 0 where no survivor existed. This is a
 * calibration-free symbol-SNR measure used for RS erasure marking.
 *
 * Returns bits written, or -1 bad args, -2 workspace too small. */
int css_viterbi_k7(const float *conf, int npairs, void *work, size_t worklen,
                   unsigned char *bits, float *energy);

/* Inverse of the transmit block interleaver, applied to the bits and their
 * confidences together: they are the same permutation and must not be
 * separated. `n` must be rows*cols. conf/conf_out may both be NULL to permute
 * bits alone. Out buffers must not alias the inputs. Returns n, or -1 on bad
 * args. */
int css_deinterleave(const unsigned char *bits, const float *conf, int n,
                     int rows, unsigned char *bits_out, float *conf_out);

/* recovered[i] = received[i] XOR received[i-1], with received[-1] = ref.
 * Not the inverse spelling of css_diff_encode, which XORs its own output;
 * this decoder XORs its own input. Safe in place. Returns n. */
int css_diff_decode(const unsigned char *bits, int n, int ref,
                    unsigned char *out);

/* MSB-first bit packing, the inverse of css_bytes_to_bits.  Writes n/8 bytes
 * and ignores a trailing partial byte, as get_bytes_from_bits does.
 * Returns bytes written, or -1. */
int css_bits_to_bytes(const unsigned char *bits, int n, unsigned char *out);

/* CRC check over seq | data[chunk_size] | crc32.
 * `raw` must hold at least 1 + chunk_size + 4 bytes.  On a match writes the
 * sequence number to *seq (if non-NULL) and returns 1; returns 0 on mismatch,
 * -1 on bad args.  The payload is raw[1 .. 1+chunk_size). */
int css_check_crc(const unsigned char *raw, int nraw, int chunk_size,
                  int *seq);

/* deinterleave -> viterbi -> diff -> bytes, the receive bit pipeline.
 *
 * Mirrors css_decoder._frame_pipeline up to (not including) Reed-Solomon.
 * `bits`/`conf` are frame_symbols long: one hard decision and one soft
 * confidence per received chirp. Geometry comes from the same css_enc_cfg
 * the transmitter used.
 *
 * `raw` receives base_bits/8 bytes (the RS-encoded inner frame, or the inner
 * frame itself when rs_parity is 0). `byte_conf`, if non-NULL, receives one
 * value per raw byte: the minimum |conf| across that byte's eight bits,
 * which is the statistic RS erasure marking thresholds on.
 *
 * `work` must be at least css_dec_workspace(g) bytes.
 * Returns bytes written to `raw`, or -1 bad args, -2 workspace too small. */
int css_dec_raw(const css_enc_cfg *g, const unsigned char *bits,
                const float *conf, int nbits, int diff_ref,
                void *work, size_t worklen,
                unsigned char *raw, int rawcap, float *byte_conf);

/* Bytes of caller memory css_dec_raw needs for this geometry. */
size_t css_dec_workspace(const css_enc_cfg *g);

#ifdef __cplusplus
}
#endif
#endif
