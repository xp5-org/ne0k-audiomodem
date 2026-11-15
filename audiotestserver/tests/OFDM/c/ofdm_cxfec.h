/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_cxfec.h -- inner FEC for complex-baseband OFDM (step 10c), freestanding C89.
 *
 * The real path's frame is RS + byte interleave + scrambler + hard decisions.
 * Complex mode can add an inner code between the scrambled RS bytes and the
 * constellation mapper:
 *
 *   wire bytes -> bits (MSB first) -> scrambler -> conv encode K=7 (133, 171
 *   octal, the 802.11 / DVB pair), 6 zero tail bits -> puncture -> bit
 *   interleave over the frame -> bit-loaded mapper
 *
 * and the receiver demaps max-log LLRs weighted by |H|^2 (a dead carrier gives
 * LLR ~0: an erasure for free), de-interleaves, de-punctures (0 for the bits
 * that were not sent) and runs a soft-decision Viterbi before the RS.
 *
 * Rates (OFDM_CXFEC_*): 1/2, 2/3 = [c0 c1][c0 -], 3/4 = [c0 c1][c0 -][- c1].
 * Interleaver: coded bit i -> frame bit position (i * P) mod n_coded, with P
 * the first value >= bps + bps/3 + 1 coprime to n_coded, so neighbouring
 * coded bits land one symbol and a third of the band apart.
 *
 * LLR sign: positive = bit 0 more likely.
 * Contract as the other cores: no malloc, no stdio, no globals.
 */
#ifndef OFDM_CXFEC_H
#define OFDM_CXFEC_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OFDM_CXFEC_NONE 0
#define OFDM_CXFEC_R12  1
#define OFDM_CXFEC_R23  2
#define OFDM_CXFEC_R34  3
#define OFDM_CXFEC_TAIL 6

/* 1 if fec is one of the codes above (NONE included). */
int  ofdm_cxfec_valid(int fec);

/* Coded bits for n_info information bits (the tail included). -1 bad args. */
long ofdm_cxfec_coded_bits(int fec, long n_info);

/* Coded index of the first transmitted bit of trellis step j (for mapping an
 * information bit back to the cell that carries it). */
long ofdm_cxfec_coded_index(int fec, long j);

/* Interleaver step for n_coded bits on a table of bps bits per symbol. */
long ofdm_cxfec_perm_step(long n_coded, int bps);

/* info[n_info] (0/1) -> coded[ofdm_cxfec_coded_bits] (0/1), tail added. */
long ofdm_cxfec_encode(int fec, const unsigned char *info, long n_info,
                       unsigned char *coded);

/* Workspace for ofdm_cxfec_decode on n_info information bits. */
size_t ofdm_cxfec_workspace(long n_info);

/* llr[ofdm_cxfec_coded_bits(fec, n_info)] in coded (de-interleaved) order ->
 * info[n_info] (0/1).  Returns 0, -1 bad args, -2 work too small. */
int ofdm_cxfec_decode(int fec, const float *llr, long n_info,
                      unsigned char *info, void *work, size_t worklen);

#ifdef __cplusplus
}
#endif
#endif
