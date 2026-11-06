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

#include "ofdm_frame.h"

int ofdm_frame_encode_il(rs255 *rs, const unsigned char *msg, int msg_len,
                         int depth, int k, int ecc,
                         unsigned char *wire, int wire_cap,
                         unsigned char *blk)
{
    int i, j, L, got;

    if (!rs || !msg || !wire || !blk) return -1;
    if (depth < 1 || k < 1 || ecc < 1 || msg_len < 0) return -1;
    L = k + ecc;
    if (wire_cap < depth * L) return -1;

    for (i = 0; i < depth; i++) {
        int base = i * k;
        for (j = 0; j < k; j++)
            blk[j] = (base + j < msg_len) ? msg[base + j] : 0;
        got = rs255_encode(rs, blk, k, blk, L);
        if (got != L) return -3;
        for (j = 0; j < L; j++) wire[j * depth + i] = blk[j];
    }
    return depth * L;
}

int ofdm_frame_decode_il(rs255 *rs, const unsigned char *wire, int wire_len,
                         int depth, int k, int ecc, int msg_len,
                         unsigned char *out, int out_cap,
                         unsigned char *blk, int *errata)
{
    int i, j, L, got, err, total = 0, written = 0;

    if (!rs || !wire || !out || !blk) return -1;
    if (depth < 1 || k < 1 || ecc < 1 || msg_len < 0) return -1;
    L = k + ecc;
    if (wire_len < depth * L) return -1;
    if (out_cap < msg_len) return -2;

    for (i = 0; i < depth; i++) {
        /* De-interleave codeword i: it occupies every depth-th wire byte. */
        for (j = 0; j < L; j++) blk[j] = wire[j * depth + i];
        err = 0;
        got = rs255_decode(rs, blk, L, blk, L, &err);
        if (got < 0) return -3;
        total += err;
        for (j = 0; j < got; j++) {
            int at = i * k + j;
            if (at < msg_len && at < out_cap) {
                out[at] = blk[j];
                if (at + 1 > written) written = at + 1;
            }
        }
    }
    if (errata) *errata = total;
    return written < msg_len ? written : msg_len;
}
