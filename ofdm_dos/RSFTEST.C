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

/* RSFTEST.C -- on-hardware self-test for rs255.c + ofdm_frame.c (RS(255) +
 * interleave, the data-plane codec layer, ported from the Linux reference at
 * OFDM/c/{rs255,ofdm_frame}.c).
 *
 * WHY THIS EXISTS SEPARATELY FROM OFDMBAT.C's own self-test (BATLOOP).
 * BATLOOP proves the bespoke BAT-table codec (batcodec.c), a simpler wire
 * format built just to move the ~35 entry bit-allocation table. This proves
 * the GENERAL RS(255) + interleave layer the eventual real data-frame modem
 * (ofdm_mod/ofdm_dem) depends on. No audio, no DSP, no DMA, no SB16 -- pure
 * CPU/memory, so it needs no VM sound config or PipeWire wiring.
 *
 * TIMING, MEASURED NOT ASSUMED. rs255/ofdm_frame are pure GF(2^8) table
 * lookups, no floating point, so unlike ofdm_probe_estimate (which has
 * measurably blown a 30s slot budget under load) there's no reason to expect
 * this to be CPU-bound. This reports its own encode/decode timing over COM1
 * rather than asserting that. Run on both the 25MHz and 50MHz test hosts: if
 * both land near the BIOS tick granularity (54.9ms at 18.2065 Hz), the codec
 * is free regardless of host speed and needs no host-specific timing budget.
 *
 * GEOMETRY. depth=2, k=24, ecc_bytes(nsym)=16, msg_len=40: two RS(40,24)-ish
 * codewords interleaved, inside the 64KB-DMA-page-sized frames this
 * project's other tests use, with enough correction budget (8 symbols/
 * codeword) to inject and recover a nontrivial error pattern per codeword.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "core/rs255.h"
#include "core/ofdm_frame.h"

static FILE *serial_out = NULL;

static void report(const char *fmt, ...)
{
    char msg[400];
    va_list ap;
    va_start(ap, fmt);
    vsprintf(msg, fmt, ap);
    va_end(ap);
    printf("%s\n", msg);
    if (serial_out) {
        fprintf(serial_out, "%s\n", msg);
        fflush(serial_out);
    }
}

#define BIOS_TICK_HZ 18.2065
static unsigned long bios_ticks(void)
{
    return *(volatile unsigned long *)0x46CUL;
}

#define NSYM  16
#define DEPTH 2
#define K     24
#define MSG_LEN 40

int main(void)
{
    unsigned char rsmem[4096];
    rs255 *rs;
    unsigned char msg[MSG_LEN], wire[DEPTH * (K + NSYM)], blk[K + NSYM];
    unsigned char out[MSG_LEN];
    int i, rc, wire_len, errata, ok = 1;
    unsigned long t0, t1, enc_ticks, dec_ticks;

    serial_out = fopen("COM1", "w");

    report("RSFTEST: rs255+ofdm_frame self-test, nsym=%d depth=%d k=%d msg_len=%d",
          NSYM, DEPTH, K, MSG_LEN);

    rc = rs255_init(&rs, rsmem, sizeof(rsmem), NSYM);
    if (rc != 0) {
        report("RSFTEST: rs255_init failed rc=%d", rc);
        report("RSFTEST: VERDICT FAIL");
        return 1;
    }

    /* Deterministic pseudo-random message, no rand() dependency on the
     * DOS/4G runtime's own seeding behaviour -- same reasoning as every
     * other fixed-pattern self-test in this project (BATLOOP included). */
    for (i = 0; i < MSG_LEN; i++) msg[i] = (unsigned char)(i * 37 + 11);

    t0 = bios_ticks();
    wire_len = ofdm_frame_encode_il(rs, msg, MSG_LEN, DEPTH, K, NSYM,
                                    wire, sizeof(wire), blk);
    t1 = bios_ticks();
    enc_ticks = t1 - t0;
    if (wire_len != DEPTH * (K + NSYM)) {
        report("RSFTEST: encode rc=%d, expected %d", wire_len, DEPTH * (K + NSYM));
        report("RSFTEST: VERDICT FAIL");
        return 1;
    }
    report("RSFTEST: encoded %d bytes in %lu tick(s) (%.1f ms @ %.4f Hz)",
          wire_len, enc_ticks, (double)enc_ticks * 1000.0 / BIOS_TICK_HZ, BIOS_TICK_HZ);

    /* Inject 8 errors (== NSYM/2, the guaranteed correction limit) into
     * EACH of the DEPTH codewords, at wire positions j*DEPTH+blk for
     * j=0..K+NSYM-1 -- see ofdm_frame.c's column-major interleave. Spread
     * rather than clustered, so this also exercises that the interleave
     * genuinely separates a burst the way its own header comment claims. */
    for (i = 0; i < DEPTH; i++) {
        int e;
        for (e = 0; e < NSYM / 2; e++) {
            int j = (e * (K + NSYM)) / (NSYM / 2);   /* spread across the codeword */
            int pos = j * DEPTH + i;
            wire[pos] ^= (unsigned char)(0x55 + e);
        }
    }

    t0 = bios_ticks();
    rc = ofdm_frame_decode_il(rs, wire, wire_len, DEPTH, K, NSYM, MSG_LEN,
                              out, sizeof(out), blk, &errata);
    t1 = bios_ticks();
    dec_ticks = t1 - t0;

    if (rc != MSG_LEN) {
        report("RSFTEST: decode rc=%d, expected %d", rc, MSG_LEN);
        ok = 0;
    } else if (memcmp(out, msg, MSG_LEN) != 0) {
        report("RSFTEST: decoded message does not match original");
        ok = 0;
    } else {
        report("RSFTEST: decoded %d bytes OK, %d total errata corrected, "
              "in %lu tick(s) (%.1f ms)",
              rc, errata, dec_ticks, (double)dec_ticks * 1000.0 / BIOS_TICK_HZ);
    }

    report(ok ? "RSFTEST: VERDICT PASS" : "RSFTEST: VERDICT FAIL");
    if (serial_out) fclose(serial_out);
    return ok ? 0 : 1;
}
