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

/* battest.c -- host round-trip harness for the BPSK BAT back-channel.
 *
 * NOT shipped to DOS. This exists so the codec is proven before a single
 * emulator run is spent on it: every previous fault in this project cost a
 * full hardware cycle to see, and an encode/decode bug is exactly the kind
 * that a host test catches in a second.
 *
 * It grades three things, because they fail differently:
 *   1. Clean round trip -- the table comes back bit-identical.
 *   2. Round trip with the receiver armed at an ARBITRARY OFFSET, which is
 *      the real operating condition (the far end arms whenever it arms, mid
 *      burst); this is what BAT_REPEATS and the sync search exist for.
 *   3. Round trip under additive noise at descending SNR, to find where it
 *      actually breaks rather than assuming the margin is fine.
 *
 * Build: gcc -std=c89 -DOFDM_C89_MATH -I core -o battest battest.c batcodec.c \
 *            core/ofdm_cfg.c core/ofdm_fft.c core/ofdm_sym.c core/ofdm_acq.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "batcodec.h"
#include "ofdm_cfg.h"

static unsigned long rng = 12345u;
static double urand(void)
{
    rng = rng * 1103515245u + 12345u;
    return (double)((rng >> 16) & 0x7FFFu) / 32768.0;
}
static double gauss(void)
{
    double u1 = urand(), u2 = urand();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

static int run_case(const ofdm_cfg *c, void *ws, size_t wslen,
                    const int *table, int n, double snr_db, long shift,
                    const char *label)
{
    long cap = bat_encode_samples(c, n);
    long total, wrote, i;
    double *tx, *rx, p = 0.0, sigma;
    int *out, rc, bad = 0;

    if (cap <= 0) { printf("  %-34s ENCODE SIZE FAILED\n", label); return 1; }

    /* Two full bursts back to back, so an arbitrary arming offset still has a
       complete frame ahead of it -- the same reason the transmitter loops. */
    total = cap * 2 + shift;
    tx  = (double *)calloc((size_t)total, sizeof(double));
    rx  = (double *)calloc((size_t)total, sizeof(double));
    out = (int *)calloc((size_t)n + 8, sizeof(int));
    if (!tx || !rx || !out) { printf("  oom\n"); return 1; }

    wrote = bat_encode(c, ws, wslen, table, n, tx, cap);
    if (wrote != cap) {
        printf("  %-34s ENCODE rc=%ld (expected %ld)\n", label, wrote, cap);
        free(tx); free(rx); free(out); return 1;
    }
    for (i = 0; i < cap; i++) { tx[cap + i] = tx[i]; }

    for (i = 0; i < cap * 2; i++) p += tx[i] * tx[i];
    p /= (double)(cap * 2);
    sigma = (snr_db > 90.0) ? 0.0 : sqrt(p / pow(10.0, snr_db / 10.0));

    for (i = 0; i < total; i++) {
        double v = (i >= shift && i < shift + cap * 2) ? tx[i - shift] : 0.0;
        rx[i] = v + sigma * gauss();
    }

    rc = bat_decode(c, ws, wslen, rx, total, out, n + 8);
    if (rc != n) {
        printf("  %-34s DECODE rc=%d (wanted %d entries)\n", label, rc, n);
        free(tx); free(rx); free(out); return 1;
    }
    for (i = 0; i < n; i++) if (out[i] != table[i]) bad++;
    printf("  %-34s %s (%d/%d entries exact)\n", label,
           bad ? "MISMATCH" : "ok", n - bad, n);
    free(tx); free(rx); free(out);
    return bad ? 1 : 0;
}

static int run_geometry(int geo)
{
    ofdm_cfg c;
    void *ws;
    size_t wslen;
    int table[128];
    int i, n, fails = 0;
    static const double snrs[6] = { 40.0, 30.0, 20.0, 15.0, 10.0, 6.0 };
    char label[64];

    /* Both geometries this project actually runs: the main audiomodem test,
       and the widest survey point. The survey has ~2.5x the carriers and a
       4x longer symbol, so it exercises the group-splitting and the frame
       length in a way the small geometry cannot. */
    if (ofdm_cfg_build(&c, geo ? 256 : 128, 0.25,
                       geo ? 200.0 : 1000.0, geo ? 18000.0 : 8000.0, 8,
                       geo ? 44100 : 22050,
                       16, 256, 2, 2.0, 20.0, 32) != 0) {
        printf("cfg_build failed\n");
        return 1;
    }
    wslen = bat_workspace(&c);
    ws = malloc(wslen);
    if (!ws) { printf("oom\n"); return 1; }

    n = c.n_data_carriers;
    if (n > 128) n = 128;
    /* A deliberately non-uniform table: an all-4s table would still decode if
       whole groups were stuck, and would prove nothing. */
    for (i = 0; i < n; i++) table[i] = (i * 7 + 3) % 7;

    printf("\n=== BAT back-channel round trip, geometry %d ===\n", geo);
    printf("  geometry n_fft=%d occ=%d data=%d sym=%d, %d entries\n",
           c.n_fft, c.n_occupied, c.n_data_carriers, c.symbol_samples, n);
    printf("  burst %ld samples (%.2f s at %d Hz), workspace %lu B\n\n",
           bat_encode_samples(&c, n),
           (double)bat_encode_samples(&c, n) / c.sample_rate, c.sample_rate,
           (unsigned long)wslen);

    fails += run_case(&c, ws, wslen, table, n, 999.0, 0, "clean, aligned");
    fails += run_case(&c, ws, wslen, table, n, 999.0, 37, "clean, offset 37");
    fails += run_case(&c, ws, wslen, table, n, 999.0, 1234, "clean, offset 1234");

    for (i = 0; i < 6; i++) {
        sprintf(label, "noise %.0f dB SNR, offset 501", snrs[i]);
        fails += run_case(&c, ws, wslen, table, n, snrs[i], 501, label);
    }

    free(ws);
    return fails;
}

int main(void)
{
    int fails = run_geometry(0) + run_geometry(1);
    printf("\n%s\n", fails ? "SOME CASES FAILED" : "ALL CASES PASSED");
    return fails ? 1 : 0;
}
