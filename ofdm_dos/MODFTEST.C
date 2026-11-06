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

/* MODFTEST.C -- on-hardware self-test for the ofdm_mod/ofdm_sync/ofdm_dem
 * data-frame codec, layered on the already-proven rs255/ofdm_frame (see
 * RSFTEST.C).
 *
 * Same discipline as RSFTEST: this is the CPU/DSP self-loop only -- encode a
 * known message with ofdm_mod, run ofdm_sync/ofdm_dem/ofdm_frame_decode_il
 * against the SAME in-memory buffer (no SB16, no DMA, no PipeWire), and
 * report PASS/FAIL plus per-stage BIOS-tick timing over COM1.
 *
 * WHY TIMING MATTERS HERE, UNLIKE RSFTEST. rs255/ofdm_frame are integer table
 * lookups (0 ticks measured on both hosts). ofdm_mod/ofdm_dem/ofdm_sync do
 * real floating-point FFTs per OFDM symbol, closer in shape to
 * ofdm_probe_estimate, which has been measured blowing a 30s slot budget
 * under load -- reporting encode/sync/demod timing separately here, before a
 * two-way exchange is built on top, is cheaper than rediscovering the same
 * slot-timing problem later.
 *
 * GEOMETRY. n_fft=128, cp_ratio=0.25, freq 1000-8000 Hz, pilot_spacing=8,
 * sample_rate=22050 -- the same bin=[6,46]/n_occupied=41 geometry OFDMBAT.C's
 * probe uses in production. QPSK (2 bits/carrier) on every data carrier, one
 * chunk_bytes=20 message, ecc_bytes=8, sync_period=8.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

#include "core/ofdm_cfg.h"
#include "core/ofdm_mod.h"
#include "core/ofdm_dem.h"
#include "core/ofdm_sync.h"
#include "core/ofdm_frame.h"
#include "core/rs255.h"

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

#define N_FFT        128
#define CP_RATIO     0.25
#define FREQ_LO      1000.0
#define FREQ_HI      8000.0
#define PILOT_SPACING 8
#define SAMPLE_RATE  22050
#define ECC_BYTES    8
#define CHUNK_BYTES  20
#define SEQ_BYTES    2
#define TAPER_MS     2.0
#define FRAME_GAP_MS 20.0
#define SYNC_PERIOD  8

static unsigned char g_modmem[16384];
static unsigned char g_demmem[65536];
static unsigned char g_syncmem[16384];
static unsigned char g_rsmem[4096];
static double        g_sig[4096];

static void report_ticks(const char *label, unsigned long t0, unsigned long t1)
{
    unsigned long dt = t1 - t0;
    report("MODFTEST: %s: %lu tick(s) (%.1f ms)", label, dt,
          (double)dt * 1000.0 / BIOS_TICK_HZ);
}

int main(void)
{
    ofdm_cfg c;
    int rc, n_data, i;
    int bits_per_bin[64];
    rs255 *rs;
    ofdm_mod *m;
    ofdm_dem *d;
    ofdm_sync *sy;
    size_t need;
    unsigned char message[CHUNK_BYTES];
    long sig_len;
    unsigned long t0, t1;

    serial_out = fopen("COM1", "w");
    report("MODFTEST: ofdm_mod+ofdm_sync+ofdm_dem+ofdm_frame self-test");

    rc = ofdm_cfg_build(&c, N_FFT, CP_RATIO, FREQ_LO, FREQ_HI, PILOT_SPACING,
                        SAMPLE_RATE, ECC_BYTES, CHUNK_BYTES, SEQ_BYTES,
                        TAPER_MS, FRAME_GAP_MS, SYNC_PERIOD);
    if (rc != 0) { report("MODFTEST: cfg build failed rc=%d", rc); goto fail; }
    report("MODFTEST: n_occupied=%d n_data_carriers=%d n_pilots=%d symbol_samples=%d",
          c.n_occupied, c.n_data_carriers, c.n_pilots, c.symbol_samples);

    n_data = c.n_data_carriers;
    for (i = 0; i < n_data; i++) bits_per_bin[i] = 2; /* QPSK */

    rc = rs255_init(&rs, g_rsmem, sizeof(g_rsmem), c.ecc_bytes);
    if (rc != 0) { report("MODFTEST: rs255_init failed rc=%d", rc); goto fail; }

    need = ofdm_mod_workspace(&c, bits_per_bin);
    if (need == 0 || need > sizeof(g_modmem)) {
        report("MODFTEST: mod workspace need=%lu, have %lu",
              (unsigned long)need, (unsigned long)sizeof(g_modmem));
        goto fail;
    }
    rc = ofdm_mod_init(&m, g_modmem, sizeof(g_modmem), &c, bits_per_bin);
    if (rc != 0) { report("MODFTEST: ofdm_mod_init failed rc=%d", rc); goto fail; }

    for (i = 0; i < CHUNK_BYTES; i++) message[i] = (unsigned char)(i * 13 + 5);

    t0 = bios_ticks();
    sig_len = ofdm_mod_encode(m, message, CHUNK_BYTES, g_sig,
                             (long)(sizeof(g_sig) / sizeof(double)));
    t1 = bios_ticks();
    report_ticks("encode", t0, t1);
    if (sig_len < 0) { report("MODFTEST: encode failed rc=%ld", sig_len); goto fail; }
    report("MODFTEST: encoded %ld samples (frame_len=%ld, payload_bytes=%d)",
          sig_len, ofdm_mod_frame_samples(m), ofdm_mod_payload_bytes(m));

    need = ofdm_sync_workspace(&c);
    if (need == 0 || need > sizeof(g_syncmem)) {
        report("MODFTEST: sync workspace need=%lu, have %lu",
              (unsigned long)need, (unsigned long)sizeof(g_syncmem));
        goto fail;
    }
    rc = ofdm_sync_init(&sy, g_syncmem, sizeof(g_syncmem), &c, ofdm_mod_sc_preamble(m));
    if (rc != 0) { report("MODFTEST: ofdm_sync_init failed rc=%d", rc); goto fail; }

    {
        int starts[8], n_starts, best, refined;
        double score;

        t0 = bios_ticks();
        n_starts = ofdm_sync_find(sy, g_sig, (int)sig_len, ofdm_mod_body_syms(m),
                                  OFDM_SC_THRESHOLD, OFDM_SC_NCC_MIN, starts, 8);
        t1 = bios_ticks();
        report_ticks("sync_find", t0, t1);
        if (n_starts < 1) { report("MODFTEST: no frame found"); goto fail; }
        best = starts[0];

        t0 = bios_ticks();
        rc = ofdm_sync_refine(sy, g_sig, (int)sig_len, best, &refined, &score);
        t1 = bios_ticks();
        report_ticks("sync_refine", t0, t1);
        if (rc != 0) { report("MODFTEST: refine failed rc=%d", rc); goto fail; }
        report("MODFTEST: sync score=%.4f raw=%d refined=%d", score, best, refined);
        best = refined;

        {
            int pilot_offsets[16], n_pilots = 0;
            double ce_re[64], ce_im[64];
            int widths[4], nw;
            int tab_b[4], tab_n[4];
            double pts_re[64], pts_im[64];
            unsigned char tab_bits[256];
            int n_pts_total = 0, n_bits_total = 0;
            unsigned char wire[128];
            ofdm_dem_out info;

            for (i = 0; i < c.n_occupied; i += c.pilot_spacing) pilot_offsets[n_pilots++] = i;
            ofdm_known_bpsk(ce_re, c.n_occupied, OFDM_SEED_CE, OFDM_REF_SYM_BOOST / sqrt(2.0));
            for (i = 0; i < c.n_occupied; i++) ce_im[i] = 0.0;

            nw = ofdm_bat_widths(bits_per_bin, n_data, widths, 4);
            rc = ofdm_mod_tables_flat(widths, nw, tab_b, tab_n, pts_re, pts_im,
                                      tab_bits, 64, 256, &n_pts_total, &n_bits_total);
            if (rc != 0) { report("MODFTEST: tables_flat failed rc=%d", rc); goto fail; }

            need = ofdm_dem_workspace(c.n_fft, c.cp_len, c.n_occupied, n_pilots,
                                      n_data, ofdm_mod_data_syms(m), c.sync_period,
                                      ofdm_mod_payload_bytes(m), n_pts_total, n_bits_total);
            if (need == 0 || need > sizeof(g_demmem)) {
                report("MODFTEST: dem workspace need=%lu, have %lu",
                      (unsigned long)need, (unsigned long)sizeof(g_demmem));
                goto fail;
            }
            rc = ofdm_dem_init(&d, g_demmem, sizeof(g_demmem),
                               c.n_fft, c.cp_len, c.bin_lo, c.n_occupied,
                               ce_re, ce_im, ce_re, ce_im,
                               pilot_offsets, n_pilots,
                               0.70710678118654752, 0.70710678118654752,
                               bits_per_bin, n_data,
                               ofdm_mod_data_syms(m), c.sync_period,
                               ofdm_mod_payload_bytes(m), OFDM_SEED_DATA,
                               tab_b, tab_n, nw, pts_re, pts_im, tab_bits);
            if (rc != 0) { report("MODFTEST: ofdm_dem_init failed rc=%d", rc); goto fail; }

            t0 = bios_ticks();
            rc = ofdm_dem_frame(d, g_sig, (int)sig_len, best, wire, sizeof(wire), &info);
            t1 = bios_ticks();
            report_ticks("dem_frame", t0, t1);
            if (rc < 0) { report("MODFTEST: dem_frame failed rc=%d", rc); goto fail; }
            report("MODFTEST: evm=%.4f data_syms=%d n_bytes=%d",
                  info.evm, info.data_syms, info.n_bytes);

            {
                int depth, k, wire_len2, errata;
                unsigned char decoded[SEQ_BYTES + CHUNK_BYTES];
                unsigned char blk[128];
                int msg_len = SEQ_BYTES + CHUNK_BYTES;

                rc = ofdm_interleave_geometry(&c, ofdm_mod_bits_per_sym(m), &depth, &k, &wire_len2);
                if (rc != 0) { report("MODFTEST: interleave_geometry failed rc=%d", rc); goto fail; }

                t0 = bios_ticks();
                rc = ofdm_frame_decode_il(rs, wire, wire_len2, depth, k, c.ecc_bytes,
                                          msg_len, decoded, sizeof(decoded), blk, &errata);
                t1 = bios_ticks();
                report_ticks("frame_decode_il", t0, t1);
                if (rc != msg_len) { report("MODFTEST: frame decode failed rc=%d", rc); goto fail; }
                report("MODFTEST: errata=%d", errata);

                {
                    unsigned seq = 0;
                    for (i = 0; i < SEQ_BYTES; i++) seq = (seq << 8) | decoded[i];
                    if (seq != 0 || memcmp(decoded + SEQ_BYTES, message, CHUNK_BYTES) != 0) {
                        report("MODFTEST: recovered message does not match (seq=%u)", seq);
                        goto fail;
                    }
                }
            }
        }
    }

    report("MODFTEST: VERDICT PASS");
    if (serial_out) fclose(serial_out);
    return 0;

fail:
    report("MODFTEST: VERDICT FAIL");
    if (serial_out) fclose(serial_out);
    return 1;
}
