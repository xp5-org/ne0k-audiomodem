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

/* OFDMDATA.C -- one-way REAL PAYLOAD exchange over the actual SB16 hardware
 * path: `OFDMDATA TX` plays a known message encoded with ofdm_mod; `OFDMDATA
 * RX` records over the SB16 ADC and recovers it via ofdm_sync -> ofdm_dem ->
 * ofdm_frame_decode_il, the same chain MODFTEST proves as a pure CPU
 * self-loop, now over real audio hardware. Deliberately ONE-WAY: prove real
 * bytes survive the hardware path before attempting a two-way exchange,
 * which adds its own scheduling problems on top of a codec correctness one.
 *
 * SYNCHRONISATION: OVERLAP, NOT A RACE. Two independent DOS VMs cannot be
 * started in lockstep -- the testlist gates on RX's own "ARMED" over COM1,
 * then types TX's command into the other machine, and that skew is larger
 * than this frame's ~107 ms airtime. So TX plays the SAME encoded frame back
 * to back via the DMA's AUTO-INIT loop while RX makes ONE capture covering
 * close to a full 64KB DMA page (~2.9 s at 22050 Hz): as long as RX's window
 * lands anywhere inside TX's longer transmission it contains at least one
 * complete frame, and ofdm_sync_find's multi-candidate return means RX does
 * not need to know WHICH repetition it caught.
 *
 * GEOMETRY. Identical to MODFTEST.C and to OFDMBAT.C's own probe (n_fft=128,
 * bin=[6,46]/n_occupied=41, sample_rate=22050).
 *
 * VERIFICATION. RX knows the message TX sends (a validation test, not a real
 * link), so VERDICT PASS/FAIL is a direct byte-for-byte and sequence-number
 * compare.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <conio.h>
#include <dos.h>
#include <i86.h>
#include <math.h>

#include "core/ofdm_cfg.h"
#include "core/ofdm_mod.h"
#include "core/ofdm_dem.h"
#include "core/ofdm_sync.h"
#include "core/ofdm_frame.h"
#include "core/rs255.h"

/* ==================== SB16 / DMA plumbing -- copied from OFDMBAT.C's
 * audio_play/audio_record and their shared low-level helpers, deliberately
 * NOT shared via a header: this is the hard-won hardware code, and
 * restructuring it to serve a new caller risks the working behaviour for a
 * cosmetic gain (see DUPLEX.C's docstring for the same reasoning). ==== */

static unsigned g_sb_base = 0x220;
static unsigned g_sb_dma  = 1;

static const unsigned char DMA_ADDR[4] = { 0x00, 0x02, 0x04, 0x06 };
static const unsigned char DMA_CNT [4] = { 0x01, 0x03, 0x05, 0x07 };
static const unsigned char DMA_PAGE[4] = { 0x87, 0x83, 0x81, 0x82 };

#define PCM_SCALE 100.0

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

static void parse_blaster(void)
{
    char *b = getenv("BLASTER");
    unsigned v;
    if (!b) return;
    while (*b) {
        char k = *b++;
        if (k >= 'a' && k <= 'z') k = (char)(k - 'a' + 'A');
        if (k == 'A') {
            v = 0;
            while (*b && *b != ' ') {
                char c = *b++;
                if (c >= '0' && c <= '9')      v = v * 16 + (unsigned)(c - '0');
                else if (c >= 'A' && c <= 'F') v = v * 16 + (unsigned)(c - 'A' + 10);
                else if (c >= 'a' && c <= 'f') v = v * 16 + (unsigned)(c - 'a' + 10);
                else break;
            }
            if (v) g_sb_base = v;
        } else if (k == 'D') {
            v = 0;
            while (*b >= '0' && *b <= '9') v = v * 10 + (unsigned)(*b++ - '0');
            if (v < 4) g_sb_dma = v;
        }
        while (*b && *b != ' ') b++;
        while (*b == ' ') b++;
    }
}

static void dsp_write(unsigned char v)
{
    unsigned long n;
    for (n = 0; n < 200000UL; n++)
        if (!(inp(g_sb_base + 0x0C) & 0x80)) break;
    outp(g_sb_base + 0x0C, v);
}

static int dsp_reset(void)
{
    unsigned long n;
    outp(g_sb_base + 0x06, 1);
    for (n = 0; n < 1000UL; n++) (void)inp(0x80);
    outp(g_sb_base + 0x06, 0);
    for (n = 0; n < 200000UL; n++) {
        if (inp(g_sb_base + 0x0E) & 0x80) {
            if ((unsigned char)inp(g_sb_base + 0x0A) == 0xAA) return 1;
        }
    }
    return 0;
}

static void dma_program_mode(unsigned long phys, unsigned len, unsigned char mode)
{
    unsigned ch = g_sb_dma & 3;
    unsigned addr = (unsigned)(phys & 0xFFFFUL);
    unsigned cnt  = len - 1;
    outp(0x0A, (unsigned char)(0x04 | ch));
    outp(0x0C, 0);
    outp(0x0B, (unsigned char)(mode | ch));
    outp(DMA_ADDR[ch], (unsigned char)(addr & 0xFF));
    outp(DMA_ADDR[ch], (unsigned char)(addr >> 8));
    outp(DMA_PAGE[ch], (unsigned char)(phys >> 16));
    outp(0x0C, 0);
    outp(DMA_CNT[ch], (unsigned char)(cnt & 0xFF));
    outp(DMA_CNT[ch], (unsigned char)(cnt >> 8));
    outp(0x0A, (unsigned char)ch);
}

#define MIX_INPUT_LEFT   0x3D
#define MIX_INPUT_RIGHT  0x3E
#define IN_MIC     0x01
#define IN_LINE_R  0x08
#define IN_LINE_L  0x10

static void mixer_write(unsigned char reg, unsigned char val)
{
    outp(g_sb_base + 4, reg);
    outp(g_sb_base + 5, val);
}

static unsigned dma_read_count(void)
{
    unsigned ch = g_sb_dma & 3, lo, hi;
    outp(0x0C, 0);
    lo = (unsigned)inp(DMA_CNT[ch]);
    hi = (unsigned)inp(DMA_CNT[ch]);
    return lo | (hi << 8);
}

/* See OFDMBAT.C's own extensive comment on alloc_dma for why this goes
 * through DPMI INT 31h AX=0100h rather than _dos_allocmem, and why the
 * caller dereferences `phys` as a flat pointer rather than through the
 * 16-bit-limited selector DPMI hands back. */
static int alloc_dma(unsigned len, unsigned *selout, unsigned *rmsegout,
                     unsigned long *physout)
{
    union REGS r;
    unsigned rmseg, sel;
    unsigned long phys, base;

    memset(&r, 0, sizeof(r));
    r.w.ax = 0x0100;
    r.w.bx = (unsigned short)((2UL * (unsigned long)len + 15) / 16);
    int386(0x31, &r, &r);
    if (r.w.cflag) {
        report("OFDMDATA: DPMI 0100h failed (err %04Xh) for %u bytes", r.w.ax, len);
        return 0;
    }
    rmseg = r.w.ax;
    sel   = r.w.dx;
    base  = (unsigned long)rmseg << 4;
    if (rmseg < 0x0100) {
        report("OFDMDATA: implausible real-mode segment %04Xh", rmseg);
        return 0;
    }
    phys = base;
    if (((phys & 0xFFFFUL) + (unsigned long)len) > 0x10000UL)
        phys = (phys + 0xFFFFUL) & 0xFFFF0000UL;

    *selout   = sel;
    *rmsegout = rmseg;
    *physout  = phys;
    return 1;
}

static void free_dma(unsigned sel)
{
    union REGS r;
    memset(&r, 0, sizeof(r));
    r.w.ax = 0x0101;
    r.w.dx = (unsigned short)sel;
    int386(0x31, &r, &r);
}

static void dma_write(unsigned long phys, const void *src, unsigned len)
{
    memcpy((void *)phys, src, (size_t)len);
}

static void dma_read(void *dst, unsigned long phys, unsigned len)
{
    memcpy(dst, (const void *)phys, (size_t)len);
}

#define BIOS_TICK_HZ 18.2065
static unsigned long bios_ticks(void)
{
    return *(volatile unsigned long *)0x46CUL;
}

#define DMA_SENTINEL 0xA5

/* Play `sig` (n doubles) `reps` times through the DAC, blocking until done.
 * AUTO-INIT DMA (0x58) loops the SAME physical buffer -- reps only lengthens
 * the CPU-side delay(), it does not grow the DMA transfer or need more than
 * one 64KB page. Copied from OFDMBAT.C's XCHG audio_play verbatim. */
static int audio_play(const ofdm_cfg *c, const double *sig, long n, int reps)
{
    unsigned sel, rmseg, blk;
    unsigned long phys;
    unsigned char *pcm;
    long i;

    if (n <= 0 || n > 65024L) {
        report("OFDMDATA: waveform %ld samples does not fit one 64KB DMA page", n);
        return 0;
    }
    pcm = (unsigned char *)malloc((size_t)n);
    if (!pcm) { report("OFDMDATA: out of memory for %ld PCM bytes", n); return 0; }
    for (i = 0; i < n; i++) {
        double v = 128.0 + PCM_SCALE * sig[i];
        if (v < 0.0)   v = 0.0;
        if (v > 255.0) v = 255.0;
        pcm[i] = (unsigned char)v;
    }
    if (!dsp_reset()) { report("OFDMDATA: no DSP at %03Xh", g_sb_base); free(pcm); return 0; }
    if (!alloc_dma((unsigned)n, &sel, &rmseg, &phys)) {
        report("OFDMDATA: cannot allocate DMA buffer"); free(pcm); return 0;
    }
    dma_write(phys, pcm, (unsigned)n);
    dma_program_mode(phys, (unsigned)n, 0x58);   /* auto-init: loops for reps */
    dsp_write(0xD1);
    dsp_write(0x41);
    dsp_write((unsigned char)((c->sample_rate >> 8) & 0xFF));
    dsp_write((unsigned char)(c->sample_rate & 0xFF));
    blk = (unsigned)n - 1;
    dsp_write(0xC6);
    dsp_write(0x00);
    dsp_write((unsigned char)(blk & 0xFF));
    dsp_write((unsigned char)(blk >> 8));

    report("OFDMDATA TX: playing %ld samples x%d reps (%.1f s total)",
          n, reps, (double)(n * reps) / (double)c->sample_rate);
    delay((unsigned)((((n * reps) * 1000L) / c->sample_rate) + 100L));

    dsp_write(0xD0);
    dsp_write(0xDA);
    dsp_reset();
    dsp_write(0xD3);
    free_dma(sel);
    free(pcm);
    return 1;
}

/* Capture n samples through the ADC into `raw`, blocking until the transfer
 * completes. Single-cycle, one 64KB DMA page max. Copied from OFDMBAT.C's
 * XCHG audio_record verbatim (mixer routing included). */
static int audio_record(const ofdm_cfg *c, unsigned char *raw, long n)
{
    unsigned sel, rmseg, blk, resid;
    unsigned long phys, tick0, tick1, grace;
    int completed;

    if (n <= 0 || n > 65024L) {
        report("OFDMDATA: capture %ld samples does not fit one 64KB DMA page", n);
        return 0;
    }
    if (!dsp_reset()) { report("OFDMDATA: no DSP at %03Xh", g_sb_base); return 0; }
    if (!alloc_dma((unsigned)n, &sel, &rmseg, &phys)) {
        report("OFDMDATA: cannot allocate DMA buffer"); return 0;
    }
    memset(raw, DMA_SENTINEL, (size_t)n);
    dma_write(phys, raw, (unsigned)n);
    mixer_write(MIX_INPUT_LEFT,  (unsigned char)(IN_LINE_L | IN_LINE_R | IN_MIC));
    mixer_write(MIX_INPUT_RIGHT, (unsigned char)(IN_LINE_L | IN_LINE_R | IN_MIC));
    dma_program_mode(phys, (unsigned)n, 0x44);   /* record, single cycle */
    dsp_write(0x42);
    dsp_write((unsigned char)((c->sample_rate >> 8) & 0xFF));
    dsp_write((unsigned char)(c->sample_rate & 0xFF));
    blk = (unsigned)n - 1;
    dsp_write(0xC8);
    dsp_write(0x00);
    dsp_write((unsigned char)(blk & 0xFF));
    dsp_write((unsigned char)(blk >> 8));

    grace = (unsigned long)(((double)n / (double)c->sample_rate) * BIOS_TICK_HZ) + 28UL;
    tick0 = bios_ticks();
    completed = 0;
    for (;;) {
        resid = dma_read_count();
        tick1 = bios_ticks();
        if (resid > (unsigned)n) { completed = 1; break; }
        if (tick1 - tick0 > grace) break;
    }
    dsp_write(0xD0);
    dsp_write(0xDA);
    dsp_reset();
    dma_read(raw, phys, (unsigned)n);
    free_dma(sel);
    if (!completed) report("OFDMDATA: capture timed out (residual=%u)", resid);
    return 1;
}

/* ==================== geometry -- identical to MODFTEST.C ==================== */

#define N_FFT         128
#define CP_RATIO      0.25
#define FREQ_LO       1000.0
#define FREQ_HI       8000.0
#define PILOT_SPACING 8
#define SAMPLE_RATE   22050
#define ECC_BYTES     8
#define CHUNK_BYTES   20
#define SEQ_BYTES     2
#define TAPER_MS      2.0
#define FRAME_GAP_MS  20.0
#define SYNC_PERIOD   8

#define TX_REPEATS    60   /* ~6.4 s total airtime -- see file header. Needs to
                              comfortably exceed (PipeWire propagation wait +
                              RX's up-to-2.95s capture + dispatch/typing skew
                              between the two testlist commands), the same
                              margin reasoning OFDMBAT.C's tx_repeats uses. */
#define CAPTURE_SAMPLES 65024L /* one 64KB DMA page, ~2.95 s at 22050 Hz */

static unsigned char g_modmem[16384];
static unsigned char g_demmem[65536];
static unsigned char g_syncmem[16384];
static unsigned char g_rsmem[4096];
static double        g_sig[4096];
static unsigned char g_raw[CAPTURE_SAMPLES];
static double        g_rx[CAPTURE_SAMPLES];

static int build_geometry(ofdm_cfg *c, int *bits_per_bin, int *n_data)
{
    int rc = ofdm_cfg_build(c, N_FFT, CP_RATIO, FREQ_LO, FREQ_HI, PILOT_SPACING,
                            SAMPLE_RATE, ECC_BYTES, CHUNK_BYTES, SEQ_BYTES,
                            TAPER_MS, FRAME_GAP_MS, SYNC_PERIOD);
    if (rc != 0) return rc;
    *n_data = c->n_data_carriers;
    { int i; for (i = 0; i < *n_data; i++) bits_per_bin[i] = 2; /* QPSK */ }
    return 0;
}

static void run_tx(int tag)
{
    ofdm_cfg c;
    int n_data, rc, i;
    int bits_per_bin[64];
    rs255 *rs;
    ofdm_mod *m;
    size_t need;
    unsigned char message[CHUNK_BYTES];
    long sig_len;

    if (build_geometry(&c, bits_per_bin, &n_data) != 0) {
        report("OFDMDATA TX: cfg build failed"); return;
    }
    rc = rs255_init(&rs, g_rsmem, sizeof(g_rsmem), c.ecc_bytes);
    if (rc != 0) { report("OFDMDATA TX: rs255_init failed rc=%d", rc); return; }
    need = ofdm_mod_workspace(&c, bits_per_bin);
    if (need == 0 || need > sizeof(g_modmem)) {
        report("OFDMDATA TX: mod workspace need=%lu", (unsigned long)need); return;
    }
    rc = ofdm_mod_init(&m, g_modmem, sizeof(g_modmem), &c, bits_per_bin);
    if (rc != 0) { report("OFDMDATA TX: ofdm_mod_init failed rc=%d", rc); return; }

    /* Fixed, known message (perturbed by `tag` so each direction of a 2-way
       exchange carries genuinely distinct, independently-verifiable content
       rather than the same bytes coincidentally traveling both ways) -- RX
       is told the same tag for verification; this is a validation test, not
       a real link. */
    for (i = 0; i < CHUNK_BYTES; i++)
        message[i] = (unsigned char)((i * 13 + 5) ^ (tag * 0x55));

    sig_len = ofdm_mod_encode(m, message, CHUNK_BYTES, g_sig,
                             (long)(sizeof(g_sig) / sizeof(double)));
    if (sig_len < 0) { report("OFDMDATA TX: encode failed rc=%ld", sig_len); return; }
    report("OFDMDATA TX: tag=%d encoded %ld samples, playing", tag, sig_len);

    if (!audio_play(&c, g_sig, sig_len, TX_REPEATS)) {
        report("OFDMDATA TX: playback failed");
        return;
    }
    report("OFDMDATA TX: done");
}

static void run_rx(int tag)
{
    ofdm_cfg c;
    int n_data, rc, i;
    int bits_per_bin[64];
    rs255 *rs;
    ofdm_sync *sy;
    size_t need;
    int pilot_offsets[16], n_pilots = 0;
    double ce_re[64], ce_im[64];
    int widths[4], nw;
    int tab_b[4], tab_n[4];
    double pts_re[64], pts_im[64];
    unsigned char tab_bits[256];
    int n_pts_total = 0, n_bits_total = 0;
    unsigned char expected[CHUNK_BYTES];
    int msg_len = SEQ_BYTES + CHUNK_BYTES;
    int found_ok = 0;

    if (build_geometry(&c, bits_per_bin, &n_data) != 0) {
        report("OFDMDATA RX: cfg build failed"); goto fail;
    }
    for (i = 0; i < CHUNK_BYTES; i++)
        expected[i] = (unsigned char)((i * 13 + 5) ^ (tag * 0x55));
    report("OFDMDATA RX: expecting tag=%d", tag);

    rc = rs255_init(&rs, g_rsmem, sizeof(g_rsmem), c.ecc_bytes);
    if (rc != 0) { report("OFDMDATA RX: rs255_init failed rc=%d", rc); goto fail; }

    /* Need ofdm_mod's own instance only for its SC preamble / geometry
       introspection (frame_samples, body_syms, bits_per_sym, payload_bytes,
       data_syms) -- RX must derive these from cfg exactly the way TX did,
       since nothing about frame length travels on the wire. */
    {
        ofdm_mod *m;
        need = ofdm_mod_workspace(&c, bits_per_bin);
        if (need == 0 || need > sizeof(g_modmem)) {
            report("OFDMDATA RX: mod workspace need=%lu", (unsigned long)need); goto fail;
        }
        rc = ofdm_mod_init(&m, g_modmem, sizeof(g_modmem), &c, bits_per_bin);
        if (rc != 0) { report("OFDMDATA RX: ofdm_mod_init failed rc=%d", rc); goto fail; }

        need = ofdm_sync_workspace(&c);
        if (need == 0 || need > sizeof(g_syncmem)) {
            report("OFDMDATA RX: sync workspace need=%lu", (unsigned long)need); goto fail;
        }
        rc = ofdm_sync_init(&sy, g_syncmem, sizeof(g_syncmem), &c, ofdm_mod_sc_preamble(m));
        if (rc != 0) { report("OFDMDATA RX: ofdm_sync_init failed rc=%d", rc); goto fail; }

        report("OFDMDATA RX: ARMED, capturing %ld samples (%.2f s)",
              (long)CAPTURE_SAMPLES, (double)CAPTURE_SAMPLES / c.sample_rate);
        if (!audio_record(&c, g_raw, CAPTURE_SAMPLES)) {
            report("OFDMDATA RX: capture failed"); goto fail;
        }
        for (i = 0; i < CAPTURE_SAMPLES; i++)
            g_rx[i] = ((double)g_raw[i] - 128.0) / PCM_SCALE;

        {
            int starts[16], n_starts, s;
            n_starts = ofdm_sync_find(sy, g_rx, (int)CAPTURE_SAMPLES,
                                      ofdm_mod_body_syms(m), OFDM_SC_THRESHOLD,
                                      OFDM_SC_NCC_MIN, starts, 16);
            report("OFDMDATA RX: sync_find n_starts=%d", n_starts);
            if (n_starts < 1) { report("OFDMDATA RX: no frame found"); goto fail; }

            for (i = 0; i < c.n_occupied; i += c.pilot_spacing) pilot_offsets[n_pilots++] = i;
            ofdm_known_bpsk(ce_re, c.n_occupied, OFDM_SEED_CE, OFDM_REF_SYM_BOOST / sqrt(2.0));
            for (i = 0; i < c.n_occupied; i++) ce_im[i] = 0.0;
            nw = ofdm_bat_widths(bits_per_bin, n_data, widths, 4);
            rc = ofdm_mod_tables_flat(widths, nw, tab_b, tab_n, pts_re, pts_im,
                                      tab_bits, 64, 256, &n_pts_total, &n_bits_total);
            if (rc != 0) { report("OFDMDATA RX: tables_flat failed rc=%d", rc); goto fail; }

            for (s = 0; s < n_starts && !found_ok; s++) {
                ofdm_dem *d;
                int refined;
                double score;
                unsigned char wire[128];
                ofdm_dem_out info;

                rc = ofdm_sync_refine(sy, g_rx, (int)CAPTURE_SAMPLES, starts[s], &refined, &score);
                if (rc != 0) continue;

                need = ofdm_dem_workspace(c.n_fft, c.cp_len, c.n_occupied, n_pilots,
                                          n_data, ofdm_mod_data_syms(m), c.sync_period,
                                          ofdm_mod_payload_bytes(m), n_pts_total, n_bits_total);
                if (need == 0 || need > sizeof(g_demmem)) {
                    report("OFDMDATA RX: dem workspace need=%lu", (unsigned long)need);
                    continue;
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
                if (rc != 0) { report("OFDMDATA RX: ofdm_dem_init failed rc=%d", rc); continue; }

                rc = ofdm_dem_frame(d, g_rx, (int)CAPTURE_SAMPLES, refined, wire, sizeof(wire), &info);
                if (rc < 0) {
                    report("OFDMDATA RX: candidate %d (start=%d score=%.3f) dem_frame rc=%d",
                          s, refined, score, rc);
                    continue;
                }

                {
                    int depth, k, wire_len2, errata;
                    unsigned char decoded[SEQ_BYTES + CHUNK_BYTES];
                    unsigned char blk[128];

                    rc = ofdm_interleave_geometry(&c, ofdm_mod_bits_per_sym(m), &depth, &k, &wire_len2);
                    if (rc != 0) continue;
                    rc = ofdm_frame_decode_il(rs, wire, wire_len2, depth, k, c.ecc_bytes,
                                              msg_len, decoded, sizeof(decoded), blk, &errata);
                    if (rc != msg_len) {
                        report("OFDMDATA RX: candidate %d (start=%d score=%.3f) frame decode rc=%d",
                              s, refined, score, rc);
                        continue;
                    }
                    {
                        unsigned seq = 0;
                        for (i = 0; i < SEQ_BYTES; i++) seq = (seq << 8) | decoded[i];
                        report("OFDMDATA RX: candidate %d (start=%d score=%.3f) decoded seq=%u "
                              "errata=%d evm=%.4f", s, refined, score, seq, errata, info.evm);
                        if (memcmp(decoded + SEQ_BYTES, expected, CHUNK_BYTES) == 0) {
                            report("OFDMDATA RX: payload matches expected pattern");
                            found_ok = 1;
                        } else {
                            report("OFDMDATA RX: payload MISMATCH");
                        }
                    }
                }
            }
        }
    }

    if (found_ok) {
        report("OFDMDATA RX: VERDICT PASS");
        if (serial_out) fclose(serial_out);
        return;
    }

fail:
    report("OFDMDATA RX: VERDICT FAIL");
    if (serial_out) fclose(serial_out);
}

int main(int argc, char **argv)
{
    int tag = 0;

    parse_blaster();
    serial_out = fopen("COM1", "w");

    if (argc < 2) {
        report("usage: OFDMDATA TX|RX [tag]");
        if (serial_out) fclose(serial_out);
        return 2;
    }
    if (argc >= 3) tag = atoi(argv[2]);

    if (strcmp(argv[1], "TX") == 0 || strcmp(argv[1], "tx") == 0) {
        run_tx(tag);
        if (serial_out) fclose(serial_out);
        return 0;
    }
    if (strcmp(argv[1], "RX") == 0 || strcmp(argv[1], "rx") == 0) {
        run_rx(tag);
        return 0;
    }
    report("OFDMDATA: unknown role '%s'", argv[1]);
    if (serial_out) fclose(serial_out);
    return 2;
}
