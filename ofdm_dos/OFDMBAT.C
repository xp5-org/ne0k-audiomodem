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

/*
 * OFDMBAT.C -- SB16 OFDM channel-probe + bit-allocation ("BAT") exchange for
 * the 86Box "echo box" A+B pair, on OpenWatcom's 32-bit DOS-extended target.
 *
 * Ports sourcedir/86box_mic_echo/OFDM/c/ofdm_probe.c (and the ofdm_cfg/
 * ofdm_fft/ofdm_sym/ofdm_acq cores it depends on, vendored unmodified into
 * ./core/) into a single two-role DOS program: `OFDMBAT TX` builds the probe
 * waveform on-device and plays it over the SB16 DAC; `OFDMBAT RX` records
 * over the SB16 ADC, then runs the estimate/allocate pass and reports
 * per-carrier SNR, the ISI ratio, and the resulting bit-allocation table (the
 * "BAT") over COM1 and the screen. Needs the 32-bit target, not a 16-bit
 * real-mode one: ofdm_cfg.h stores sample_rate/n_fft/etc. as plain `int`,
 * which overflows at 16 bits.
 *
 * ONE binary, not two: a later round of work turns this into a genuine
 * two-way exchange (both ends probe each other and negotiate), which only
 * makes sense from a binary that already knows how to be either side.
 * `argv[1]` picks the role.
 *
 * All modem geometry comes from OFDMBAT.INI in the current directory. CLI is
 * kept to just the role, plus one optional override (RX's output WAV path).
 *
 * SB16 DMA plumbing (dsp_reset/alloc_dma/dma_program/dma_read_count) is the
 * exact 8-bit-only subset SB16ECHO.C already uses and validates. `_dos_allocmem`
 * returns a protected-mode SELECTOR under DOS/4GW, not a real-mode paragraph
 * segment, so its value shifted left by 4 is NOT a physical address and must
 * never be handed to the 8237. The buffer therefore comes from DPMI INT 31h
 * AX=0100h, which returns the real-mode segment and a selector separately --
 * see alloc_dma's comment below for the full derivation.
 *
 * Build: wcl386 -l=dos4g -3s -fpc -oneatx -DOFDM_C89_MATH -i=core OFDMBAT.C
 *        core/ofdm_cfg.c core/ofdm_fft.c core/ofdm_sym.c core/ofdm_acq.c
 *        core/ofdm_probe.c
 * Usage: ofdmbat TX
 *        ofdmbat RX [output.wav]
 *        ofdmbat RATESCAN -- sweeps ADC and DAC across programmed rates and
 *                            times each against the BIOS tick, to separate a
 *                            mis-scaled rate from an ignored one. Needs no
 *                            signal and no second machine.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <conio.h>
#include <dos.h>
#include <i86.h>
#include <math.h>

#include "ofdm_cfg.h"
#include "ofdm_fft.h"
#include "ofdm_probe.h"
#include "batcodec.h"

/* ---- SB16 defaults, DMA register tables -- identical to SPIKE.C/
 * SB16ECHO.C. 8-bit auto-init DMA only; this test does not need 16-bit
 * fidelity to prove the channel-probe pipeline works end to end. ---- */
static unsigned g_sb_base = 0x220;
static unsigned g_sb_dma  = 1;

static const unsigned char DMA_ADDR[4] = { 0x00, 0x02, 0x04, 0x06 };
static const unsigned char DMA_CNT [4] = { 0x01, 0x03, 0x05, 0x07 };
static const unsigned char DMA_PAGE[4] = { 0x87, 0x83, 0x81, 0x82 };

/* OFDM samples out of ofdm_probe_build are roughly unit-power per carrier and
   sum coherently at the SC preamble/CE peaks -- measured range on this INI's
   default geometry was about [-2.5, +2.1]. 40.0 leaves comfortable headroom
   (128 +/- 40*2.5 = 228..28, inside 0..255) without the clipping this
   project's mic-echo test already found the hard way at higher amplitudes. */
#define PCM_SCALE 40.0

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

/* The 8237 sequence, with the transfer mode chosen by the caller. Three modes
   are in use here and they must agree with the DSP command that follows:
     0x58  single mode + AUTO-INIT + read   -- TX playback, loops on purpose
     0x48  single mode + read, no auto-init -- RATESCAN playback, one pass
     0x44  single mode + write, no auto-init -- RX record, one pass then stop
   Kept as one function rather than copies, since the page/count/flip-flop
   order is the part that silently breaks when duplicated. */
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

static void dma_program(unsigned long phys, unsigned len, int record)
{
    dma_program_mode(phys, len, (unsigned char)(record ? 0x44 : 0x58));
}

/* ---- CT1745 mixer input routing -- copied from SB16ECHO.C, which has always
 * done this and always recorded real audio. OFDMBAT did NOT, and that is why
 * its first working capture came back a constant 128: with no source selected
 * for the ADC, the card faithfully digitises nothing, and 128 IS 8-bit
 * unsigned silence. Reg 0x3D selects the sources feeding the LEFT ADC channel,
 * 0x3E the RIGHT; a mono record drains only the left entry, so 0x3D is what
 * decides what this program hears. Line in on both, since the testlist routes
 * box1_out.monitor into box2's virtual mic via a PipeWire loopback. ---- */
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

static unsigned char mixer_read(unsigned char reg)
{
    outp(g_sb_base + 4, reg);
    return (unsigned char)inp(g_sb_base + 5);
}

static unsigned dma_read_count(void)
{
    unsigned ch = g_sb_dma & 3, lo, hi;
    outp(0x0C, 0);
    lo = (unsigned)inp(DMA_CNT[ch]);
    hi = (unsigned)inp(DMA_CNT[ch]);
    return lo | (hi << 8);
}

/* ---- ROOT CAUSE of the garbled playback and the low-memory crashes ----

   `_dos_allocmem` does NOT return a real-mode paragraph segment under this
   target -- under DOS/4GW it returns a protected-mode SELECTOR (8-byte
   aligned, far too low to be a paragraph segment a program could own: e.g.
   seg=01B8h puts 0x1B8<<4 about 7KB into memory, inside DOS itself).

   So `phys = seg << 4` was never a physical address: dma_program aimed the
   8237 at DOS's own low memory, so playback rendered DOS's data structures as
   a buffer-length loop, and a memset against that "address" scribbled over
   the DOS kernel. Reading the BIOS timer tick at PHYSICAL 0040:006C through a
   plain flat pointer returns a correctly-advancing count, which proves flat
   pointers DO reach physical memory here and leaves the address as the only
   suspect.

   Fix: DPMI INT 31h AX=0100h, "Allocate DOS Memory Block", which returns both
   halves explicitly -- AX = the real-mode segment, whose <<4 is a genuine
   physical address for the 8237, and DX = a selector covering the same bytes
   for CPU access.

   2*len is allocated so a sub-block can slide up past a 64KB physical
   boundary, which ISA 8-bit DMA cannot cross -- exactly as SB16ECHO.C's own
   alloc_dma does. */
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
        report("alloc_dma: DPMI 0100h failed (err %04Xh, largest free block "
              "%u paragraphs) for %u bytes", r.w.ax, r.w.bx, len);
        return 0;
    }
    rmseg = r.w.ax;
    sel   = r.w.dx;
    base  = (unsigned long)rmseg << 4;

    /* A conventional-memory block handed to a program never lives below
       0x0100 paragraphs -- DOS is down there. If this trips, something has
       again returned a selector where a real-mode segment was expected, and
       the one thing not to do is hand it to the DMA controller. */
    if (rmseg < 0x0100) {
        report("alloc_dma: implausible real-mode segment %04Xh -- refusing to "
              "use it as a physical address", rmseg);
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

/* Offset of the (possibly boundary-adjusted) buffer within the allocated
   block. Reported in the checkpoints because it is load-bearing: see below. */
#define DMA_OFF(phys, rmseg) ((unsigned)((phys) - ((unsigned long)(rmseg) << 4)))

/* ---- Why these go through the FLAT pointer and not the DPMI selector ----

   The selector DPMI 0100h returns is a 16-BIT selector with a 64KB limit, and
   alloc_dma's 2*len allocation can slide the buffer up near the far end of a
   block well past that (e.g. off=33408 + 57252 bytes = 90660 against a FFFFh
   limit -- a real GP fault on the RX copy-out). The flat pointer has no such
   limit and is already proven to reach physical memory (the BIOS tick read at
   PHYSICAL 0040:006C above), so `phys` is dereferenced directly; the selector
   is kept only as the handle free_dma needs. */
static void dma_write(unsigned long phys, const void *src, unsigned len)
{
    memcpy((void *)phys, src, (size_t)len);
}

static void dma_read(void *dst, unsigned long phys, unsigned len)
{
    memcpy(dst, (const void *)phys, (size_t)len);
}

/* BIOS timer tick at PHYSICAL 0040:006C, 18.2065 Hz, read through the flat
   pointer -- the same landmark whose correct advance proved flat addressing
   works in the first place. Timing the transfer against a clock the SB16 does
   not drive is what makes "how fast did the card ACTUALLY clock samples"
   measurable, which is the resampling question this whole test exists to ask:
   a card told 22050 Hz that really delivers 22050*(48000/44100) is exactly
   what a PipeWire rate mismatch would look like from in here. */
#define BIOS_TICK_HZ 18.2065
static unsigned long bios_ticks(void)
{
    return *(volatile unsigned long *)0x46CUL;
}

#define DMA_SENTINEL 0xA5

/* ==================== OFDMBAT.INI ==================== */

typedef struct {
    int    n_fft;
    double cp_ratio;
    double freq_lo;
    double freq_hi;
    int    pilot_spacing;
    int    sample_rate;
    double taper_ms;
    double frame_gap_ms;
    int    n_probe_syms;
    int    n_meas_syms;
    double snr_margin_db;
    int    max_bits_per_bin;
    int    sync_period;
    int    tx_repeats;
} ofdm_ini;

/* ecc/chunk/seq bytes are ofdm_cfg_build's data-frame-only fields -- this
   test never gets past PROBE/ALLOCATE, so they are fixed constants rather
   than more ini surface nobody would tune here. */
#define INI_ECC_BYTES   16
#define INI_CHUNK_BYTES 256
#define INI_SEQ_BYTES   2

static void ini_defaults(ofdm_ini *ini)
{
    /* Chosen for a 25 MHz 486, not for realism: n_fft=128 keeps the FFT count
       and workspace small (verified on host: ~78 KB workspace, ~13000
       samples / 0.6 s of probe audio at these defaults). freq_hi=8000 stays
       under 22050's Nyquist with margin. Push these up from OFDMBAT.INI
       once real on-device timing is measured -- these are a starting point,
       not a ceiling. */
    ini->n_fft            = 128;
    ini->cp_ratio         = 0.25;
    ini->freq_lo          = 1000.0;
    ini->freq_hi          = 8000.0;
    ini->pilot_spacing    = 8;
    ini->sample_rate      = 22050;
    ini->taper_ms         = 2.0;
    ini->frame_gap_ms     = 20.0;
    /* n_probe_syms >= OFDM_ISI_MIN_SYMS (64) or the ISI ratio is withheld as
       untrustworthy (see ofdm_probe.h) -- this is the periodic block length,
       NOT n_meas_syms; swapping the two by mistake silently disables the ISI
       measurement without any error, which is exactly what happened once
       while this file was being written. */
    ini->n_probe_syms     = 64;
    ini->n_meas_syms      = 16;
    ini->snr_margin_db    = 6.0;
    ini->max_bits_per_bin = 6;
    ini->sync_period      = 32;
    /* TX plays the probe back to back this many times. The DSP is already in
       AUTO-INIT mode (command 0xC6), so the buffer loops for free -- this
       just holds the transfer open longer. It exists because arming two
       independent DOS VMs cannot be synchronised tightly: the testlist has to
       gate on RX's "ADC armed" appearing over COM1, then type a command into
       the other machine, and that skew was comfortably larger than a single
       0.6 s probe. Repeating turns a race that had to be won into a window
       that only has to be overlapped. */
    ini->tx_repeats       = 5;
}

/* Minimal line-oriented "key = value" reader -- no sections, no quoting.
   Matches on the KEY starting a line (after leading whitespace) so a key
   that happens to be a substring of another (there are none here, but the
   check costs nothing) can't cross-match. Comment lines start with ';' or
   '#'. Missing file or missing key: silently keep the default -- this ini
   only overrides tuning, so a fresh D: with no ini yet still runs. */
static int ini_find(const char *buf, const char *key, char *out, size_t outcap)
{
    const char *p = buf;
    size_t klen = strlen(key);
    while ((p = strstr(p, key)) != NULL) {
        const char *line_start = p;
        while (line_start > buf && line_start[-1] != '\n') line_start--;
        {
            const char *q = line_start;
            while (*q == ' ' || *q == '\t') q++;
            if (q == p) {
                const char *v = p + klen;
                while (*v == ' ' || *v == '\t') v++;
                if (*v == '=') {
                    size_t n = 0;
                    v++;
                    while (*v == ' ' || *v == '\t') v++;
                    while (*v && *v != '\r' && *v != '\n' && n + 1 < outcap)
                        out[n++] = *v++;
                    out[n] = '\0';
                    return 1;
                }
            }
        }
        p += klen;
    }
    return 0;
}

static void ini_load(const char *path, ofdm_ini *ini)
{
    FILE *f;
    static char buf[8192];
    char val[64];
    size_t n;

    ini_defaults(ini);

    f = fopen(path, "rb");
    if (!f) {
        report("OFDMBAT: no %s -- using built-in defaults", path);
        return;
    }
    n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = '\0';
    fclose(f);

    if (ini_find(buf, "n_fft", val, sizeof(val)))            ini->n_fft = atoi(val);
    if (ini_find(buf, "cp_ratio", val, sizeof(val)))         ini->cp_ratio = atof(val);
    if (ini_find(buf, "freq_lo", val, sizeof(val)))          ini->freq_lo = atof(val);
    if (ini_find(buf, "freq_hi", val, sizeof(val)))          ini->freq_hi = atof(val);
    if (ini_find(buf, "pilot_spacing", val, sizeof(val)))    ini->pilot_spacing = atoi(val);
    if (ini_find(buf, "sample_rate", val, sizeof(val)))      ini->sample_rate = atoi(val);
    if (ini_find(buf, "taper_ms", val, sizeof(val)))         ini->taper_ms = atof(val);
    if (ini_find(buf, "frame_gap_ms", val, sizeof(val)))     ini->frame_gap_ms = atof(val);
    if (ini_find(buf, "n_probe_syms", val, sizeof(val)))     ini->n_probe_syms = atoi(val);
    if (ini_find(buf, "n_meas_syms", val, sizeof(val)))      ini->n_meas_syms = atoi(val);
    if (ini_find(buf, "snr_margin_db", val, sizeof(val)))    ini->snr_margin_db = atof(val);
    if (ini_find(buf, "max_bits_per_bin", val, sizeof(val))) ini->max_bits_per_bin = atoi(val);
    if (ini_find(buf, "sync_period", val, sizeof(val)))      ini->sync_period = atoi(val);
    if (ini_find(buf, "tx_repeats", val, sizeof(val)))       ini->tx_repeats = atoi(val);

    report("OFDMBAT: loaded %s", path);
}

static int build_cfg(const ofdm_ini *ini, ofdm_cfg *c)
{
    int rc = ofdm_cfg_build(c, ini->n_fft, ini->cp_ratio, ini->freq_lo,
                            ini->freq_hi, ini->pilot_spacing, ini->sample_rate,
                            INI_ECC_BYTES, INI_CHUNK_BYTES, INI_SEQ_BYTES,
                            ini->taper_ms, ini->frame_gap_ms, ini->sync_period);
    if (rc != 0) {
        report("OFDMBAT: ofdm_cfg_build rejected this geometry, rc=%d", rc);
        return 0;
    }
    report("OFDMBAT: cfg n_fft=%d cp_len=%d bin=[%d,%d] n_occupied=%d "
          "symbol_samples=%d sample_rate=%d",
          c->n_fft, c->cp_len, c->bin_lo, c->bin_hi, c->n_occupied,
          c->symbol_samples, c->sample_rate);
    return 1;
}

/* ==================== WAV header (RX only) -- verbatim from SB16ECHO.C ==================== */

static void wav_put32(unsigned char *p, unsigned long v)
{
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)((v >> 8) & 0xFF);
    p[2] = (unsigned char)((v >> 16) & 0xFF);
    p[3] = (unsigned char)((v >> 24) & 0xFF);
}

static void wav_put16(unsigned char *p, unsigned v)
{
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)((v >> 8) & 0xFF);
}

static void wav_header(unsigned char *hdr, unsigned long bytes, long rate)
{
    memcpy(hdr, "RIFF", 4);
    wav_put32(hdr + 4, 36UL + bytes);
    memcpy(hdr + 8, "WAVEfmt ", 8);
    wav_put32(hdr + 16, 16UL);
    wav_put16(hdr + 20, 1);
    wav_put16(hdr + 22, 1);                 /* mono               */
    wav_put32(hdr + 24, (unsigned long)rate);
    wav_put32(hdr + 28, (unsigned long)rate);   /* 1 byte/sample * mono */
    wav_put16(hdr + 32, 1);                 /* block align        */
    wav_put16(hdr + 34, 8);                 /* bits               */
    memcpy(hdr + 36, "data", 4);
    wav_put32(hdr + 40, bytes);
}

/* ==================== TX: build + play ==================== */

static void run_tx(const ofdm_ini *ini)
{
    ofdm_cfg c;
    ofdm_probe *p;
    size_t ws;
    long total, i;
    void *mem;
    double *tx_d;
    unsigned char *pcm;
    unsigned long tick0, tick1;
    unsigned resid;
    int reps;
    unsigned sel, rmseg, off;
    unsigned long phys;
    unsigned blk;

    if (!build_cfg(ini, &c)) return;

    /* TX only needs ofdm_probe_build_samples() here, which depends on the
       geometry/n_probe_syms/n_meas_syms, not on max_capture_samples -- but
       the API still requires a fully init'd ofdm_probe* to ask it, and
       ofdm_probe_workspace REJECTS (returns 0) any max_capture_samples below
       c.symbol_samples (ofdm_probe.c: "if (max_capture_samples <
       c->symbol_samples) return 0;"). Passing 1 here failed exactly that
       check on real hardware -- c.symbol_samples is the smallest value that
       passes it, and TX never uses this sizing instance for anything else. */
    ws = ofdm_probe_workspace(&c, ini->n_probe_syms, ini->n_meas_syms, c.symbol_samples);
    if (ws == 0) { report("OFDMBAT TX: ofdm_probe_workspace rejected params"); return; }
    mem = malloc(ws);
    if (!mem) { report("OFDMBAT TX: cannot allocate %lu-byte workspace", (unsigned long)ws); return; }
    if (ofdm_probe_init(&p, mem, ws, &c, ini->n_probe_syms, ini->n_meas_syms, c.symbol_samples) != 0) {
        report("OFDMBAT TX: ofdm_probe_init failed");
        free(mem);
        return;
    }

    total = ofdm_probe_build_samples(p);
    tx_d = (double *)malloc(sizeof(double) * (size_t)total);
    pcm  = (unsigned char *)malloc((size_t)total);
    if (!tx_d || !pcm) {
        report("OFDMBAT TX: cannot allocate %ld-sample waveform buffer", total);
        free(mem); free(tx_d); free(pcm);
        return;
    }
    total = ofdm_probe_build(p, tx_d, total);
    report("OFDMBAT TX: waveform built, %ld samples (%.2f s at %d Hz)",
          total, (double)total / c.sample_rate, c.sample_rate);

    /* Single-shot buffer, sized to the WHOLE waveform, not a small
       continuously-refilled ring: a real run found that touching a DMA
       buffer via movedata WHILE the 8237 is actively cycling through it
       corrupts the transfer, even when the bytes written are unchanged (see
       ofdm_spike_src/SPIKE.C's run_play comment for the full story -- it
       turned a clean 1000 Hz tone into broadband hissing garbage). The fix
       is to never touch this buffer again once armed: write the whole thing
       in up front, wait untouched, stop. That only works because these
       probe waveforms are short enough to fit in one 64KB ISA DMA page --
       if a future geometry change makes `total` exceed that, this needs
       chunking (arm/wait/stop per chunk, still never touching a live
       buffer) instead of silently truncating. */
    if (total > 65536L) {
        report("OFDMBAT TX: %ld samples exceeds one 64KB DMA page -- "
              "reduce n_probe_syms/n_meas_syms or n_fft in the ini", total);
        free(tx_d); free(mem); free(pcm);
        return;
    }

    for (i = 0; i < total; i++) {
        double v = 128.0 + PCM_SCALE * tx_d[i];
        if (v < 0.0) v = 0.0;
        if (v > 255.0) v = 255.0;
        pcm[i] = (unsigned char)v;
    }
    free(tx_d);
    free(mem);

    /* Same checkpoints as run_rx -- see that function's comment. */
    report("OFDMBAT TX: checkpoint 1 (before dsp_reset)");
    if (!dsp_reset()) { report("OFDMBAT TX: no DSP answering at %03Xh", g_sb_base); free(pcm); return; }
    report("OFDMBAT TX: checkpoint 2 (dsp_reset ok, before alloc_dma)");
    if (!alloc_dma((unsigned)total, &sel, &rmseg, &phys)) {
        report("OFDMBAT TX: cannot allocate DMA buffer");
        free(pcm);
        return;
    }
    off = DMA_OFF(phys, rmseg);
    report("OFDMBAT TX: checkpoint 3 (alloc_dma ok, rmseg=%04Xh sel=%04Xh "
          "phys=%08lXh off=%u, before buffer fill)", rmseg, sel, phys, off);

    /* Whole waveform in ONE copy, before arming -- see the comment above for
       why this replaced the old continuously-refilled ring. */
    dma_write(phys, pcm, (unsigned)total);
    report("OFDMBAT TX: checkpoint 4 (buffer filled, before dma_program)");

    dma_program(phys, (unsigned)total, 0 /* playback */);
    report("OFDMBAT TX: checkpoint 5 (dma_program ok, before dsp_write sequence)");
    /* Same rule as run_rx: say it BEFORE arming, never between the arm and
       the clock read. TX's error from this was small (~1.3% diluted across a
       24 s playback, indistinguishable from tick noise) which is exactly why
       it went unnoticed here while the same bug read as +28% on RX's 2.3 s
       capture -- a good reminder that a measurement which looks fine at one
       duration can be quietly wrong at another. */
    reps = ini->tx_repeats > 0 ? ini->tx_repeats : 1;
    /* The RATE is in this line so a testlist running several geometries in one
       run can gate on a phrase unique to each. test_filecontains re-reads the
       whole log every poll with no offset tracking, so a bare "DAC armed"
       would be matched by the PREVIOUS point's line and let the run race
       ahead -- the same stale-phrase trap the RX side needed the output
       filename for. */
    report("OFDMBAT TX: DAC armed at %d Hz, playing %ld samples x%d repeats "
          "(%.2f s total, auto-init loop -- buffer never touched)",
          c.sample_rate, total, reps, (double)total * reps / c.sample_rate);

    dsp_write(0xD1);
    dsp_write(0x41);
    dsp_write((unsigned char)((c.sample_rate >> 8) & 0xFF));
    dsp_write((unsigned char)(c.sample_rate & 0xFF));
    blk = (unsigned)total - 1;
    dsp_write(0xC6);
    dsp_write(0x00);
    dsp_write((unsigned char)(blk & 0xFF));
    dsp_write((unsigned char)(blk >> 8));
    tick0 = bios_ticks();
    delay((unsigned)((((total * reps) * 1000L) / c.sample_rate) + 100L));
    resid = dma_read_count();
    tick1 = bios_ticks();

    dsp_write(0xD0);
    dsp_write(0xDA);
    dsp_reset();
    dsp_write(0xD3);
    free_dma(sel);
    free(pcm);

    /* The TX half of the rate measurement -- see bios_ticks(). Comparing this
       against RX's identical number is what turns "the 44.1/48k mismatch is
       somewhere in the path" into a figure with a sign and a magnitude. */
    {
        long moved = total * reps - (long)resid - 1L;
        double elapsed_s = (double)(tick1 - tick0) / BIOS_TICK_HZ;
        double measured_hz;
        if (moved < 0) moved = 0;
        measured_hz = (elapsed_s > 0.0) ? (double)moved / elapsed_s : 0.0;
        report("OFDMBAT TX: DMA residual=%u -> %ld of %ld samples played in "
              "%.3f s -> measured %.1f Hz vs %d Hz programmed (%+.2f%%)",
              resid, moved, total * reps, elapsed_s, measured_hz, c.sample_rate,
              (c.sample_rate > 0 && measured_hz > 0.0)
                  ? (measured_hz - c.sample_rate) * 100.0 / c.sample_rate : 0.0);
    }

    report("OFDMBAT TX: done");
}

/* ==================== RATESCAN: is it a ratio or a fixed rate? ============ */

/* Arms one transfer at a given programmed rate WITH NO SIGNAL PRESENT and
   times it against the BIOS tick. Silence is fine and in fact preferable --
   this measures how fast the card CLOCKS samples, which has nothing to do with
   what is in them. (A report() call sitting inside the timed window costs
   ~0.64s of serial I/O on a 25MHz 486 and will read as phantom clock skew --
   say everything BEFORE arming, same rule as the RX arm sequence.)

   Kept as a regression instrument. If a real fault ever appears, sweeping the
   rate separates the candidates by the SHAPE of the error rather than by
   argument:

     constant RATIO across rates  -> the requested rate is being scaled wrong
                                     (a divisor or clock-constant bug)
     constant Hz across rates     -> the guest's request is being IGNORED and
                                     some fixed host/device rate is showing
                                     through (device selection / resampler)
     neither                      -> not a rate mapping at all; look at how
                                     samples are fed into the ring instead

   The DAC is swept too, as the control for the control: if playback tracks
   every programmed rate accurately while capture does not, the fault is
   isolated to the capture path with no further argument needed. */
static void rate_probe(long rate, int record, long nsamp_override)
{
    unsigned sel, rmseg, off;
    unsigned long phys, tick0, tick1, grace;
    unsigned resid, blk;
    long nsamp, moved;
    int completed;
    double elapsed_s, measured;
    static unsigned char fill[65024];

    /* ~2.5 s per point where it fits, so one BIOS tick (55 ms) is ~2% rather
       than swamping the measurement -- a 30% error stays unmistakable. */
    nsamp = (nsamp_override > 0) ? nsamp_override : (rate * 5L / 2L);
    if (nsamp > 65024L) nsamp = 65024L;

    if (!dsp_reset()) { report("RATESCAN: no DSP at %03Xh", g_sb_base); return; }
    if (!alloc_dma((unsigned)nsamp, &sel, &rmseg, &phys)) {
        report("RATESCAN: cannot allocate %ld-sample buffer", nsamp);
        return;
    }
    off = DMA_OFF(phys, rmseg);
    memset(fill, 128, (size_t)nsamp);
    dma_write(phys, fill, (unsigned)nsamp);

    if (record) {
        mixer_write(MIX_INPUT_LEFT,  (unsigned char)(IN_LINE_L | IN_LINE_R | IN_MIC));
        mixer_write(MIX_INPUT_RIGHT, (unsigned char)(IN_LINE_L | IN_LINE_R | IN_MIC));
    }
    dma_program_mode(phys, (unsigned)nsamp, (unsigned char)(record ? 0x44 : 0x48));
    if (!record) dsp_write(0xD1);
    dsp_write((unsigned char)(record ? 0x42 : 0x41));
    dsp_write((unsigned char)((rate >> 8) & 0xFF));
    dsp_write((unsigned char)(rate & 0xFF));
    blk = (unsigned)nsamp - 1;
    dsp_write((unsigned char)(record ? 0xC8 : 0xC0));   /* 8-bit single-cycle */
    dsp_write(0x00);
    dsp_write((unsigned char)(blk & 0xFF));
    dsp_write((unsigned char)(blk >> 8));

    grace = (unsigned long)(((double)nsamp / (double)rate) * BIOS_TICK_HZ) + 55UL;
    tick0 = bios_ticks();
    completed = 0;
    for (;;) {
        resid = dma_read_count();
        tick1 = bios_ticks();
        if (resid > (unsigned)nsamp) { completed = 1; break; }
        if (tick1 - tick0 > grace) break;
    }

    dsp_write(0xD0);
    dsp_write(0xDA);
    dsp_reset();
    if (!record) dsp_write(0xD3);
    free_dma(sel);

    moved = completed ? nsamp : (nsamp - (long)resid - 1L);
    if (moved < 0) moved = 0;
    elapsed_s = (double)(tick1 - tick0) / BIOS_TICK_HZ;
    measured = (elapsed_s > 0.0) ? (double)moved / elapsed_s : 0.0;

    report("RATESCAN %s: programmed %5ld -> %-7s %6ld samples in %6.3f s = "
          "%8.1f Hz  ratio %.3f  excess %+ld samples",
          record ? "ADC" : "DAC", rate, completed ? "done" : "TIMEOUT",
          moved, elapsed_s, measured,
          rate > 0 ? measured / (double)rate : 0.0,
          moved - (long)(elapsed_s * (double)rate));
}

static void run_ratescan(void)
{
    static const long rates[6] = { 8000L, 11025L, 16000L, 22050L, 32000L, 44100L };
    int i;

    report("OFDMBAT RATESCAN: timing SB16 transfers at a range of programmed "
          "rates, with no signal present.");
    report("  constant RATIO  -> the requested rate is scaled wrong (divisor/clock bug)");
    report("  constant Hz     -> the request is ignored; a fixed host/device rate shows through");
    report("  DAC accurate but ADC not -> the fault is in the capture path only");

    for (i = 0; i < 6; i++) rate_probe(rates[i], 1, 0L);
    for (i = 0; i < 6; i++) rate_probe(rates[i], 0, 0L);

    /* DURATION SWEEP, same rate, different transfer lengths. This tests a
       different shape of fault from the rate sweep above: a FIXED EXCESS at
       the start of a transfer (a burst, a pre-armed timer, a ring already
       holding samples) is not a rate error at all, but it inflates a short
       measurement enormously and a long one barely. If measured Hz walks DOWN
       toward the programmed rate as the transfer lengthens while "excess
       samples" stays roughly CONSTANT, the clock is fine and there is a
       startup burst; if the ratio holds steady while the excess grows in
       proportion, the clock really is fast. */
    report("OFDMBAT RATESCAN: duration sweep at 22050 Hz ADC -- constant EXCESS"
          " means a startup burst, constant RATIO means a genuine clock error");
    rate_probe(22050L, 1, 11025L);   /* ~0.5 s */
    rate_probe(22050L, 1, 22050L);   /* ~1.0 s */
    rate_probe(22050L, 1, 44100L);   /* ~2.0 s */
    rate_probe(22050L, 1, 65024L);   /* ~2.9 s, the size RX actually uses */
    report("OFDMBAT RATESCAN: done");
}

/* ==================== RX: record + estimate + allocate ==================== */

static void run_rx(const ofdm_ini *ini, const char *out_path)
{
    ofdm_cfg c;
    ofdm_probe *p;
    size_t ws;
    long build_samples, capture_samples;
    void *mem;
    unsigned sel, rmseg, off;
    unsigned long phys;
    unsigned blk;
    unsigned char hdr[44];
    FILE *f;
    unsigned char *raw;
    unsigned long tick0, tick1, grace;
    unsigned resid;
    int completed;
    int is_control;
    long moved, untouched, k;
    int vmin, vmax;
    double elapsed_s, measured_hz;

    if (!build_cfg(ini, &c)) return;

    /* Same sizing-only pass TX does, and the same fix: max_capture_samples
       must be >= c.symbol_samples or ofdm_probe_workspace rejects it outright
       (returns 0) -- passing 1 here is what "sizing failed" on real hardware
       turned out to be. This instance is thrown away once build_samples is
       known; the real one below is sized for the actual capture length. */
    ws = ofdm_probe_workspace(&c, ini->n_probe_syms, ini->n_meas_syms, c.symbol_samples);
    if (ws == 0) {
        report("OFDMBAT RX: ofdm_probe_workspace rejected params (sizing pass)");
        return;
    }
    mem = malloc(ws);
    if (!mem || ofdm_probe_init(&p, mem, ws, &c, ini->n_probe_syms, ini->n_meas_syms, c.symbol_samples) != 0) {
        report("OFDMBAT RX: ofdm_probe_init sizing pass failed");
        free(mem);
        return;
    }
    build_samples = ofdm_probe_build_samples(p);
    free(mem);

    /* TWICE the probe length, plus a little margin for TX/RX arm skew. Since TX
       loops the probe back to back (tx_repeats), a window of 2*build_samples is
       guaranteed to contain one complete copy no matter what phase the capture
       starts at. A wider survey geometry can push build_samples up enough that
       silently clamping the window to 65024 would cut it below one probe
       length -- which would look like a sync/channel failure rather than a
       capture too short to hold what it's looking for, so this fails loudly
       instead of clamping. */
    capture_samples = 2L * build_samples + (long)c.sample_rate / 10L;
    if (capture_samples > 65024L) {
        report("OFDMBAT RX: geometry needs %ld capture samples (2 x %ld probe "
              "+ margin) but one 64KB ISA DMA page holds 65024 -- reduce n_fft, "
              "n_probe_syms/n_meas_syms, or sample_rate in the ini",
              capture_samples, build_samples);
        return;
    }

    /* Whole capture has to fit in one un-refilled 64KB ISA DMA page -- see
       run_tx's comment for why continuous refilling during a live transfer
       is no longer done at all (it corrupted a real run on hardware). If a
       future geometry increase pushes capture_samples past this, it needs
       chunking (arm/wait/stop per chunk, never touching a live buffer), not
       a bigger single buffer. */
    if (capture_samples > 65536L) {
        report("OFDMBAT RX: %ld samples exceeds one 64KB DMA page -- "
              "reduce n_probe_syms/n_meas_syms or n_fft in the ini",
              capture_samples);
        return;
    }

    ws = ofdm_probe_workspace(&c, ini->n_probe_syms, ini->n_meas_syms, (int)capture_samples);
    if (ws == 0) { report("OFDMBAT RX: ofdm_probe_workspace rejected params"); return; }
    mem = malloc(ws);
    if (!mem) { report("OFDMBAT RX: cannot allocate %lu-byte workspace", (unsigned long)ws); return; }
    if (ofdm_probe_init(&p, mem, ws, &c, ini->n_probe_syms, ini->n_meas_syms, (int)capture_samples) != 0) {
        report("OFDMBAT RX: ofdm_probe_init failed");
        free(mem);
        return;
    }
    report("OFDMBAT RX: capturing %ld samples (%.2f s at %d Hz, %lu-byte workspace)",
          capture_samples, (double)capture_samples / c.sample_rate, c.sample_rate,
          (unsigned long)ws);

    raw = (unsigned char *)malloc((size_t)capture_samples);
    if (!raw) {
        report("OFDMBAT RX: cannot allocate %ld-byte capture buffer", capture_samples);
        free(mem);
        return;
    }

    /* Checkpoints kept: touching this buffer via movedata WHILE the 8237 is
       actively cycling through it corrupts the transfer with no error message
       (see run_tx's buffer comment). Fix: single-shot capture into a buffer
       sized for the WHOLE window, never touched again until the transfer
       stops. */
    report("OFDMBAT RX: checkpoint 1 (before dsp_reset)");
    if (!dsp_reset()) { report("OFDMBAT RX: no DSP at %03Xh", g_sb_base); free(mem); free(raw); return; }
    report("OFDMBAT RX: checkpoint 2 (dsp_reset ok, before alloc_dma)");
    if (!alloc_dma((unsigned)capture_samples, &sel, &rmseg, &phys)) {
        report("OFDMBAT RX: cannot allocate DMA buffer");
        free(mem);
        free(raw);
        return;
    }
    off = DMA_OFF(phys, rmseg);
    report("OFDMBAT RX: checkpoint 3 (alloc_dma ok, rmseg=%04Xh sel=%04Xh "
          "phys=%08lXh off=%u, before dma_program)", rmseg, sel, phys, off);

    /* Pre-fill with a sentinel the ADC cannot plausibly produce. A capture that
       comes back ALL ZEROS is ambiguous: 0x00 is both "the DMA never wrote
       anything" and "the ADC railed to the negative limit". Sentinel bytes
       still present afterwards were never written by the DMA, which separates
       the two without guessing (8-bit unsigned silence is 0x80, not 0x00). */
    memset(raw, DMA_SENTINEL, (size_t)capture_samples);
    dma_write(phys, raw, (unsigned)capture_samples);

    mixer_write(MIX_INPUT_LEFT,  (unsigned char)(IN_LINE_L | IN_LINE_R | IN_MIC));
    mixer_write(MIX_INPUT_RIGHT, (unsigned char)(IN_LINE_L | IN_LINE_R | IN_MIC));
    report("OFDMBAT RX: ADC input select 3Dh=%02Xh 3Eh=%02Xh (read back)",
          mixer_read(MIX_INPUT_LEFT), mixer_read(MIX_INPUT_RIGHT));

    dma_program(phys, (unsigned)capture_samples, 1 /* record */);
    report("OFDMBAT RX: checkpoint 4 (dma_program ok, before dsp_write sequence)");
    /* NOTHING may print between arming the transfer (the final block-length
       write) and reading tick0 -- a report() call there costs ~0.64s of
       serial I/O on a 25MHz 486 and reads back as phantom clock skew (see
       RATESCAN's rate_probe, which has no reports in its timed window). */
    grace = (unsigned long)(((double)capture_samples / (double)c.sample_rate)
                            * BIOS_TICK_HZ) + 28UL;
    report("OFDMBAT RX: ADC armed, capturing %ld samples untouched -> %s",
          capture_samples, out_path);

    dsp_write(0x42);
    dsp_write((unsigned char)((c.sample_rate >> 8) & 0xFF));
    dsp_write((unsigned char)(c.sample_rate & 0xFF));
    blk = (unsigned)capture_samples - 1;
    /* SINGLE-CYCLE (0xC8), not auto-init. Auto-init wraps the 8237 and keeps
       going, leaving the buffer a jumble of overlapping passes (fatal for a
       sync-then-estimate pipeline) and making the residual count ambiguous
       (which wrap it corresponds to is unknowable from the count alone).
       Single-cycle stops at the end of the buffer, so the recording is
       contiguous and the residual means exactly one thing. (TX stays
       auto-init on purpose -- there, looping IS the feature.) */
    dsp_write(0xC8);
    dsp_write(0x00);
    dsp_write((unsigned char)(blk & 0xFF));
    dsp_write((unsigned char)(blk >> 8));
    tick0 = bios_ticks();               /* the instant the transfer starts */

    /* Wait for the transfer to ACTUALLY finish rather than for the time it
       ought to take. In single-cycle mode the 8237's count wraps to FFFFh at
       terminal count, an unambiguous "done" signal; the BIOS tick at that
       moment gives the real sample rate the card clocked, which a fixed
       delay() cannot. Polling the count is a port read, not a touch of the
       DMA buffer, so it's safe while the transfer is live. The grace period
       is the nominal duration plus ~1.5s so a card running slower than
       programmed still completes instead of being cut off. */
    completed = 0;
    for (;;) {
        resid = dma_read_count();
        tick1 = bios_ticks();
        if (resid > (unsigned)capture_samples) { completed = 1; break; }
        if (tick1 - tick0 > grace) break;
    }

    dsp_write(0xD0);
    dsp_write(0xDA);
    dsp_reset();
    report("OFDMBAT RX: checkpoint 7 (DMA stopped, before copy-out)");

    dma_read(raw, phys, (unsigned)capture_samples);
    free_dma(sel);

    /* ---- did the transfer actually happen, and at what real rate? ---- */
    untouched = 0;
    vmin = 255; vmax = 0;
    for (k = 0; k < capture_samples; k++) {
        if (raw[k] == DMA_SENTINEL) untouched++;
        if (raw[k] < vmin) vmin = raw[k];
        if (raw[k] > vmax) vmax = raw[k];
    }
    /* The 8237 counts DOWN from len-1, so residual+1 is what is still
       outstanding and the rest is what it moved -- unless it hit terminal
       count, where the whole buffer moved and the count has wrapped to FFFFh. */
    moved = completed ? capture_samples : (capture_samples - (long)resid - 1L);
    if (moved < 0) moved = 0;
    elapsed_s = (double)(tick1 - tick0) / BIOS_TICK_HZ;
    measured_hz = (elapsed_s > 0.0) ? (double)moved / elapsed_s : 0.0;

    report("OFDMBAT RX: DMA %s (residual=%u) -> %ld of %ld samples moved; "
          "%ld bytes still sentinel; value range [%d,%d]",
          completed ? "completed" : "TIMED OUT", resid, moved, capture_samples,
          untouched, vmin, vmax);
    report("OFDMBAT RX: transfer took %.3f s by BIOS tick -> measured %.1f Hz "
          "vs %d Hz programmed (%+.2f%%)",
          elapsed_s, measured_hz, c.sample_rate,
          (c.sample_rate > 0 && measured_hz > 0.0)
              ? (measured_hz - c.sample_rate) * 100.0 / c.sample_rate : 0.0);

    /* A control run (deliberately armed with nothing transmitting, to prove
       the ADC isn't hearing leakage or stale ring content) is SUPPOSED to come
       back as digital silence -- the "flat rail, check your mixer" warning
       below only fires for a live run, detected from the output name rather
       than a new flag (the testlist already gates on that name too). */
    is_control = (strstr(out_path, "CTRL") != NULL);

    if (untouched == capture_samples) {
        report("OFDMBAT RX: *** the DMA wrote NOTHING -- every byte is still "
              "the sentinel. This is a capture-path fault (ADC not clocking, "
              "wrong DMA channel, or no input device), NOT a signal-level or "
              "sync problem. ***");
    } else if (vmin == vmax && is_control) {
        report("OFDMBAT RX: control run captured a constant %d (8-bit silence) "
              "-- EXPECTED, and the point of the control: nothing was "
              "transmitting, so the ADC hearing nothing is what makes the live "
              "run's result trustworthy. The VERDICT FAIL below is also "
              "expected.", vmin);
    } else if (vmin == vmax) {
        report("OFDMBAT RX: *** capture is a constant %d -- the DMA ran but "
              "the ADC delivered a flat rail (0=negative rail, 128=silence). "
              "Check the input device/mixer source, not the modem. ***", vmin);
    }

    f = fopen(out_path, "wb");
    if (!f) {
        report("OFDMBAT RX: could not open %s for writing", out_path);
        free(mem);
        free(raw);
        return;
    }
    wav_header(hdr, (unsigned long)capture_samples, c.sample_rate);
    fwrite(hdr, 1, sizeof(hdr), f);
    fwrite(raw, 1, (size_t)capture_samples, f);
    fclose(f);
    free(raw);

    report("OFDMBAT RX: capture done, %ld samples -> %s", capture_samples, out_path);

    /* ---- analysis: no real-time deadline past this point ---- */
    {
        double *rx_d;
        double *H_re, *H_im, *snr_lin, *snr_db, *snr_db_periodic;
        int *bits_out;
        ofdm_probe_info info;
        long i;
        unsigned char *raw;
        int rc, n_alloc, carriers_on, tot_bits;
        double minsnr, maxsnr, sumsnr;

        raw = (unsigned char *)malloc((size_t)capture_samples);
        rx_d = (double *)malloc(sizeof(double) * (size_t)capture_samples);
        if (!raw || !rx_d) {
            report("OFDMBAT RX: VERDICT FAIL (out of memory for analysis pass)");
            free(mem); free(raw); free(rx_d);
            return;
        }
        f = fopen(out_path, "rb");
        fseek(f, 44L, SEEK_SET);
        fread(raw, 1, (size_t)capture_samples, f);
        fclose(f);
        for (i = 0; i < capture_samples; i++)
            rx_d[i] = ((double)raw[i] - 128.0) / PCM_SCALE;
        free(raw);

        H_re            = (double *)malloc(sizeof(double) * c.n_occupied);
        H_im            = (double *)malloc(sizeof(double) * c.n_occupied);
        snr_lin         = (double *)malloc(sizeof(double) * c.n_occupied);
        snr_db          = (double *)malloc(sizeof(double) * c.n_occupied);
        snr_db_periodic = (double *)malloc(sizeof(double) * c.n_occupied);
        bits_out        = (int *)malloc(sizeof(int) * c.n_data_carriers);

        rc = ofdm_probe_estimate(p, rx_d, (int)capture_samples, 0.0, H_re, H_im, snr_lin,
                                 snr_db, snr_db_periodic, &info);
        report("OFDMBAT RX: estimate rc=%d valid=%d error=%d coherence=%.3f "
              "frame_start=%d n_syms=%d n_meas_syms=%d",
              rc, info.valid, info.error, info.coherence, info.frame_start,
              info.n_syms, info.n_meas_syms);

        /* Where the probe actually landed in the capture window. This is the
           end-to-end LATENCY of the whole path -- RX arm, TX arm, DAC, the
           host audio graph, ADC -- measured in the RX's own samples, and the
           number to watch run-to-run for arm-skew jitter. It is reported
           whether or not sync succeeded, because an onset with no sync is a
           very different failure from no onset at all. */
        {
            long onset = -1;
            double peak = 0.0;
            for (i = 0; i < capture_samples; i++) {
                double a = rx_d[i] < 0.0 ? -rx_d[i] : rx_d[i];
                if (a > peak) peak = a;
            }
            for (i = 0; i < capture_samples; i++) {
                double a = rx_d[i] < 0.0 ? -rx_d[i] : rx_d[i];
                if (peak > 0.0 && a > peak * 0.25) { onset = i; break; }
            }
            if (onset >= 0)
                report("OFDMBAT RX: signal onset at sample %ld (%.1f ms into "
                      "the capture), peak amplitude %.3f",
                      onset, (double)onset * 1000.0 / c.sample_rate, peak);
            else
                report("OFDMBAT RX: no signal onset found (peak amplitude "
                      "%.4f) -- nothing detectable arrived", peak);
        }

        if (info.valid)
            report("OFDMBAT RX: sync frame_start=%d (%.1f ms), coherence=%.3f",
                  info.frame_start,
                  (double)info.frame_start * 1000.0 / c.sample_rate,
                  info.coherence);

        if (rc != 0 || !info.valid) {
            if (is_control)
                report("OFDMBAT RX: VERDICT FAIL (no probe found, error=%d) -- "
                      "EXPECTED for a control run with no transmitter", info.error);
            else
                report("OFDMBAT RX: VERDICT FAIL (no probe found, error=%d)", info.error);
            free(mem); free(rx_d);
            free(H_re); free(H_im); free(snr_lin); free(snr_db); free(snr_db_periodic);
            free(bits_out);
            return;
        }

        if (info.has_isi_penalty)
            report("OFDMBAT RX: ISI penalty (periodic - measured median) = %.2f dB",
                  info.isi_penalty_db);
        if (info.has_beta)
            report("OFDMBAT RX: ISI ratio (beta) = %.4f", info.isi_ratio);
        else
            report("OFDMBAT RX: ISI ratio withheld (n_probe_syms=%d < %d minimum)",
                  info.n_syms, 64);

        minsnr = 1e18; maxsnr = -1e18; sumsnr = 0.0;
        for (i = 0; i < c.n_occupied; i++) {
            if (snr_db[i] < minsnr) minsnr = snr_db[i];
            if (snr_db[i] > maxsnr) maxsnr = snr_db[i];
            sumsnr += snr_db[i];
            if (serial_out)
                fprintf(serial_out, "OFDMBAT RX: carrier %ld snr_db=%.2f\n", i, snr_db[i]);
        }
        report("OFDMBAT RX: snr_db min=%.2f max=%.2f mean=%.2f (n_occupied=%d)",
              minsnr, maxsnr, sumsnr / c.n_occupied, c.n_occupied);

        n_alloc = ofdm_allocate(snr_db, c.n_occupied, c.pilot_spacing,
                                ini->snr_margin_db, ini->max_bits_per_bin,
                                info.has_beta ? info.isi_ratio : 0.0,
                                bits_out, c.n_data_carriers);
        carriers_on = 0; tot_bits = 0;
        for (i = 0; i < n_alloc; i++) {
            if (bits_out[i] > 0) carriers_on++;
            tot_bits += bits_out[i];
            if (serial_out)
                fprintf(serial_out, "OFDMBAT RX: BAT[%ld] = %d bits\n", i, bits_out[i]);
        }

        /* The BAT itself, ON SCREEN and not only down COM1 -- this table IS
           the product of the exchange, so watching a run should show it. One
           digit per data carrier, 32 per row with the starting index, which
           fits 80 columns and stays readable as the geometry grows. */
        report("OFDMBAT RX: BAT (%d data carriers, bits per carrier):", n_alloc);
        {
            char row[96];
            int col = 0, pos = 0, start = 0;
            for (i = 0; i < n_alloc; i++) {
                row[pos++] = (char)(bits_out[i] > 9 ? '+' : ('0' + bits_out[i]));
                if (++col == 32 || i == n_alloc - 1) {
                    row[pos] = '\0';
                    report("  [%3d] %s", start, row);
                    col = 0; pos = 0; start = (int)i + 1;
                }
            }
        }
        report("OFDMBAT RX: BAT allocated %d/%d carriers on, %d total bits/symbol",
              carriers_on, n_alloc, tot_bits);

        /* Simple, deliberately loose gate: half the data carriers usable at
           all. This is a channel-probe feasibility test, not a link-quality
           bar -- the actual numbers are what this test exists to produce. */
        if (n_alloc > 0 && carriers_on * 2 >= n_alloc)
            report("OFDMBAT RX: VERDICT PASS");
        else
            report("OFDMBAT RX: VERDICT FAIL (only %d/%d carriers usable)",
                  carriers_on, n_alloc);

        free(mem); free(rx_d);
        free(H_re); free(H_im); free(snr_lin); free(snr_db); free(snr_db_periodic);
        free(bits_out);
    }
}

/* ==================== two-way BAT exchange ==================== */

/* ---- slot scheduling ----------------------------------------------------
 * Every slot is a FIXED wall-clock window measured against the BIOS tick, and
 * both ends run the same schedule, so a receiver that finishes early WAITS OUT
 * the rest of its slot rather than racing ahead (a fixed-repetition TX / exit-
 * on-success RX schedule lets the two machines' clocks drift apart). The only
 * remaining misalignment is the constant skew between the two program starts
 * (a few seconds, set by how fast the testlist types the second command),
 * which the slot absorbs.
 *
 * 90s accounts for probe-listen attempts (run_probe_rx), not just BAT-table
 * listens (xchg_recv_bat, ~3-4s each): ofdm_probe_estimate's coarse
 * acquisition runs a fine-step coherence scan across the whole capture
 * window (chosen over the cheaper Schmidl-Cox because SC false-locks on the
 * audio stream's startup transient -- see its own comment) and can take 35+s
 * on a loaded host. A shorter slot starves the NEXT slot of any receive
 * attempts at all with no probe/decode fault involved; 90s leaves enough
 * margin for a retry at the cost of tripling the exchange's total time. */
#define XCHG_SLOT_SECS 90.0

static unsigned long xchg_t0;

static unsigned long xchg_slot_end(int slot)
{
    return xchg_t0 + (unsigned long)(XCHG_SLOT_SECS * BIOS_TICK_HZ * (double)slot);
}

/* Seconds still left in `slot`, 0 if it has already passed. */
static long xchg_slot_left(int slot)
{
    unsigned long now = bios_ticks(), end = xchg_slot_end(slot);
    if (now >= end) return 0;
    return (long)((double)(end - now) / BIOS_TICK_HZ);
}

/* Hold until the slot boundary, so an end that finished early does not run
   ahead of the one that did not. */
static void xchg_wait_slot(int slot)
{
    unsigned long end = xchg_slot_end(slot);
    long waited = 0;
    while (bios_ticks() < end) {
        delay(250);
        waited++;
    }
    if (waited) report("XCHG: idle %ld ms to slot boundary", waited * 250L);
}

/* Audio primitives for the exchange. These duplicate a little of run_tx/run_rx
   rather than refactoring them, deliberately: those two paths work on real
   hardware and restructuring them to serve a new caller risks that for a
   cosmetic gain. The parts that matter -- alloc_dma, dma_program_mode, the
   arm sequences, the never-touch-a-live-buffer rule -- are the same
   functions underneath. */

/* Play `sig` (n doubles) `reps` times through the DAC, blocking until done. */
static int audio_play(const ofdm_cfg *c, const double *sig, long n, int reps)
{
    unsigned sel, rmseg, blk;
    unsigned long phys;
    unsigned char *pcm;
    long i;

    if (n <= 0 || n > 65024L) {
        report("XCHG: waveform %ld samples does not fit one 64KB DMA page", n);
        return 0;
    }
    pcm = (unsigned char *)malloc((size_t)n);
    if (!pcm) { report("XCHG: out of memory for %ld PCM bytes", n); return 0; }
    for (i = 0; i < n; i++) {
        double v = 128.0 + PCM_SCALE * sig[i];
        if (v < 0.0)   v = 0.0;
        if (v > 255.0) v = 255.0;
        pcm[i] = (unsigned char)v;
    }
    if (!dsp_reset()) { report("XCHG: no DSP at %03Xh", g_sb_base); free(pcm); return 0; }
    if (!alloc_dma((unsigned)n, &sel, &rmseg, &phys)) {
        report("XCHG: cannot allocate DMA buffer"); free(pcm); return 0;
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
   completes. Single-cycle, never touched while live -- the two rules this
   project paid for the hard way. */
static int audio_record(const ofdm_cfg *c, unsigned char *raw, long n)
{
    unsigned sel, rmseg, blk, resid;
    unsigned long phys, tick0, tick1, grace;
    int completed;

    if (n <= 0 || n > 65024L) {
        report("XCHG: capture %ld samples does not fit one 64KB DMA page", n);
        return 0;
    }
    if (!dsp_reset()) { report("XCHG: no DSP at %03Xh", g_sb_base); return 0; }
    if (!alloc_dma((unsigned)n, &sel, &rmseg, &phys)) {
        report("XCHG: cannot allocate DMA buffer"); return 0;
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
    if (!completed) report("XCHG: capture timed out (residual=%u)", resid);
    return 1;
}

/* Send a table. Repeated for the whole slot so the far end can arm at any
   moment inside it -- the same reasoning as the probe transmitter's loop. */
static int xchg_send_bat(const ofdm_cfg *c, const int *bits, int n, int slot)
{
    void  *ws;
    double *sig;
    long   need, wrote;
    int    ok = 0;

    need = bat_encode_samples(c, n);
    if (need <= 0) { report("XCHG: bat_encode_samples rejected %d entries", n); return 0; }
    ws  = malloc(bat_workspace(c));
    sig = (double *)malloc(sizeof(double) * (size_t)need);
    if (!ws || !sig) { report("XCHG: out of memory for BAT burst"); free(ws); free(sig); return 0; }
    wrote = bat_encode(c, ws, bat_workspace(c), bits, n, sig, need);
    if (wrote != need) {
        report("XCHG: bat_encode failed rc=%ld", wrote);
    } else {
        long secs = xchg_slot_left(slot);
        int  reps = (int)((secs * c->sample_rate) / need);
        if (reps < 1) reps = 1;
        report("XCHG: sending BAT, %d entries, %ld samples x%d reps "
              "(%ld s left in slot %d)", n, need, reps, secs, slot);
        ok = audio_play(c, sig, need, reps);
    }
    free(ws); free(sig);
    return ok;
}

/* Listen for a table, retrying across the far end's transmit slot. Retries
   because the two machines are started by hand seconds apart and their slots
   cannot be assumed aligned; each attempt is a fresh capture, so a miss costs
   one window rather than the exchange. */
static int xchg_recv_bat(const ofdm_cfg *c, int *bits, int cap, int slot)
{
    void  *ws;
    unsigned char *raw;
    double *rx;
    long   burst, window, i;
    unsigned long attempt_ticks = 0, t0;
    int    a, rc = -3;

    burst  = bat_encode_samples(c, cap);
    window = 2L * burst + c->sample_rate / 10L;
    if (window > 65024L) window = 65024L;

    ws  = malloc(bat_workspace(c));
    raw = (unsigned char *)malloc((size_t)window);
    rx  = (double *)malloc(sizeof(double) * (size_t)window);
    if (!ws || !raw || !rx) {
        report("XCHG: out of memory for BAT capture");
        free(ws); free(raw); free(rx);
        return -1;
    }
    /* audio_record's own grace timeout bounds the CAPTURE, but the decode that
       follows it is unbounded CPU time and can run far longer than the
       capture on a loaded host -- an attempt started with little time left in
       the slot can silently consume the NEXT slot's window too. Once one
       attempt's cost is known, refuse to start another unless the slot has at
       least that much time left. */
    for (a = 1; ; a++) {
        long left = (long)(xchg_slot_end(slot) - bios_ticks());
        if (left <= 0) {
            if (a == 1)
                report("XCHG: slot %d already expired before the first BAT "
                      "listen attempt -- a prior slot overran its budget",
                      slot);
            break;
        }
        if (a > 1 && (unsigned long)left < attempt_ticks) {
            report("XCHG: %ld s left in slot %d, less than the last attempt's "
                  "%ld s -- stopping early rather than overrunning into the "
                  "next slot", xchg_slot_left(slot), slot,
                  (long)(attempt_ticks / BIOS_TICK_HZ));
            break;
        }
        t0 = bios_ticks();
        if (!audio_record(c, raw, window)) break;
        for (i = 0; i < window; i++) rx[i] = ((double)raw[i] - 128.0) / PCM_SCALE;
        rc = bat_decode(c, ws, bat_workspace(c), rx, window, bits, cap);
        attempt_ticks = bios_ticks() - t0;
        /* The sync plateau numbers go out on every failure. "No frame" and
           "a plateau of the wrong length" are different faults -- the first
           means nothing was heard, the second means something was heard and
           misread -- and on hardware that distinction is the whole
           investigation. Guessing between them costs a full run. */
        report("XCHG: BAT listen attempt %d -> rc=%d%s (plateau len=%ld end=%ld, "
              "window=%ld, %ld s left in slot %d)",
              a, rc,
              rc > 0 ? " decoded" :
              rc == -3 ? " no frame" :
              rc == -4 ? " frame found but CRC failed" : "",
              bat_dbg_runlen, bat_dbg_runend, window,
              xchg_slot_left(slot), slot);
        if (rc > 0) break;
    }
    free(ws); free(raw); free(rx);
    return rc;
}

static void print_bat(const char *label, const int *bits, int n)
{
    char row[96];
    int col = 0, pos = 0, start = 0, i, on = 0, tot = 0;
    for (i = 0; i < n; i++) { if (bits[i] > 0) on++; tot += bits[i]; }
    report("XCHG: %s -- %d/%d carriers on, %d bits/symbol", label, on, n, tot);
    for (i = 0; i < n; i++) {
        row[pos++] = (char)(bits[i] > 9 ? '+' : ('0' + bits[i]));
        if (++col == 32 || i == n - 1) {
            row[pos] = '\0';
            report("  [%3d] %s", start, row);
            col = 0; pos = 0; start = i + 1;
        }
    }
}

/* The exchange itself. Four half-duplex slots, mirrored between the ends:

     slot  A                      B
       1   TX probe               RX probe  -> B measures BAT_AB
       2   RX probe  -> BAT_BA    TX probe
       3   RX bat    <- BAT_AB    TX bat (BAT_AB)
       4   TX bat (BAT_BA)        RX bat  <- BAT_BA

   The receiver computes the table for the direction it HEARD, so each end
   ends up holding the table for the direction it TRANSMITS -- which is the
   one it needs. That is the whole point of the exchange and the reason a
   one-way probe, however good its numbers, is not one. */
/* Build the probe once and loop it for the slot. Same waveform the one-way
   TX produces; only the plumbing differs. */
static int run_probe_tx(const ofdm_cfg *c, const ofdm_ini *ini, int slot)
{
    ofdm_probe *p;
    void   *mem;
    size_t  ws;
    double *tx_d;
    long    total;
    int     ok = 0;

    ws = ofdm_probe_workspace(c, ini->n_probe_syms, ini->n_meas_syms,
                              c->symbol_samples);
    if (ws == 0) { report("XCHG: probe workspace rejected"); return 0; }
    mem = malloc(ws);
    if (!mem || ofdm_probe_init(&p, mem, ws, c, ini->n_probe_syms,
                                ini->n_meas_syms, c->symbol_samples) != 0) {
        report("XCHG: probe init failed");
        free(mem);
        return 0;
    }
    total = ofdm_probe_build_samples(p);
    tx_d  = (double *)malloc(sizeof(double) * (size_t)total);
    if (!tx_d) { report("XCHG: out of memory for probe"); free(mem); return 0; }
    if (ofdm_probe_build(p, tx_d, total) == total) {
        /* Repetitions are derived from the time actually left in the slot, so
           transmission ends WITH the slot rather than at some fixed count
           that has drifted away from it. */
        long secs = xchg_slot_left(slot);
        int  reps = (int)((secs * c->sample_rate) / total);
        if (reps < 1) reps = 1;
        report("XCHG: probe %ld samples x%d reps (%ld s left in slot %d)",
              total, reps, secs, slot);
        ok = audio_play(c, tx_d, total, reps);
    } else {
        report("XCHG: ofdm_probe_build failed");
    }
    free(tx_d);
    free(mem);
    return ok;
}

/* Capture and grade the far end's probe, returning the allocated table's
   length. Retries for the same reason xchg_recv_bat does: the two machines
   are not started in lockstep. */
static int run_probe_rx(const ofdm_cfg *c, const ofdm_ini *ini,
                        int *bits, int cap, int slot)
{
    ofdm_probe *p;
    void   *mem;
    size_t  ws;
    unsigned char *raw;
    double *rx_d, *H_re, *H_im, *snr_lin, *snr_db, *snr_dbp;
    ofdm_probe_info info;
    long    build_samples, window, i;
    unsigned long attempt_ticks = 0, t0;
    int     a, n_alloc = -1;

    ws = ofdm_probe_workspace(c, ini->n_probe_syms, ini->n_meas_syms,
                              c->symbol_samples);
    if (ws == 0) { report("XCHG: probe workspace rejected (sizing)"); return -1; }
    mem = malloc(ws);
    if (!mem || ofdm_probe_init(&p, mem, ws, c, ini->n_probe_syms,
                                ini->n_meas_syms, c->symbol_samples) != 0) {
        report("XCHG: probe init failed (sizing)"); free(mem); return -1;
    }
    build_samples = ofdm_probe_build_samples(p);
    free(mem);

    window = 2L * build_samples + c->sample_rate / 10L;
    if (window > 65024L) {
        report("XCHG: probe window %ld exceeds one 64KB DMA page", window);
        return -1;
    }
    ws = ofdm_probe_workspace(c, ini->n_probe_syms, ini->n_meas_syms, (int)window);
    mem = malloc(ws);
    if (!mem || ofdm_probe_init(&p, mem, ws, c, ini->n_probe_syms,
                                ini->n_meas_syms, (int)window) != 0) {
        report("XCHG: probe init failed"); free(mem); return -1;
    }

    raw     = (unsigned char *)malloc((size_t)window);
    rx_d    = (double *)malloc(sizeof(double) * (size_t)window);
    H_re    = (double *)malloc(sizeof(double) * c->n_occupied);
    H_im    = (double *)malloc(sizeof(double) * c->n_occupied);
    snr_lin = (double *)malloc(sizeof(double) * c->n_occupied);
    snr_db  = (double *)malloc(sizeof(double) * c->n_occupied);
    snr_dbp = (double *)malloc(sizeof(double) * c->n_occupied);
    if (!raw || !rx_d || !H_re || !H_im || !snr_lin || !snr_db || !snr_dbp) {
        report("XCHG: out of memory for probe capture");
    } else {
        /* Keep listening for the WHOLE slot rather than a fixed number of
           tries -- an attempt-counted budget has nothing to do with when the
           far end is actually transmitting. audio_record's grace timeout
           bounds the capture but not the estimate that follows it (observed
           running far longer than the capture on a loaded host), so refuse to
           start another attempt unless the slot has at least as much time
           left as the last attempt took. */
        for (a = 1; ; a++) {
            long left = (long)(xchg_slot_end(slot) - bios_ticks());
            if (left <= 0) {
                if (a == 1)
                    report("XCHG: slot %d already expired before the first "
                          "probe listen attempt -- a prior slot overran its "
                          "budget", slot);
                break;
            }
            if (a > 1 && (unsigned long)left < attempt_ticks) {
                report("XCHG: %ld s left in slot %d, less than the last "
                      "attempt's %ld s -- stopping early rather than "
                      "overrunning into the next slot", xchg_slot_left(slot),
                      slot, (long)(attempt_ticks / BIOS_TICK_HZ));
                break;
            }
            t0 = bios_ticks();
            if (!audio_record(c, raw, window)) break;
            for (i = 0; i < window; i++)
                rx_d[i] = ((double)raw[i] - 128.0) / PCM_SCALE;
            attempt_ticks = bios_ticks() - t0;
            if (ofdm_probe_estimate(p, rx_d, (int)window, 0.0, H_re, H_im,
                                    snr_lin, snr_db, snr_dbp, &info) == 0
                && info.valid) {
                n_alloc = ofdm_allocate(snr_db, c->n_occupied, c->pilot_spacing,
                                        ini->snr_margin_db, ini->max_bits_per_bin,
                                        info.has_beta ? info.isi_ratio : 0.0,
                                        bits, cap);
                report("XCHG: probe heard on attempt %d, coherence=%.3f, "
                      "beta=%.4f, %d entries allocated",
                      a, info.coherence,
                      info.has_beta ? info.isi_ratio : 0.0, n_alloc);
                break;
            }
            report("XCHG: probe listen attempt %d -> no probe (error=%d, "
                  "%ld s left in slot %d)", a, info.error,
                  xchg_slot_left(slot), slot);
        }
    }
    free(raw); free(rx_d); free(H_re); free(H_im);
    free(snr_lin); free(snr_db); free(snr_dbp);
    free(mem);
    return n_alloc;
}

static void run_exchange(const ofdm_ini *ini, int is_a)
{
    ofdm_cfg c;
    int mine[BAT_MAX_CARRIERS];     /* table for the direction I RECEIVE  */
    int theirs[BAT_MAX_CARRIERS];   /* table for the direction I TRANSMIT */
    int n_mine = 0, n_theirs = 0;
    int rc = -3;

    if (!build_cfg(ini, &c)) return;
    if (c.n_data_carriers > BAT_MAX_CARRIERS) {
        report("XCHG: %d data carriers exceeds BAT_MAX_CARRIERS %d",
              c.n_data_carriers, BAT_MAX_CARRIERS);
        return;
    }
    /* One shared clock for both ends. Every slot boundary below is derived
       from it, so a side that finishes early idles instead of racing. */
    xchg_t0 = bios_ticks();
    report("OFDMBAT XCHG: role %s, %d data carriers, %.0f s per slot",
          is_a ? "A" : "B", c.n_data_carriers, XCHG_SLOT_SECS);

    if (is_a) {
        report("XCHG: slot 1/4 -- transmitting probe");
        if (!run_probe_tx(&c, ini, 1)) return;
        xchg_wait_slot(1);

        report("XCHG: slot 2/4 -- listening for B's probe");
        n_mine = run_probe_rx(&c, ini, mine, BAT_MAX_CARRIERS, 2);
        xchg_wait_slot(2);
        if (n_mine <= 0) { report("XCHG: FAILED, never heard B's probe"); return; }
        print_bat("BAT for B->A (measured here)", mine, n_mine);

        report("XCHG: slot 3/4 -- listening for B's table");
        rc = xchg_recv_bat(&c, theirs, c.n_data_carriers, 3);
        xchg_wait_slot(3);

        /* Slot 4 runs even if slot 3 came up empty. A one-sided result is
           still worth having -- it tells the far end its table arrived, and
           it distinguishes "the back-channel is broken in one direction" from
           "it is broken in both", which aborting here would hide. */
        report("XCHG: slot 4/4 -- sending my table to B");
        if (!xchg_send_bat(&c, mine, n_mine, 4)) return;
        xchg_wait_slot(4);
        if (rc <= 0) { report("XCHG: FAILED, never decoded B's table (rc=%d)", rc); return; }
        n_theirs = rc;
    } else {
        report("XCHG: slot 1/4 -- listening for A's probe");
        n_mine = run_probe_rx(&c, ini, mine, BAT_MAX_CARRIERS, 1);
        xchg_wait_slot(1);
        if (n_mine <= 0) { report("XCHG: FAILED, never heard A's probe"); return; }
        print_bat("BAT for A->B (measured here)", mine, n_mine);

        report("XCHG: slot 2/4 -- transmitting probe");
        if (!run_probe_tx(&c, ini, 2)) return;
        xchg_wait_slot(2);

        report("XCHG: slot 3/4 -- sending my table to A");
        if (!xchg_send_bat(&c, mine, n_mine, 3)) return;
        xchg_wait_slot(3);

        report("XCHG: slot 4/4 -- listening for A's table");
        rc = xchg_recv_bat(&c, theirs, c.n_data_carriers, 4);
        xchg_wait_slot(4);
        if (rc <= 0) { report("XCHG: FAILED, never decoded A's table (rc=%d)", rc); return; }
        n_theirs = rc;
    }

    print_bat(is_a ? "BAT for A->B (received from B, use this to transmit)"
                   : "BAT for B->A (received from A, use this to transmit)",
              theirs, n_theirs);
    report("OFDMBAT XCHG: VERDICT PASS -- both tables held (%d measured, "
          "%d received)", n_mine, n_theirs);
}

/* In-memory codec self-test, no audio at all. Proves batcodec works on THIS
   cpu/compiler before any of it is blamed on the sound path -- the host
   harness (battest.c) already proves the algorithm, so a disagreement here
   is a 486/Watcom problem specifically. */
static void run_batloop(const ofdm_ini *ini)
{
    ofdm_cfg c;
    void *ws;
    double *sig;
    int *in, *out;
    long need, wrote;
    int i, n, rc, bad = 0;

    if (!build_cfg(ini, &c)) return;
    n = c.n_data_carriers;
    if (n > BAT_MAX_CARRIERS) n = BAT_MAX_CARRIERS;
    need = bat_encode_samples(&c, n);
    ws   = malloc(bat_workspace(&c));
    sig  = (double *)malloc(sizeof(double) * (size_t)need);
    in   = (int *)malloc(sizeof(int) * (size_t)n);
    out  = (int *)malloc(sizeof(int) * (size_t)n);
    if (!ws || !sig || !in || !out) { report("BATLOOP: out of memory"); return; }

    for (i = 0; i < n; i++) in[i] = (i * 7 + 3) % 7;
    wrote = bat_encode(&c, ws, bat_workspace(&c), in, n, sig, need);
    report("BATLOOP: encoded %d entries into %ld samples (rc=%ld)", n, need, wrote);
    if (wrote != need) { report("BATLOOP: VERDICT FAIL (encode)"); return; }

    rc = bat_decode(&c, ws, bat_workspace(&c), sig, need, out, n);
    if (rc != n) { report("BATLOOP: VERDICT FAIL (decode rc=%d)", rc); return; }
    for (i = 0; i < n; i++) if (in[i] != out[i]) bad++;
    if (bad) report("BATLOOP: VERDICT FAIL (%d/%d entries differ)", bad, n);
    else     report("BATLOOP: VERDICT PASS (%d entries round-tripped exactly)", n);
    free(ws); free(sig); free(in); free(out);
}

static void usage(void)
{
    report("usage: ofdmbat TX");
    report("       ofdmbat RX [output.wav]");
    report("       ofdmbat RATESCAN   (clock diagnostic, no signal needed)");
    report("       ofdmbat XCHG A|B   (two-way probe + BAT exchange)");
    report("       ofdmbat BATLOOP    (codec self-test, no audio)");
}

int main(int argc, char **argv)
{
    ofdm_ini ini;

    serial_out = fopen("COM1", "w");
    parse_blaster();
    ini_load("OFDMBAT.INI", &ini);

    if (argc < 2) { usage(); return 1; }

    if (stricmp(argv[1], "BATLOOP") == 0) {
        run_batloop(&ini);
        return 0;
    }
    if (stricmp(argv[1], "XCHG") == 0) {
        if (argc < 3) { usage(); return 1; }
        run_exchange(&ini, stricmp(argv[2], "A") == 0);
        return 0;
    }
    if (stricmp(argv[1], "RATESCAN") == 0) {
        run_ratescan();
        return 0;
    }
    if (stricmp(argv[1], "TX") == 0) {
        run_tx(&ini);
        return 0;
    } else if (stricmp(argv[1], "RX") == 0) {
        const char *out = (argc >= 3) ? argv[2] : "PROBE.WAV";
        run_rx(&ini, out);
        return 0;
    }
    usage();
    return 1;
}
