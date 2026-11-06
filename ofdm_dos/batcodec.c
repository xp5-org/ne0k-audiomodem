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

/* batcodec.c -- see batcodec.h for the scheme and why it is shaped this way. */
#include <string.h>
#include <math.h>

#include "batcodec.h"
#include "ofdm_sym.h"
#include "ofdm_fft.h"
#include "ofdm_acq.h"

/* ---- frame layout ------------------------------------------------------
 * header : magic(8) | n_entries(8) | crc16(16)          = 32 bits
 * body   : n_entries * 3 bits, MSB first
 * The CRC covers the body only; magic and count are checked first so a
 * mis-synced frame is rejected before its length is ever trusted as a
 * buffer bound. Getting that order wrong is how a bad decode becomes a
 * memory bug rather than a returned error. */
#define BAT_HDR_BITS   32
#define BAT_BITS_PER_ENTRY 3

static unsigned bat_crc16(const unsigned char *p, int n)
{
    /* CCITT, bitwise -- no table, because a 256-entry table costs more DOS
       data than the handful of milliseconds it would save on a 300-bit
       payload. */
    unsigned crc = 0xFFFFu;
    int i, b;
    for (i = 0; i < n; i++) {
        crc ^= (unsigned)p[i] << 8;
        for (b = 0; b < 8; b++)
            crc = (crc & 0x8000u) ? (unsigned)(((crc << 1) ^ 0x1021u) & 0xFFFFu)
                                  : (unsigned)((crc << 1) & 0xFFFFu);
    }
    return crc & 0xFFFFu;
}

static int bat_frame_bits(int n_entries)
{
    return BAT_HDR_BITS + n_entries * BAT_BITS_PER_ENTRY;
}

static int bat_frame_syms(int n_entries)
{
    int bits = bat_frame_bits(n_entries);
    return (bits + BAT_GROUPS - 1) / BAT_GROUPS;
}

/* Pack/unpack one bit into a bit array held as one byte per bit -- clearer
   than bit-twiddling a packed buffer, and the arrays are a few hundred
   entries. */
static void bat_put_bits(unsigned char *bits, int *pos, unsigned long v, int n)
{
    int i;
    for (i = n - 1; i >= 0; i--)
        bits[(*pos)++] = (unsigned char)((v >> i) & 1u);
}

static unsigned long bat_get_bits(const unsigned char *bits, int *pos, int n)
{
    unsigned long v = 0;
    int i;
    for (i = 0; i < n; i++)
        v = (v << 1) | (unsigned long)(bits[(*pos)++] & 1u);
    return v;
}

/* Byte view of the body, for the CRC. Body bits are a multiple of 3, not of
   8, so the last byte is zero-padded; both ends pad identically. */
static int bat_body_bytes(const unsigned char *bits, int nbits,
                          unsigned char *out, int cap)
{
    int nbytes = (nbits + 7) / 8;
    int i;
    if (nbytes > cap) return -1;
    memset(out, 0, (size_t)nbytes);
    for (i = 0; i < nbits; i++)
        if (bits[i]) out[i >> 3] |= (unsigned char)(0x80u >> (i & 7));
    return nbytes;
}

/* ---- workspace ---------------------------------------------------------
 * One block shared by encoder and decoder: the symbol builder, an FFT, the
 * known reference, and room for a symbol plus its spectrum. Sized for the
 * larger of the two so a caller needs only one allocation for either. */
struct bat_ws {
    ofdm_sym  *sym;
    ofdm_fft   fft;
    double    *known;      /* n_occupied BPSK reference                 */
    double    *zero;       /* n_occupied zeros, for the imaginary part  */
    double    *occ_re;     /* n_occupied working carriers               */
    double    *tsym;       /* one symbol, symbol_samples + tapers       */
    ofdm_cpx  *spec;       /* n_fft/2+1                                 */
    ofdm_real *tbuf;       /* n_fft, FFT input -- in the workspace and not
                              on the stack: as doubles that is 32KB at
                              n_fft=4096, which a DOS stack will not carry */
    double    *h_re;       /* channel estimate                          */
    double    *h_im;
    double     dc;         /* capture mean, removed before correlating  */
};

/* Last decode's sync numbers, for the caller's failure report. Globals rather
   than out-params because every caller wants them only when something went
   wrong, and threading them through the signature would clutter the common
   path. Single-threaded DOS, so this is safe. */
long bat_dbg_runlen = 0;
long bat_dbg_runend = 0;

static size_t bat_align8(size_t v) { return (v + 7u) & ~(size_t)7u; }

static size_t bat_parts(const ofdm_cfg *c, size_t *sym_ws, size_t *fft_ws)
{
    size_t n;
    if (!c || c->n_occupied <= 0 || c->n_fft <= 0) return 0;
    *sym_ws = ofdm_sym_workspace(c);
    *fft_ws = ofdm_fft_workspace(c->n_fft);
    if (*sym_ws == 0 || *fft_ws == 0) return 0;
    n  = bat_align8(sizeof(struct bat_ws));
    n += bat_align8(*sym_ws);
    n += bat_align8(*fft_ws);
    n += bat_align8(sizeof(double) * (size_t)c->n_occupied) * 5; /* known,zero,occ,h_re,h_im */
    n += bat_align8(sizeof(double) * (size_t)(c->symbol_samples + 2 * c->taper_samples + 8));
    n += bat_align8(sizeof(ofdm_cpx) * (size_t)(c->n_fft / 2 + 1));
    n += bat_align8(sizeof(ofdm_real) * (size_t)c->n_fft);
    return n;
}

size_t bat_workspace(const ofdm_cfg *c)
{
    size_t a, b;
    return bat_parts(c, &a, &b);
}

static struct bat_ws *bat_ws_init(const ofdm_cfg *c, void *mem, size_t memlen)
{
    size_t sym_ws, fft_ws, need;
    unsigned char *p = (unsigned char *)mem;
    struct bat_ws *w;
    size_t occ_bytes, tsym_bytes;

    need = bat_parts(c, &sym_ws, &fft_ws);
    if (need == 0 || mem == NULL || memlen < need) return NULL;

    occ_bytes  = bat_align8(sizeof(double) * (size_t)c->n_occupied);
    tsym_bytes = bat_align8(sizeof(double) *
                            (size_t)(c->symbol_samples + 2 * c->taper_samples + 8));

    w = (struct bat_ws *)p;             p += bat_align8(sizeof(struct bat_ws));
    if (ofdm_sym_init(&w->sym, p, sym_ws, c) != 0) return NULL;
    p += bat_align8(sym_ws);
    if (ofdm_fft_init(&w->fft, c->n_fft, p, fft_ws) != 0) return NULL;
    p += bat_align8(fft_ws);
    w->known  = (double *)p;  p += occ_bytes;
    w->zero   = (double *)p;  p += occ_bytes;
    w->occ_re = (double *)p;  p += occ_bytes;
    w->h_re   = (double *)p;  p += occ_bytes;
    w->h_im   = (double *)p;  p += occ_bytes;
    w->tsym   = (double *)p;  p += tsym_bytes;
    w->spec   = (ofdm_cpx *)p;
    p += bat_align8(sizeof(ofdm_cpx) * (size_t)(c->n_fft / 2 + 1));
    w->tbuf   = (ofdm_real *)p;

    /* amp 1/sqrt(2) matches the probe's own sequences (see ofdm_cfg.h's note
       on REF_SYM_BOOST -- the probe deliberately does NOT boost). */
    ofdm_known_bpsk(w->known, c->n_occupied, BAT_SEED_SYNC, 0.7071067811865476);
    memset(w->zero, 0, sizeof(double) * (size_t)c->n_occupied);
    return w;
}

/* Which group a carrier belongs to. Contiguous slices rather than
   interleaved: a real channel impairment is usually a contiguous band (a
   filter edge, a notch), and interleaving would spread one bad band across
   EVERY group instead of hurting one. Contiguous grouping means a band that
   dies takes one bit with it and the CRC catches that, rather than degrading
   all eight bits at once. */
static int bat_group_of(int carrier, int n_occupied)
{
    long g = (long)carrier * BAT_GROUPS / (long)n_occupied;
    if (g < 0) g = 0;
    if (g >= BAT_GROUPS) g = BAT_GROUPS - 1;
    return (int)g;
}

long bat_encode_samples(const ofdm_cfg *c, int n_entries)
{
    long syms;
    if (!c || n_entries <= 0 || n_entries > BAT_MAX_CARRIERS) return -1;
    syms = (long)BAT_SYNC_SYMS + (long)bat_frame_syms(n_entries);
    syms *= BAT_REPEATS;
    return syms * (long)c->symbol_samples;
}

long bat_encode(const ofdm_cfg *c, void *mem, size_t memlen,
                const int *bits, int n_entries, double *out, long cap)
{
    struct bat_ws *w;
    unsigned char framebits[BAT_HDR_BITS + BAT_MAX_CARRIERS * BAT_BITS_PER_ENTRY];
    unsigned char body[(BAT_MAX_CARRIERS * BAT_BITS_PER_ENTRY + 7) / 8];
    int pos, i, k, s, rep, nbody, nsyms, nb;
    unsigned crc;
    long need, wrote = 0;

    if (!c || !bits || !out || n_entries <= 0 || n_entries > BAT_MAX_CARRIERS)
        return -1;
    need = bat_encode_samples(c, n_entries);
    if (need <= 0) return -1;
    if (cap < need) return -2;
    w = bat_ws_init(c, mem, memlen);
    if (!w) return -3;

    /* body first, so its CRC can go in the header ahead of it */
    nbody = n_entries * BAT_BITS_PER_ENTRY;
    pos = 0;
    for (i = 0; i < n_entries; i++) {
        int v = bits[i];
        if (v < 0) v = 0;
        if (v > 7) v = 7;
        bat_put_bits(framebits + BAT_HDR_BITS, &pos, (unsigned long)v,
                     BAT_BITS_PER_ENTRY);
    }
    nb = bat_body_bytes(framebits + BAT_HDR_BITS, nbody, body, (int)sizeof(body));
    if (nb < 0) return -1;
    crc = bat_crc16(body, nb);

    pos = 0;
    bat_put_bits(framebits, &pos, (unsigned long)BAT_MAGIC, 8);
    bat_put_bits(framebits, &pos, (unsigned long)n_entries, 8);
    bat_put_bits(framebits, &pos, (unsigned long)crc, 16);

    nsyms = bat_frame_syms(n_entries);

    for (rep = 0; rep < BAT_REPEATS; rep++) {
        /* preamble: every carrier the plain reference */
        for (s = 0; s < BAT_SYNC_SYMS; s++) {
            for (k = 0; k < c->n_occupied; k++) w->occ_re[k] = w->known[k];
            (void)ofdm_sym_make(w->sym, w->occ_re, w->zero, 0, w->tsym);
            for (i = 0; i < c->symbol_samples; i++) out[wrote++] = w->tsym[i];
        }
        /* payload: BAT_GROUPS bits per symbol, each spread over its group */
        for (s = 0; s < nsyms; s++) {
            for (k = 0; k < c->n_occupied; k++) {
                int g  = bat_group_of(k, c->n_occupied);
                int bi = s * BAT_GROUPS + g;
                int bit = (bi < bat_frame_bits(n_entries)) ? framebits[bi] : 0;
                w->occ_re[k] = bit ? -w->known[k] : w->known[k];
            }
            (void)ofdm_sym_make(w->sym, w->occ_re, w->zero, 0, w->tsym);
            for (i = 0; i < c->symbol_samples; i++) out[wrote++] = w->tsym[i];
        }
    }
    return wrote;
}

/* Demodulate the symbol starting at `off`, leaving the equalised correlation
   per group in `acc`. Shared by the channel estimate and the payload, so the
   two cannot drift apart in how they read a symbol off the wire. */
static void bat_sym_spectrum(struct bat_ws *w, const ofdm_cfg *c,
                             const double *rx, long off)
{
    int i;
    int n = c->n_fft;
    /* skip the cyclic prefix, transform the body */
    for (i = 0; i < n; i++) w->tbuf[i] = (ofdm_real)rx[off + c->cp_len + i];
    ofdm_rfft(&w->fft, w->tbuf, w->spec);
}

/* Equalised correlation per group for the symbol at `off`. The channel
   estimate carries an arbitrary phase, so the payload MUST be equalised
   before a sign is taken -- correlating the raw spectrum against a real
   reference would decide bits on cos(phase) and flip them wholesale
   whenever the channel rotated past 90 degrees. */
static void bat_group_corr(struct bat_ws *w, const ofdm_cfg *c,
                           const double *rx, long off,
                           double *acc, int *cnt)
{
    int i, k;
    for (i = 0; i < BAT_GROUPS; i++) { acc[i] = 0.0; cnt[i] = 0; }
    bat_sym_spectrum(w, c, rx, off);
    for (k = 0; k < c->n_occupied; k++) {
        int g = bat_group_of(k, c->n_occupied);
        double hr = w->h_re[k], hi = w->h_im[k];
        double mag = hr * hr + hi * hi;
        double corr;
        if (mag <= 0.0) continue;
        corr = ((double)w->spec[c->bin_lo + k].re * hr +
                (double)w->spec[c->bin_lo + k].im * hi) / mag;
        acc[g] += corr * w->known[k];
        cnt[g]++;
    }
}

/* One decode attempt at an assumed body offset: estimate H from the preamble
   that would precede it, read the header, and verify the CRC. Returns entries
   recovered, or negative. Split out because alignment is found by TRYING
   candidates and letting the CRC arbitrate -- see bat_decode. */
static int bat_try(struct bat_ws *w, const ofdm_cfg *c, const double *rx, long n,
                   long body_off, int *bits_out, int cap, int *saw_frame)
{
    unsigned char framebits[BAT_HDR_BITS + BAT_MAX_CARRIERS * BAT_BITS_PER_ENTRY];
    unsigned char body[(BAT_MAX_CARRIERS * BAT_BITS_PER_ENTRY + 7) / 8];
    double acc[BAT_GROUPS];
    int    cnt[BAT_GROUPS];
    long   pre_off = body_off - (long)BAT_SYNC_SYMS * c->symbol_samples;
    int    i, k, s, pos, nsyms, nb, n_entries, magic;
    unsigned crc_rx;

    if (pre_off < 0) return -3;
    if (body_off + 4L * c->symbol_samples > n) return -3;

    /* Channel estimate from THIS candidate's preamble. It has to be redone per
       candidate rather than once: a misaligned window is a cyclic shift, which
       is a phase ramp across carriers, and the whole reason misalignment is
       survivable is that H picks up the same ramp and cancels it in the
       payload. An H measured at a different offset would not. */
    for (k = 0; k < c->n_occupied; k++) { w->h_re[k] = 0.0; w->h_im[k] = 0.0; }
    for (s = 0; s < BAT_SYNC_SYMS; s++) {
        long off = pre_off + (long)s * c->symbol_samples;
        if (off + c->symbol_samples > n) break;
        bat_sym_spectrum(w, c, rx, off);
        for (k = 0; k < c->n_occupied; k++) {
            double kr = w->known[k];
            if (kr == 0.0) continue;
            w->h_re[k] += (double)w->spec[c->bin_lo + k].re / kr;
            w->h_im[k] += (double)w->spec[c->bin_lo + k].im / kr;
        }
    }

    /* header, to learn the length before trusting it as a bound */
    pos = 0;
    for (s = 0; s < 4 && pos < BAT_HDR_BITS; s++) {
        bat_group_corr(w, c, rx, body_off + (long)s * c->symbol_samples, acc, cnt);
        for (i = 0; i < BAT_GROUPS && pos < BAT_HDR_BITS; i++)
            framebits[pos++] = (unsigned char)((cnt[i] && acc[i] < 0.0) ? 1 : 0);
    }
    if (pos < BAT_HDR_BITS) return -3;

    pos = 0;
    magic     = (int)bat_get_bits(framebits, &pos, 8);
    n_entries = (int)bat_get_bits(framebits, &pos, 8);
    crc_rx    = (unsigned)bat_get_bits(framebits, &pos, 16);
    if (magic != (int)BAT_MAGIC)                        return -3;
    *saw_frame = 1;   /* magic matched: this really is a frame, so a later
                         failure is corruption (-4) rather than absence (-3) */
    if (n_entries <= 0 || n_entries > BAT_MAX_CARRIERS)  return -4;
    if (n_entries > cap)                                 return -5;

    nsyms = bat_frame_syms(n_entries);
    if (body_off + (long)nsyms * c->symbol_samples > n)  return -4;
    for (s = 0; s < nsyms; s++) {
        bat_group_corr(w, c, rx, body_off + (long)s * c->symbol_samples, acc, cnt);
        for (i = 0; i < BAT_GROUPS; i++) {
            int bi = s * BAT_GROUPS + i;
            if (bi < bat_frame_bits(n_entries))
                framebits[bi] = (unsigned char)((cnt[i] && acc[i] < 0.0) ? 1 : 0);
        }
    }

    nb = bat_body_bytes(framebits + BAT_HDR_BITS,
                        n_entries * BAT_BITS_PER_ENTRY, body, (int)sizeof(body));
    if (nb < 0) return -4;
    if (bat_crc16(body, nb) != crc_rx) return -4;

    pos = BAT_HDR_BITS;
    for (i = 0; i < n_entries; i++)
        bits_out[i] = (int)bat_get_bits(framebits, &pos, BAT_BITS_PER_ENTRY);
    return n_entries;
}

int bat_decode(const ofdm_cfg *c, void *mem, size_t memlen,
               const double *rx, long n, int *bits_out, int cap)
{
    struct bat_ws *w;
    long L, run_end = -1;
    int  i, rc, saw_frame = 0, saw_toobig = 0;

    if (!c || !rx || !bits_out || cap <= 0) return -1;
    if (c->n_fft > 4096) return -1;
    w = bat_ws_init(c, mem, memlen);
    if (!w) return -2;
    L = c->symbol_samples;

    /* ---- coarse: Schmidl-Cox style autocorrelation, not a matched filter.
       Three reasons, the first two correctness rather than speed:

       (a) A payload symbol carrying a 0 bit is BIT-IDENTICAL to a preamble
           symbol (both are +known on every carrier), so "the position whose
           correlation against the reference is largest" happily locks onto
           payload. The host harness caught exactly that -- synced fine at
           offset 0 and 37, failed at 501 and 1234.
       (b) What distinguishes the preamble is that it is BAT_SYNC_SYMS
           IDENTICAL symbols in a row. Autocorrelating the capture against
           itself one symbol apart finds that structure directly.
       (c) O(n) with sliding sums and no FFT per candidate, which the matched
           filter would have needed thousands of times over. */
    {
        long d, run_start = -1, run_len = 0, best_len = 0;
        double P = 0.0, R = 0.0, rmax = 0.0, r2, dc = 0.0;

        if (n < 3 * L) return -3;

        /* Remove DC before correlating. An 8-bit ADC whose silence level is
           not exactly 128 leaves a constant offset, and a constant correlates
           PERFECTLY with itself one symbol later -- so silence reads as
           P/R ~ 1 and manufactures a plateau wherever there is no signal,
           which is the opposite of what this detector is looking for. This
           container has already been bitten by an SB16 DC offset once (see
           the sb16echo DC-offset fix in the history), and the measured
           captures here sit around 122 rather than 128. The signal itself is
           zero-mean by construction, so subtracting the capture mean costs
           nothing and removes the failure mode entirely. */
        for (d = 0; d < n; d++) dc += rx[d];
        dc /= (double)n;
        w->dc = dc;

        for (i = 0; i < (int)L; i++) {
            P += (rx[i] - dc) * (rx[i + L] - dc);
            R += (rx[i + L] - dc) * (rx[i + L] - dc);
        }
        r2 = R;
        for (d = 0; d + 2 * L < n; d++) {
            if (r2 > rmax) rmax = r2;
            r2 -= (rx[d + L] - dc) * (rx[d + L] - dc);
            r2 += (rx[d + 2 * L] - dc) * (rx[d + 2 * L] - dc);
        }
        if (rmax <= 0.0) return -3;

        for (d = 0; d + 2 * L < n; d++) {
            double m = (R > 1e-12) ? (P / R) : 0.0;
            if (m > 0.5 && R > rmax * 0.10) {
                if (run_start < 0) { run_start = d; run_len = 0; }
                run_len++;
                if (run_len > best_len) {
                    best_len = run_len;
                    run_end  = run_start + run_len - 1;
                }
            } else {
                run_start = -1;
                run_len   = 0;
            }
            P -= (rx[d] - dc) * (rx[d + L] - dc);
            P += (rx[d + L] - dc) * (rx[d + 2 * L] - dc);
            R -= (rx[d + L] - dc) * (rx[d + L] - dc);
            R += (rx[d + 2 * L] - dc) * (rx[d + 2 * L] - dc);
        }
        bat_dbg_runlen = best_len;
        bat_dbg_runend = run_end;
        if (run_end < 0 || best_len < (long)(BAT_SYNC_SYMS - 3) * L) return -3;
        /* A run VASTLY longer than the preamble is not a preamble -- it is
           silence, a stuck rail, or a tone. Rejecting it here keeps the
           candidate search from being aimed at the end of a bogus plateau,
           where it would waste every attempt and report a misleading "no
           frame" instead of "that was not a frame". */
        if (best_len > (long)(BAT_SYNC_SYMS + 4) * L) return -3;
    }

    /* ---- fine: let the CRC arbitrate the alignment.
       The plateau's edges are NOT a reliable frame boundary. A frame's
       trailing payload symbols are zero-padded, so they are identical to each
       other AND to the preamble that follows, and the run merges BACKWARDS
       across the boundary -- observed locking 189 samples early, where the
       usable tolerance is one cyclic prefix (32). Rather than deriving the
       edge analytically and being wrong in a new way, try candidate offsets
       and accept the one whose CRC passes. That is precisely what a CRC is
       for, it is already being computed, and it makes a successful decode
       self-verifying instead of merely plausible.

       The plateau END is the better anchor of the two (it is pinned by the
       preamble/payload transition, where the header's magic guarantees a
       differing symbol), so candidates are centred on it and stepped by half
       a prefix so at least one lands inside the ISI-free window. */
    {
        long est = run_end + L;
        long step = (c->cp_len > 3) ? (long)(c->cp_len / 2) : 4L;
        long span = 3L * L;
        long off;
        for (off = est - span; off <= est + span; off += step) {
            if (off < (long)BAT_SYNC_SYMS * L) continue;
            rc = bat_try(w, c, rx, n, off, bits_out, cap, &saw_frame);
            if (rc > 0) return rc;
            /* Do NOT abort the search on a bad candidate. Noise can make a
               misaligned header decode a plausible magic and an absurd entry
               count, and treating that as a verdict throws away the remaining
               candidates -- one of which is the correctly aligned one. Only a
               CRC-verified decode ends the search; everything else is just
               this offset being wrong. (Observed: a 10 dB case failed while
               6 dB passed, purely because a spurious header aborted early.) */
            if (rc == -5) saw_toobig = 1;
        }
    }
    if (saw_toobig) return -5;
    return saw_frame ? -4 : -3;
}
