/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
#include "css_enc.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Rate-1/2 K=7 NASA generators, 0o171 and 0o133. The register is 7 bits with
 * the newest bit entering at the MSB, matching css_encoder.conv_encode_k7.
 * The opposite convention (new bit at the LSB) produces a valid but
 * different code. */
#define CONV_G1   0x79      /* 0b1111001 */
#define CONV_G2   0x5B      /* 0b1011011 */
#define CONV_K    7
#define CONV_TAIL (CONV_K - 1)
#define CONV_MASK ((1 << CONV_K) - 1)

#define CRC_BYTES 4

/* Population-count parity of a value that never exceeds 7 bits here. */
static int parity(unsigned int v)
{
    int p = 0;

    while (v) {
        p ^= (int)(v & 1u);
        v >>= 1;
    }
    return p;
}

int css_enc_geometry(css_enc_cfg *g, int chunk_size, int rs_t,
                     int conv_code, int diff_enc, int interleave_rows)
{
    int cols;

    if (!g || chunk_size < 1 || rs_t < 0 || interleave_rows < 0) return -1;
    if (conv_code != 0 && conv_code != 1) return -1;

    g->chunk_size      = chunk_size;
    g->rs_parity       = rs_t * 2;
    g->base_bits       = (1 + chunk_size + CRC_BYTES + g->rs_parity) * 8;
    g->conv_code       = conv_code;
    g->diff_enc        = diff_enc ? 1 : 0;
    g->interleave_rows = (interleave_rows > 1) ? interleave_rows : 0;
    g->conv_bits       = conv_code ? (g->base_bits + CONV_TAIL) * 2
                                   : g->base_bits;

    if (g->interleave_rows > 1) {
        /* ceil division, spelled without the -(-a//b) idiom Python uses */
        cols = (g->conv_bits + g->interleave_rows - 1) / g->interleave_rows;
        g->frame_symbols = g->interleave_rows * cols;
    } else {
        g->frame_symbols = g->conv_bits;
    }
    return 0;
}

unsigned long css_crc32(const unsigned char *buf, int n)
{
    unsigned long crc;
    int i, k;

    if (!buf || n < 0) return 0uL;
    crc = 0xFFFFFFFFuL;
    for (i = 0; i < n; i++) {
        crc ^= (unsigned long)buf[i];
        for (k = 0; k < 8; k++) {
            /* Reflected form: shift right, conditionally xor the reversed
             * polynomial. The 0xFFFFFFFFuL masks are load-bearing: C89 only
             * guarantees unsigned long is at least 32 bits, and on an LP64
             * host it is 64, where an unmasked shift keeps bits a 32-bit
             * target would have dropped. */
            if (crc & 1uL) crc = ((crc >> 1) ^ 0xEDB88320uL) & 0xFFFFFFFFuL;
            else           crc = (crc >> 1) & 0xFFFFFFFFuL;
        }
    }
    return (crc ^ 0xFFFFFFFFuL) & 0xFFFFFFFFuL;
}

int css_enc_inner(const css_enc_cfg *g, int seq, const unsigned char *data,
                  unsigned char *out)
{
    unsigned long crc;
    int i, n;

    if (!g || !data || !out || seq < 0 || seq > 255) return -1;

    out[0] = (unsigned char)seq;
    for (i = 0; i < g->chunk_size; i++) out[1 + i] = data[i];
    n = 1 + g->chunk_size;

    /* Big-endian on the wire, matching crc.to_bytes(4, "big"). */
    crc = css_crc32(out, n);
    out[n + 0] = (unsigned char)((crc >> 24) & 0xFFuL);
    out[n + 1] = (unsigned char)((crc >> 16) & 0xFFuL);
    out[n + 2] = (unsigned char)((crc >>  8) & 0xFFuL);
    out[n + 3] = (unsigned char)( crc        & 0xFFuL);
    return n + CRC_BYTES;
}

int css_bytes_to_bits(const unsigned char *bytes, int n, unsigned char *bits)
{
    int i, k;

    if (!bytes || !bits || n < 0) return -1;
    /* Descending k so the MSB lands first, as get_bits() does. The expansion
     * is 8:1 so out == in aliasing is impossible here. */
    for (i = 0; i < n; i++)
        for (k = 0; k < 8; k++)
            bits[i * 8 + k] = (unsigned char)((bytes[i] >> (7 - k)) & 1);
    return n * 8;
}

int css_conv_encode_k7(const unsigned char *bits, int n, unsigned char *out)
{
    unsigned int reg;
    int i, b, total;

    if (!bits || !out || n < 0) return -1;

    reg   = 0u;
    total = n + CONV_TAIL;
    /* CONV_TAIL zero bits appended to flush the register back to state 0,
     * letting the Viterbi terminate at a known state instead of picking the
     * best survivor. Generated here rather than expected in `bits`. */
    for (i = 0; i < total; i++) {
        b   = (i < n) ? (bits[i] ? 1 : 0) : 0;
        reg = ((reg >> 1) | ((unsigned int)b << (CONV_K - 1))) & CONV_MASK;
        out[i * 2 + 0] = (unsigned char)parity(reg & CONV_G1);
        out[i * 2 + 1] = (unsigned char)parity(reg & CONV_G2);
    }
    return total * 2;
}

int css_diff_encode(const unsigned char *bits, int n, int ref,
                    unsigned char *out)
{
    int i, prev, d;

    if (!bits || !out || n < 0) return -1;

    prev = ref ? 1 : 0;
    for (i = 0; i < n; i++) {
        /* out[i] = in[i] XOR out[i-1]: the running output, not the running
         * input. The decoder's inverse, diff_decode_bits, runs the other
         * way: recovered[i] = received[i] XOR received[i-1]. Reading
         * bits[i] before writing out[i] makes out == in safe. */
        d      = (bits[i] ? 1 : 0) ^ prev;
        out[i] = (unsigned char)d;
        prev   = d;
    }
    return n;
}

int css_interleave(const unsigned char *bits, int n, int rows,
                   unsigned char *out)
{
    int cols, r, c, w;

    if (!bits || !out || n < 0 || rows < 0) return -1;
    if (rows <= 1) {
        for (r = 0; r < n; r++) out[r] = bits[r] ? 1 : 0;
        return n;
    }

    /* Write row-by-row into a rows x cols block, read out column-by-column.
     * Zero-padded to fill; the decoder trims back to base_bits. Done as a
     * direct index map rather than a transpose, so `out` must not alias
     * `bits`. */
    cols = (n + rows - 1) / rows;
    w    = 0;
    for (c = 0; c < cols; c++)
        for (r = 0; r < rows; r++) {
            int src = r * cols + c;
            out[w++] = (src < n) ? (bits[src] ? 1 : 0) : 0;
        }
    return w;
}

int css_enc_scratch(const css_enc_cfg *g)
{
    if (!g) return 0;
    /* One stage buffer. frame_symbols is the largest any intermediate gets:
     * base_bits <= conv_bits <= frame_symbols at every legal config. */
    return g->frame_symbols;
}

int css_enc_bits(const css_enc_cfg *g, const unsigned char *frame, int nframe,
                 int diff_ref, unsigned char *bits, int cap,
                 unsigned char *scratch, int scap)
{
    unsigned char *a, *b, *swap;
    int nb;

    if (!g || !frame || !bits || !scratch || nframe < 0) return -1;
    if (cap < g->frame_symbols || scap < css_enc_scratch(g)) return -2;
    if (nframe * 8 != g->base_bits) return -1;

    /* Ping-pong between the caller's output buffer and the scratch, ending on
     * `bits`. A caller-owned scratch buffer rather than in-place stages: two
     * of the three transforms change length (conv expands 1 -> 2 plus a
     * tail, interleave pads up to a whole block) and the interleave is a
     * transpose, which in place would need a cycle-following permutation or
     * a shift-and-refill. frame_symbols bytes is at most a few kB. */
    a = bits;
    b = scratch;

    nb = css_bytes_to_bits(frame, nframe, a);
    if (nb < 0) return -1;

    if (g->diff_enc) {
        /* Safe in place: reads bits[i] before writing out[i]. */
        nb = css_diff_encode(a, nb, diff_ref, a);
        if (nb < 0) return -1;
    }
    if (g->conv_code) {
        nb = css_conv_encode_k7(a, nb, b);
        if (nb < 0) return -1;
        swap = a; a = b; b = swap;
    }
    if (g->interleave_rows > 1) {
        nb = css_interleave(a, nb, g->interleave_rows, b);
        if (nb < 0) return -1;
        swap = a; a = b; b = swap;
    }

    if (nb != g->frame_symbols) return -1;
    if (a != bits) {
        int i;
        for (i = 0; i < nb; i++) bits[i] = a[i];
    }
    return nb;
}

int css_enc_chirp(float *out, int n, double f0, double f1, double duration)
{
    css_phase t, k, phi;
    double dt;
    int i;

    if (!out || n < 1 || duration <= 0.0) return -1;

    /* t[i] = i * duration / n, k = (f1 - f0) / duration. dt is formed in
     * double regardless of css_phase: it is one division whose result is
     * then rounded into the working precision. */
    dt = duration / (double)n;
    k  = (css_phase)((f1 - f0) / duration);
    for (i = 0; i < n; i++) {
        t   = (css_phase)((double)i * dt);
        phi = (css_phase)(2.0 * M_PI) * ((css_phase)f0 * t
                                         + (css_phase)0.5 * k * t * t);
        out[i] = (float)sin((double)phi);
    }
    return 0;
}
