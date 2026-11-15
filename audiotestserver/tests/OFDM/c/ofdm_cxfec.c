/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

#include "ofdm_cxfec.h"

#define G0 0133u
#define G1 0171u
#define NST 64

static int parity7(unsigned v)
{
    v &= 0x7Fu;
    v ^= v >> 4; v ^= v >> 2; v ^= v >> 1;
    return (int)(v & 1u);
}

/* Puncture pattern: per trellis step within the period, which of (c0, c1) go out. */
static int period_of(int fec)
{
    return fec == OFDM_CXFEC_R23 ? 2 : fec == OFDM_CXFEC_R34 ? 3 : 1;
}

static void keep_of(int fec, long j, int *k0, int *k1)
{
    int p = (int)(j % period_of(fec));
    *k0 = 1; *k1 = 1;
    if (fec == OFDM_CXFEC_R23 && p == 1) *k1 = 0;
    if (fec == OFDM_CXFEC_R34) {
        if (p == 1) *k1 = 0;
        if (p == 2) *k0 = 0;
    }
}

int ofdm_cxfec_valid(int fec)
{
    return fec >= OFDM_CXFEC_NONE && fec <= OFDM_CXFEC_R34;
}

long ofdm_cxfec_coded_bits(int fec, long n_info)
{
    long steps, j, n = 0;
    int  k0, k1;
    if (!ofdm_cxfec_valid(fec) || fec == OFDM_CXFEC_NONE || n_info < 0) return -1;
    steps = n_info + OFDM_CXFEC_TAIL;
    for (j = 0; j < period_of(fec) && j < steps; j++) {   /* whole periods + rest */
        keep_of(fec, j, &k0, &k1);
        n += (long)(k0 + k1) * ((steps - j + period_of(fec) - 1) / period_of(fec));
    }
    return n;
}

long ofdm_cxfec_coded_index(int fec, long j)
{
    long per = period_of(fec), m = j / per, p = j % per;
    if (fec == OFDM_CXFEC_R12) return 2 * j;
    if (fec == OFDM_CXFEC_R23) return 3 * m + (p ? 2 : 0);
    if (fec == OFDM_CXFEC_R34) return 4 * m + (p == 0 ? 0 : p == 1 ? 2 : 3);
    return -1;
}

static long gcd_l(long a, long b)
{
    while (b) { long t = a % b; a = b; b = t; }
    return a;
}

long ofdm_cxfec_perm_step(long n_coded, int bps)
{
    long p;
    if (n_coded < 2) return 1;
    p = (long)bps + bps / 3 + 1;
    p %= n_coded;
    if (p < 1) p = 1;
    while (gcd_l(p, n_coded) != 1) p = (p + 1) % n_coded ? (p + 1) % n_coded : 1;
    return p;
}

long ofdm_cxfec_encode(int fec, const unsigned char *info, long n_info,
                       unsigned char *coded)
{
    long j, n = 0, steps;
    unsigned st = 0, reg;
    int  k0, k1;
    if (!coded || (!info && n_info > 0) || fec == OFDM_CXFEC_NONE || !ofdm_cxfec_valid(fec))
        return -1;
    steps = n_info + OFDM_CXFEC_TAIL;
    for (j = 0; j < steps; j++) {
        unsigned u = j < n_info ? (unsigned)(info[j] & 1u) : 0u;
        reg = (u << 6) | st;
        keep_of(fec, j, &k0, &k1);
        if (k0) coded[n++] = (unsigned char)parity7(reg & G0);
        if (k1) coded[n++] = (unsigned char)parity7(reg & G1);
        st = reg >> 1;
    }
    return n;
}

/* work: path metrics 2 x NST floats, decisions (steps x 64 bits) as 2 unsigned
 * long words of 32 bits each per step. */
static size_t al8(size_t v) { return (v + 7u) & ~(size_t)7u; }

size_t ofdm_cxfec_workspace(long n_info)
{
    if (n_info < 0) return 0;
    return al8(2 * NST * sizeof(float))
         + al8((size_t)(n_info + OFDM_CXFEC_TAIL) * 2 * sizeof(unsigned long));
}

int ofdm_cxfec_decode(int fec, const float *llr, long n_info,
                      unsigned char *info, void *work, size_t worklen)
{
    float *pm, *nm;
    unsigned long *dec;
    long   steps, j, ci = 0;
    int    s, k0, k1;
    unsigned char *p = (unsigned char *)work;
    unsigned char out0[NST], out1[NST];

    if (!llr || !info || !work || fec == OFDM_CXFEC_NONE || !ofdm_cxfec_valid(fec) || n_info < 0)
        return -1;
    if (worklen < ofdm_cxfec_workspace(n_info)) return -2;
    steps = n_info + OFDM_CXFEC_TAIL;
    pm  = (float *)p;            p += al8(2 * NST * sizeof(float));
    nm  = pm + NST;
    dec = (unsigned long *)p;

    /* expected outputs for input 0 from each state (input 1 = flip both: both
     * generators include the newest bit) */
    for (s = 0; s < NST; s++) {
        unsigned reg = (unsigned)s;                     /* u = 0 */
        out0[s] = (unsigned char)parity7(reg & G0);
        out1[s] = (unsigned char)parity7(reg & G1);
    }
    for (s = 0; s < NST; s++) pm[s] = s == 0 ? 0.0f : -1e30f;

    for (j = 0; j < steps; j++) {
        float l0 = 0.0f, l1 = 0.0f;
        unsigned long d0 = 0, d1 = 0;
        int ns;
        keep_of(fec, j, &k0, &k1);
        if (k0) l0 = llr[ci++];
        if (k1) l1 = llr[ci++];
        /* new state ns = (u << 5) | (s >> 1); predecessors s = 2*(ns & 31) + {0,1} */
        for (ns = 0; ns < NST; ns++) {
            unsigned u = (unsigned)ns >> 5;
            int sa = (ns & 31) << 1, sb = sa | 1, best;
            float ma, mb, bm;
            int c0, c1;
            c0 = out0[sa] ^ (int)u; c1 = out1[sa] ^ (int)u;
            bm = (c0 ? -l0 : l0) + (c1 ? -l1 : l1);
            ma = pm[sa] + bm;
            c0 = out0[sb] ^ (int)u; c1 = out1[sb] ^ (int)u;
            bm = (c0 ? -l0 : l0) + (c1 ? -l1 : l1);
            mb = pm[sb] + bm;
            best = mb > ma;
            nm[ns] = best ? mb : ma;
            if (best) {
                if (ns < 32) d0 |= 1ul << ns; else d1 |= 1ul << (ns - 32);
            }
        }
        dec[2 * j] = d0; dec[2 * j + 1] = d1;
        { float *t = pm; pm = nm; nm = t; }
        /* renormalise */
        { float mx = pm[0]; for (s = 1; s < NST; s++) if (pm[s] > mx) mx = pm[s];
          for (s = 0; s < NST; s++) pm[s] -= mx; }
    }
    /* traceback from state 0 (terminated) */
    s = 0;
    for (j = steps - 1; j >= 0; j--) {
        unsigned long w = s < 32 ? dec[2 * j] : dec[2 * j + 1];
        int b = (int)((w >> (s < 32 ? s : s - 32)) & 1ul);
        int u = s >> 5;
        if (j < n_info) info[j] = (unsigned char)u;
        s = ((s & 31) << 1) | b;
    }
    return 0;
}
