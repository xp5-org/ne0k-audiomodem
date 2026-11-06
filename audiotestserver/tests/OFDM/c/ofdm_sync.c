/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

#include "ofdm_sync.h"
#include <math.h>

struct ofdm_sync {
    ofdm_cfg c;
    int      L;          /* half-symbol correlation lag, n_fft/2 */
    double  *ref;        /* symbol_samples, mean-removed, unit norm */
    double  *ring;       /* symbol_samples metric history          */
    int      ring_n;
};

static size_t al8(size_t v) { return (v + 7u) & ~(size_t)7u; }

/* Must stay the term-for-term sum of the bumps ofdm_sync_init performs. */
size_t ofdm_sync_workspace(const ofdm_cfg *c)
{
    if (!c || c->symbol_samples < 1 || c->n_fft < 4) return 0;
    return al8(sizeof(struct ofdm_sync))
         + al8((size_t)c->symbol_samples * sizeof(double))   /* ref  */
         + al8((size_t)c->symbol_samples * sizeof(double));  /* ring */
}

int ofdm_sync_init(ofdm_sync **sp, void *mem, size_t memlen,
                   const ofdm_cfg *c, const double *sc_ref)
{
    struct ofdm_sync *s;
    unsigned char *p = (unsigned char *)mem;
    size_t need;
    int    i, SYM;
    double mean = 0.0, nrm = 0.0;

    if (!sp || !mem || !c || !sc_ref) return -1;
    need = ofdm_sync_workspace(c);
    if (need == 0) return -1;
    if (memlen < need) return -2;
    SYM = c->symbol_samples;

    s = (struct ofdm_sync *)p; p += al8(sizeof(struct ofdm_sync));
    s->ref  = (double *)p;     p += al8((size_t)SYM * sizeof(double));
    s->ring = (double *)p;
    s->c      = *c;
    s->L      = c->n_fft / 2;
    s->ring_n = SYM;

    /* Mean-removed and unit-norm, so the correlation below is a normalised
     * cross-correlation in [0,1] by Cauchy-Schwarz and therefore comparable
     * across capture levels. */
    for (i = 0; i < SYM; i++) mean += sc_ref[i];
    mean /= (double)SYM;
    for (i = 0; i < SYM; i++) {
        s->ref[i] = sc_ref[i] - mean;
        nrm += s->ref[i] * s->ref[i];
    }
    nrm = sqrt(nrm);
    if (nrm > 0.0)
        for (i = 0; i < SYM; i++) s->ref[i] /= nrm;
    for (i = 0; i < SYM; i++) s->ring[i] = 0.0;

    *sp = s;
    return 0;
}

int ofdm_sync_refine(ofdm_sync *s, const double *rx, int n, int d,
                     int *best, double *score)
{
    int    SYM, lo, hi, i, j, bi;
    double bs;

    if (!s || !rx || d < 0) return -1;
    SYM = s->c.symbol_samples;
    if (n < SYM) { if (best) *best = d; if (score) *score = 1.0; return 0; }

    lo = d - s->c.cp_len;
    if (lo < 0) lo = 0;
    hi = d + s->c.cp_len;
    if (hi > n - SYM) hi = n - SYM;
    if (hi <= lo) { if (best) *best = d; if (score) *score = 1.0; return 0; }

    bi = d;
    bs = -1.0;
    for (i = lo; i <= hi; i++) {
        double num = 0.0, eng = 0.0, v;
        for (j = 0; j < SYM; j++) {
            num += rx[i + j] * s->ref[j];
            eng += rx[i + j] * rx[i + j];
        }
        /* |correlation| / sqrt(energy): the reference takes the absolute value,
         * so a preamble arriving inverted (a polarity flip anywhere in the
         * analogue path) still scores as a match. */
        v = (num < 0.0 ? -num : num) / (sqrt(eng) + 1e-12);
        if (v > bs) { bs = v; bi = i; }
    }
    if (best)  *best  = bi;
    if (score) *score = bs;
    return 0;
}

/* Schmidl-Cox metric at one position, from running sums.
 *
 *     M[d] = P[d]^2 / (R1[d] * R2[d])
 *
 * with P the correlation of the two half-symbol windows and R1/R2 their
 * energies.  By Cauchy-Schwarz M <= 1, so a fixed threshold is meaningful.
 *
 * R1*R2 in the denominator, NOT R2^2: the older form let a DECAYING transient
 * (loud first half, quiet second) drive M far above 1 -- an audio-stream
 * startup click scored ~317 and swamped the real preamble. */
static double sc_metric_at(double P, double R1, double R2)
{
    double den = R1 * R2;
    if (den <= 1e-18) return 0.0;
    return P * P / (den + 1e-24);
}

int ofdm_sync_find(ofdm_sync *s, const double *rx, int n, int n_body_syms,
                   double threshold, double ncc_min,
                   int *starts, int max_starts)
{
    int    SYM, L, md, d, cnt = 0, min_gap;
    int    last, in_region, region_d0, region_end;
    double P = 0.0, R1 = 0.0, R2 = 0.0, region_max = 0.0;
    double last_sc = 0.0, last_rmax = 0.0;
    int    i;

    if (!s || !rx || !starts || max_starts < 1 || n_body_syms < 1) return -1;
    SYM = s->c.symbol_samples;
    L   = s->L;
    if (n < 2 * L + 1) return 0;

    /* Positions d for which both half-windows fit: same bound the reference's
     * cumulative-sum arithmetic produces. */
    md = n - 2 * L;
    if (md < 1) return 0;

    min_gap = ofdm_frame_min_gap(&s->c, n_body_syms);
    last = -(min_gap * 2);
    in_region = 0;
    region_d0 = 0;
    region_end = 0;

    /* Prime the windows at d = 0. */
    for (i = 0; i < L; i++) {
        P  += rx[i] * rx[i + L];
        R1 += rx[i] * rx[i];
        R2 += rx[i + L] * rx[i + L];
    }

    for (d = 0; d < md; d++) {
        double M = sc_metric_at(P, R1, R2);

        s->ring[d % s->ring_n] = M;

        if (!in_region) {
            if (M > threshold) {
                /* A rising edge.  The region is one symbol wide, matching the
                 * reference's refine window. */
                in_region  = 1;
                region_d0  = d;
                region_end = d + SYM;
                if (region_end > md) region_end = md;
                region_max = M;
            }
        } else {
            if (M > region_max) region_max = M;
        }

        if (in_region && d + 1 >= region_end) {
            /* Region complete: the accepted position is the FIRST sample at or
             * above 95% of the region max -- the plateau's leading edge, not
             * its peak, because the metric is flat across the cyclic prefix. */
            int    ref_pos = region_d0;
            int    k;
            double gate = 0.95 * region_max;
            for (k = region_d0; k < region_end; k++) {
                if (s->ring[k % s->ring_n] >= gate) { ref_pos = k; break; }
            }
            in_region = 0;

            int inside = region_d0 - last < min_gap;
            if (!inside || (cnt > 0 && region_max > last_rmax)) {
                int    bi = ref_pos;
                double sc = 1.0;
                ofdm_sync_refine(s, rx, n, ref_pos, &bi, &sc);
                /* A candidate that does not look like a preamble is skipped
                 * AND DOES NOT ADVANCE `last`.  That is the whole point: the
                 * delay-correlator scores high on any decaying transient, so a
                 * startup click that consumed the min_gap slot would suppress
                 * the real first frame behind it. */
                if (sc >= ncc_min) {
                    if (!inside) {
                        if (cnt < max_starts) starts[cnt] = bi;
                        cnt++;
                        last = ref_pos; last_sc = sc; last_rmax = region_max;
                    } else if (sc >= OFDM_SC_REPLACE_RATIO * last_sc) {
                        /* Junk that passed ncc_min (the keyup tone) took this
                         * slot just ahead of a real preamble: replace it. */
                        if (cnt <= max_starts) starts[cnt - 1] = bi;
                        last = ref_pos; last_sc = sc; last_rmax = region_max;
                    }
                }
            }
        }

        /* Slide the windows one sample. */
        if (d + 1 < md) {
            P  -= rx[d] * rx[d + L];
            R1 -= rx[d] * rx[d];
            R2 -= rx[d + L] * rx[d + L];
            P  += rx[d + L] * rx[d + 2 * L];
            R1 += rx[d + L] * rx[d + L];
            R2 += rx[d + 2 * L] * rx[d + 2 * L];
        }
    }
    return cnt > max_starts ? max_starts : cnt;
}
