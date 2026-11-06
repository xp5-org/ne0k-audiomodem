/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_fdd.c -- see ofdm_fdd.h. */
#include <math.h>
#include "ofdm_fdd.h"

static int ceil_div(int a, int b) { return (a + b - 1) / b; }

int ofdm_fdd_min_occupied(double min_baud, double symbol_rate, int pilot_spacing)
{
    int need, n;
    if (min_baud <= 0.0 || symbol_rate <= 0.0 || pilot_spacing < 1)
        return 1;
    need = (int)ceil(min_baud / symbol_rate);
    n = need;
    /* range(0, n, spacing) has ceil(n/spacing) pilots. */
    while (n - ceil_div(n, pilot_spacing) < need)
        n++;
    return n;
}

int ofdm_fdd_plan_bands(ofdm_fdd_plan *p, double freq_lo, double freq_hi,
                        double dl_frac, double guard_hz, int n_fft,
                        double cp_ratio, int sample_rate, int pilot_spacing,
                        double min_baud, int dl_side)
{
    double sc_spc, symbol_rate, span, usable, min_bw, dl_bw, ul_bw;
    double lo_b0, lo_b1, hi_b0, hi_b1;
    int cp_len, min_occ;

    if (!p || n_fft < 4 || sample_rate <= 0 || pilot_spacing < 1 || freq_hi <= freq_lo)
        return -1;
    sc_spc = (double)sample_rate / n_fft;
    cp_len = (int)(n_fft * cp_ratio + 0.5);
    if (cp_len < 1) cp_len = 1;
    symbol_rate = (double)sample_rate / (n_fft + cp_len);

    span = freq_hi - freq_lo;
    usable = span - guard_hz;
    if (usable <= 0.0)
        return -2;

    min_occ = ofdm_fdd_min_occupied(min_baud, symbol_rate, pilot_spacing);
    min_bw = min_occ * sc_spc;
    if (2.0 * min_bw > usable)
        return -3;

    dl_bw = dl_frac * usable;
    if (dl_bw > usable - min_bw) dl_bw = usable - min_bw;
    if (dl_bw < min_bw) dl_bw = min_bw;
    ul_bw = usable - dl_bw;

    if (dl_side == OFDM_FDD_SIDE_LOW) {
        p->dl_lo = freq_lo;               p->dl_hi = freq_lo + dl_bw;
        p->ul_lo = freq_lo + dl_bw + guard_hz;  p->ul_hi = freq_hi;
    } else {
        p->ul_lo = freq_lo;               p->ul_hi = freq_lo + ul_bw;
        p->dl_lo = freq_lo + ul_bw + guard_hz;  p->dl_hi = freq_hi;
    }

    /* Splatter travels upward: does the lower band's 2nd/3rd harmonic land in
     * the upper one?  The guard only helps against the skirt, not a harmonic. */
    if (p->ul_lo < p->dl_lo) {
        lo_b0 = p->ul_lo; lo_b1 = p->ul_hi; hi_b0 = p->dl_lo; hi_b1 = p->dl_hi;
    } else {
        lo_b0 = p->dl_lo; lo_b1 = p->dl_hi; hi_b0 = p->ul_lo; hi_b1 = p->ul_hi;
    }
    p->harm2_overlap = (2.0 * lo_b0 < hi_b1 && 2.0 * lo_b1 > hi_b0);
    p->harm3_overlap = (3.0 * lo_b0 < hi_b1 && 3.0 * lo_b1 > hi_b0);

    p->guard_lo = p->dl_hi < p->ul_hi ? p->dl_hi : p->ul_hi;
    p->guard_hi = p->dl_lo > p->ul_lo ? p->dl_lo : p->ul_lo;
    p->dl_frac_actual = dl_bw / usable;
    p->min_bw_hz = min_bw;
    p->symbol_rate_hz = symbol_rate;
    p->carrier_spacing_hz = sc_spc;
    p->ratio_clamped = fabs(dl_bw - dl_frac * usable) > 0.5 * sc_spc;
    p->within_one_octave = freq_hi < 2.0 * freq_lo;
    return 0;
}

int ofdm_fdd_chunk_for_frame_len(const ofdm_cfg *c, int bits_per_ofdm_sym,
                                 double max_frame_s)
{
    static const int chunks[] = { 256, 192, 128, 96, 64, 48, 32, 24, 16 };
    int i, over, sp, n_syms;
    if (!c || c->sample_rate <= 0)
        return 16;
    if (bits_per_ofdm_sym < 1) bits_per_ofdm_sym = 1;
    sp = c->sync_period < 1 ? 1 : c->sync_period;
    over = c->seq_bytes + c->ecc_bytes;
    for (i = 0; i < (int)(sizeof chunks / sizeof chunks[0]); i++) {
        n_syms = ceil_div((chunks[i] + over) * 8, bits_per_ofdm_sym);
        n_syms += 2 + ceil_div(n_syms, sp);
        if ((double)n_syms * c->symbol_samples / c->sample_rate <= max_frame_s)
            return chunks[i];
    }
    return 16;
}

int ofdm_fdd_sub_bat(const ofdm_cfg *full, const int *full_bat,
                     const ofdm_cfg *sub, int *out)
{
    int i, b, k, n;
    if (!full || !full_bat || !sub || !out)
        return -1;
    n = 0;
    for (i = 0; i < sub->n_occupied; i++) {
        if (i % sub->pilot_spacing == 0)
            continue;                              /* a pilot in the sub-band */
        b = sub->bin_lo + i;
        out[n] = 1;
        k = b - full->bin_lo;
        if (k >= 0 && k < full->n_occupied && k % full->pilot_spacing != 0) {
            /* BAT index = occupied index minus the pilots before it
             * (indices 0, ps, 2ps ... below k: ceil(k/ps) of them). */
            int idx = k - ceil_div(k, full->pilot_spacing);
            if (idx >= 0 && idx < full->n_data_carriers)
                out[n] = full_bat[idx];
        }
        n++;
    }
    return n == sub->n_data_carriers ? 0 : -1;
}
