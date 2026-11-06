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

#include "ofdm_dem.h"
#include <math.h>

/* Guards copied from the reference, and each one means something different:
 *   DEM_H_EPS   a channel estimate this small is a dead carrier -- equalising
 *               by it would amplify noise without bound, so pass Y through.
 *   DEM_KNOWN_EPS  the reference writes `ce_known[i] or 1e-12`, i.e. substitute
 *               a tiny divisor rather than divide by zero.  The BPSK reference
 *               sequences are +-1/sqrt(2) so this never fires; kept because
 *               dropping it would be a silent behaviour change.
 *   DEM_REF_EPS the +1e-20 under the EVM ratio, so an all-zero reference gives
 *               0 rather than NaN.
 * These are ABSOLUTE and inherited from a float64 reference.  A float32 build
 * runs them ~7 orders above its own resolution, which is why they still behave;
 * if this core is ever cut to a smaller type, re-derive them from the dtype
 * rather than carrying the literals across. */
#define DEM_H_EPS      1e-9
#define DEM_KNOWN_EPS  1e-12
#define DEM_REF_EPS    1e-20

struct ofdm_dem {
    int n_fft, cp_len, bin_lo, n_occupied, sym;
    int n_pilots, n_data_carriers, data_syms, sync_period, payload_bytes;
    int n_syncs, total_body, n_tabs;
    unsigned prbs_seed;
    ofdm_real pilot_re, pilot_im;
    ofdm_fft   fft;
    ofdm_cpx  *ce;        /* n_occupied                              */
    ofdm_cpx  *sync;      /* n_occupied                              */
    ofdm_cpx  *H;         /* n_occupied  (live channel estimate)     */
    ofdm_real *win;       /* n_fft       scratch                     */
    ofdm_cpx  *spec;      /* n_fft/2+1   scratch                     */
    unsigned char *pilot_mask;  /* n_occupied                        */
    int       *bpb;       /* n_data_carriers                         */
    int       *pilots;    /* n_pilots                                */
    /* constellation tables, flat */
    int       *tb;        /* n_tabs: bits per carrier                */
    int       *tn;        /* n_tabs: point count                     */
    int       *toff;      /* n_tabs: index of first point            */
    int       *tboff;     /* n_tabs: index of first bit              */
    ofdm_cpx  *pts;       /* n_pts_total                             */
    unsigned char *tbits; /* n_bits_total                            */
    unsigned char *bits;  /* payload_bytes*8 scratch                 */
};

static size_t al8(size_t v) { return (v + 7u) & ~(size_t)7u; }

static int plan_ok(int n_fft, int cp_len, int n_occupied, int n_pilots,
                   int n_data_carriers, int data_syms, int sync_period,
                   int payload_bytes, int n_pts_total, int n_bits_total)
{
    if (n_fft < 4 || (n_fft & (n_fft - 1)) != 0) return 0;
    if (cp_len < 0 || n_occupied < 1) return 0;
    if (n_occupied > n_fft / 2 + 1) return 0;
    if (n_pilots < 0 || n_pilots > n_occupied) return 0;
    if (n_data_carriers < 1 || n_data_carriers > n_occupied) return 0;
    if (data_syms < 0 || sync_period < 0) return 0;
    if (payload_bytes < 1) return 0;
    if (n_pts_total < 1 || n_bits_total < 1) return 0;
    return 1;
}

/* Must stay the term-for-term sum of the bumps ofdm_dem_init performs, in the
 * same order.  Edit the two together. */
size_t ofdm_dem_workspace(int n_fft, int cp_len, int n_occupied,
                          int n_pilots, int n_data_carriers,
                          int data_syms, int sync_period,
                          int payload_bytes, int n_pts_total,
                          int n_bits_total)
{
    size_t fftw;
    if (!plan_ok(n_fft, cp_len, n_occupied, n_pilots, n_data_carriers,
                 data_syms, sync_period, payload_bytes,
                 n_pts_total, n_bits_total)) return 0;
    fftw = ofdm_fft_workspace(n_fft);
    if (fftw == 0) return 0;

    return al8(sizeof(struct ofdm_dem))
         + fftw
         + al8((size_t)n_occupied      * sizeof(ofdm_cpx))   /* ce     */
         + al8((size_t)n_occupied      * sizeof(ofdm_cpx))   /* sync   */
         + al8((size_t)n_occupied      * sizeof(ofdm_cpx))   /* H      */
         + al8((size_t)n_fft           * sizeof(ofdm_real))  /* win    */
         + al8((size_t)(n_fft / 2 + 1) * sizeof(ofdm_cpx))   /* spec   */
         + al8((size_t)n_occupied      * sizeof(unsigned char)) /* mask */
         + al8((size_t)n_data_carriers * sizeof(int))        /* bpb    */
         + al8((size_t)(n_pilots > 0 ? n_pilots : 1) * sizeof(int))
         + al8((size_t)4 * 4 * sizeof(int))                  /* tb/tn/toff/tboff */
         + al8((size_t)n_pts_total     * sizeof(ofdm_cpx))   /* pts    */
         + al8((size_t)n_bits_total    * sizeof(unsigned char))
         + al8((size_t)payload_bytes * 8 * sizeof(unsigned char));
}

int ofdm_dem_init(ofdm_dem **dp, void *mem, size_t memlen,
                  int n_fft, int cp_len, int bin_lo, int n_occupied,
                  const double *ce_re, const double *ce_im,
                  const double *sy_re, const double *sy_im,
                  const int *pilot_offsets, int n_pilots,
                  double pilot_re, double pilot_im,
                  const int *bits_per_bin, int n_data_carriers,
                  int data_syms, int sync_period, int payload_bytes,
                  unsigned prbs_seed,
                  const int *tab_b, const int *tab_n, int n_tabs,
                  const double *pts_re, const double *pts_im,
                  const unsigned char *tab_bits)
{
    struct ofdm_dem *d;
    unsigned char *p = (unsigned char *)mem;
    size_t need, fftw;
    int i, t, n_pts_total = 0, n_bits_total = 0, rc;

    if (!dp || !mem || !ce_re || !ce_im || !sy_re || !sy_im) return -1;
    if (!bits_per_bin || !tab_b || !tab_n || !pts_re || !pts_im || !tab_bits)
        return -1;
    if (n_tabs < 1 || n_tabs > 4) return -1;
    if (bin_lo < 0 || bin_lo + n_occupied > n_fft / 2 + 1) return -1;
    for (t = 0; t < n_tabs; t++) {
        if (tab_b[t] < 1 || tab_n[t] < 1) return -1;
        n_pts_total  += tab_n[t];
        n_bits_total += tab_n[t] * tab_b[t];
    }
    if (!plan_ok(n_fft, cp_len, n_occupied, n_pilots, n_data_carriers,
                 data_syms, sync_period, payload_bytes,
                 n_pts_total, n_bits_total)) return -1;

    /* Every non-zero bits_per_bin must have a table, or a carrier would be
     * silently skipped mid-frame and every later bit would shift. */
    for (i = 0; i < n_data_carriers; i++) {
        int b = bits_per_bin[i], found = 0;
        if (b == 0) continue;
        for (t = 0; t < n_tabs; t++) if (tab_b[t] == b) { found = 1; break; }
        if (!found) return -4;
    }

    need = ofdm_dem_workspace(n_fft, cp_len, n_occupied, n_pilots,
                              n_data_carriers, data_syms, sync_period,
                              payload_bytes, n_pts_total, n_bits_total);
    if (need == 0 || memlen < need) return -2;
    fftw = ofdm_fft_workspace(n_fft);

    d = (struct ofdm_dem *)p; p += al8(sizeof(struct ofdm_dem));
    rc = ofdm_fft_init(&d->fft, n_fft, p, fftw);
    if (rc != 0) return -3;
    p += fftw;

    d->ce   = (ofdm_cpx *)p; p += al8((size_t)n_occupied * sizeof(ofdm_cpx));
    d->sync = (ofdm_cpx *)p; p += al8((size_t)n_occupied * sizeof(ofdm_cpx));
    d->H    = (ofdm_cpx *)p; p += al8((size_t)n_occupied * sizeof(ofdm_cpx));
    d->win  = (ofdm_real *)p; p += al8((size_t)n_fft * sizeof(ofdm_real));
    d->spec = (ofdm_cpx *)p; p += al8((size_t)(n_fft/2+1) * sizeof(ofdm_cpx));
    d->pilot_mask = p;        p += al8((size_t)n_occupied * sizeof(unsigned char));
    d->bpb  = (int *)p;       p += al8((size_t)n_data_carriers * sizeof(int));
    d->pilots = (int *)p;     p += al8((size_t)(n_pilots > 0 ? n_pilots : 1)
                                       * sizeof(int));
    d->tb   = (int *)p;       p += al8((size_t)4 * 4 * sizeof(int));
    d->tn   = d->tb + 4;
    d->toff = d->tb + 8;
    d->tboff= d->tb + 12;
    d->pts  = (ofdm_cpx *)p;  p += al8((size_t)n_pts_total * sizeof(ofdm_cpx));
    d->tbits= p;              p += al8((size_t)n_bits_total * sizeof(unsigned char));
    d->bits = p;

    d->n_fft = n_fft; d->cp_len = cp_len; d->bin_lo = bin_lo;
    d->n_occupied = n_occupied; d->sym = n_fft + cp_len;
    d->n_pilots = n_pilots; d->n_data_carriers = n_data_carriers;
    d->data_syms = data_syms; d->sync_period = sync_period;
    d->payload_bytes = payload_bytes; d->prbs_seed = prbs_seed;
    d->pilot_re = (ofdm_real)pilot_re; d->pilot_im = (ofdm_real)pilot_im;
    d->n_tabs = n_tabs;

    /* SYNC count is ceil(data/period), NOT an integer divide -- a SYNC is
     * inserted BEFORE every period-th data symbol, so 5 data at period 32 is
     * still one SYNC, and truncating here would misalign the whole body. */
    d->n_syncs = (sync_period > 0 && data_syms > 0)
               ? (data_syms + sync_period - 1) / sync_period : 0;
    d->total_body = 1 + data_syms + d->n_syncs;

    for (i = 0; i < n_occupied; i++) {
        d->ce[i].re   = (ofdm_real)ce_re[i];
        d->ce[i].im   = (ofdm_real)ce_im[i];
        d->sync[i].re = (ofdm_real)sy_re[i];
        d->sync[i].im = (ofdm_real)sy_im[i];
        d->pilot_mask[i] = 0;
    }
    for (i = 0; i < n_pilots; i++) {
        int off = pilot_offsets[i];
        if (off < 0 || off >= n_occupied) return -1;
        d->pilots[i] = off;
        d->pilot_mask[off] = 1;
    }
    for (i = 0; i < n_data_carriers; i++) d->bpb[i] = bits_per_bin[i];

    {
        int po = 0, bo = 0;
        for (t = 0; t < n_tabs; t++) {
            d->tb[t] = tab_b[t]; d->tn[t] = tab_n[t];
            d->toff[t] = po; d->tboff[t] = bo;
            po += tab_n[t]; bo += tab_n[t] * tab_b[t];
        }
        for (i = 0; i < n_pts_total; i++) {
            d->pts[i].re = (ofdm_real)pts_re[i];
            d->pts[i].im = (ofdm_real)pts_im[i];
        }
        for (i = 0; i < n_bits_total; i++) d->tbits[i] = tab_bits[i] ? 1 : 0;
    }

    *dp = d;
    return 0;
}

int ofdm_dem_total_body_syms(const ofdm_dem *d) { return d ? d->total_body : 0; }

/* numpy's complex divide (Smith): keeps the rounding identical to the
 * reference rather than multiplying by a reciprocal. */
static void cdiv(ofdm_real ar, ofdm_real ai, ofdm_real br, ofdm_real bi,
                 ofdm_real *qr, ofdm_real *qi)
{
    ofdm_real r, den;
    if (fabs((double)br) >= fabs((double)bi)) {
        r = bi / br; den = br + bi * r;
        *qr = (ar + ai * r) / den;
        *qi = (ai - ar * r) / den;
    } else {
        r = br / bi; den = br * r + bi;
        *qr = (ar * r + ai) / den;
        *qi = (ai * r - ar) / den;
    }
}

/* One symbol's half spectrum into d->spec. */
static void sym_fft(ofdm_dem *d, const double *rx, int pos)
{
    int i;
    const double *src = rx + pos + d->cp_len;
    for (i = 0; i < d->n_fft; i++) d->win[i] = (ofdm_real)src[i];
    ofdm_rfft(&d->fft, d->win, d->spec);
}

/* H[i] = (X[bin_lo+i]/sqrt(N)) / known[i]
 *
 * `blend` averages the new estimate with the one already in d->H instead of
 * replacing it.  Used for exactly one symbol per frame -- see the first-SYNC
 * case in ofdm_dem_frame.  Done in place, per carrier, so it costs no extra
 * workspace; the two ofdm_dem_workspace/ofdm_dem_init term lists are unchanged
 * and stay term-for-term. */
static void estimate_h(ofdm_dem *d, const ofdm_cpx *known, int blend)
{
    int i;
    ofdm_real scale = (ofdm_real)(1.0 / sqrt((double)d->n_fft));
    for (i = 0; i < d->n_occupied; i++) {
        ofdm_cpx X = d->spec[d->bin_lo + i];
        ofdm_real kr = known[i].re, ki = known[i].im;
        ofdm_real nr, ni;
        if (fabs((double)kr) < DEM_KNOWN_EPS &&
            fabs((double)ki) < DEM_KNOWN_EPS) kr = (ofdm_real)DEM_KNOWN_EPS;
        cdiv(X.re * scale, X.im * scale, kr, ki, &nr, &ni);
        if (blend) {
            d->H[i].re = (ofdm_real)0.5 * (d->H[i].re + nr);
            d->H[i].im = (ofdm_real)0.5 * (d->H[i].im + ni);
        } else {
            d->H[i].re = nr;
            d->H[i].im = ni;
        }
    }
}

/* Pilot residual for one pilot: r = (X/sqrt(N)) / (H * pilot), plus its
 * weight |H|^2.  Returns 0 if the pilot is unusable. */
static int pilot_resid(ofdm_dem *d, int pi, double *rr, double *ri, double *w)
{
    ofdm_cpx  Hk = d->H[pi];
    ofdm_cpx  X  = d->spec[d->bin_lo + pi];
    ofdm_real scale = (ofdm_real)(1.0 / sqrt((double)d->n_fft));
    ofdm_real dr, di, qr, qi;

    *w = (double)Hk.re * (double)Hk.re + (double)Hk.im * (double)Hk.im;
    if (*w < 1e-18) return 0;
    if (ofdm_hypot(Hk.re, Hk.im) < DEM_H_EPS) return 0;
    dr = Hk.re * d->pilot_re - Hk.im * d->pilot_im;   /* H * pilot_sym */
    di = Hk.re * d->pilot_im + Hk.im * d->pilot_re;
    cdiv(X.re * scale, X.im * scale, dr, di, &qr, &qi);
    *rr = (double)qr; *ri = (double)qi;
    return 1;
}

/* Pilot phase residual as intercept + slope: phase(p) = a + b*p.
 *
 * WHY A SLOPE AND NOT JUST A CONSTANT.  Two sound cards never share a sample
 * clock, and consumer parts sit tens of ppm apart.  That offset is not a
 * carrier rotation a scalar can absorb -- it is a TIMING drift, and a timing
 * error of n samples is a phase 2*pi*n*k/N, LINEAR IN CARRIER INDEX.  It also
 * grows across the frame.  At 50 ppm over a 33-symbol body that is ~1.7
 * samples, ~1.8 rad across a 400-8000 Hz carrier set.
 *
 * Measured over the simulated rooms (48 kHz, 14 dB, 6 seeds), frames decoded
 * on the mild `desk` room at 30 ppm: scalar 10/36, slope 18/36, and EVM 34.9%
 * against 30.9%.  On a clean channel at 60 ppm the scalar loses every frame
 * and the slope loses none.
 *
 * The slope comes from DIFFERENTIAL phase between adjacent pilots -- the angle
 * of sum(w_j*w_j-1 * r_j * conj(r_j-1)) -- not from unwrapping and regressing.
 * Residual angles wrap at +-pi and an unwrapper is precisely the thing that
 * fails silently under noise; the differential form has no wrap to undo while
 * the step BETWEEN ADJACENT PILOTS stays under pi, which at spacing 8 and
 * n_fft 1024 is 64 samples of timing error -- far beyond what the cyclic
 * prefix tolerates anyway, so the range is not the binding constraint.
 *
 * Weights are |H|^2 throughout.  Frequency-selective fading drops pilots into
 * nulls where the residual angle is near-uniform noise; at unit weight those
 * vote as loudly as a pilot on a strong carrier and drag the correction
 * applied to every data carrier in the symbol.  |H|^2 is not a tuned weight:
 * the residual's angle variance goes as 1/(|H|^2 * SNR), so inverse-variance
 * weighting IS |H|^2 weighting.
 *
 * Three passes over the pilots, each recomputing the residual rather than
 * buffering it, so the workspace term list does not move.  n_pilots is ~21 and
 * this runs once per data symbol, against a whole 64-QAM demap per carrier --
 * it does not show up in the profile. */
static void estimate_cpe_ramp(ofdm_dem *d, ofdm_real *a_out, ofdm_real *b_out)
{
    int j, prev_p = -1, gap = 0;
    double rr, ri, w, prr = 0.0, pri = 0.0, pw = 0.0;
    double accr = 0.0, acci = 0.0, ssum = 0.0, csum = 0.0, b = 0.0;

    *a_out = (ofdm_real)0; *b_out = (ofdm_real)0;

    /* pass 1: smallest gap between consecutive USABLE pilots */
    for (j = 0; j < d->n_pilots; j++) {
        int pi = d->pilots[j];
        if (!pilot_resid(d, pi, &rr, &ri, &w)) continue;
        if (prev_p >= 0 && (gap == 0 || pi - prev_p < gap)) gap = pi - prev_p;
        prev_p = pi;
    }

    /* pass 2: slope, from adjacent pairs at THAT gap only.  Restricting to one
     * gap keeps every term the same angle-per-carrier so they can be summed as
     * complex numbers; mixing gaps would need each term divided by its own gap
     * first, reintroducing the wrap this form exists to avoid. */
    if (gap > 0) {
        prev_p = -1;
        for (j = 0; j < d->n_pilots; j++) {
            int pi = d->pilots[j];
            if (!pilot_resid(d, pi, &rr, &ri, &w)) continue;
            if (prev_p >= 0 && pi - prev_p == gap) {
                double t = w * pw;                  /* r_j * conj(r_j-1) */
                accr += t * (rr * prr + ri * pri);
                acci += t * (ri * prr - rr * pri);
            }
            prev_p = pi; prr = rr; pri = ri; pw = w;
        }
        if (accr != 0.0 || acci != 0.0)
            b = atan2(acci, accr) / (double)gap;
    }

    /* pass 3: intercept, from the slope-derotated weighted circular mean */
    for (j = 0; j < d->n_pilots; j++) {
        int pi = d->pilots[j];
        double cb, sb, dr_, di_, e;
        if (!pilot_resid(d, pi, &rr, &ri, &w)) continue;
        cb = cos(b * (double)pi); sb = -sin(b * (double)pi);
        dr_ = rr * cb - ri * sb;
        di_ = ri * cb + rr * sb;
        e = atan2(di_, dr_);
        ssum += w * sin(e); csum += w * cos(e);
    }
    if (ssum != 0.0 || csum != 0.0) *a_out = (ofdm_real)atan2(ssum, csum);
    *b_out = (ofdm_real)b;
}

int ofdm_dem_frame(ofdm_dem *d, const double *rx, int n, int frame_start,
                   unsigned char *out, int out_cap, ofdm_dem_out *info)
{
    int pos, s, i, t, nbits = 0, data_cnt = 0, body_sym = 0;
    int want_bits, nby;
    double evm_acc = 0.0;
    ofdm_real scale;
    unsigned state;

    if (!d || !rx || !out || !info || frame_start < 0) return -1;
    if (out_cap < d->payload_bytes) return -4;
    if (frame_start + d->sym * (d->total_body + 1) > n) return -2;
    scale = (ofdm_real)(1.0 / sqrt((double)d->n_fft));

    pos = frame_start + d->sym;          /* skip the SC preamble */
    sym_fft(d, rx, pos);
    estimate_h(d, d->ce, 0);
    pos += d->sym;

    want_bits = d->payload_bytes * 8;

    for (s = 0; s < d->data_syms + d->n_syncs; s++) {
        int is_sync;
        if (pos + d->cp_len + d->n_fft > n) break;
        sym_fft(d, rx, pos);
        pos += d->sym;

        /* Body layout is SYNC, data*P, SYNC, data*P, ... so a SYNC sits at
         * every (sync_period+1)-th body index, counting the SYNC itself. */
        is_sync = (d->sync_period > 0 &&
                   body_sym % (d->sync_period + 1) == 0);
        body_sym++;
        if (is_sync) {
            /* The FIRST SYNC sits immediately after the CE symbol and carries
             * the IDENTICAL known sequence (both are _SEED_CE in the
             * reference; ce[] and sync[] are handed over separately only so
             * this core does not have to know that).  Overwriting discarded a
             * whole full-band reference symbol, one symbol old, every frame.
             * Averaging halves the estimator noise for nothing.
             *
             * Later SYNCs must still REPLACE: they exist because the channel
             * has moved, so blending them with a stale estimate would smear
             * exactly what they are there to track. */
            estimate_h(d, d->sync, body_sym == 1);
            continue;
        }

        {
            /* SIGN: the pilot residual r = Y/(H*pilot) has angle +phase, so
             * removing it needs H_eff = H*e^{+j*phase}; eq = Y/H_eff then
             * divides the error out.  This was e^{-j*phase}, which DOUBLED the
             * error every symbol instead of cancelling it.
             *
             * Nothing caught it because nothing ever handed it a phase error
             * worth doubling: the simulated channel was a zero-phase magnitude
             * filter with no delay and no sample-clock offset, and 2*0 == 0.
             * It is also why the sign and the slope only pay TOGETHER -- with
             * a scalar estimator there was almost nothing to get wrong, and
             * feeding the inverted sign a good slope is worse than no slope at
             * all (2/36 frames against 18/36 on desk at 30 ppm). */
            ofdm_real cpe_a, cpe_b;
            double err_sum = 0.0, ref_sum = 0.0;
            int cnt = 0, data_i = 0;
            estimate_cpe_ramp(d, &cpe_a, &cpe_b);

            for (i = 0; i < d->n_occupied; i++) {
                ofdm_cpx X, Heff;
                ofdm_real eqr, eqi, hm, ph, cr, ci;
                int b_val, ti = -1, best = 0, k;
                double bestd = 0.0;

                if (d->pilot_mask[i]) continue;
                b_val = d->bpb[data_i];
                data_i++;
                if (b_val == 0) continue;

                for (t = 0; t < d->n_tabs; t++)
                    if (d->tb[t] == b_val) { ti = t; break; }
                if (ti < 0) continue;      /* rejected at init; belt and braces */

                /* Per-carrier phase: a + b*i.  cos/sin per carrier mirrors
                 * the reference exactly.  A target build can replace this with
                 * a complex recurrence (multiply by e^{j*b} each step) once
                 * bit-equivalence with numpy stops being the bar -- it is the
                 * same arithmetic to within accumulated rounding. */
                ph = cpe_a + cpe_b * (ofdm_real)i;
                cr = (ofdm_real)cos((double)ph);
                ci = (ofdm_real)sin((double)ph);
                X = d->spec[d->bin_lo + i];
                Heff.re = d->H[i].re * cr - d->H[i].im * ci;
                Heff.im = d->H[i].re * ci + d->H[i].im * cr;
                hm = (ofdm_real)ofdm_hypot(Heff.re, Heff.im);
                if (hm > (ofdm_real)DEM_H_EPS)
                    cdiv(X.re * scale, X.im * scale, Heff.re, Heff.im,
                         &eqr, &eqi);
                else { eqr = X.re * scale; eqi = X.im * scale; }

                /* Nearest constellation point.  The reference takes
                 * argmin(|eq - pts|**2); |.| is monotone in |.|**2 so the
                 * decision is the same, and ties break to the lowest index on
                 * both sides because the scan is ascending with a strict <. */
                for (k = 0; k < d->tn[ti]; k++) {
                    ofdm_cpx P = d->pts[d->toff[ti] + k];
                    double dr = (double)(eqr - P.re), di = (double)(eqi - P.im);
                    double dd = ofdm_hypot(dr, di);
                    dd = dd * dd;
                    if (k == 0 || dd < bestd) { bestd = dd; best = k; }
                }
                {
                    ofdm_cpx P = d->pts[d->toff[ti] + best];
                    const unsigned char *bp =
                        d->tbits + d->tboff[ti] + (size_t)best * b_val;
                    double dr = (double)(eqr - P.re), di = (double)(eqi - P.im);
                    double e  = ofdm_hypot(dr, di);
                    double rr = ofdm_hypot(P.re, P.im);
                    for (k = 0; k < b_val; k++)
                        if (nbits < want_bits) d->bits[nbits++] = bp[k];
                    err_sum += e * e;
                    ref_sum += rr * rr;
                    cnt++;
                }
            }
            if (cnt > 0)
                evm_acc += sqrt((err_sum / cnt) / ((ref_sum / cnt) + DEM_REF_EPS));
            data_cnt++;
        }
    }

    if (data_cnt == 0) return -3;

    /* PRBS descramble, then pack MSB-first -- np.packbits' order. */
    state = d->prbs_seed & 0xFFFFu;
    for (i = 0; i < nbits; i++) {
        unsigned fb = ((state >> 15) ^ (state >> 14) ^ (state >> 12)
                       ^ (state >> 3)) & 1u;
        unsigned bit = (state >> 15) & 1u;
        state = ((state << 1) | fb) & 0xFFFFu;
        d->bits[i] ^= (unsigned char)bit;
    }
    nby = nbits / 8;
    if (nby > d->payload_bytes) nby = d->payload_bytes;
    for (i = 0; i < nby; i++) {
        int k; unsigned char v = 0;
        for (k = 0; k < 8; k++) v = (unsigned char)((v << 1) | d->bits[i*8 + k]);
        out[i] = v;
    }

    info->evm       = evm_acc / data_cnt;
    info->data_syms = data_cnt;
    info->n_bytes   = nby;
    return 0;
}
