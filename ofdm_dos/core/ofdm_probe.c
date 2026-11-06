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

#include "ofdm_probe.h"
#include "ofdm_sym.h"
#include "ofdm_acq.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct ofdm_probe {
    ofdm_cfg   c;
    int        n_probe_syms;
    int        n_meas_syms;
    int        max_cap;
    int        acq_step;
    ofdm_sym  *sym;
    ofdm_fft   fft;
    ofdm_acq  *acq;
    double    *probe_known;   /* n_occupied */
    double    *ce_known;      /* n_occupied */
    double    *mk;            /* n_occupied, measurement pattern, on demand */
    ofdm_real *win;           /* n_fft   rfft input scratch  */
    ofdm_cpx  *spec;          /* half    rfft output scratch */
    double    *hr, *hi;       /* n_occupied, one H row       */
    double    *rr, *ri;       /* n_occupied, reference row   */
    /* Welford accumulators over per-symbol H. */
    double    *mre, *mim, *m2re, *m2im;
    double    *hmr, *hmi;     /* n_occupied, saved mean H    */
    double    *spl;           /* n_occupied, periodic snr_lin*/
    double    *pdb;           /* n_occupied, periodic snr_db */
    double    *srt;           /* n_occupied, sort scratch    */
    int       *strong;        /* OFDM_COH_CARRIERS indices   */
    double    *symbuf;        /* symbol_samples + 2*taper    */
    double    *metric;        /* acquisition candidates      */
    int        metric_cap;
    void      *isc;           /* isi scratch                 */
    size_t     isc_len;
};

static size_t al8(size_t v) { return (v + 7u) & ~(size_t)7u; }

static int acq_step_for(const ofdm_cfg *c)
{
    int a = c->symbol_samples, b = c->cp_len / 2, g;
    if (b < 8) b = 8;
    while (b) { g = a % b; a = b; b = g; }
    return a < 1 ? 1 : a;
}

size_t ofdm_isi_scratch(int n_fft)
{
    if (n_fft < 4) return 0;
    return al8((size_t)n_fft * sizeof(double))
         + al8((size_t)(n_fft / 2 + 1) * sizeof(ofdm_cpx));
}

/* Must stay the term-for-term sum of the bumps ofdm_probe_init performs. */
size_t ofdm_probe_workspace(const ofdm_cfg *c, int n_probe_syms,
                            int n_meas_syms, int max_capture_samples)
{
    size_t symw, acqw;
    int    half, step, n_pos, n_cand, span, stride, T;

    if (!c || n_probe_syms < 1 || n_meas_syms < 0) return 0;
    if (max_capture_samples < c->symbol_samples) return 0;
    symw = ofdm_sym_workspace(c);
    if (symw == 0) return 0;
    step = acq_step_for(c);
    acqw = ofdm_acq_workspace(c->n_fft, c->cp_len, c->n_occupied,
                              OFDM_COH_SCAN_SYMS, step, OFDM_COH_CARRIERS);
    if (acqw == 0) return 0;
    half = c->n_fft / 2 + 1;
    T    = c->taper_samples;

    stride = c->symbol_samples / step;
    span   = stride * (OFDM_COH_SCAN_SYMS - 1);
    n_pos  = (max_capture_samples - c->cp_len - c->n_fft) / step + 1;
    n_cand = n_pos - span;
    if (n_cand < 1) n_cand = 1;

    return al8(sizeof(struct ofdm_probe))
         + symw
         + ofdm_fft_workspace(c->n_fft)
         + acqw
         + al8((size_t)c->n_occupied * sizeof(double)) * 16   /* the double rows */
         + al8((size_t)c->n_fft * sizeof(ofdm_real))          /* win    */
         + al8((size_t)half * sizeof(ofdm_cpx))               /* spec   */
         + al8((size_t)OFDM_COH_CARRIERS * sizeof(int))       /* strong */
         + al8((size_t)(c->symbol_samples + 2 * T) * sizeof(double))
         + al8((size_t)n_cand * sizeof(double))               /* metric */
         + ofdm_isi_scratch(c->n_fft);
}

int ofdm_probe_init(ofdm_probe **pp, void *mem, size_t memlen,
                    const ofdm_cfg *c, int n_probe_syms, int n_meas_syms,
                    int max_capture_samples)
{
    struct ofdm_probe *p;
    unsigned char *q = (unsigned char *)mem;
    size_t need, symw, fftw, acqw, row;
    int    rc, half, T, step, stride, span, n_pos, n_cand;

    if (!pp || !mem || !c) return -1;
    need = ofdm_probe_workspace(c, n_probe_syms, n_meas_syms,
                                max_capture_samples);
    if (need == 0) return -1;
    if (memlen < need) return -2;

    symw = ofdm_sym_workspace(c);
    fftw = ofdm_fft_workspace(c->n_fft);
    step = acq_step_for(c);
    acqw = ofdm_acq_workspace(c->n_fft, c->cp_len, c->n_occupied,
                              OFDM_COH_SCAN_SYMS, step, OFDM_COH_CARRIERS);
    half = c->n_fft / 2 + 1;
    T    = c->taper_samples;
    row  = al8((size_t)c->n_occupied * sizeof(double));

    stride = c->symbol_samples / step;
    span   = stride * (OFDM_COH_SCAN_SYMS - 1);
    n_pos  = (max_capture_samples - c->cp_len - c->n_fft) / step + 1;
    n_cand = n_pos - span;
    if (n_cand < 1) n_cand = 1;

    p = (struct ofdm_probe *)q; q += al8(sizeof(struct ofdm_probe));
    p->c            = *c;
    p->n_probe_syms = n_probe_syms;
    p->n_meas_syms  = n_meas_syms;
    p->max_cap      = max_capture_samples;
    p->acq_step     = step;
    p->metric_cap   = n_cand;

    rc = ofdm_sym_init(&p->sym, q, symw, c);
    if (rc != 0) return -3;
    q += symw;
    rc = ofdm_fft_init(&p->fft, c->n_fft, q, fftw);
    if (rc != 0) return -3;
    q += fftw;

    p->probe_known = (double *)q; q += row;
    p->ce_known    = (double *)q; q += row;
    p->mk          = (double *)q; q += row;
    p->hr          = (double *)q; q += row;
    p->hi          = (double *)q; q += row;
    p->rr          = (double *)q; q += row;
    p->ri          = (double *)q; q += row;
    p->mre         = (double *)q; q += row;
    p->mim         = (double *)q; q += row;
    p->m2re        = (double *)q; q += row;
    p->m2im        = (double *)q; q += row;
    p->hmr         = (double *)q; q += row;
    p->hmi         = (double *)q; q += row;
    p->spl         = (double *)q; q += row;
    p->pdb         = (double *)q; q += row;
    p->srt         = (double *)q; q += row;
    p->win    = (ofdm_real *)q; q += al8((size_t)c->n_fft * sizeof(ofdm_real));
    p->spec   = (ofdm_cpx *)q;  q += al8((size_t)half * sizeof(ofdm_cpx));
    p->strong = (int *)q;       q += al8((size_t)OFDM_COH_CARRIERS * sizeof(int));
    p->symbuf = (double *)q;    q += al8((size_t)(c->symbol_samples + 2 * T)
                                         * sizeof(double));
    p->metric = (double *)q;    q += al8((size_t)n_cand * sizeof(double));
    /* Advance past the ISI scratch before carving the acquisition workspace.
     * These two overlapped in the first version, so the first estimate()'s
     * isi_ratio call quietly demolished the acquisition plan and every
     * subsequent call failed to acquire -- one good capture, then nothing. */
    p->isc     = (void *)q;
    p->isc_len = ofdm_isi_scratch(c->n_fft);
    q += p->isc_len;

    ofdm_known_bpsk(p->probe_known, c->n_occupied, OFDM_SEED_PROBE,
                    1.0 / sqrt(2.0));
    ofdm_known_bpsk(p->ce_known, c->n_occupied, OFDM_SEED_CE, 1.0 / sqrt(2.0));

    /* The acquisition scan keys on the PERIODIC pattern with an all-ones symbol
     * key: every probe symbol is identical, which is exactly the repetition the
     * coherence metric is built to find. */
    {
        double *kr = p->hr, *ki = p->hi;
        int i;
        for (i = 0; i < c->n_occupied; i++) {
            kr[i] = p->probe_known[i];
            ki[i] = 0.0;
        }
        rc = ofdm_acq_init(&p->acq, q, acqw, kr, ki, 0,
                           c->n_fft, c->cp_len, c->bin_lo, c->n_occupied,
                           OFDM_COH_SCAN_SYMS, step, OFDM_COH_CARRIERS);
        if (rc != 0) return -3;
    }
    *pp = p;
    return 0;
}

long ofdm_probe_build_samples(const ofdm_probe *p)
{
    int nsym;
    if (!p) return -1;
    nsym = 1 + p->n_probe_syms + p->n_meas_syms;   /* CE + periodic + meas */
    return (long)p->c.symbol_samples
         + (long)nsym * p->c.symbol_samples
         + (p->c.taper_samples > 0 ? 2L * p->c.taper_samples : 0L);
}

long ofdm_probe_build(ofdm_probe *p, double *out, long cap)
{
    const ofdm_cfg *c;
    long need;
    int  i, j, si = 0, len;
    double *occ_re, *occ_im;

    if (!p || !out) return -1;
    c = &p->c;
    need = ofdm_probe_build_samples(p);
    if (cap < need) return -2;

    occ_re = p->hr;
    occ_im = p->hi;
    for (i = 0; i < c->n_occupied; i++) occ_im[i] = 0.0;

    /* SC preamble, untapered and concatenated ahead of the body. */
    for (i = 0; i < c->n_occupied; i++) occ_re[i] = 0.0;
    {
        int n_even = (c->n_occupied + 1) / 2;
        ofdm_known_bpsk(p->srt, n_even, OFDM_SEED_SC, 1.0 / sqrt(2.0));
        for (i = 0; i < n_even; i++)
            if (2 * i < c->n_occupied) occ_re[2 * i] = p->srt[i];
        ofdm_sym_make(p->sym, occ_re, occ_im, 0, p->symbuf);
        for (i = 0; i < c->symbol_samples; i++) out[i] = p->symbuf[i];
    }
    for (i = 0; i < need - c->symbol_samples; i++)
        out[c->symbol_samples + i] = 0.0;

    /* CE, then the periodic block, then the non-periodic measurement block --
     * overlap-added on one stride, exactly as _build_ola does. */
    for (i = 0; i < c->n_occupied; i++) occ_re[i] = p->ce_known[i];
    len = ofdm_sym_make(p->sym, occ_re, occ_im, 1, p->symbuf);
    for (i = 0; i < len; i++)
        out[c->symbol_samples + (long)si * c->symbol_samples + i] += p->symbuf[i];
    si++;

    for (j = 0; j < p->n_probe_syms; j++) {
        for (i = 0; i < c->n_occupied; i++) occ_re[i] = p->probe_known[i];
        len = ofdm_sym_make(p->sym, occ_re, occ_im, 1, p->symbuf);
        for (i = 0; i < len; i++)
            out[c->symbol_samples + (long)si * c->symbol_samples + i] +=
                p->symbuf[i];
        si++;
    }
    for (j = 0; j < p->n_meas_syms; j++) {
        ofdm_known_bpsk(occ_re, c->n_occupied,
                        OFDM_SEED_MEAS + (unsigned)j, 1.0 / sqrt(2.0));
        len = ofdm_sym_make(p->sym, occ_re, occ_im, 1, p->symbuf);
        for (i = 0; i < len; i++)
            out[c->symbol_samples + (long)si * c->symbol_samples + i] +=
                p->symbuf[i];
        si++;
    }
    return need;
}

/* ------------------------------------------------------------------ helpers */

/* One symbol's H row: rfft the window at `start`+CP, scale, divide by the
 * known pattern.  `known` is real for every caller (BPSK +-1/sqrt(2)), so the
 * division is one real divide per carrier rather than a complex one.
 * Returns 0 ok, -1 the window runs outside rx. */
static int H_row(ofdm_probe *p, const double *rx, int n, int start,
                 const double *known, double *hr, double *hi)
{
    const ofdm_cfg *c = &p->c;
    int    i, N = c->n_fft, CP = c->cp_len;
    double scale = 1.0 / sqrt((double)N);

    if (start < 0 || start + CP + N > n) return -1;
    for (i = 0; i < N; i++) p->win[i] = (ofdm_real)rx[start + CP + i];
    ofdm_rfft(&p->fft, p->win, p->spec);
    for (i = 0; i < c->n_occupied; i++) {
        double k = known[i];
        ofdm_cpx X = p->spec[c->bin_lo + i];
        if (k == 0.0) k = 1e-12;
        hr[i] = (double)X.re * scale / k;
        hi[i] = (double)X.im * scale / k;
    }
    return 0;
}

/* Coherent energy sum_k |mean_j H_jk|^2 over `nsyms` periodic symbols. */
static double coh_energy(ofdm_probe *p, const double *rx, int n, int start,
                         int nsyms)
{
    const ofdm_cfg *c = &p->c;
    int    i, j;
    double e = 0.0;

    if (start < 0 ||
        start + (nsyms - 1) * c->symbol_samples + c->cp_len + c->n_fft > n)
        return 0.0;
    for (i = 0; i < c->n_occupied; i++) { p->mre[i] = 0.0; p->mim[i] = 0.0; }
    for (j = 0; j < nsyms; j++) {
        if (H_row(p, rx, n, start + j * c->symbol_samples, p->probe_known,
                  p->hr, p->hi) != 0) return 0.0;
        for (i = 0; i < c->n_occupied; i++) {
            p->mre[i] += p->hr[i];
            p->mim[i] += p->hi[i];
        }
    }
    for (i = 0; i < c->n_occupied; i++) {
        double a = p->mre[i] / nsyms, b = p->mim[i] / nsyms;
        e += a * a + b * b;
    }
    return e;
}

/* Linear-interpolated percentile of a[0..n), destroying a.  Matches numpy's
 * default: sort, then interpolate between the two straddling ranks. */
static double pctile(double *a, int n, double q)
{
    int i, j, lo, hi;
    double pos, frac;
    for (i = 1; i < n; i++) {                    /* insertion sort: n is small */
        double v = a[i];
        for (j = i - 1; j >= 0 && a[j] > v; j--) a[j + 1] = a[j];
        a[j + 1] = v;
    }
    if (n < 1) return 0.0;
    if (n == 1) return a[0];
    pos = (q / 100.0) * (double)(n - 1);
    lo  = (int)floor(pos);
    hi  = lo + 1;
    if (hi >= n) return a[n - 1];
    frac = pos - (double)lo;
    return a[lo] * (1.0 - frac) + a[hi] * frac;
}

/* Indices of the k largest values of mag[0..n).  Order within the set does not
 * matter -- every caller averages over it. */
static void top_k_idx(const double *mag, int n, int k, int *out)
{
    int i, j, cnt = 0;
    for (i = 0; i < n && cnt < k; i++) out[cnt++] = i;
    for (; i < n; i++) {
        int worst = 0;
        for (j = 1; j < cnt; j++)
            if (mag[out[j]] < mag[out[worst]]) worst = j;
        if (mag[i] > mag[out[worst]]) out[worst] = i;
    }
}

/* Exact symbol timing for the measurement block, by matched filter against the
 * first measurement symbol.
 *
 * The periodic block does not need this: identical symbols make any sub-symbol
 * offset a pure phase ramp, which is why acquisition never resolves it.  A
 * NON-periodic block cannot tolerate that -- an offset makes the FFT window
 * straddle two different symbols and the H estimate collapses into noise.
 *
 * Two cheaper metrics were tried and both fail, instructively.  Coherent energy
 * is FLAT across the whole prefix (that is what a cyclic prefix is FOR), so its
 * argmax lands wherever noise puts it while the consequence is cliff-edged.
 * Cyclic-prefix correlation is sharp on a single-tap channel but AMBIGUOUS in
 * multipath and can lock onto an echo -- and multipath is the entire point of
 * this measurement.  Scoring on SNR would be wrong for a third reason: that is
 * selecting on the quantity being measured. */
static int align_meas(ofdm_probe *p, const double *rx, int n, int approx)
{
    const ofdm_cfg *c = &p->c;
    int    i, j, lo, hi, reflen, best = approx;
    double bs = -1.0;

    ofdm_known_bpsk(p->mk, c->n_occupied, OFDM_SEED_MEAS, 1.0 / sqrt(2.0));
    for (i = 0; i < c->n_occupied; i++) p->ri[i] = 0.0;
    /* The reference is UNTAPERED (_ofdm_td), while the transmitted symbol comes
     * from the windowed form, which prepends taper_samples for the overlap-add
     * cross-fade.  The correlation therefore peaks exactly one taper late, and
     * that offset is removed below. */
    reflen = ofdm_sym_make(p->sym, p->mk, p->ri, 0, p->symbuf);

    lo = approx - c->symbol_samples;
    if (lo < 0) lo = 0;
    hi = approx + c->symbol_samples;
    if (hi > n - reflen) hi = n - reflen;
    if (hi < lo) return approx;

    for (i = lo; i <= hi; i++) {
        double num = 0.0, eng = 0.0, v;
        for (j = 0; j < reflen; j++) {
            num += rx[i + j] * p->symbuf[j];
            eng += rx[i + j] * rx[i + j];
        }
        /* Normalised, so a loud echo cannot win on energy alone. */
        v = (num < 0.0 ? -num : num) / (sqrt(eng) + 1e-12);
        if (v > bs) { bs = v; best = i; }
    }
    return best - c->taper_samples;
}

/* ------------------------------------------------------------ Welford block */

static void acc_reset(ofdm_probe *p)
{
    int i;
    for (i = 0; i < p->c.n_occupied; i++) {
        p->mre[i] = 0.0; p->mim[i] = 0.0;
        p->m2re[i] = 0.0; p->m2im[i] = 0.0;
    }
}

/* Welford rather than sum/sum-of-squares: the variance here IS the measurement
 * (it is the additive-noise floor the SNR is divided by), and the naive
 * computational formula loses it to cancellation when |H| is large and the
 * noise small -- exactly the high-SNR carriers the allocator cares most about. */
static void acc_push(ofdm_probe *p, int cnt)
{
    int i;
    for (i = 0; i < p->c.n_occupied; i++) {
        double d = p->hr[i] - p->mre[i];
        p->mre[i] += d / (double)cnt;
        p->m2re[i] += d * (p->hr[i] - p->mre[i]);
        d = p->hi[i] - p->mim[i];
        p->mim[i] += d / (double)cnt;
        p->m2im[i] += d * (p->hi[i] - p->mim[i]);
    }
}

/* snr = max(2*|mean H|^2 / max(var_re + var_im, 1e-12), 1e-10), ddof=1. */
static void acc_snr(ofdm_probe *p, int cnt, double *snr_lin)
{
    int i;
    for (i = 0; i < p->c.n_occupied; i++) {
        double nv = (cnt > 1)
                  ? (p->m2re[i] + p->m2im[i]) / (double)(cnt - 1) : 1e-12;
        double s;
        if (nv < 1e-12) nv = 1e-12;
        s = 2.0 * (p->mre[i] * p->mre[i] + p->mim[i] * p->mim[i]) / nv;
        snr_lin[i] = s < 1e-10 ? 1e-10 : s;
    }
}

/* Walk the periodic grid from lo_g upward, accumulating at most n_probe_syms
 * symbols.  Returns the count accumulated. */
static int accumulate_periodic(ofdm_probe *p, const double *rx, int n,
                               int lo_g, int hi_g, int dly)
{
    int g, cnt = 0;
    acc_reset(p);
    for (g = lo_g; g <= hi_g && cnt < p->n_probe_syms;
         g += p->c.symbol_samples) {
        if (H_row(p, rx, n, g + dly, p->probe_known, p->hr, p->hi) != 0)
            continue;
        cnt++;
        acc_push(p, cnt);
    }
    return cnt;
}

/* ------------------------------------------------------------- ISI and BAT */

double ofdm_isi_ratio(ofdm_fft *fft, const double *H_re, const double *H_im,
                      int n_occ, int n_fft, int bin_lo, int cp_len,
                      const double *snr_lin, int taper_samples,
                      void *scratch, size_t scratch_len)
{
    double *h;
    ofdm_cpx *X;
    unsigned char *q = (unsigned char *)scratch;
    int    i, half, peak = 0, guard, pre, n_out = 0;
    double tot = 0.0, e_in = 0.0, e_out = 0.0, emax = -1.0;

    if (!fft || !H_re || !H_im || !scratch) return 0.0;
    if (n_occ < 1 || n_fft < 4) return 0.0;
    if (scratch_len < ofdm_isi_scratch(n_fft)) return 0.0;
    half = n_fft / 2 + 1;
    h = (double *)q; q += al8((size_t)n_fft * sizeof(double));
    X = (ofdm_cpx *)q;

    /* Hann-taper the occupied band before transforming.  Only n_occ of the
     * n_fft/2 bins are measured, and that hard band edge leaks a sinc skirt
     * across EVERY delay tap: untapered, a perfectly echo-free single-tap
     * channel reads beta=0.025 -- a phantom 16 dB ceiling in an anechoic room.
     * The taper costs a slightly wider main lobe, irrelevant at the CP scale,
     * and buys an essentially unbiased estimate. */
    for (i = 0; i < half; i++) { X[i].re = (ofdm_real)0; X[i].im = (ofdm_real)0; }
    for (i = 0; i < n_occ; i++) {
        double w = (n_occ > 1)
                 ? 0.5 - 0.5 * cos(2.0 * M_PI * (double)i / (double)(n_occ - 1))
                 : 1.0;
        X[bin_lo + i].re = (ofdm_real)(H_re[i] * w);
        X[bin_lo + i].im = (ofdm_real)(H_im[i] * w);
    }
    {
        /* The transform writes ofdm_real, which is float in the 32-bit build.
         * Widening DOWNWARD in place is safe because h[i] occupies bytes at
         * 8i while td[i] sits at 4i, and 8i >= 4(i+1) for every i >= 1 -- the
         * write is always above the next read. */
        ofdm_real *td = (ofdm_real *)h;
        ofdm_irfft(fft, X, td);
        for (i = n_fft - 1; i >= 0; i--) h[i] = (double)td[i];
    }
    for (i = 0; i < n_fft; i++) {
        h[i] = h[i] * h[i];
        tot += h[i];
        if (h[i] > emax) { emax = h[i]; peak = i; }
    }
    if (tot <= 0.0) return 0.0;

    /* The guard is the prefix MINUS the taper: the overlap-add puts the
     * previous symbol's falling taper over the first taper_samples of every
     * prefix, so those samples are not ISI-free. */
    guard = cp_len - taper_samples;
    if (guard < 1) guard = 1;
    /* A little energy BEFORE the peak counts as covered: a band-limited impulse
     * response rings symmetrically about its peak, and that pre-ringing is an
     * artefact of measuring only n_occ carriers, not a real pre-echo. */
    pre = (int)(0.125 * cp_len);

    for (i = 0; i < n_fft; i++) {
        int d = (i - peak) % n_fft;
        if (d < 0) d += n_fft;
        if (d < guard || d >= n_fft - pre) e_in += h[i];
        else { e_out += h[i]; n_out++; }
    }
    if (snr_lin) {
        double p_sig = 0.0, p_noise = 0.0;
        for (i = 0; i < n_occ; i++) {
            double m = H_re[i] * H_re[i] + H_im[i] * H_im[i];
            double s = snr_lin[i] < 1e-12 ? 1e-12 : snr_lin[i];
            p_sig   += m;
            p_noise += m / s;
        }
        if (p_sig > 0.0) {
            double frac = p_noise / (p_sig + p_noise);
            double e_noise;
            if (frac > 0.9) frac = 0.9;
            e_noise = tot * frac;
            e_out -= e_noise * (double)n_out / (double)n_fft;
            if (e_out < 0.0) e_out = 0.0;
        }
    }
    return e_out / (e_in < 1e-30 ? 1e-30 : e_in);
}

void ofdm_derate_snr_lin(double *snr_lin, int n, double beta)
{
    int i;
    if (!snr_lin || beta <= 0.0) return;
    for (i = 0; i < n; i++)
        snr_lin[i] = snr_lin[i] / (1.0 + beta * snr_lin[i]);
}

int ofdm_allocate(const double *snr_db_occupied, int n_occupied,
                  int pilot_spacing, double snr_margin_db, int max_bits,
                  double isi_beta, int *bits_out, int cap)
{
    static const int    lv[4] = { 6, 4, 2, 1 };
    static const double fl[4] = { OFDM_FLOOR_64QAM, OFDM_FLOOR_16QAM,
                                  OFDM_FLOOR_QPSK,  OFDM_FLOOR_BPSK };
    int    i, j, cnt = 0;
    double m_lin;

    if (!snr_db_occupied || !bits_out || n_occupied < 1 || pilot_spacing < 1)
        return -1;
    m_lin = pow(10.0, snr_margin_db / 10.0);

    for (i = 0; i < n_occupied; i++) {
        double lin, usable;
        int    assigned = 0;
        if (i % pilot_spacing == 0) continue;      /* pilot */
        if (cnt >= cap) return -1;

        /* Margin FIRST, then the ISI ceiling on what is left.  Applying both as
         * independent dB penalties double-counts: they answer the same question
         * ("how much worse is data than the probe says"), and at beta=0.25
         * subtracting a 9 dB margin from an already capped 6 dB ceiling mutes
         * every carrier.  Spending the margin first keeps its real meaning --
         * headroom against fade migration between probe and data -- while
         * leaving the ceiling intact, because the floor is not something margin
         * can buy off. */
        lin = pow(10.0, snr_db_occupied[i] / 10.0) / m_lin;
        if (isi_beta > 0.0) lin = lin / (1.0 + isi_beta * lin);
        usable = 10.0 * log10(lin < 1e-12 ? 1e-12 : lin);

        for (j = 0; j < 4; j++) {
            if (lv[j] > max_bits) continue;
            if (usable >= fl[j]) { assigned = lv[j]; break; }
        }
        bits_out[cnt++] = assigned;
    }
    return cnt;
}

/* ---------------------------------------------------------------- estimate */

int ofdm_probe_estimate(ofdm_probe *p, const double *rx, int n,
                        double threshold,
                        double *H_re, double *H_im,
                        double *snr_lin, double *snr_db,
                        double *snr_db_periodic,
                        ofdm_probe_info *info)
{
    const ofdm_cfg *c;
    ofdm_acq_info ai;
    int    i, j, N, CP, SYM, span, rc, n_cand, best, bc, cnt, kk;
    int    lo_g, hi_g, first, meas_at = -1, dly = 0, meas_cnt = 0;
    double gate, peak, floor_v, e, ebest;
    double *pdb;

    if (!p || !rx || !info || !H_re || !H_im || !snr_lin || !snr_db) return -1;
    c   = &p->c;
    N   = c->n_fft; CP = c->cp_len; SYM = c->symbol_samples;
    pdb = snr_db_periodic ? snr_db_periodic : p->pdb;

    info->valid = 0; info->error = OFDM_PROBE_OK;
    info->coherence = 0.0; info->noise_floor = 0.0;
    info->frame_start = 0; info->n_syms = 0; info->n_meas_syms = 0;
    info->has_isi_penalty = 0; info->isi_penalty_db = 0.0;
    info->has_beta = 0; info->isi_ratio = 0.0;

    gate = threshold > OFDM_COH_GATE ? threshold : OFDM_COH_GATE;
    span = (OFDM_COH_SCAN_SYMS - 1) * SYM + CP + N;
    if (n < span + SYM || n > p->max_cap) {
        info->error = OFDM_PROBE_TOO_SHORT;
        return 1;
    }

    /* -- 1. coarse scan: where is the probe?  A coherence search, NOT
     * Schmidl-Cox: SC false-locks on the audio-stream startup click (a decaying
     * transient scores ~1 on any delay-correlator) and its min_gap then blocks
     * the real preamble for a whole probe length.  Coherence separates probe
     * from noise by ~0.99 against ~0.36 on real captures, where SC managed
     * 0.3 against 0.5 -- the wrong way round. */
    rc = ofdm_acq_run(p->acq, rx, n, p->metric, p->metric_cap, &ai);
    if (rc != 0 || ai.n_cand < 1) { info->error = OFDM_PROBE_TOO_SHORT; return 1; }
    n_cand = ai.n_cand;
    bc = 0;
    peak = p->metric[0];
    for (i = 1; i < n_cand; i++)
        if (p->metric[i] > peak) { peak = p->metric[i]; bc = i; }
    best = bc * p->acq_step;
    /* Noise floor = low percentile of candidate coherence, a DIAGNOSTIC only;
     * the absolute gate is the discriminator.  A relative peak/floor test is
     * wrong here: when the probe fills most of the capture, most candidates
     * land inside it and score high, so the floor legitimately approaches the
     * peak.  (This sorts p->metric, which is finished with.) */
    floor_v = pctile(p->metric, n_cand, 25.0);
    info->coherence  = peak;
    info->noise_floor = floor_v;
    if (peak < gate) { info->error = OFDM_PROBE_SYNC_FAIL; return 1; }

    /* -- 2. fine align: maximise coherent energy within +-CP.  Exact timing is
     * not required -- the probe is a periodic excitation, so a start offset is
     * only a phase ramp and |H| is unaffected -- but it costs little. */
    {
        int b2 = best, lo_s = best - CP > 0 ? best - CP : 0, hi_s = best + CP;
        ebest = -1.0;
        for (i = lo_s; i < hi_s; i += 4) {
            e = coh_energy(p, rx, n, i, OFDM_COH_SCAN_SYMS);
            if (e > ebest) { ebest = e; b2 = i; }
        }
        best = b2;
    }

    /* -- 3. build the local reference from the winner, then keep only grid
     * symbols whose H matches it.  The coarse winner can sit anywhere inside
     * the probe (all probe symbols are identical, so coherence is flat across
     * it), so walking the grid is what drops the silence either side, the SC
     * preamble, and the CE symbol -- which carries a different sequence and so
     * scores ~0.1 here. */
    for (i = 0; i < c->n_occupied; i++) { p->rr[i] = 0.0; p->ri[i] = 0.0; }
    for (j = 0; j < OFDM_COH_SCAN_SYMS; j++) {
        if (H_row(p, rx, n, best + j * SYM, p->probe_known, p->hr, p->hi) != 0)
            break;
        for (i = 0; i < c->n_occupied; i++) {
            p->rr[i] += p->hr[i] / OFDM_COH_SCAN_SYMS;
            p->ri[i] += p->hi[i] / OFDM_COH_SCAN_SYMS;
        }
    }
    for (i = 0; i < c->n_occupied; i++)
        p->srt[i] = sqrt(p->rr[i] * p->rr[i] + p->ri[i] * p->ri[i]);
    kk = OFDM_COH_CARRIERS < c->n_occupied ? OFDM_COH_CARRIERS : c->n_occupied;
    top_k_idx(p->srt, c->n_occupied, kk, p->strong);
    for (j = 0; j < kk; j++) {
        int s = p->strong[j];
        double m = p->srt[s] + 1e-12;
        p->rr[s] /= m;
        p->ri[s] /= m;
    }

    first = best % SYM;
    lo_g = best;
    hi_g = best;
    /* Walk outward from the winner instead of scoring the whole grid: the
     * reference keeps a match array the length of the capture, and only ever
     * uses the contiguous run containing the winner. */
    for (;;) {
        int g = lo_g - SYM;
        double mr = 0.0, mi = 0.0, sc;
        if (g < first) break;
        if (H_row(p, rx, n, g, p->probe_known, p->hr, p->hi) != 0) break;
        for (j = 0; j < kk; j++) {
            int s = p->strong[j];
            double a = sqrt(p->hr[s] * p->hr[s] + p->hi[s] * p->hi[s]) + 1e-12;
            double ur = p->hr[s] / a, ui = p->hi[s] / a;
            mr += ur * p->rr[s] + ui * p->ri[s];
            mi += ui * p->rr[s] - ur * p->ri[s];
        }
        sc = sqrt(mr * mr + mi * mi) / (double)kk;
        if (sc <= OFDM_COH_MATCH) break;
        lo_g = g;
    }
    for (;;) {
        int g = hi_g + SYM;
        double mr = 0.0, mi = 0.0, sc;
        if (g >= n - CP - N) break;
        if (H_row(p, rx, n, g, p->probe_known, p->hr, p->hi) != 0) break;
        for (j = 0; j < kk; j++) {
            int s = p->strong[j];
            double a = sqrt(p->hr[s] * p->hr[s] + p->hi[s] * p->hi[s]) + 1e-12;
            double ur = p->hr[s] / a, ui = p->hi[s] / a;
            mr += ur * p->rr[s] + ui * p->ri[s];
            mi += ui * p->rr[s] - ur * p->ri[s];
        }
        sc = sqrt(mr * mr + mi * mi) / (double)kk;
        if (sc <= OFDM_COH_MATCH) break;
        hi_g = g;
    }

    cnt = accumulate_periodic(p, rx, n, lo_g, hi_g, 0);
    if (cnt < 2) { info->error = OFDM_PROBE_TOO_FEW_SYMS; return 1; }
    info->frame_start = lo_g;

    /* -- 4. put BOTH blocks on ONE prefix-aligned grid before measuring.
     * frame_start is only good to a fraction of a symbol (the periodic block
     * never needed better), and a measured 318-sample error -- past the
     * 256-sample prefix -- had the periodic estimates straddling the previous
     * symbol and reading ~9 dB LOW on a clean channel.  Comparing a misaligned
     * periodic SNR against a properly aligned measured one measures the
     * misalignment, not the channel.
     *
     * The delay MUST be measured on the measurement block: prefix correlation
     * is degenerate in the periodic block, where identical symbols make the
     * signal periodic with SYM and every offset matches equally well. */
    if (p->n_meas_syms > 0 && hi_g + SYM < n - CP - N) {
        int m0 = hi_g + SYM;
        meas_at = align_meas(p, rx, n, m0);
        dly = meas_at - m0;
        if (dly > -SYM && dly < SYM) {
            int c2 = accumulate_periodic(p, rx, n, lo_g, hi_g, dly);
            if (c2 >= 2) {
                cnt = c2;
                info->frame_start = lo_g + dly;
            } else {
                accumulate_periodic(p, rx, n, lo_g, hi_g, 0);
            }
        } else {
            dly = 0;
            meas_at = m0;
        }
    }

    info->n_syms = cnt;
    acc_snr(p, cnt, p->spl);
    for (i = 0; i < c->n_occupied; i++) {
        p->hmr[i] = p->mre[i];
        p->hmi[i] = p->mim[i];
        pdb[i] = 10.0 * log10(p->spl[i] + 1e-10);
        snr_lin[i] = p->spl[i];
        snr_db[i]  = pdb[i];
        H_re[i] = p->hmr[i];
        H_im[i] = p->hmi[i];
    }

    /* -- 5. the honest measurement: the NON-PERIODIC block, same grid and same
     * alignment, so only the excitation differs. */
    if (meas_at >= 0) {
        acc_reset(p);
        meas_cnt = 0;
        for (j = 0; j < p->n_meas_syms; j++) {
            ofdm_known_bpsk(p->mk, c->n_occupied,
                            OFDM_SEED_MEAS + (unsigned)j, 1.0 / sqrt(2.0));
            if (H_row(p, rx, n, meas_at + j * SYM, p->mk, p->hr, p->hi) != 0)
                break;
            meas_cnt++;
            acc_push(p, meas_cnt);
        }
        if (meas_cnt >= 4) {
            double med_p, med_m;
            acc_snr(p, meas_cnt, snr_lin);
            for (i = 0; i < c->n_occupied; i++)
                snr_db[i] = 10.0 * log10(snr_lin[i] + 1e-10);
            for (i = 0; i < c->n_occupied; i++) p->srt[i] = pdb[i];
            med_p = pctile(p->srt, c->n_occupied, 50.0);
            for (i = 0; i < c->n_occupied; i++) p->srt[i] = snr_db[i];
            med_m = pctile(p->srt, c->n_occupied, 50.0);
            info->n_meas_syms = meas_cnt;
            info->has_isi_penalty = 1;
            info->isi_penalty_db = med_p - med_m;
        }
    }

    /* -- 6. ISI ceiling.  Always MEASURED (it is the single most useful number
     * about a reverberant room and costs one transform); whether it is APPLIED
     * is the allocator's decision, because the margin lives there too and the
     * two only compose correctly in one place. */
    if (cnt >= OFDM_ISI_MIN_SYMS) {
        info->has_beta = 1;
        info->isi_ratio = ofdm_isi_ratio(&p->fft, p->hmr, p->hmi,
                                         c->n_occupied, N, c->bin_lo, CP,
                                         snr_lin, c->taper_samples,
                                         p->isc, p->isc_len);
    }
    info->valid = 1;
    return 0;
}
