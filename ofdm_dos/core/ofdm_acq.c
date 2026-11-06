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

#include "ofdm_acq.h"
#include <math.h>

/* The grid and the metric accumulators are float32 in the numpy reference
 * (complex64 grid, float32 sums) regardless of the FFT precision, so they are
 * float32 here too.  See ofdm_acq.h. */
typedef struct { float re, im; } acq_cpx;

/* Guard from the reference's `nsyms*den + 1e-18`.  This one is safe to carry
 * across precisions: den is a sum of squared magnitudes, so the term only
 * matters when den is 0 (a dead carrier), where it just avoids 0/0.  It is NOT
 * a precision-tuned epsilon -- contrast the derived epsilons that a float32
 * build does have to re-derive. */
#define ACQ_DEN_EPS 1e-18f

struct ofdm_acq {
    int n_fft, cp_len, bin_lo, n_occupied, nsyms, step, top_k;
    int stride;      /* (n_fft+cp_len)/step                     */
    int span;        /* stride*(nsyms-1)                        */
    int ring_rows;   /* span+1                                  */
    ofdm_fft   fft;
    ofdm_cpx  *known;    /* n_occupied                          */
    ofdm_real *q;        /* nsyms                               */
    ofdm_real *win;      /* n_fft      scratch, real FFT input   */
    ofdm_cpx  *spec;     /* n_fft/2+1  scratch, half spectrum    */
    acq_cpx   *ring;     /* ring_rows * n_occupied               */
    acq_cpx   *acc;      /* n_occupied scratch                   */
    float     *den;      /* n_occupied scratch                   */
    float     *mag;      /* n_occupied scratch                   */
    float     *ck;       /* n_occupied scratch                   */
    float     *sel;      /* n_occupied scratch, destroyed by the
                          * quickselect in top_k_mean            */
};

static size_t al8(size_t v) { return (v + 7u) & ~(size_t)7u; }

static int plan_ok(int n_fft, int cp_len, int n_occupied,
                   int nsyms, int step, int top_k)
{
    if (n_fft < 4 || (n_fft & (n_fft - 1)) != 0) return 0;
    if (cp_len < 0 || n_occupied < 1 || nsyms < 1) return 0;
    if (step < 1 || top_k < 1 || top_k > n_occupied) return 0;
    if (n_occupied > n_fft / 2 + 1) return 0;
    if ((n_fft + cp_len) % step != 0) return 0;   /* step must divide SYM */
    return 1;
}

/* Must stay the term-for-term sum of the bumps ofdm_acq_init performs, in the
 * same order.  Edit the two together. */
size_t ofdm_acq_workspace(int n_fft, int cp_len, int n_occupied,
                          int nsyms, int step, int top_k)
{
    int stride, ring_rows;
    size_t fftw;
    if (!plan_ok(n_fft, cp_len, n_occupied, nsyms, step, top_k)) return 0;
    fftw = ofdm_fft_workspace(n_fft);
    if (fftw == 0) return 0;
    stride    = (n_fft + cp_len) / step;
    ring_rows = stride * (nsyms - 1) + 1;

    return al8(sizeof(struct ofdm_acq))
         + fftw
         + al8((size_t)n_occupied            * sizeof(ofdm_cpx))   /* known */
         + al8((size_t)nsyms                 * sizeof(ofdm_real))  /* q     */
         + al8((size_t)n_fft                 * sizeof(ofdm_real))  /* win   */
         + al8((size_t)(n_fft / 2 + 1)       * sizeof(ofdm_cpx))   /* spec  */
         + al8((size_t)ring_rows * n_occupied * sizeof(acq_cpx))   /* ring  */
         + al8((size_t)n_occupied            * sizeof(acq_cpx))    /* acc   */
         + al8((size_t)n_occupied            * sizeof(float))      /* den   */
         + al8((size_t)n_occupied            * sizeof(float))      /* mag   */
         + al8((size_t)n_occupied            * sizeof(float))      /* ck    */
         + al8((size_t)n_occupied            * sizeof(float));     /* sel   */
}

int ofdm_acq_init(ofdm_acq **ap, void *mem, size_t memlen,
                  const double *known_re, const double *known_im,
                  const double *q,
                  int n_fft, int cp_len, int bin_lo, int n_occupied,
                  int nsyms, int step, int top_k)
{
    struct ofdm_acq *a;
    unsigned char *p = (unsigned char *)mem;
    size_t need, fftw;
    int i, rc, stride, ring_rows;

    if (!ap || !mem || !known_re || !known_im) return -1;
    if (bin_lo < 0 || bin_lo + n_occupied > n_fft / 2 + 1) return -1;
    if (!plan_ok(n_fft, cp_len, n_occupied, nsyms, step, top_k)) {
        /* Separate the one precondition a caller is most likely to get wrong
         * from generic bad args: the grid only lands on symbol boundaries if
         * step divides n_fft+cp_len (callers use gcd(SYM, desired_step)). */
        if (step >= 1 && (n_fft + cp_len) % step != 0) return -4;
        return -1;
    }
    need = ofdm_acq_workspace(n_fft, cp_len, n_occupied, nsyms, step, top_k);
    if (need == 0 || memlen < need) return -2;

    stride    = (n_fft + cp_len) / step;
    ring_rows = stride * (nsyms - 1) + 1;
    fftw      = ofdm_fft_workspace(n_fft);

    a = (struct ofdm_acq *)p; p += al8(sizeof(struct ofdm_acq));
    rc = ofdm_fft_init(&a->fft, n_fft, p, fftw);
    if (rc != 0) return -3;
    p += fftw;

    a->known = (ofdm_cpx  *)p; p += al8((size_t)n_occupied * sizeof(ofdm_cpx));
    a->q     = (ofdm_real *)p; p += al8((size_t)nsyms      * sizeof(ofdm_real));
    a->win   = (ofdm_real *)p; p += al8((size_t)n_fft      * sizeof(ofdm_real));
    a->spec  = (ofdm_cpx  *)p; p += al8((size_t)(n_fft/2+1)* sizeof(ofdm_cpx));
    a->ring  = (acq_cpx   *)p; p += al8((size_t)ring_rows * n_occupied
                                        * sizeof(acq_cpx));
    a->acc   = (acq_cpx   *)p; p += al8((size_t)n_occupied * sizeof(acq_cpx));
    a->den   = (float     *)p; p += al8((size_t)n_occupied * sizeof(float));
    a->mag   = (float     *)p; p += al8((size_t)n_occupied * sizeof(float));
    a->ck    = (float     *)p; p += al8((size_t)n_occupied * sizeof(float));
    a->sel   = (float     *)p;

    a->n_fft = n_fft; a->cp_len = cp_len; a->bin_lo = bin_lo;
    a->n_occupied = n_occupied; a->nsyms = nsyms; a->step = step;
    a->top_k = top_k; a->stride = stride; a->span = stride * (nsyms - 1);
    a->ring_rows = ring_rows;

    for (i = 0; i < n_occupied; i++) {
        a->known[i].re = (ofdm_real)known_re[i];
        a->known[i].im = (ofdm_real)known_im[i];
    }
    for (i = 0; i < nsyms; i++)
        a->q[i] = (ofdm_real)(q ? q[i] : 1.0);

    *ap = a;
    return 0;
}

int ofdm_acq_ring_rows(const ofdm_acq *a) { return a ? a->ring_rows : 0; }
int ofdm_acq_nfft(const ofdm_acq *a)      { return a ? a->n_fft     : 0; }

/* Complex division matching numpy's Smith algorithm.  The reference computes
 * (X * scale) / known with `known` complex128; `known` is real (+-1/sqrt(2))
 * for every caller today, but doing the true division rather than multiplying
 * by a precomputed reciprocal keeps the rounding identical to numpy and costs
 * one divide per carrier against a whole FFT per row.
 *
 * A target build that no longer needs numpy bit-equivalence can hoist this to
 * a precomputed scale/known table -- one extra rounding, ~n_occupied divides
 * per row saved. */
static void cdiv(ofdm_real ar, ofdm_real ai, ofdm_real br, ofdm_real bi,
                 ofdm_real *qr, ofdm_real *qi)
{
    ofdm_real r, den;
    if (fabs((double)br) >= fabs((double)bi)) {
        r   = bi / br;
        den = br + bi * r;
        *qr = (ar + ai * r) / den;
        *qi = (ai - ar * r) / den;
    } else {
        r   = br / bi;
        den = br * r + bi;
        *qr = (ar * r + ai) / den;
        *qi = (ai * r - ar) / den;
    }
}

/* One grid row: FFT the window at `pos`, normalise, store as float32. */
static void row_at(ofdm_acq *a, const double *rx, int pos, acq_cpx *out)
{
    int i;
    ofdm_real scale = (ofdm_real)(1.0 / sqrt((double)a->n_fft));
    const double *src = rx + (size_t)pos * a->step + a->cp_len;

    for (i = 0; i < a->n_fft; i++)
        a->win[i] = (ofdm_real)src[i];
    ofdm_rfft(&a->fft, a->win, a->spec);

    for (i = 0; i < a->n_occupied; i++) {
        ofdm_cpx  X = a->spec[a->bin_lo + i];
        ofdm_real nr, ni;
        cdiv(X.re * scale, X.im * scale,
             a->known[i].re, a->known[i].im, &nr, &ni);
        out[i].re = (float)nr;
        out[i].im = (float)ni;
    }
}

/* Value of the k-th largest element of a[0..n), destroying a.  Quickselect,
 * median-of-three pivot: O(n) expected against O(n log n) for a full sort.
 *
 * The first version of this was a partial selection sort -- k passes, each
 * rescanning the already-picked list -- which is O(k^2 n): 512k operations per
 * candidate at k=40, n=320, and it dominated the whole scan (the C measured 5x
 * SLOWER than numpy until this was replaced).  On the MCU that cost would be
 * just as real, so it is fixed here rather than papered over on the host. */
#define ACQ_SWAP(x, y) do { float t_ = (x); (x) = (y); (y) = t_; } while (0)

static float kth_largest(float *a, int n, int k)
{
    int lo = 0, hi = n - 1, target = k - 1;   /* 0-based rank, descending */
    while (lo < hi) {
        int   mid = lo + (hi - lo) / 2, i, store;
        float p;
        /* Median-of-three: order a[lo] >= a[mid] >= a[hi], then park the
         * median at a[hi] as the pivot. */
        if (a[mid] > a[lo]) ACQ_SWAP(a[mid], a[lo]);
        if (a[hi]  > a[lo]) ACQ_SWAP(a[hi],  a[lo]);
        if (a[mid] > a[hi]) ACQ_SWAP(a[mid], a[hi]);
        p = a[hi];

        /* Lomuto partition, descending: strictly-greater to the left. */
        store = lo;
        for (i = lo; i < hi; i++)
            if (a[i] > p) { ACQ_SWAP(a[i], a[store]); store++; }
        ACQ_SWAP(a[store], a[hi]);
        /* a[lo..store-1] > p, a[store] == p, a[store+1..hi] <= p */

        if (target == store) return a[store];
        if (target <  store) hi = store - 1;
        else                 lo = store + 1;
    }
    return a[lo];
}

/* Mean of ck over the top_k carriers by mag.
 *
 * Threshold-and-scan rather than materialising the index set: quickselect for
 * the k-th largest magnitude, then one ascending pass taking everything
 * strictly above it plus as many ties as are still needed.  Scanning ascending
 * makes the tie-break lowest-index and deterministic; numpy's argpartition
 * makes no tie-break guarantee.
 *
 * This is the ONE place the C and numpy visibly disagree, and it is inherent,
 * not a bug.  "The top 40 carriers" is ambiguous when carriers sit within
 * round-off of the cut: both sides compute float32 magnitudes correctly but
 * with different roundings, so two carriers a single ULP apart can order
 * either way.  Swapping one carrier of 40 moves the metric by ~1e-5 -- two
 * orders above the 3e-7 arithmetic floor, which is why it stands out.
 * Measured over 3 geometries x 6 signals: 4 candidates out of ~40000, every
 * one a boundary near-tie, all at low metric values far from any gate, with
 * argmax and every gate decision unchanged.
 *
 * Do not "fix" this by matching numpy's introselect: that is unspecified
 * behaviour which can change between numpy versions, and the target has no
 * numpy to match.  Lowest-index is deterministic and reproducible on the MCU,
 * which is what actually matters. */
static float top_k_mean(const float *mag, const float *ck, float *scratch,
                        int n, int k)
{
    int i, taken = 0, ties_left;
    float thr, s = 0.0f;

    for (i = 0; i < n; i++) scratch[i] = mag[i];
    thr = kth_largest(scratch, n, k);

    /* Everything strictly above the threshold is in, unconditionally. */
    for (i = 0; i < n; i++)
        if (mag[i] > thr) { s += ck[i]; taken++; }
    /* Then fill from the ties, lowest index first. */
    ties_left = k - taken;
    for (i = 0; i < n && ties_left > 0; i++)
        if (mag[i] == thr) { s += ck[i]; ties_left--; }

    return s / (float)k;
}

int ofdm_acq_run(ofdm_acq *a, const double *rx, int n,
                 double *metric_out, int metric_cap, ofdm_acq_info *info)
{
    int n_pos, n_cand, p, c, i, j;

    if (!a || !rx || !metric_out || !info) return -1;
    if (n < a->cp_len + a->n_fft) {
        info->n_cand = 0; info->n_pos = 0; info->start_step = a->step;
        return 0;
    }
    n_pos = (n - a->cp_len - a->n_fft) / a->step + 1;
    if (n_pos < a->span + 1) {
        info->n_cand = 0; info->n_pos = n_pos; info->start_step = a->step;
        return 0;
    }
    n_cand = n_pos - a->span;
    if (metric_cap < n_cand) return -2;

    info->n_cand = n_cand; info->n_pos = n_pos; info->start_step = a->step;

    for (p = 0; p < n_pos; p++) {
        row_at(a, rx, p, a->ring + (size_t)(p % a->ring_rows) * a->n_occupied);

        c = p - a->span;
        if (c < 0) continue;

        for (i = 0; i < a->n_occupied; i++) {
            a->acc[i].re = a->acc[i].im = 0.0f;
            a->den[i] = a->mag[i] = 0.0f;
        }
        /* Sequential accumulation, deliberately: the numpy reference was made
         * sequential to match this, so a difference here is a port bug rather
         * than a reduction-tree artifact. */
        for (j = 0; j < a->nsyms; j++) {
            int      r   = (c + j * a->stride) % a->ring_rows;
            acq_cpx *row = a->ring + (size_t)r * a->n_occupied;
            float    qj  = (float)a->q[j];
            for (i = 0; i < a->n_occupied; i++) {
                float re = row[i].re, im = row[i].im;
                /* hypotf, and den from the ROUNDED magnitude, because the
                 * reference is `a = np.abs(Hj); den += a*a`: numpy's complex
                 * abs is hypot, and it squares the float32 result rather than
                 * summing re^2+im^2.  Both details are worth ~1 ulp each and
                 * both would otherwise show up as a phantom port bug. */
                float m = ofdm_hypotf(re, im);
                a->acc[i].re += re * qj;
                a->acc[i].im += im * qj;
                a->den[i] += m * m;
                a->mag[i] += m;
            }
        }
        for (i = 0; i < a->n_occupied; i++) {
            float m = ofdm_hypotf(a->acc[i].re, a->acc[i].im);   /* np.abs(acc)**2 */
            a->ck[i] = (m * m)
                     / ((float)a->nsyms * a->den[i] + ACQ_DEN_EPS);
        }
        metric_out[c] = (double)top_k_mean(a->mag, a->ck, a->sel,
                                           a->n_occupied, a->top_k);
    }
    return 0;
}

int ofdm_acq_row(ofdm_acq *a, const double *rx, int n, int pos, float *out)
{
    int i, n_pos;
    acq_cpx *tmp;
    if (!a || !rx || !out || pos < 0) return -1;
    if (n < a->cp_len + a->n_fft) return -2;
    n_pos = (n - a->cp_len - a->n_fft) / a->step + 1;
    if (pos >= n_pos) return -2;
    tmp = a->ring;                       /* row 0 is free scratch here */
    row_at(a, rx, pos, tmp);
    for (i = 0; i < a->n_occupied; i++) {
        out[2 * i]     = tmp[i].re;
        out[2 * i + 1] = tmp[i].im;
    }
    return 0;
}
