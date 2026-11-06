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

#include "ofdm_sym.h"
#include "ofdm_fft.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct ofdm_sym {
    ofdm_cfg   c;
    ofdm_fft   fft;
    ofdm_cpx  *X;    /* n_fft/2+1 half spectrum      */
    ofdm_real *td;   /* n_fft inverse-transform out  */
    double    *win;  /* 2*taper raised-cosine        */
};

static size_t al8(size_t v) { return (v + 7u) & ~(size_t)7u; }

size_t ofdm_sym_workspace(const ofdm_cfg *c)
{
    size_t fftw;
    int    half, T;

    if (!c) return 0;
    fftw = ofdm_fft_workspace(c->n_fft);
    if (fftw == 0) return 0;
    half = c->n_fft / 2 + 1;
    T    = c->taper_samples;

    return al8(sizeof(struct ofdm_sym))
         + fftw
         + al8((size_t)half * sizeof(ofdm_cpx))
         + al8((size_t)c->n_fft * sizeof(ofdm_real))
         + al8((size_t)(T > 0 ? 2 * T : 1) * sizeof(double));
}

int ofdm_sym_init(ofdm_sym **sp, void *mem, size_t memlen, const ofdm_cfg *c)
{
    struct ofdm_sym *s;
    unsigned char *p = (unsigned char *)mem;
    size_t need, fftw;
    int    i, rc, half, T;

    if (!sp || !mem || !c) return -1;
    need = ofdm_sym_workspace(c);
    if (need == 0) return -1;
    if (memlen < need) return -2;
    fftw = ofdm_fft_workspace(c->n_fft);
    half = c->n_fft / 2 + 1;
    T    = c->taper_samples;

    s = (struct ofdm_sym *)p; p += al8(sizeof(struct ofdm_sym));
    s->c = *c;
    rc = ofdm_fft_init(&s->fft, c->n_fft, p, fftw);
    if (rc != 0) return -3;
    p += fftw;
    s->X   = (ofdm_cpx *)p;  p += al8((size_t)half * sizeof(ofdm_cpx));
    s->td  = (ofdm_real *)p; p += al8((size_t)c->n_fft * sizeof(ofdm_real));
    s->win = (double *)p;

    /* Raised-cosine rise with a fall that is its exact complement, so the
     * overlap-add of consecutive symbols sums back to unity. */
    if (T > 0) {
        for (i = 0; i < T; i++) {
            double x = (i + 0.5) / (double)T;
            s->win[i]     = 0.5 * (1.0 - cos(M_PI * x));
            s->win[T + i] = 0.5 * (1.0 + cos(M_PI * x));
        }
    }
    *sp = s;
    return 0;
}

int ofdm_sym_max_len(const ofdm_sym *s)
{
    if (!s) return 0;
    return s->c.symbol_samples + 2 * s->c.taper_samples;
}

int ofdm_sym_make(ofdm_sym *s, const double *occ_re, const double *occ_im,
                  int use_taper, double *out)
{
    const ofdm_cfg *c;
    int    N, CP, T, half, i, len;
    double scale;

    if (!s || !occ_re || !out) return -1;
    c    = &s->c;
    N    = c->n_fft;
    CP   = c->cp_len;
    T    = use_taper ? c->taper_samples : 0;
    half = N / 2 + 1;
    scale = sqrt((double)N);

    /* Only the positive-frequency half is filled: the inverse real transform IS
     * the Hermitian inverse, so the conjugate mirror is implied rather than
     * materialised.  ofdm_cfg_build guarantees bin_lo >= 1 and
     * bin_hi <= N/2-1, so no occupied bin lands on DC or Nyquist and the half
     * spectrum is unambiguous. */
    for (i = 0; i < half; i++) {
        s->X[i].re = (ofdm_real)0;
        s->X[i].im = (ofdm_real)0;
    }
    for (i = 0; i < c->n_occupied; i++) {
        s->X[c->bin_lo + i].re = (ofdm_real)occ_re[i];
        s->X[c->bin_lo + i].im = (ofdm_real)(occ_im ? occ_im[i] : 0.0);
    }
    ofdm_irfft(&s->fft, s->X, s->td);

    if (T == 0) {
        for (i = 0; i < CP; i++) out[i] = (double)s->td[N - CP + i] * scale;
        for (i = 0; i < N; i++)  out[CP + i] = (double)s->td[i] * scale;
        return CP + N;
    }
    len = CP + N + 2 * T;
    for (i = 0; i < CP + T; i++) out[i] = (double)s->td[N - CP - T + i] * scale;
    for (i = 0; i < N; i++)      out[CP + T + i] = (double)s->td[i] * scale;
    for (i = 0; i < T; i++)      out[CP + T + N + i] = (double)s->td[i] * scale;
    for (i = 0; i < T; i++) {
        out[i]           *= s->win[i];
        out[len - T + i] *= s->win[T + i];
    }
    return len;
}
