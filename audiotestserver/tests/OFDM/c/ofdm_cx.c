/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

#include "ofdm_cx.h"
#include "ofdm_fft.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------------------------------------------------------------- geometry */

static int is_pow2(int n) { return n >= 4 && (n & (n - 1)) == 0; }

int ofdm_cxcfg_build(ofdm_cxcfg *x, int n_fft, double cp_ratio,
                     double freq_lo, double freq_hi, int pilot_spacing,
                     int sample_rate, int ecc_bytes, int chunk_bytes,
                     int seq_bytes, double taper_ms, double frame_gap_ms,
                     int sync_period, int dc_null)
{
    ofdm_cfg *c;
    double sc_spc;
    int    cp_len, bin_lo, bin_hi, n_occ, raw_tap, half_cp, has_dc;

    if (!x) return -3;
    if (!is_pow2(n_fft)) return -1;
    if (sample_rate <= 0 || pilot_spacing < 1) return -3;
    if (ecc_bytes < 1 || chunk_bytes < 1 || seq_bytes < 1) return -3;
    if (sync_period < 0 || cp_ratio <= 0.0 || freq_hi < freq_lo) return -3;
    c = &x->c;

    cp_len = (int)(n_fft * cp_ratio + 0.5);
    if (cp_len < 1) cp_len = 1;
    sc_spc = (double)sample_rate / (double)n_fft;

    /* Signed: bin -k is frequency -k*spacing.  -n_fft/2 (Nyquist) is excluded
     * for the same reason as the real path: it is its own mirror. */
    bin_lo = (int)ceil(freq_lo / sc_spc);
    if (bin_lo < -n_fft / 2 + 1) bin_lo = -n_fft / 2 + 1;
    bin_hi = (int)floor(freq_hi / sc_spc);
    if (bin_hi > n_fft / 2 - 1) bin_hi = n_fft / 2 - 1;
    if (bin_lo > bin_hi) return -2;

    has_dc = dc_null && bin_lo <= 0 && bin_hi >= 0;
    n_occ  = bin_hi - bin_lo + 1 - (has_dc ? 1 : 0);
    if (n_occ < 1) return -2;
    /* A band that only touches DC at an edge still loses the edge bin; move
     * the reported edge inward so freq_lo_hz/freq_hi_hz stay occupied bins. */
    if (has_dc && bin_lo == 0) { bin_lo = 1; has_dc = 0; }
    if (has_dc && bin_hi == 0) { bin_hi = -1; has_dc = 0; }

    raw_tap = (int)(taper_ms * sample_rate / 1000.0 + 0.5);
    half_cp = cp_len / 2;
    c->taper_samples = raw_tap < half_cp ? raw_tap : half_cp;
    if (c->taper_samples < 0) c->taper_samples = 0;

    c->n_fft             = n_fft;
    c->cp_len            = cp_len;
    c->bin_lo            = bin_lo;
    c->bin_hi            = bin_hi;
    c->n_occupied        = n_occ;
    c->pilot_spacing     = pilot_spacing;
    c->n_pilots          = (n_occ + pilot_spacing - 1) / pilot_spacing;
    c->n_data_carriers   = n_occ - c->n_pilots;
    c->sample_rate       = sample_rate;
    c->symbol_samples    = n_fft + cp_len;
    c->frame_gap_samples = (int)(frame_gap_ms * sample_rate / 1000.0 + 0.5);
    c->ecc_bytes         = ecc_bytes;
    c->chunk_bytes       = chunk_bytes;
    c->seq_bytes         = seq_bytes;
    c->sync_period       = sync_period;
    c->cp_ratio          = cp_ratio;
    c->sc_spacing_hz     = sc_spc;
    c->freq_lo_hz        = bin_lo * sc_spc;
    c->freq_hi_hz        = bin_hi * sc_spc;
    x->dc_null           = dc_null ? 1 : 0;
    x->dc_slot           = has_dc ? -bin_lo : -1;
    return 0;
}

int ofdm_cx_bin(const ofdm_cxcfg *x, int i)
{
    if (!x || i < 0 || i >= x->c.n_occupied) return 0;
    if (x->dc_slot >= 0 && i >= x->dc_slot) return x->c.bin_lo + i + 1;
    return x->c.bin_lo + i;
}

/* ------------------------------------------------------------------ symbol */

struct ofdm_cxsym {
    ofdm_cxcfg x;
    ofdm_cfft  fft;
    ofdm_cpx  *X;     /* n_fft complex: spectrum, then time domain (in place) */
    double    *win;   /* 2*taper raised-cosine, as ofdm_sym                  */
};

static size_t al8(size_t v) { return (v + 7u) & ~(size_t)7u; }

size_t ofdm_cxsym_workspace(const ofdm_cxcfg *x)
{
    size_t fftw;
    int    T;
    if (!x) return 0;
    fftw = ofdm_cfft_workspace(x->c.n_fft);
    if (fftw == 0) return 0;
    T = x->c.taper_samples;
    return al8(sizeof(struct ofdm_cxsym))
         + fftw
         + al8((size_t)x->c.n_fft * sizeof(ofdm_cpx))
         + al8((size_t)(T > 0 ? 2 * T : 1) * sizeof(double));
}

int ofdm_cxsym_init(ofdm_cxsym **sp, void *mem, size_t memlen, const ofdm_cxcfg *x)
{
    struct ofdm_cxsym *s;
    unsigned char *p = (unsigned char *)mem;
    size_t need, fftw;
    int    i, T;

    if (!sp || !mem || !x) return -1;
    need = ofdm_cxsym_workspace(x);
    if (need == 0) return -1;
    if (memlen < need) return -2;
    fftw = ofdm_cfft_workspace(x->c.n_fft);
    T    = x->c.taper_samples;

    s = (struct ofdm_cxsym *)p; p += al8(sizeof(struct ofdm_cxsym));
    s->x = *x;
    if (ofdm_cfft_init(&s->fft, x->c.n_fft, p, fftw) != 0) return -3;
    p += fftw;
    s->X   = (ofdm_cpx *)p; p += al8((size_t)x->c.n_fft * sizeof(ofdm_cpx));
    s->win = (double *)p;

    /* Identical window to ofdm_sym: rise and its exact complement, so the
     * overlap-add of consecutive symbols sums back to unity. */
    for (i = 0; i < T; i++) {
        double t = (i + 0.5) / (double)T;
        s->win[i]     = 0.5 * (1.0 - cos(M_PI * t));
        s->win[T + i] = 0.5 * (1.0 + cos(M_PI * t));
    }
    *sp = s;
    return 0;
}

int ofdm_cxsym_max_len(const ofdm_cxsym *s)
{
    if (!s) return 0;
    return s->x.c.symbol_samples + 2 * s->x.c.taper_samples;
}

/* signed bin -> natural-order FFT index */
static int fidx(int bin, int n) { return bin < 0 ? bin + n : bin; }

int ofdm_cxsym_make(ofdm_cxsym *s, const double *occ_re, const double *occ_im,
                    int use_taper, double *out_iq)
{
    const ofdm_cfg *c;
    int    N, CP, T, i, len, src;
    double scale;

    if (!s || !occ_re || !out_iq) return -1;
    c  = &s->x.c;
    N  = c->n_fft;
    CP = c->cp_len;
    T  = use_taper ? c->taper_samples : 0;
    scale = sqrt((double)N);

    for (i = 0; i < N; i++) {
        s->X[i].re = (ofdm_real)0;
        s->X[i].im = (ofdm_real)0;
    }
    for (i = 0; i < c->n_occupied; i++) {
        int k = fidx(ofdm_cx_bin(&s->x, i), N);
        s->X[k].re = (ofdm_real)occ_re[i];
        s->X[k].im = (ofdm_real)(occ_im ? occ_im[i] : 0.0);
    }
    ofdm_cfft_inv(&s->fft, s->X);                 /* s->X is now the time domain */

    /* [CP+T | body | T] taken cyclically from the body, as ofdm_sym_make. */
    len = CP + N + 2 * T;
    for (i = 0; i < len; i++) {
        src = (i - CP - T) % N;
        if (src < 0) src += N;
        out_iq[2 * i]     = (double)s->X[src].re * scale;
        out_iq[2 * i + 1] = (double)s->X[src].im * scale;
    }
    for (i = 0; i < T; i++) {
        double wr = s->win[i], wf = s->win[T + i];
        out_iq[2 * i]                 *= wr;
        out_iq[2 * i + 1]             *= wr;
        out_iq[2 * (len - T + i)]     *= wf;
        out_iq[2 * (len - T + i) + 1] *= wf;
    }
    return len;
}

int ofdm_cxsym_demod(ofdm_cxsym *s, const double *body_iq,
                     double *occ_re, double *occ_im)
{
    const ofdm_cfg *c;
    int    N, i;
    double inv;

    if (!s || !body_iq || !occ_re || !occ_im) return -1;
    c   = &s->x.c;
    N   = c->n_fft;
    inv = 1.0 / sqrt((double)N);   /* make scaled ifft (1/N) by sqrt(N): fwd fft / sqrt(N) undoes it */
    for (i = 0; i < N; i++) {
        s->X[i].re = (ofdm_real)body_iq[2 * i];
        s->X[i].im = (ofdm_real)body_iq[2 * i + 1];
    }
    ofdm_cfft_fwd(&s->fft, s->X);
    for (i = 0; i < c->n_occupied; i++) {
        int k = fidx(ofdm_cx_bin(&s->x, i), N);
        occ_re[i] = (double)s->X[k].re * inv;
        occ_im[i] = (double)s->X[k].im * inv;
    }
    return c->n_occupied;
}
