/* Compiled twice: -DPREC_F32 (-DOFDM_ACQ_F32, ofdm_real=float) and default (double).
 * The build renames the ofdm_fft entry points per precision (-Dofdm_rfft=...). */
#include <stddef.h>
#include "ofdm_fft.h"
#include "kernels.h"

#ifdef PREC_F32
#define FFTK fftk_f32
#else
#define FFTK fftk_f64
#endif

#define BUF_REAL(n) ((n) * sizeof(ofdm_real))

unsigned FFTK(int inverse, int n, unsigned reps, unsigned *ws)
{
    ofdm_fft f;
    size_t need = ofdm_fft_workspace(n);
    size_t xo = (need + 7u) & ~7u;
    size_t Xo = xo + ((BUF_REAL((size_t)n) + 7u) & ~7u);
    ofdm_real *x;
    ofdm_cpx *X;
    unsigned h = 2166136261u, r;
    int i;
    *ws = (unsigned)(Xo + (size_t)(n / 2 + 1) * sizeof(ofdm_cpx));
    if (need == 0 || *ws > EV_ARENA_BYTES) return 0;
    x = (ofdm_real *)(ev_arena + xo);
    X = (ofdm_cpx *)(ev_arena + Xo);
    if (ofdm_fft_init(&f, n, ev_arena, need) != 0) return 1;
    ev_seed(12345u);
    for (i = 0; i < n; i++) x[i] = (ofdm_real)((int)(ev_rand() & 0xFFFFu) - 32768) / (ofdm_real)32768;
    ofdm_rfft(&f, x, X);                       /* spectrum for the inverse path */
    for (r = 0; r < reps; r++) {
        if (inverse) ofdm_irfft(&f, X, x);
        else         ofdm_rfft(&f, x, X);
    }
    h = ev_fnv(h, x, (unsigned)BUF_REAL((size_t)n));
    h = ev_fnv(h, X, (unsigned)((size_t)(n / 2 + 1) * sizeof(ofdm_cpx)));
    return h;
}

#ifdef PREC_F32
/* CSS-style dechirp stand-in: n complex multiplies by a chirp reference, then the FFT
 * peak search happens in fftk_f32. Measures the multiply stage alone. */
unsigned cmulk_f32(int n, unsigned reps, unsigned *ws)
{
    ofdm_cpx *a = (ofdm_cpx *)ev_arena, *b = a + n, *o = b + n;
    unsigned h = 2166136261u, r;
    int i;
    *ws = (unsigned)(3u * (unsigned)n * sizeof(ofdm_cpx));
    if (*ws > EV_ARENA_BYTES) return 0;
    ev_seed(777u);
    for (i = 0; i < n; i++) {
        a[i].re = (float)((int)(ev_rand() & 0xFFFFu) - 32768) / 32768.0f;
        a[i].im = (float)((int)(ev_rand() & 0xFFFFu) - 32768) / 32768.0f;
        b[i].re = (float)ev_cos(0.01 * (double)i * (double)i);
        b[i].im = (float)ev_sin(0.01 * (double)i * (double)i);
    }
    for (r = 0; r < reps; r++)
        for (i = 0; i < n; i++) {
            o[i].re = a[i].re * b[i].re - a[i].im * b[i].im;
            o[i].im = a[i].re * b[i].im + a[i].im * b[i].re;
        }
    return ev_fnv(h, o, (unsigned)((unsigned)n * sizeof(ofdm_cpx)));
}
#endif
