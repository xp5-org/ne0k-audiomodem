/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
/* css_util.h — numpy reduction semantics the PLL decoder port has to mirror:
 * pairwise summation (np.sum / np.mean on contiguous float64), np.median
 * (mean of the two middle values for an even count), banker's rounding
 * (Python round / np.round), and Python's floor-mod. */
#ifndef CSS_UTIL_H
#define CSS_UTIL_H
#include <math.h>
#include <stddef.h>

/* numpy pairwise_sum (umath loops): < 8 sequential; <= 128 eight
 * accumulators; above that split at n/2 rounded down to a multiple of 8. */
static inline double css_psum(const double *a, size_t n)
{
    if (n < 8) {
        size_t i;
        double res = 0.0;
        for (i = 0; i < n; i++) res += a[i];
        return res;
    } else if (n <= 128) {
        double r[8], res;
        size_t i;
        for (i = 0; i < 8; i++) r[i] = a[i];
        for (i = 8; i < n - (n % 8); i += 8) {
            r[0] += a[i + 0]; r[1] += a[i + 1]; r[2] += a[i + 2]; r[3] += a[i + 3];
            r[4] += a[i + 4]; r[5] += a[i + 5]; r[6] += a[i + 6]; r[7] += a[i + 7];
        }
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    } else {
        size_t n2 = n / 2;
        n2 -= n2 % 8;
        return css_psum(a, n2) + css_psum(a + n2, n - n2);
    }
}

static inline double css_pmean(const double *a, size_t n)
{
    return n ? css_psum(a, n) / (double)n : NAN;
}

/* Python round() / np.round: half to even. */
static inline double css_round_even(double x) { return nearbyint(x); }
static inline long css_iround(double x) { return (long)nearbyint(x); }

/* numpy/Python floating mod: result has the sign of the divisor. */
static inline double css_pymod(double a, double b)
{
    double m = fmod(a, b);
    if (m != 0.0 && ((b < 0) != (m < 0))) m += b;
    return m;
}

/* In-place quickselect of the k-th smallest (0-based) in a[0..n). */
#define CSS_DEFINE_SELECT(NAME, T)                                            \
static inline T NAME(T *a, size_t n, size_t k)                                       \
{                                                                             \
    size_t lo = 0, hi = n - 1;                                                \
    while (lo < hi) {                                                         \
        T piv = a[lo + (hi - lo) / 2];                                        \
        size_t i = lo, j = hi;                                                \
        while (i <= j) {                                                      \
            while (a[i] < piv) i++;                                           \
            while (a[j] > piv) { if (j == 0) break; j--; }                    \
            if (i <= j) {                                                     \
                T t = a[i]; a[i] = a[j]; a[j] = t;                            \
                i++; if (j == 0) break; j--;                                  \
            }                                                                 \
        }                                                                     \
        if (k <= j) hi = j; else if (k >= i) lo = i; else break;              \
    }                                                                         \
    return a[k];                                                              \
}
CSS_DEFINE_SELECT(css_select_d, double)
CSS_DEFINE_SELECT(css_select_f, float)

/* np.median over a scratch copy (destroyed). Even n: mean of the middle two,
 * evaluated in the array's own type like numpy. */
static inline double css_median_d(double *a, size_t n)
{
    double hi, lo;
    size_t i;
    if (n == 0) return NAN;
    hi = css_select_d(a, n, n / 2);
    if (n % 2) return hi;
    lo = a[0];
    for (i = 1; i < n / 2; i++) if (a[i] > lo) lo = a[i];
    return (lo + hi) / 2.0;
}

static inline float css_median_f(float *a, size_t n)
{
    float hi, lo;
    size_t i;
    if (n == 0) return NAN;
    hi = css_select_f(a, n, n / 2);
    if (n % 2) return hi;
    lo = a[0];
    for (i = 1; i < n / 2; i++) if (a[i] > lo) lo = a[i];
    return (float)((lo + hi) / 2.0f);
}

#endif
