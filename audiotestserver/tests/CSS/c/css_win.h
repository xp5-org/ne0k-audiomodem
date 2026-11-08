/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
/* css_win.h — numpy's window formulas, evaluated exactly as numpy does
 * (n = arange(1-M, M, 2); 0.5 + 0.5*cos(pi*n/(M-1))), not the textbook
 * 0.5 - 0.5*cos(2*pi*k/(M-1)) spelling, which rounds differently. */
#ifndef CSS_WIN_H
#define CSS_WIN_H
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline double css_hanning(int k, int M)
{
    double n;
    if (M == 1) return 1.0;
    n = (double)(1 - M + 2 * k);
    return 0.5 + 0.5 * cos(M_PI * n / (double)(M - 1));
}

static inline double css_hamming(int k, int M)
{
    double n;
    if (M == 1) return 1.0;
    n = (double)(1 - M + 2 * k);
    return 0.54 + 0.46 * cos(M_PI * n / (double)(M - 1));
}

#endif
