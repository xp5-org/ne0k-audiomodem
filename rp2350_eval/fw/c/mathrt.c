/* Minimal libm for the kernels' one-time setup (twiddles, scale factors). Shared with
 * the host reference build so results are bit-identical on M33 and x86. Not tuned. */
#include "mathrt.h"
#define PI 3.14159265358979323846

double ev_sqrt(double x)
{
    double r, p;
    int i;
    if (x <= 0.0) return 0.0;
    r = x > 1.0 ? x : 1.0;
    for (i = 0; i < 60; i++) {
        p = r;
        r = 0.5 * (r + x / r);
        if (r == p) break;
    }
    return r;
}

/* sin/cos by reduction to [-pi/4, pi/4] then Taylor to x^17. */
static double ksin(double x)
{
    double x2 = x * x;
    return x * (1.0 + x2 * (-1.0 / 6 + x2 * (1.0 / 120 + x2 * (-1.0 / 5040 + x2 * (1.0 / 362880 +
           x2 * (-1.0 / 39916800 + x2 * (1.0 / 6227020800.0 + x2 * (-1.0 / 1307674368000.0))))))));
}
static double kcos(double x)
{
    double x2 = x * x;
    return 1.0 + x2 * (-0.5 + x2 * (1.0 / 24 + x2 * (-1.0 / 720 + x2 * (1.0 / 40320 +
           x2 * (-1.0 / 3628800 + x2 * (1.0 / 479001600 + x2 * (-1.0 / 87178291200.0)))))));
}
static void reduce(double x, int *q, double *r)
{
    double k = x * (2.0 / PI);
    long n = (long)(k < 0 ? k - 0.5 : k + 0.5);
    *r = x - (double)n * (PI / 2);
    *q = (int)(n & 3);
}
double ev_sin(double x)
{
    int q; double r;
    reduce(x, &q, &r);
    switch (q) { case 0: return ksin(r); case 1: return kcos(r); case 2: return -ksin(r); default: return -kcos(r); }
}
double ev_cos(double x)
{
    int q; double r;
    reduce(x, &q, &r);
    switch (q) { case 0: return kcos(r); case 1: return -ksin(r); case 2: return -kcos(r); default: return ksin(r); }
}
