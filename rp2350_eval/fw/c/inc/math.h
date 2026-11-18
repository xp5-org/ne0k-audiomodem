/* Freestanding stand-in: the kernels only need these (mathrt.c implements them). */
#ifndef EVAL_MATH_H
#define EVAL_MATH_H
double sqrt(double);
double sin(double);
double cos(double);
double atan2(double, double);
double hypot(double, double);
float hypotf(float, float);
#endif
