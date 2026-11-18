#include <stddef.h>
#include "kernels.h"
#include "rs255.h"

unsigned char ev_arena[EV_ARENA_BYTES] __attribute__((aligned(8)));
static unsigned lcg;
void ev_seed(unsigned s) { lcg = s; }
unsigned ev_rand(void) { lcg = lcg * 1664525u + 1013904223u; return lcg >> 8; }
unsigned ev_fnv(unsigned h, const void *p, unsigned n)
{
    const unsigned char *b = (const unsigned char *)p;
    while (n--) { h ^= *b++; h *= 16777619u; }
    return h;
}

#define RS_K(nsym) (255 - (nsym))

static unsigned k_rs(unsigned id, unsigned nsym, unsigned reps, unsigned *ws)
{
    rs255 *rs;
    unsigned char *msg = ev_arena + 8192, *cw = msg + 256, *out = cw + 256;
    size_t need = rs255_workspace((int)nsym);
    unsigned h = 2166136261u, r, i;
    int k = RS_K((int)nsym), e, errata = 0;
    *ws = (unsigned)need + 3u * 256u;
    if (!need || rs255_init(&rs, ev_arena, 8192, (int)nsym) != 0) return 1;
    ev_seed(4242u);
    for (i = 0; i < (unsigned)k; i++) msg[i] = (unsigned char)ev_rand();
    rs255_encode(rs, msg, k, cw, 256);
    if (id == K_RS_DEC_ERR) /* nsym/2 symbol errors: the correction limit */
        for (e = 0; e < (int)nsym / 2; e++) cw[(e * 37 + 3) % 255] ^= (unsigned char)(0x5A + e);
    for (r = 0; r < reps; r++) {
        if (id == K_RS_ENC) rs255_encode(rs, msg, k, cw, 256);
        else                rs255_decode(rs, cw, 255, out, 256, &errata);
    }
    h = ev_fnv(h, cw, 255);
    if (id != K_RS_ENC) { h = ev_fnv(h, out, (unsigned)k); h = ev_fnv(h, &errata, 4); }
    return h;
}

/* Sliding-correlator inner loop: one output = dot(window, ref) over `len` taps. */
static unsigned k_dot_f32(unsigned len, unsigned reps, unsigned *ws)
{
    float *a = (float *)ev_arena, *b = a + len, acc = 0.0f;
    unsigned r, i;
    *ws = 2u * len * 4u;
    if (*ws > EV_ARENA_BYTES) return 0;
    ev_seed(9u);
    for (i = 0; i < len; i++) { a[i] = (float)((int)(ev_rand() & 0xFFFF) - 32768) / 32768.0f; b[i] = (float)((int)(ev_rand() & 0xFFFF) - 32768) / 32768.0f; }
    for (r = 0; r < reps; r++) { float s = 0.0f; for (i = 0; i < len; i++) s += a[i] * b[i]; acc += s; }
    return ev_fnv(2166136261u, &acc, 4);
}
static unsigned k_dot_f64(unsigned len, unsigned reps, unsigned *ws)
{
    double *a = (double *)ev_arena, *b = a + len, acc = 0.0;
    unsigned r, i;
    *ws = 2u * len * 8u;
    if (*ws > EV_ARENA_BYTES) return 0;
    ev_seed(9u);
    for (i = 0; i < len; i++) { a[i] = (double)((int)(ev_rand() & 0xFFFF) - 32768) / 32768.0; b[i] = (double)((int)(ev_rand() & 0xFFFF) - 32768) / 32768.0; }
    for (r = 0; r < reps; r++) { double s = 0.0; for (i = 0; i < len; i++) s += a[i] * b[i]; acc += s; }
    return ev_fnv(2166136261u, &acc, 8);
}
/* Q15 FIR, one output per rep; int32 accumulate, the shape the M33 DSP MACs help with. */
static unsigned k_fir_q15(unsigned taps, unsigned reps, unsigned *ws)
{
    short *x = (short *)ev_arena, *c = x + taps;
    int acc = 0;
    unsigned r, i;
    *ws = 2u * taps * 2u;
    if (*ws > EV_ARENA_BYTES) return 0;
    ev_seed(31u);
    for (i = 0; i < taps; i++) { x[i] = (short)ev_rand(); c[i] = (short)ev_rand(); }
    for (r = 0; r < reps; r++) { int s = 0; for (i = 0; i < taps; i++) s += (int)x[i] * (int)c[i]; acc += s >> 15; }
    return ev_fnv(2166136261u, &acc, 4);
}

unsigned kernel_run(unsigned id, unsigned p, unsigned reps, unsigned *ws)
{
    *ws = 0;
    switch (id) {
    case K_NOP:         return 0;
    case K_RFFT_F64:    return fftk_f64(0, (int)p, reps, ws);
    case K_RFFT_F32:    return fftk_f32(0, (int)p, reps, ws);
    case K_IRFFT_F64:   return fftk_f64(1, (int)p, reps, ws);
    case K_IRFFT_F32:   return fftk_f32(1, (int)p, reps, ws);
    case K_RS_ENC: case K_RS_DEC_CLEAN: case K_RS_DEC_ERR: return k_rs(id, p, reps, ws);
    case K_DOT_F32:     return k_dot_f32(p, reps, ws);
    case K_DOT_F64:     return k_dot_f64(p, reps, ws);
    case K_FIR_Q15:     return k_fir_q15(p, reps, ws);
    case K_CMUL_F32:    return cmulk_f32((int)p, reps, ws);
    }
    return 0xFFFFFFFFu;
}
