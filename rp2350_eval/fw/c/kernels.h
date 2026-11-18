#ifndef EVAL_KERNELS_H
#define EVAL_KERNELS_H
/* One entry per kernel id. Every kernel: sets up once, runs `reps` iterations of the
 * measured work, returns an FNV-1a checksum of its output, and stores the bytes of
 * working memory it needed in *ws. Deterministic input (LCG), so the host reference
 * build must produce the same checksum bit for bit. */
enum {
    K_NOP = 0,
    K_RFFT_F64, K_RFFT_F32, K_IRFFT_F64, K_IRFFT_F32,
    K_RS_ENC, K_RS_DEC_CLEAN, K_RS_DEC_ERR,
    K_DOT_F32, K_DOT_F64, K_FIR_Q15, K_CMUL_F32,
    K_COUNT
};
unsigned kernel_run(unsigned id, unsigned param, unsigned reps, unsigned *ws);

/* per-precision FFT kernels (kern_fft.c compiled twice) */
unsigned fftk_f64(int inverse, int n, unsigned reps, unsigned *ws);
unsigned fftk_f32(int inverse, int n, unsigned reps, unsigned *ws);
unsigned cmulk_f32(int n, unsigned reps, unsigned *ws);

unsigned ev_fnv(unsigned h, const void *p, unsigned n);
unsigned ev_rand(void);
void ev_seed(unsigned s);
extern unsigned char ev_arena[];
#define EV_ARENA_BYTES (160u * 1024u)
#endif
